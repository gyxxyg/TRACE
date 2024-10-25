import os
import io
import json
import argparse
import torch
from PIL import Image
import torchvision
import numpy as np
import decord
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
from tqdm import tqdm
from accelerate import Accelerator
import random
import numbers
import torch.backends.cudnn as cudnn
import sys
import re
sys.path.append('yourpath/projects')
sys.path.append('yourpath/projects/Trace')
sys.path.append('yourpath/projects/Trace/trace')
sys.path.append('./')
from Trace.trace.conversation import conv_templates
from Trace.trace.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from Trace.trace.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, process_image
from Trace.trace.model.builder import load_pretrained_model
from Trace.trace.conversation import conv_templates, SeparatorStyle
from Trace.trace.mm_utils import process_video, tokenizer_MMODAL_token_all, get_model_name_from_path, KeywordsStoppingCriteria
import logging
from pathlib import Path

from transformers import StoppingCriteria, StoppingCriteriaList
from math import ceil

decord.bridge.set_bridge('torch')

from torchvision import transforms
import pdb
import json
import time
import datetime
from tqdm import tqdm

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2)
                                   for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1]
                                       for x in img_group], axis=2)
            else:
                #print(np.concatenate(img_group, axis=2).shape)
                # print(img_group[0].shape)
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(
                    pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class MVBench_dataset(Dataset):
    def __init__(self, processor, data_dir, data_list, num_segments=8, resolution=224):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })

        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }

        self.num_segments = num_segments
        self.processor = processor

        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std)
        ])

    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])

        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k] / option_list[k] * 100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct / total * 100:.2f}%"
        return res.rstrip()

    def __len__(self):
        return len(self.data_list)

    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        msg = [[f / fps] for f in frame_indices]
        return frame_indices, msg

    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        images_group = list()
        frame_indices, msg = self.get_index(bound, fps, max_frame, first_idx=0)

        # images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]

        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())
            images_group.append(img)
        # images_group = [expand2square(image, tuple(int(x*255) for x in self.processor.image_mean)) for image in images_group]
        # torch_imgs = self.processor.preprocess(images_group, return_tensors='pt')['pixel_values']
        torch_imgs = self.transform(images_group).view(128, 3, 336, 336)
        return torch_imgs, msg

    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1

        images_group = list()
        frame_indices, msg = self.get_index(bound, fps, max_frame, first_idx=0)
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        # images_group = [expand2square(image, tuple(int(x*255) for x in self.processor.image_mean)) for image in images_group]
        # torch_imgs = self.processor.preprocess(images_group, return_tensors='pt')['pixel_values']
        torch_imgs = self.transform(images_group).view(128, 3, 336, 336)
        return torch_imgs, msg

    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices, msg = self.get_index(bound, fps, max_frame, first_idx=1)  # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        torch_imgs = self.transform(images_group).view(128, 3, 336, 336)
        # images_group = [expand2square(image, tuple(int(x*255) for x in self.processor.image_mean)) for image in images_group]
        # torch_imgs = self.processor.preprocess(images_group, return_tensors='pt')['pixel_values']
        return torch_imgs, msg

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        question, answer = self.qa_template(self.data_list[idx]['data'])
        # return {
        #     'video': video_path,
        #     'question': question,
        #     'answer': answer,
        #     'task_type': self.data_list[idx]['task_type'],
        # }
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        try:
            torch_imgs, msg = decord_method(video_path, bound)
        except:
            print(f"Error in {video_path}")
            return None
        question, answer = self.qa_template(self.data_list[idx]['data'])

        return {
            'video': torch_imgs,
            'question': question,
            'answer': answer,
            'task_type': self.data_list[idx]['task_type'],
            'msg': msg
        }


def infer_mvbench(
        args,
        model,
        data_sample,
        conv_mode,
        tokenizer,
        processor,
        system="",
        question_prompt='',  # add in the end of question
        system_llm=False,
        n_frms=128
):
    # chat_state = conv_llava_llama_2.copy()
    # chat_state.system = system

    # tensor, video_timestamps = process_video(data_sample['video'], processor, model.config.image_aspect_ratio, n_frms)
    default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
    modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
    tensor = data_sample['video']
    # model = model.to(dtype=torch.float32)
    tensor = tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
    video_timestamps = data_sample['msg']
    tensor = [tensor]
    video_timestamps = [video_timestamps]
    heads = [1]
    modal_list = ['video']

    if system_llm:
        question = default_mm_token + "\n" + system + data_sample['question'] + question_prompt
    else:
        question = default_mm_token + "\n" + data_sample['question'] + question_prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    cur_prompt = conv.get_prompt()
    cur_prompt += '<sync>'
    print(cur_prompt)

    input_ids = tokenizer_MMODAL_token_all(cur_prompt, tokenizer, return_tensors='pt').unsqueeze(0).to('cuda')
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
    # keywords = ["<s>", "</s>"]
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    do_sample = False

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images_or_videos=tensor, 
            modal_list=modal_list,
            do_sample=do_sample,
            temperature=0.2 if do_sample else 0.0,
            max_new_tokens=128,
            use_cache=True,
            # stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            video_timestamps=video_timestamps,
            heads=heads
        )

    outputs = {
        'timestamps': [],
        'scores': [],
        'captions': [],
    }
    cur_timestamps = []
    cur_timestamp = []
    cur_scores = []
    cur_score = []
    cur_caption = []
    # print(output_ids)
    for idx in output_ids[0]:
        if idx <= 32000:
            if idx == 32000:
                new_caption = tokenizer.decode(cur_caption, skip_special_tokens=True)
                outputs['captions'].append(new_caption)
                cur_caption = []
                if stop_str in new_caption:
                    break
            else:
                cur_caption.append(idx)
        elif idx <= 32013: # 32001 <sync>; 32002 <sep>
            if idx == 32001:
                if len(cur_timestamp) > 0:
                    cur_timestamps.append(float(''.join(cur_timestamp)))
                outputs['timestamps'].append(cur_timestamps)
                cur_timestamps = []
                cur_timestamp = []
            elif idx == 32002:
                if len(cur_timestamp) > 0:
                    cur_timestamps.append(float(''.join(cur_timestamp)))
                cur_timestamp = []
            else:
                cur_timestamp.append(model.get_model().time_tokenizer.decode(idx - 32001))
        else: # 32014 <sync>; 32015 <sep>
            if idx == 32014:
                if len(cur_score) > 0:
                    cur_scores.append(float(''.join(cur_score)))
                outputs['scores'].append(cur_scores)
                cur_scores = []
                cur_score = []
            elif idx == 32015:
                if len(cur_score) > 0:
                    cur_scores.append(float(''.join(cur_score)))
                cur_score = []
            else:
                cur_score.append(model.get_model().score_tokenizer.decode(idx - 32014))
    if len(cur_caption):
        outputs['captions'].append(tokenizer.decode(cur_caption, skip_special_tokens=True))
    print(outputs)
    return outputs['captions'][0]
    # TC, H, W = video.shape
    # video = video.reshape(1, TC // 3, 3, H, W).permute(0, 2, 1, 3, 4).to("cuda:0")  # [b, c, t, h, w]

    # video_list = []
    # with torch.no_grad():
    #     if chat.model.qformer_text_input:
    #         # timestamp
    #         timestamps = msg.split('at')[1].replace('seconds.', '').strip().split(
    #             ',')  # extract timestamps from msg
    #         timestamps = [f'This frame is sampled at {t.strip()} second.' for t in timestamps]
    #         timestamps = chat.model.llama_tokenizer(
    #             timestamps,
    #             return_tensors="pt",
    #             padding="longest",
    #             max_length=32,
    #             truncation=True,
    #         )
    #     else:
    #         timestamps = None
    #     video_emb, _ = chat.model.encode_videoQformer_visual(video, timestamp=timestamps)
    # video_list.append(video_emb)
    # #     video_list.append(torch.zeros_like(video_emb))
    # chat_state.append_message(chat_state.roles[0], "<Video><ImageHere></Video> " + msg)

    # if system_llm:
    #     prompt = system + data_sample['question'] + question_prompt
    # else:
    #     prompt = data_sample['question'] + question_prompt

    # #print(f"prompt: {prompt}\n\n")
    # chat.ask(prompt, chat_state)

    # llm_message = chat.answer(conv=chat_state,
    #                           img_list=video_list,
    #                           num_beams=1,
    #                           temperature=args.temperature,
    #                           top_p=args.top_p,
    #                           max_new_tokens=512,
    #                           max_length=3000)[0]
    # print(f"chat_state: {chat_state}\n\n")
    # # remove potential explanation
    # llm_message = llm_message.strip().split('\n')[0]
    # print(llm_message)
    # print(f"GT: {data_sample['answer']}")
    # return llm_message


def check_ans(pred, gt):
    flag = False

    try:
        pred_list = pred.lower()
        pred_list = re.findall(r'\(*\s*([a-z])\s*[\).]', pred.lower())
        pred_option = '(' + pred_list[0] + ')'
        # pred_option, pred_content = f'({pred_list[0]})', ' '.join(pred_list[1:]).strip()
    except:  # random answer
        pred_option = '(a)'
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]

    print(pred_option, gt_option)
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True

    return flag


def main(args):
    correct = 0
    total = 0
    res_list = []
    acc_dict = {}

    # load model
    device = torch.device(f"cuda:{args.gpu_id}")
    args.options = []

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, None, model_name)
    model = model.to(device)
    # conv_mode = 'mistral_instruct'
    conv_mode = 'llama_2'

    # load data
    resolution = 336
    data_list = {
        "Action Sequence": ("action_sequence.json", f"{args.video_path}/star/Charades_v1_480/", "video", True),
        # has start & end
        "Action Prediction": ("action_prediction.json", f"{args.video_path}/star/Charades_v1_480/", "video", True),
        # has start & end
        "Action Antonym": ("action_antonym.json", f"{args.video_path}/ssv2_video/", "video", False),
        "Fine-grained Action": (
        "fine_grained_action.json", f"{args.video_path}/Moments_in_Time_Raw/videos/", "video", False),
        "Unexpected Action": ("unexpected_action.json", f"{args.video_path}/FunQA_test/test/", "video", False),
        "Object Existence": ("object_existence.json", f"{args.video_path}/clevrer/video_validation/", "video", False),
        "Object Interaction": ("object_interaction.json", f"{args.video_path}/star/Charades_v1_480/", "video", True),
        # has start & end
        "Object Shuffle": ("object_shuffle.json", f"{args.video_path}/perception/videos/", "video", False),
        "Moving Direction": ("moving_direction.json", f"{args.video_path}/clevrer/video_validation/", "video", False),
        "Action Localization": ("action_localization.json", f"{args.video_path}/sta/sta_video/", "video", True),
        # has start & end
        "Scene Transition": ("scene_transition.json", f"{args.video_path}/scene_qa/video/", "video", False),
        "Action Count": ("action_count.json", f"{args.video_path}/perception/videos/", "video", False),
        "Moving Count": ("moving_count.json", f"{args.video_path}/clevrer/video_validation/", "video", False),
        "Moving Attribute": ("moving_attribute.json", f"{args.video_path}/clevrer/video_validation/", "video", False),
        "State Change": ("state_change.json", f"{args.video_path}/perception/videos/", "video", False),
        "Fine-grained Pose": ("fine_grained_pose.json", f"{args.video_path}/nturgbd/", "video", False),
        "Character Order": ("character_order.json", f"{args.video_path}/perception/videos/", "video", False),
        "Egocentric Navigation": ("egocentric_navigation.json", f"{args.video_path}/vlnqa/", "video", False),
        "Episodic Reasoning": ("episodic_reasoning.json", f"{args.video_path}/tvqa/frames_fps3_hq/", "frame", True),
        # has start & end, read frame
        "Counterfactual Inference": (
        "counterfactual_inference.json", f"{args.video_path}/clevrer/video_validation/", "video", False),
    }

    dataset = MVBench_dataset(processor, args.anno_path, data_list, num_segments=args.num_frames, resolution=resolution)

    for example in tqdm(dataset):
        if example is None:
            continue
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0]  # correct, total

        try:
            pred = infer_mvbench(
                args,
                model,
                example,
                conv_mode,
                tokenizer,
                processor,
                system="Watch the video carefully, noticing the cause and sequence of events, and then choose the best option for the given question.\n",
                question_prompt="\nPlease think step by step and only give the best option that matches the question best.",
                system_llm=True,
                n_frms=args.num_frames
            )
        except:
            continue
        gt = example['answer']
        res_list.append({
            'pred': pred,
            'gt': gt
        })
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1
        acc_dict[task_type][1] += 1
        total += 1
        print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        print(f"Total Acc: {correct / total * 100 :.2f}%")
        print('-' * 30, task_type, '-' * 30)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.output_dir}/test.json", "w") as f:
        json.dump({
            "acc_dict": acc_dict,
            "res_list": res_list
        }, f)

    final_res = dict()
    correct = 0
    total = 0
    for k, v in acc_dict.items():
        final_res[k] = v[0] / v[1] * 100
        correct += v[0]
        total += v[1]
    final_res['Avg'] = correct / total * 100

    print(final_res)

    with open(f"{args.output_dir}/upload_leaderboard.json", "w+") as f:
        json.dump(final_res, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='../eval_configs/timechat.yaml')
    parser.add_argument('--anno_path', type=str, default='/home/v-shuhuairen/mycontainer/data/MVBench/json')
    parser.add_argument('--video_path', type=str, default='/home/v-shuhuairen/mycontainer/data/MVBench/video')
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--output_dir', default='debug')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--debug', action='store_true', help='the debug mode will only use 10 data samples')
    parser.add_argument('--model_path',
                        default='../ckpt/timechat/train_stage2_llama2_7b_time64k_valley72k_bz32_f96_epoch3_open_i_instruct_qformer_lora_bind_time_ws32_mfp96_mtl2048/20231026060/checkpoint_2.pth')
    args = parser.parse_args()
    accelerate = Accelerator()
    main(args)