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


class MME_dataset(Dataset):
    def __init__(self, data_prefix, anno_path, num_segments=16, resolution=224, max_subtitle_len=4096):
        self.data_prefix = data_prefix
        with open(anno_path, 'r') as f:
            self.data_list = json.load(f)
            
        self.num_segments = num_segments
        self.max_subtitle_len = max_subtitle_len
        
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
        task_dict = {}
        total = 0
        for data in self.data_list:
            if data['duration_category'] not in ans_dict:
                task_dict[data['duration_category']] = {}
            for q in data['questions']:
                if q['task_type'] not in ans_dict[data['duration_category']]:
                    ans_dict[data['duration_category']][q['task_type']] = 0
                ans_dict[data['duration_category']][q['task_type']] += 1
                total += 1

        res = f"There are {len(self.data_list)} videos.\n"
        res += f"There are {total} QAs.\n"
        for k, v in task_dict.items():
            res += f"------{k}------\n"
            for kk, vv in task_dict.items():
                res += f"{kk}: {vv}\n"
                
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

    def read_frame(self, video_path, bound=None):
        video_path = os.path.join(video_path, str(self.num_segments))
        
        if os.path.exists(video_path):
            frame_list = [p for p in os.listdir(video_path)]
        else:
            raise Exception
            
        images_group = list()
        
        for frame_name in frame_list:
            img = Image.open(os.path.join(video_path, frame_name))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices, msg = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group).view(128, 3, 336, 336)
        return torch_imgs, msg

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer = f"({answer}) {data['options'][ord(answer) - ord('A')][3:]}"
        for idx, c in enumerate(data['options']):
            cur_choice, cur_text = c[0], c[3:]
            question += f"({cur_choice}) {cur_text}\n"
        question = question.rstrip()
        return question, answer

    def __getitem__(self, idx):
        video_name = self.data_list[idx]['url'].split("watch?v=")[1]
        video_path = os.path.join(self.data_prefix, "data", video_name + '.mp4')

        # We store the videos with only 16 or 32 frames for testing,
        # since directly reading the whold videos cost a lot of time.
        # You can also read the whole video via self.read_video(video_path)
        try:
            torch_imgs, msg = self.read_video(video_path)
        except:
            print(f"Error in {video_path}")
            return None
        print(self.data_list[idx])
        duration_category = self.data_list[idx]['duration']
        qa_list = []
        qa_list.append(self.qa_template(self.data_list[idx]))

        subtitle = ""
        try:
            subtitle_path = os.path.join(self.data_prefix, "subtitle", video_name + ".vtt")
            if os.path.exists(subtitle_path):
                subtitle = read_vtt_and_concatenate(subtitle_path, model.mistral_tokenizer, self.max_subtitle_len)
        except Exception:
            subtitle = ""
            print(f"Error for {subtitle_path}")
                
        return {
            'subtitle': subtitle,
            'video': torch_imgs, 
            'qa_list': qa_list,
            'duration_category': duration_category,
            'msg': msg
        }

def infer_mme(
        args,
        model,
        data_sample, 
        conv_mode,
        tokenizer,
        processor,
        system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False,
        add_subtitle=False,
        n_frms=128
    ):

    subtitle = f"This video's subtitles are listed below: {data_sample['subtitle']}"
    default_mm_token = f'{subtitle}\n' + DEFAULT_MMODAL_TOKEN["VIDEO"] if (add_subtitle and data_sample['subtitle'] != '') else DEFAULT_MMODAL_TOKEN["VIDEO"]
    modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
    tensor = data_sample['video']
    # model = model.to(dtype=torch.float32)
    tensor = tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
    video_timestamps = data_sample['msg']
    tensor = [tensor]
    video_timestamps = [video_timestamps]
    heads = [1]
    modal_list = ['video']

    assert system_q == False, "do not support system_q now"
    # video = data_sample["video"]
    # TC, H, W = video.shape
    # video = video.reshape(1, TC//3, 3, H, W).to("cuda:0")
    
    # video_list = []
    # with torch.no_grad():
    #     if system_q:
    #         raise NotImplementedError
    #     else:
    #         video_emb, _ = model.encode_img(video, system)
    # video_list.append(video_emb)

    pred_list = []
    gt_list = []
    for idx, qa in enumerate(data_sample['qa_list']):
        print(f"----------qa_{idx}---------", flush=True)
        if system_llm:
            question = default_mm_token + "\n" + system + qa[0] + question_prompt
        else:
            question = default_mm_token + "\n" + qa[0] + question_prompt
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
        # remove potential explanation
        llm_message = outputs['captions'][0]
        pred_option = re.findall(r'\(*\s*([a-z])\s*[\).]', llm_message.lower())
        pred_option = pred_option[0]
        pred_list.append(pred_option)
        gt_list.append(qa[1][1].lower())
        print(f"Pred: {pred_option}", flush=True)
        print(f"GT: {qa[1][1]}", flush=True)
    return pred_list, gt_list


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

    dataset = MME_dataset(args.data_dir, args.anno_path, num_segments=args.num_frames, resolution=resolution)

    with open(args.anno_path, 'r') as f:
        res_json_data = json.load(f)

    correct = 0
    total = 0
    res_list = []
    acc_dict = {}

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for idx, example in enumerate(tqdm(dataset)):
        if example is None:
            continue
        try:
            duration_category = example['duration_category']
            if duration_category not in acc_dict:
                acc_dict[duration_category] = [0, 0] # correct, total
            qa_count = len(example['qa_list'])
            acc_dict[duration_category][1] += qa_count
            total += qa_count
            pred_list, gt_list = infer_mme(
                args,
                model,
                example, 
                conv_mode,
                tokenizer,
                processor,
                "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n",
                question_prompt="\nOnly give the best option.",
                answer_prompt="Best option:(",
                return_prompt='(',
                system_q=False,
                print_res=False,
                system_llm=True,
                # add_subtitle=True, # Comment this line to add subtitles, we use the whole subtitles by default.
            )
            res_list.append({
                'pred': pred_list,
                'gt': gt_list
            })
            qa_idx = 0
            for pred, gt in zip(pred_list, gt_list):
                if pred == gt:
                    acc_dict[duration_category][0] += 1
                    correct += 1
                res_json_data[idx]['response'] = pred
                qa_idx += 1
            print(f"Part  Acc: {acc_dict[duration_category][0] / acc_dict[duration_category][1] * 100 :.2f}%")
            print(f"Total Acc: {correct / total * 100 :.2f}%")
            print('-' * 50, duration_category, '-' * 50)
        except:
            continue

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
    parser.add_argument('--anno_path', type=str, default='/group/40065/data/Video-MME/test.json')
    parser.add_argument('--video_path', type=str, default='/home/v-shuhuairen/mycontainer/data/MVBench/video')
    parser.add_argument('--data_dir', type=str, default='/group/40065/data/Video-MME')
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