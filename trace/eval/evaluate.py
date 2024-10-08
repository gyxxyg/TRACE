import copy
import os
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print('Using NPU')
except:
    # print(e)
    print('Using GPU')
import argparse
import traceback

import sys
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

from transformers import StoppingCriteria, StoppingCriteriaList
from math import ceil
from PIL import Image
import numpy as np
import torch.backends.cudnn as cudnn
import decord

decord.bridge.set_bridge('torch')
import logging
from torchvision.transforms.functional import InterpolationMode

from torchvision import transforms
import pdb
import json
from pathlib import Path
import time
import datetime
from tqdm import tqdm
import random

random.seed(1234)
# from utils.format_dvc import format_dvc_output
# from utils.format_tvg import format_tvg_output
# from utils.format_vhd import format_vhd_output


def read_txt(path):
    with open(path, "r") as fin:
        data = fin.readline().strip()
    return data


def load_data(args, anno_path, split=None):
    '''
    anno data example:
    {"annotations":
        [
            {
                "image_id": "xHr8X2Wpmno.mp4"
                ...
            },
            ...
        ]
    }
    '''
    file_path = os.path.join(anno_path, f'{split}.caption_coco_format.json')
    with open(file_path, 'r') as f:
        data = json.load(f)["annotations"]

    if args.debug:
        data = data[:10]
    return data


def merge_seg_caps(results):
    """merge mulple generated captions from a same video into paragraph."""
    merge_results = {}
    for jterm in results:
        vname = jterm["vname"]
        cap = jterm["generated_cap"]
        postfix = vname.split(".mp4")[-1]
        start_time, end_time = float(postfix.split("_")[-2]), float(postfix.split("_")[-1])
        vid = vname.split(".mp4")[0] + ".mp4"
        if vid not in merge_results:
            merge_results[vid] = []
        merge_results[vid].append({"timestamp": [start_time, end_time], "caption": cap})
    return merge_results


def save_result(args, output_dir, results, split_name='test', format=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_name = f'{args.dataset}_{split_name}_f{args.num_frames}_result.json'
    if args.timestamp:
        if args.timestamp_file != '':
            file_name = f'{args.dataset}_{split_name}_f{args.num_frames}_result_with_pred_timestamp.json'
        else:
            file_name = f'{args.dataset}_{split_name}_f{args.num_frames}_result_with_gt_timestamp.json'
    if args.debug:
        file_name = 'debug_' + file_name
    if format:
        file_name = 'fmt_' + file_name
    with open(os.path.join(output_dir, file_name), 'w') as f:
        json.dump(results, f)
    return


def get_timestamp_from_file(timestamp_file):
    timestamp = {}
    with open(timestamp_file, 'r') as f:
        data = json.load(f)
        for vid, vlist in data.items():
            timestamp[vid] = []
            for vterm in vlist:
                timestamp[vid].append(vterm["timestamp"])
    return timestamp


def format_dvc(datas):
    fmt_datas = {}
    timestamp_count = []
    cnt = 0
    for i, jterm in enumerate(datas):
        vid = jterm["vname"]
        caption = jterm["generated_cap"]
        timestamps, sents = format_dvc_output(caption)
        if len(timestamps) == 0:
            cnt += 1
            print(vid, caption)
        fmt_datas[vid] = []
        for j in range(len(timestamps)):
            fmt_datas[vid].append({"timestamp": timestamps[j], "caption": sents[j]})
        timestamp_count.append(len(timestamps))
    print(f"predict avg {sum(timestamp_count) / len(timestamp_count)} events per video")
    print(f'parse failed number: {cnt}')
    return fmt_datas


def format_tvg(datas):
    fmt_datas = {}
    cnt = 0
    for i, jterm in enumerate(datas):
        vid = jterm["vname"]
        query = jterm["query"]
        gcap = jterm["generated_cap"]
        qid = int(jterm["id"])
        timestamps = format_tvg_output(gcap)
        if len(timestamps) == 0:
            cnt += 1
            print(vid, query + "\n", gcap, "\n")
        fmt_datas[qid] = {"timestamp": timestamps, "query": query, "vid": vid}
    print(f'parse failed number: {cnt}')
    return fmt_datas


def format_vhd(datas, gts):
    vid2gts = {}
    for jterm in gts:
        vid2gts[jterm["image_id"]] = jterm
    fmt_datas = []
    cnt = 0
    for i, jterm in enumerate(datas):
        vid = jterm["vname"]
        query = jterm["query"]
        gcap = jterm["generated_cap"]
        qid = jterm["id"]
        highlights, clipscores = format_vhd_output(gcap, vid2gts[vid])
        if len(highlights) == 0:
            cnt += 1
            print(vid, query + "\n", gcap + "\n")
            # pdb.set_trace()
        else:
            # print(gcap)
            # print(timestamps)
            pass
        result = {}
        result["qid"] = qid
        result["query"] = query
        result["vid"] = vid
        result["pred_saliency_scores"] = clipscores
        fmt_datas.append(result)
    print(f'parse failed number: {cnt}')
    return fmt_datas


def generate(chat, gr_videos, user_messages, num_beams, temperature, top_p, n_frms, chat_states=None, img_lists=None):
    N = len(user_messages)
    if chat_states is None:
        chat_states = []
        for i in range(N):
            if args.model_type == 'vicuna':
                chat_state = default_conversation.copy()
            else:
                chat_state = conv_llava_llama_2.copy()
            chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
            chat_states.append(chat_state)
    if img_lists is None:
        img_lists = [[] for i in range(N)]
        llm_message = chat.upload_video_without_audio(gr_videos, chat_states, img_lists, n_frms=n_frms)

    for user_message, chat_state in zip(user_messages, chat_states):
        chat.ask(user_message, chat_state)

    # print(len(img_lists[0][0][0]))

    responses = chat.answer(convs=chat_states,
                            img_lists=img_lists,
                            num_beams=num_beams,
                            temperature=temperature,
                            top_p=top_p,
                            max_new_tokens=1024,
                            max_length=3000)[0]
    return responses, chat_states, img_lists


def main(args):
    num_beams = 1
    temperature = args.temperature
    top_p = args.top_p
    n_frms = args.num_frames
    eval_start_time = time.time()
    prompt = read_txt(args.prompt_file)

    # load model
    device = torch.device(int(args.gpu_id))
    args.options = []

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # set after init_distributed_mode() to only log on master.

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, None, model_name)
    model = model.to(device)
    # conv_mode = 'mistral_instruct'
    conv_mode = 'llama_2'
    # conv_mode = 'v1'
    print('Initialization Finished')

    # load data
    video_path = args.video_path
    anno_path = args.anno_path
    anno_data = load_data(args, anno_path, split=args.split)
    print('Load Annotation Finished')
    if args.timestamp_file != '':
        pred_timestamps = get_timestamp_from_file(args.timestamp_file)
    vids = []
    vnames = []
    captions = []
    qids = []
    missing_videos = []
    if args.sample_num > 0:
        # sample part data to evaluate
        anno_data = random.sample(anno_data, args.sample_num)
    for jterm in anno_data:
        vname = jterm["image_id"].split("/")[-1]
        vid_path = os.path.join(video_path, vname)
        # print(vid_path)
        if not os.path.exists(vid_path):
            missing_videos.append(vid_path)
            continue
        if args.timestamp:
            duration = int(jterm["duration"])
            if args.timestamp_file == '':  # input the gt timestamps
                timestamp = jterm["segments"]
            else:  # input the pred timestamps
                timestamp = pred_timestamps[vname]
            for (start_time, end_time) in timestamp:
                # process anno timestamp error
                if start_time >= end_time or end_time > duration or start_time >= duration:
                    continue
                vids.append(vid_path)
                vnames.append(vname + "_" + str(start_time) + "_" + str(end_time))
                # image_emb, _ = model.encode_img(video)
                # img_lists.append([image_emb])
        else:
            vids.append(vid_path)
            vnames.append(vname)
            captions.append(jterm["caption"])
            qids.append(jterm["id"])

    print(f'Parse Data Done, {len(vnames)} videos')

    results = []
    bz = args.batch_size
    # evaluate using batch
    epoch = ceil(len(vnames) / bz)
    # epoch = 1
    for i in tqdm(range(epoch)):
        question = ''
        # load video
        path = vids[i]
        image_id = qids[i]
        final_prompt = copy.deepcopy(prompt)
            # final_prompt = f'{final_prompt} Transcribed speech: {final_asr}'
        if args.task in ["tvg", "vhd"]:
            # prompts.append(final_prompt.format(args.dataset, captions[idx].strip('.')))
            question = final_prompt.format(captions[i].strip())
        else:
            question = final_prompt

        if i < 5:
            print(question)

        try:
            tensor, video_timestamps = process_video(path, processor, model.config.image_aspect_ratio, n_frms)
            tensor = tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
            # print(tensor.shape)
            default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
            modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
            tensor = [tensor]
            video_timestamps = [video_timestamps]
            heads = [1]
            modal_list = ['video']
            # print(tensor.shape)

            # 3. text preprocess (tag process & generate prompt).
            question = default_mm_token + "\n" + question
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            cur_prompt = conv.get_prompt()
            cur_prompt += '<sync>'
            # print(cur_prompt)
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
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    # stopping_criteria=[stopping_criteria],
                    pad_token_id=tokenizer.eos_token_id,
                    video_timestamps=video_timestamps,
                    heads=heads
                )

            # print(output_ids)
            outputs = {
                'video': path,
                'id': image_id,
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
                results.append(outputs)
            print(outputs)
        except Exception as e:
            traceback.print_exc()
            print(f'generate for video {path} failed')
            break
            # continue

            # with open(output_file, 'a') as f:
            #     print(json.dumps(results[-1]), file=f, flush=True)

    save_result(args, args.output_dir, results, args.split, format=True)

    total_time = time.time() - eval_start_time
    # convert seconds to date
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))

    # with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
    #     f.write(json.dumps(cfg.to_dict(), indent=4) + "\n")
    #     f.write(message + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='eval_configs/vtgllm.yaml')
    parser.add_argument('--anno_path', type=str, default='data/YouCook2-BB/YouCook2_asr_denseCap/')
    parser.add_argument('--video_path', type=str, default='data/YouCook2-BB/YouCook2_asr_denseCap/youcook2_6fps_224/')
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--task',
                        default='dvc')  # dvc for dense video captioning; tvg for temporal video grounding; vhd for video highlight detection
    parser.add_argument('--dataset', default='youcook')
    parser.add_argument('--output_dir', default='debug')
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--timestamp', action='store_true', help='input the gt/predicted timestamps to the model')
    parser.add_argument('--timestamp_file', type=str, default='', help='the predcited timestamps file')
    parser.add_argument('--debug', action='store_true', help='the debug mode will only use 10 data samples')
    parser.add_argument('--prompt_file', default='prompts/dvc_description.txt')
    parser.add_argument('--model_path',
                        default='ckpt/vtgllm/train_stage2_llama2_7b_time64k_valley72k_bz32_f96_epoch3_open_i_instruct_qformer_lora_bind_time_ws32_mfp96_mtl2048/20231026060/checkpoint_2.pth')
    parser.add_argument('--sample_num', type=int, default=-1, help='fast inference by sampling N instances to evaluate')
    parser.add_argument('--example_output', action='store_true', help='output the example results')
    parser.add_argument('--no_lora', action='store_true')
    parser.add_argument('--post_check', action='store_true', help='post check the format of generated captions')
    parser.add_argument('--post_check_prompt_file', type=str, default='prompts/dvc_post_check.txt')
    parser.add_argument('--asr', action='store_true')
    parser.add_argument('--asr_path', type=str,
                        default='data/YouCook2-BB/YouCook2_asr_denseCap/whisper_outputs_with_time/small.en.cleaned/')
    args = parser.parse_args()
    main(args)
