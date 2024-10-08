import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str, default='/home/yaolinli/code/Ask-Anything/video_chat/results/eval_7b_instruct1.2k_youcook2_bz8_f8_epoch3_val.json')
parser.add_argument('--out_file', type=str, default='/home/yaolinli/code/Ask-Anything/video_chat/results/eval_7b_instruct1.2k_youcook2_bz8_f8_epoch3_val.json')
parser.add_argument('--gtpath', default="/cfs/cfs-lugcocyb/yongxinguo/data/TimeIT/data/video_highlight_detection/qvhighlights/val.caption_coco_format.json")
args = parser.parse_args()

def read_json(path):
    with open(path, "r") as fin:
        datas = json.load(fin)
    return datas

def format_vhd_output(timestamps, scores, gts):
    # map timestamps and scores to clip ids
    print('*' * 50)
    print(timestamps)
    print(scores)
    print(gts)
    print('*' * 50)
    gt_duration = gts["duration"]
    clip_num = int(gt_duration/2)
    clip_scores = []
    cid2score = np.zeros(clip_num)
    cid2num = np.zeros(clip_num)
    for time, score in zip(timestamps, scores):
        if len(time) == 0 or len(score) == 0:
            continue
        t, s = time[0], score[0]
        if t > gt_duration:
            continue
        clip_id = max(0, int(t/2) - 1)
        cid2score[clip_id] += s
        cid2num[clip_id] += 1
    for cid in range(clip_num):
        if cid2num[cid] == 0:
            clip_scores.append(0.0)
        else:
            clip_scores.append(cid2score[cid]/cid2num[cid])
    # print('*' * 50)
    # print(paras)
    # print(highlights)
    # print(clip_scores)
    # print('*' * 50)
    return clip_scores


gts = read_json(args.gtpath)["annotations"]
vid2gts = {}
for jterm in gts:
    vid2gts[jterm["image_id"]] = jterm

with open(args.pred_file, 'r') as f:
    pred_data = json.load(f)

fmt_data = []

for item in pred_data:
    new_item = {
            'query': item['captions'][0],
            'vid': item['video'].split('/')[-1],
            'qid': item['id']
        }
    if len(item['scores']) == 0 or len(item['timestamps']) == 0:
        scores, timestamps = [], []
    else:
        scores = item['scores']
        timestamps = item['timestamps']
    if len(scores) < len(timestamps):
        scores += [0.0] * (len(timestamps) - len(scores))
    else:
        scores = scores[:len(timestamps)]

    print(scores, timestamps)
    
    clip_scores = format_vhd_output(timestamps, scores, vid2gts[new_item['vid']])
    new_item["pred_saliency_scores"] = clip_scores
    
    print(new_item)
        
    fmt_data.append(new_item)

with open(args.out_file, 'w+') as f:
    json.dump(fmt_data, f)