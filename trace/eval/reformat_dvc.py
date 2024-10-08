import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str, default='/home/yaolinli/code/Ask-Anything/video_chat/results/eval_7b_instruct1.2k_youcook2_bz8_f8_epoch3_val.json')
parser.add_argument('--out_file', type=str, default='/home/yaolinli/code/Ask-Anything/video_chat/results/eval_7b_instruct1.2k_youcook2_bz8_f8_epoch3_val.json')
args = parser.parse_args()

with open(args.pred_file, 'r') as f:
    pred_data = json.load(f)

fmt_data = {}

for item in pred_data:
    new_item = []
    for time, caption in zip(item['timestamps'], item['captions']):
        if len(time) != 2:
            continue
        new_item.append({
            'caption': caption,
            'timestamp': time
        })
    fmt_data[item['video'].split('/')[-1]] = new_item

with open(args.out_file, 'w+') as f:
    json.dump(fmt_data, f)