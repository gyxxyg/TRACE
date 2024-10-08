#!/bin/bash

DIR="yourpath/projects/Trace"
MODEL_DIR="yourpath/trace_vllava/sft_v3_128_v4_sep_final_v5"
# MODEL_DIR="trace_vllava/stage1_v4/"
# MODEL_DIR="trace_vllava/stage2_sft_v3_64/checkpoint-600"

# TASK='dvc'
# ANNO_DIR='data/TimeIT/data/dense_video_captioning/anet'
# VIDEO_DIR='data/activitynet/videos'
# DATASET='activitynet'
# SPLIT='test'
# PROMPT_FILE="projects/Trace/trace/prompts/dvc-anet.txt"
# # PROMPT_FILE="projects/VTG-LLM/prompts/${TASK}-1.txt"
# GT_FILE="${ANNO_DIR}/${SPLIT}.caption_coco_format.json"

TASK='dvc'
ANNO_DIR='yourpath/data/VTG-IT/dense_video_caption/Youcook2'
VIDEO_DIR='data/youcook2/raw_videos/validation'
DATASET='youcook'
SPLIT='val'
PROMPT_FILE="yourpath/projects/VTG-LLM/prompts/${TASK}-1.txt"
GT_FILE="${ANNO_DIR}/${SPLIT}.caption_coco_format.json"


# TASK='tvg'
# ANNO_DIR="data/VTG-IT/moment_retrieval/Charades"
# VIDEO_DIR="data/charades/videos"
# DATASET='charades'
# SPLIT='test'
# PROMPT_FILE="projects/Trace/trace/prompts/mr.txt"
# GT_FILE="${ANNO_DIR}/${SPLIT}.caption_coco_format.json"

# TASK='tvg'
# ANNO_DIR="data/TimeIT/data/temporal_video_grounding/anet"
# VIDEO_DIR="data/activitynet/videos"
# DATASET='activitynet'
# SPLIT='test'
# PROMPT_FILE="projects/Trace/trace/prompts/mr.txt"
# GT_FILE="${ANNO_DIR}/${SPLIT}.caption_coco_format.json"

# TASK='vhd'
# ANNO_DIR='data/VTG-IT/video_highlight_detection/QVHighlights'
# VIDEO_DIR='data/qvhighlights/videos/val'
# DATASET='qvhighlights'
# SPLIT='val'
# PROMPT_FILE="projects/VTG-LLM/prompts/vhd-1.txt"
# GT_FILE="data/TimeIT/data/video_highlight_detection/qvhighlights/annotations_raw/highlight_val_release.jsonl"

NUM_FRAME=128
OUTPUT_DIR=${DIR}/eval_results/${TASK}_trace_embed_128
CFG_PATH=""


CUDA_VISIBLE_DEVICES=0 python yourpath/projects/Trace/trace/eval/evaluate.py --anno_path ${ANNO_DIR} --video_path ${VIDEO_DIR} --gpu_id 0 \
--task ${TASK} --dataset ${DATASET} --output_dir ${OUTPUT_DIR} --split ${SPLIT} --num_frames ${NUM_FRAME} --batch_size 1 \
--prompt_file ${PROMPT_FILE} --model_path ${MODEL_DIR}

python yourpath/projects/Trace/trace/eval/reformat_${TASK}.py --pred_file "${OUTPUT_DIR}/fmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.json" --out_file "${OUTPUT_DIR}/refmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.json"

cd yourpath/projects/Trace/trace/metrics/${TASK}
python eval_${TASK}.py --pred_file "${OUTPUT_DIR}/refmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.json" --gt_file ${GT_FILE} | tee "${OUTPUT_DIR}/fmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.txt"
cd ../..


# cd projects/Trace/trace/metrics/${TASK}
# python eval_${TASK}.py --pred_file "projects/VTG-LLM/tvg_videollama_slot_fm96_s256_fmt/fmt_anet_test_f96_result.json" --gt_file ${GT_FILE} | tee "${OUTPUT_DIR}/fmt_${DATASET}_${SPLIT}_f${NUM_FRAME}_result.txt"
# cd ../..
