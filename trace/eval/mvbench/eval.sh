#!/bin/bash

DIR="yourpath/projects/Trace/eval_results"
MODEL_DIR="yourpath/trace_vllava/sft_v3_128_v4_sep_final_v5"

TASK='mvbench'
ANNO_DIR='data/MVBench/json'
VIDEO_DIR='data/MVBench/video'


NUM_FRAME=128
OUTPUT_DIR=${DIR}/${TASK}-ori

python yourpath/projects/Trace/trace/eval/mvbench/evaluate.py --anno_path ${ANNO_DIR} --video_path ${VIDEO_DIR} \
--output_dir ${OUTPUT_DIR} --num_frames ${NUM_FRAME} \
--model_path ${MODEL_DIR}