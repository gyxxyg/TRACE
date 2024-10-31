#!/bin/bash

DIR="yourpath/projects/Trace/eval_results"
MODEL_DIR="yourpath/trace_vllava/sft_v3_128_v4_sep_final_v5"

TASK='videomme'
ANNO_DIR='yourpath/data/Video-MME/test.json'
VIDEO_DIR='yourpath/data/Video-MME'


NUM_FRAME=128
OUTPUT_DIR=${DIR}/${TASK}

python yourpath/projects/Trace/trace/eval/videomme/evaluate.py --anno_path ${ANNO_DIR} --data_dir ${VIDEO_DIR} \
--output_dir ${OUTPUT_DIR} --num_frames ${NUM_FRAME} \
--model_path ${MODEL_DIR}