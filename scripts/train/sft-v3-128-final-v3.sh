#!/bin/bash

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=16
MASTER_ADDR="127.0.0.1"
MASTER_PORT=16666
RANK=0

# Training Arguments
GLOBAL_BATCH_SIZE=128
GRADIENT_ACCUMULATION_STEPS=2
LOCAL_BATCH_SIZE=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$GRADIENT_ACCUMULATION_STEPS)]
echo $LOCAL_BATCH_SIZE

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=trace_vllava
export NCCL_P2P_LEVEL=NVL
export HCCL_BUFFSIZE=1024
RUN_NAME=trace_vllava
DATA_DIR=datasets
OUTP_DIR=yourpath/

ASCEND_LAUNCH_BLOCKING=1 torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE  \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    yourpath/projects/Trace/trace/train_mt_npu.py \
    --deepspeed yourpath/projects/Trace/scripts/zero3.json \
    --version v1_mistral \
    --vision_tower model/clip-vit-large-patch14-336 \
    --mm_projector_type spatial_slot \
    --freeze_mm_mlp_adapter False \
    --tune_mm_mlp_adapter True \
    --tune_mm_embed_head True \
    --tune_lm_embed_head True \
    --model_name_or_path yourpath/trace_vllava/stage1_v4_128 \
    --data_path yourpath/data/VTG-MD-IT/vtg-it/stage-2-v5-shorten.json \
    --data_folder data/ \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --downsample_num 1 \
    --image_aspect_ratio pad \
    --freeze_backbone False \
    --num_frames 128 \
    --bf16 True \
    --tf32 False \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/sft_v3_128_v4_sep_final_v5 \
    --num_train_epochs 2 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 99 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --run_name $RUN_NAME \
    --lazy_preprocess True \
    --sample_scheme "rand" \
    2> ${OUTP_DIR}/${WANDB_PROJECT}/log_128_sep_final_v5.err
    # --report_to tensorboard \
