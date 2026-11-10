#!/bin/bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29530
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200

export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export MEMORY_FRAGMENTATION=1
export ENCRYPTION_WEIGHT=false
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export INF_NAN_MODE_FORCE_DISABLE=1

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_NPUS=8

export TASK_NAME="minimax_h3_t2va"
export H3_VAE_BACKEND="lightning"
export INF_VRAM_BLOCKS_NUM=1

torchrun --nproc_per_node=$N_NPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT ../infer.py \
         --model MiniMax-H3 \
         --pretrained_model_name_or_path ../weights/MiniMax-H3 \
         --task_type t2va \
         --save_path ./${TASK_NAME}.mp4 \
         --num_inference_steps 50 \
         --width 1344 \
         --height 768 \
         --frames 124 \
         --seed 42 \
         --save_fps 24 \
         --prompt "integrated_multimodal_description: [Shot 1] A small robot walks through a sunlit garden. The camera follows slowly.\noverall_soundscape: Gentle footsteps and birdsong.\nnon_diegetic_music: Soft instrumental music." \
         --matmul_quant int8 \
         --vae_matmul_quant int8 \
         --atten laser \
         --rope_fused \
         --h3_vae_backend "$H3_VAE_BACKEND" \
         --text_encoder_tp $N_NPUS \
         --adaln_cache_file ../weights/MiniMax-H3/adaln_cache/adaln_cache_51step.safetensors \
         --transformer_path ../weights/MiniMax-H3/transformer_adaln_cache \
         --sp $N_NPUS \
         --phaa_num 7
