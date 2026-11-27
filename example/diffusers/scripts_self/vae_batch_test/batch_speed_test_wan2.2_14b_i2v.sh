SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29505

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MEMORY_FRAGMENTATION=1
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export TOKENIZERS_PARALLELISM=false

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

N_NPUS=8
torchrun --nproc_per_node=$N_NPUS --master_addr $MASTER_ADDR --master_port $MASTER_PORT "$REPO_ROOT/tools/evaluate_vae.py" \
         --model Wan2.2-I2V-A14B \
         --pretrained_model_name_or_path ../../weights/Wan-AI/Wan2.2-I2V-A14B-Diffusers \
         --task_type i2v \
         --num_inference_steps 40 \
         --sp $N_NPUS \
         --fsdp \
         --batch_input_dir aigc_i2v_dataset/ \
         --eval_mode speed \
         --eval_resolutions 832x480,1280x720 \
         --eval_frames 81,49,17 \
         --eval_targets none,encoder,decoder,encoder_and_decoder \
         --batch_sample_size 1 \
         --eval_report_dir speed_test_results/ \
         --atten sage \
         --matmul_quant int8 \
         --rope_fused \
         --guidance_scale 3.5 \
         --guidance_scale_2 3.5 \
         --seed 42 \
         --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

