FPS_MAX_FRAMES=64 VIDEO_MAX_PIXELS=50176 \
export NPROC_PER_NODE=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model c \
    --train_type lora \
    --dataset  dataset/train.jsonl  \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 64 \
    --lora_alpha 16 \
    --freeze_vit True \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --eval_steps 50 \
    --save_steps 900 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --output_dir save_qwenvl3 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --use_chat_template False \
    --max_length 200000

FPS_MAX_FRAMES=64 \
VIDEO_MAX_PIXELS=50176 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model models/sft/v2-20251215-091134/checkpoint-844-merged \
    --train_type lora \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.75 \
    --vllm_tensor_parallel_size 4 \
    --dataset dataset/grpo.jsonl  \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --eval_steps 1000 \
    --save_steps 500 \
    --learning_rate 1e-6 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --output_dir output_grpo_text \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_completion_length 4096 \
    --reward_funcs coderm \
    --external_plugins plugin.py \
    --num_generations 8 \
    --temperature 1.0 \
    --log_completions true \
    --async_generate false \
    --move_model_batches 16 \
    --offload_optimizer true \
    --offload_model true \
    --sleep_level 0 \
