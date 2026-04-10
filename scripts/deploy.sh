TORCH_SYMM_MEM_DISABLE_MULTICAST=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
FPS_MAX_FRAMES=64 \
VIDEO_MAX_PIXELS=50176 \
swift deploy \
--model models/grpo/v6-20251215-121705/checkpoint-5090-merged \
    --infer_backend vllm \
    --torch_dtype bfloat16 \
    --port 8007 \
    --vllm_tensor_parallel_size 8 \
    --served_model_name "qwen3vl" 
