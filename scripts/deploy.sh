TORCH_SYMM_MEM_DISABLE_MULTICAST=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
FPS_MAX_FRAMES=64 \
VIDEO_MAX_PIXELS=50176 \
swift deploy \
--model /inspire/hdd/global_user/lichenglin-253208540324/model/videopro_grpo-5090-merged \
    --infer_backend vllm \
    --torch_dtype bfloat16 \
    --port 8007 \
    --vllm_tensor_parallel_size 4 \
    --served_model_name "qwen3vl" 
