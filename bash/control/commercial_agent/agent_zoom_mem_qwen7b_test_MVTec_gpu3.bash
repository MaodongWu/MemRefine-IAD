nohup env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3 \
/mnt/nfs/wmd/conda_envs/vllm/bin/python /mnt/nfs/wmd/research/MemRefine-IAD/scripts/run_qwen_iadr1_zoom_eval_memory.py \
  --model-path /mnt/nfs/wmd/model/Qwen2.5-VL-7B-Instruct \
  --data_path /mnt/nfs/wmd/data/Industrial_test_real \
  --test_json_root /mnt/nfs/wmd/research/IAD-R1-main/data/Test \
  --test_dataset test_MVTec \
  --name agent_zoom_mem_qwen7b \
  --result_root /mnt/nfs/wmd/research/IAD-R1-main/result \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.90 \
  --batch_size 8 \
  --concurrency 8 \
  --few_shot_model 0 \
  --normal_flag good \
  --enable_working_memory \
  --memory_max_chars 240 \
  > /mnt/nfs/wmd/research/MemRefine-IAD/logs/agent_zoom_mem_qwen7b_test_MVTec_gpu3_c8.log 2>&1 &
