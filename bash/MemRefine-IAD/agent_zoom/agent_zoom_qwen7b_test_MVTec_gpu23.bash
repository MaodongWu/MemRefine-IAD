nohup env CUDA_VISIBLE_DEVICES=2,3 python /mnt/nfs/wmd/research/MemRefine-IAD/scripts/run_qwen_iadr1_zoom_eval.py \
  --model-path /mnt/nfs/wmd/model/Qwen2.5-VL-7B-Instruct \
  --name agent_zoom_qwen7b \
  --test_dataset test_MVTec \
  --few_shot_model 0 \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.90 \
  --batch_size 16 \
  --concurrency 16 \
  --temperature 0.7 \
  --top_p 0.8 \
  --max_tokens 512 \
  --data_path /mnt/nfs/wmd/data/Industrial_test_real \
  --test_json_root /mnt/nfs/wmd/research/IAD-R1-main/data/Test \
  --result_root /mnt/nfs/wmd/research/IAD-R1-main/result \
> /mnt/nfs/wmd/research/MemRefine-IAD/logs/agent_zoom_qwen7b_test_MVTec_gpu23_c16.log 2>&1 &