nohup /mnt/nfs/wmd/conda_envs/vllm/bin/python /mnt/nfs/wmd/research/MemRefine-IAD/scripts/commercial/raw/run_api_commercial_raw_eval.py \
  --api_url https://api.vectorengine.ai/v1/chat/completions \
  --api_key_file /mnt/nfs/wmd/research/MemRefine-IAD/.dashscope_key \
  --model claude-sonnet-4-6 \
  --name memrefine_raw_claude_sonnet4_api \
  --run_all_benches \
  --reproduce \
  --concurrency 16 \
  --few_shot_model 0 \
  --data_path /mnt/nfs/wmd/data/Industrial_test_real \
  --test_json_root /mnt/nfs/wmd/research/IAD-R1-main/data/Test \
  --result_root /mnt/nfs/wmd/research/IAD-R1-main/result \
  > /mnt/nfs/wmd/research/MemRefine-IAD/logs/memrefine_raw_claude_sonnet4_api_all6_c16.log 2>&1 &
