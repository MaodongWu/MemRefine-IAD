nohup bash -lc '
for BENCH in  test_MVTec test_MPDD test_SDD; do
  echo "[RUN] ${BENCH}"
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=2 \
  /mnt/nfs/wmd/conda_envs/vllm/bin/python \
  /mnt/nfs/wmd/research/MemRefine-IAD/scripts/MemRefineIAD/run_llava_onevision_zoom_eval_memory.py \
    --model-path /mnt/nfs/wmd/model/MemRefine/MemRefine-IAD_LLaVA-OneVision-SI-7B_SFT_v1/merged-checkpoint-100 \
    --name agent_zoom_mem_MemRefine_LVOV_fixchat \
    --test_dataset "${BENCH}" \
    --data_path /mnt/nfs/wmd/data/Industrial_test_real \
    --test_json_root /mnt/nfs/wmd/research/IAD-R1-main/data/Test \
    --result_root /mnt/nfs/wmd/research/IAD-R1-main/result \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.90 \
    --max_model_len 16384 \
    --batch_size 16 \
    --concurrency 16 \
    --few_shot_model 0 \
    --normal_flag good \
    --enable_working_memory \
    --memory_max_chars 240 \
    --reproduce
done
' > /mnt/nfs/wmd/research/MemRefine-IAD/logs/agent_zoom_mem_MemRefine_LVOV_fixchat_remaining4_gpu2_c16.log 2>&1 &
