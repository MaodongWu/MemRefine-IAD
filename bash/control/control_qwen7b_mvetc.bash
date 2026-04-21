nohup bash -lc '
cd /mnt/nfs/wmd/research/IAD-R1-main/scripts/Inference/Pretrain-Inference
for BENCH in test_MVTec test_DAGM test_DTD test_MPDD test_SDD test_VisA; do
  echo "[RUN] ${BENCH}"
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3 /mnt/nfs/wmd/conda_envs/vllm/bin/python vLLM_Qwen_detect.py \
    --batch_size 16 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.90 \
    --few_shot_model 0 \
    --test_dataset "${BENCH}" \
    --model-path /mnt/nfs/wmd/model/Qwen2.5-VL-3B-Instruct \
    --name control_qwen3b
done
' > /mnt/nfs/wmd/research/MemRefine-IAD/logs/control_qwen3b_remaining6_gpu3.log 2>&1 &
