# 1) 先看 GPU3 是否干净
nvidia-smi

# 2) 跑 all6（全新日志）
LOG=/mnt/nfs/wmd/research/MemRefine-IAD/logs/control_llavaonevision7b_all6_gpu3_retry_$(date +%Y%m%d_%H%M%S).log

nohup bash -lc '
set -e
cd /mnt/nfs/wmd/research/IAD-R1-main/scripts/Inference/IAD-R1-Inference

MODEL="/mnt/nfs/wmd/model/IAD-R1-UPDATE/IAD-R1(LLaVA-OneVision-SI-7B)"
PY="/mnt/nfs/wmd/conda_envs/vllm/bin/python"

for BENCH in test_DAGM test_DTD test_KSDD2 test_MTD test_MVTec test_VisA; do
  echo "[RUN] ${BENCH}"
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3 "$PY" vLLM_LLaVA_detect_format.py \
    --model-path "$MODEL" \
    --name control_llavaonevision7b \
    --test_dataset "$BENCH" \
    --data_path /mnt/nfs/wmd/data/Industrial_test_real \
    --test_json_root /mnt/nfs/wmd/research/IAD-R1-main/data/Test \
    --result_root /mnt/nfs/wmd/research/IAD-R1-main/result \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.85 \
    --max_model_len 4096 \
    --batch_size 8 \
    --few_shot_model 0 \
    --reproduce
done
' > "$LOG" 2>&1 &

echo "LOG=$LOG"
