cd /mnt/nfs/wmd/research/IAD-R1-main

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline
export DISABLE_VERSION_CHECK=1
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=/mnt/nfs/wmd/research/IAD-R1-main/train/stage_sft
export MASTER_PORT=$(/mnt/nfs/wmd/conda_envs/sft/bin/python - <<'PY'
import socket
s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()
PY
)

/mnt/nfs/wmd/conda_envs/sft/bin/torchrun \
  --nproc_per_node=2 \
  --master_port "${MASTER_PORT}" \
  /mnt/nfs/wmd/research/IAD-R1-main/train/stage_sft/train.py \
  --stage sft \
  --do_train \
  --model_name_or_path "/mnt/nfs/wmd/model/IAD-R1-UPDATE/IAD-R1(Qwen2.5-VL-Instruct-3B)" \
  --dataset_dir /mnt/nfs/wmd/research/MemRefine-IAD/agent/real_build \
  --dataset MemRefine_sft_qwen \
  --image_dir / \
  --template qwen2_vl \
  --finetuning_type lora \
  --output_dir /mnt/nfs/wmd/model/MemRefine/MemRefine-IAD_Qwen2.5-VL-3B_SFT_v1 \
  --overwrite_cache \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --logging_steps 5 \
  --save_steps 50 \
  --save_total_limit 3 \
  --num_train_epochs 2 \
  --cutoff_len 4096 \
  --bf16 \
  --ddp_find_unused_parameters False
