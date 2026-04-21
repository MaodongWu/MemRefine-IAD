PYTHONNOUSERSITE=1 /mnt/nfs/wmd/conda_envs/sft/bin/python - << 'PY'
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from peft import PeftModel
import torch, os

base = "/mnt/nfs/wmd/model/IAD-R1-UPDATE/IAD-R1(LLaVA-OneVision-SI-7B)"
lora = "/mnt/nfs/wmd/model/MemRefine/MemRefine-IAD_LLaVA-OneVision-SI-7B_SFT_v1/checkpoint-100"
out  = "/mnt/nfs/wmd/model/MemRefine/MemRefine-IAD_LLaVA-OneVision-SI-7B_SFT_v1/merged-checkpoint-100"

os.makedirs(out, exist_ok=True)

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    base, torch_dtype=torch.bfloat16, trust_remote_code=True
)
model = PeftModel.from_pretrained(model, lora)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")

processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
processor.save_pretrained(out)

print("merged model saved to:", out)
PY
