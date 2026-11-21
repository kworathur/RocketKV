import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CompressedTensorsConfig,
)

model_id = "RedHatAI/Llama-3.2-3B-Instruct-FP8"
output_dir = "./models/Llama-3.2-3B-FP8-dequantized"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    quantization_config=CompressedTensorsConfig(dequantize=True),
    attn_implementation="sdpa",
)

assert not hasattr(model.config, "quantization_config"), (
    "Quantization configuration still present"
)

model.save_pretrained(output_dir, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(output_dir)

print("Saved dequantized model to:", output_dir)
