from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from transformers import BitsAndBytesConfig
import numpy
from config import Settings
import torch
import os

ACCESS_TOKEN = os.environ.get("HF_TOKEN", None)
# 量化模型
nf4_config = BitsAndBytesConfig(
    # 启用 4-bit 量化
    load_in_4bit=True,
    # 使用 NF4 格式 (更高效的 4-bit 量化格式，推荐)
    bnb_4bit_quant_type="nf4",
    # 双重量化 (减少平均内存占用，推荐)
    bnb_4bit_use_double_quant=True,
    # 4-bit 权重的数据类型 (推荐 BFloat16/Float16)
    # 如果你的GPU不支持 BF16，使用 FP16 或 Float16。
    bnb_4bit_compute_dtype=torch.float16,
)
# print(f"正在加载模型{Settings.HF_GEN_MODEL}...")

tokenizer = AutoTokenizer.from_pretrained(Settings.HF_GEN_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    Settings.HF_GEN_MODEL,
    quantization_config=nf4_config,
    device_map="auto",
    torch_dtype=torch.float16,
    token=ACCESS_TOKEN,
)
pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)


def generate_local(prompt: str, system_prompt: str = "", max_new_tokens=2048) -> str:
    """
    使用llm推理
    """
    full_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(full_input, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant\n" in response:
        return response.split("assistant\n")[-1].strip()
    return response
