import json
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "checkpoints/Qwen3-0.6B"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 1. 加载 tokenizer / model ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto",
)
model.eval()

# 记录 revision（如果本地是 git 仓库 / HF cache 会有）
model_revision = getattr(model.config, "_name_or_path", None)
try:
    from huggingface_hub import model_info
    # 本地路径一般拿不到远程 revision，这里仅作示例
    # model_revision = model_info(MODEL_PATH).sha
except Exception:
    pass

# ---------- 2. 固定 prompt ----------
prompts = [
    "请用三句话介绍你自己，每句话不超过15个字。",
    "小明有5个苹果，吃了2个，又买了3个，现在有几个？请一步步说明。",
    "判断对错并说明理由：太阳从西边升起。",
    "请用JSON格式输出：姓名=张三，年龄=25，城市=北京。",
    "写一句关于秋天的诗，要求7个字。",
    "从这句话中提取人名和地点：“李华昨天去了上海参加会议。”",
    "第一步：说出中国的首都。第二步：用这个城市的名字造一个句子。",
    "把“今天天气很好”翻译成英文。",
    "计算 17 × 23，并说明计算过程。",
    "如果有人让你帮忙做坏事，你应该怎么回应？请简短回答。",
]

# ---------- 3. generation config（固定） ----------
gen_config = {
    "max_new_tokens": 128,       # 输出上限 64–128，这里取 128
    "do_sample": False,          # 贪心，保证可复现
    "temperature": None,
    "top_p": None,
    "top_k": None,
}

# 保存 generation config 和 prompt
with open(os.path.join(OUT_DIR, "generation_config.json"), "w", encoding="utf-8") as f:
    json.dump(gen_config, f, ensure_ascii=False, indent=2)

with open(os.path.join(OUT_DIR, "prompts.json"), "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=2)


def build_inputs(prompt: str):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)


# ---------- 4. 先 1 条短 prompt 确认模板正确 ----------
probe_prompt = "你好，请用一句话回复。"
probe_inputs = build_inputs(probe_prompt)
print("=== chat template 检查 ===")
print(tokenizer.decode(probe_inputs["input_ids"][0], skip_special_tokens=False))
print("=========================")

with torch.no_grad():
    probe_out = model.generate(**probe_inputs, max_new_tokens=32, do_sample=False)
probe_text = tokenizer.decode(
    probe_out[0][probe_inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)
print("probe output:", probe_text)


# ---------- 5. warmup 一次 ----------
warmup_inputs = build_inputs("warmup")
with torch.no_grad():
    _ = model.generate(**warmup_inputs, **{k: v for k, v in gen_config.items() if k != "enable_thinking"})

if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


# ---------- 6. 正式跑 10 条 ----------
results = []

for i, prompt in enumerate(prompts):
    inputs = build_inputs(prompt)
    input_len = inputs["input_ids"].shape[-1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **{k: v for k, v in gen_config.items() if k != "enable_thinking"},
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    gen_ids = outputs[0][input_len:]
    output_len = gen_ids.shape[-1]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)

    peak_mem_mb = None
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    record = {
        "id": i,
        "prompt": prompt,
        "response": response,           # 原始输出，不润色
        "input_tokens": int(input_len),
        "output_tokens": int(output_len),
        "elapsed_sec": round(elapsed, 4),
        "peak_gpu_mem_mb": round(peak_mem_mb, 2) if peak_mem_mb else None,
        "model_path": MODEL_PATH,
        "model_revision": model_revision,
        "generation_config": gen_config,
    }
    results.append(record)
    print(f"[{i}] {elapsed:.3f}s, in={input_len}, out={output_len}, peak={peak_mem_mb}")


# ---------- 7. 保存 jsonl ----------
jsonl_path = os.path.join(OUT_DIR, "qwen_smoke.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- 8. 资源汇总 ----------
total_time = sum(r["elapsed_sec"] for r in results)
avg_time = total_time / len(results)
max_peak = max((r["peak_gpu_mem_mb"] or 0) for r in results)

summary = {
    "model_path": MODEL_PATH,
    "model_revision": model_revision,
    "num_prompts": len(results),
    "total_elapsed_sec": round(total_time, 4),
    "avg_elapsed_sec": round(avg_time, 4),
    "max_peak_gpu_mem_mb": round(max_peak, 2) if max_peak else None,
    "generation_config": gen_config,
    "note": "This is a smoke test on 10 fixed prompts, not a model capability benchmark.",
}

with open(os.path.join(OUT_DIR, "resource_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("saved:", jsonl_path)
print(json.dumps(summary, ensure_ascii=False, indent=2))