---
tags:
  - SFT
  - 监督微调
created: 2026-10-14
updated: 2026-10-14
related:
---

# SFT（Supervised Fine-Tuning）

> 本文正式进入 LLM 后训练(Post-Training)的基础。Pre-Training让LLM学会了预测next token，而后训练对LLM进行行为对齐、偏好对齐、能力激活。SFT为后训练的基础，为后续的RLHF、DPO、DRPO等方法奠定了前提。

---

## 一、SFT

### 1.1 SFT是什么？

LLM训练通常分为三个阶段，Pre-Training、监督微调、强化学习。后两者可以归到Post-Training中。
Pre-Training赋予模型会说话的能力，预训练出来的模型叫Base Model，本质上是个续写器：
- 给它`今天天气`，它可能会续写 `真好，适合出去玩`
- 问它`今天天气怎么样？`，可能续写`今天天气怎么样？明天呢？`,也可能直接回答问题
- 它不会区分 “补全文本”和“回答问题”

SFT就是要解决Pre-Training存在的问题：
1. 指令遵循：看到指令，输出对应的回答，而不是续写
2. 输出格式：学会对话格式、system prompt、角色标记
3. 行为对齐的起点：为后续的RLHF/DPO提供一个“能用”的初始策略
4. 能力激活：把预训练学到的海量知识，用“回答”的方式释放

### 1.2 SFT的核心公式与含义

SFT的损失函数是：
$$
\mathcal{L} = - \mathbb{E}_{(x,y) \sim \mathcal{D}_{SFT}}\left[ log \pi_{\theta}\left( y | x\right)\right]
$$
其中$x$是prompt，$y$是期望的回答，$\pi_\theta$ 是模型，$\mathcal{D}_{SFT}$是SFT数据集，$\pi_{\theta}(y|x)$是给定输入x，模型输出y的概率，$\mathbb{E}$是期望。这里与Pre-Training的loss形式上一样，但是SFT的数据是(x,y) = (prompt,response)对，而且只在response部分计算loss，不对prompt计算loss。 Pre-training对所有token计算loss。

实际训练时$y$是一个token序列$y=(y_1,y_2,...,y_T)$，对上面式子进行展开：
$$
\pi_{\theta}\left( y | x\right) = \prod^T_{t=1}\pi_{\theta} \left( y_t | x, y_{<t}\right)
$$
代入原式:
$$
\mathcal{L} = - \mathbb{E}_{(x,y)}\left[ \sum^T_{t=1} log\pi_{\theta} \left( y_t | x, y_{<t}\right)\right]
$$
实际上就是对每个response的token算交叉熵，然后求和。

### 1.3 SFT数据的格式

#### 1.3.1 SFT数据的本质

SFT数据是一个个的(prompt,response)对:
$$
\mathcal{D}_{SFT} = \{ (x^{(i)},y^{(i)})\}^N_{i=1}
$$
- $x$: prompt，可以包含system prompt、问题、历史对话
- $y$: 期望的response，模型模仿的目标

SFT是模仿学习，不是偏好学习。数据告诉模型“应该输出什么”，而不是告诉模型“哪个输出更好”

#### 1.3.2 SFT常见数据格式
**单轮指令**
最基础的形式：
```json
{
  "instruction": "用一句话解释什么是过拟合。",
  "response": "过拟合是指模型在训练数据上表现很好，但在新数据上表现差。"
}
```

**带input的指令**
有些任务需要额外输入：
```json
{
  "instruction": "把下面的句子翻译成英文。",
  "input": "今天天气很好。",
  "response": "The weather is nice today."
}
```

**多轮对话**
```json
{
  "messages": [
    {"role": "system", "content": "你是一个简洁的助手。"},
    {"role": "user", "content": "什么是SFT？"},
    {"role": "assistant", "content": "SFT是监督微调……"},
    {"role": "user", "content": "它和RLHF有什么区别？"},
    {"role": "assistant", "content": "SFT是模仿，RLHF是偏好对齐……"}
  ]
}
```

**function calling**
```json
{
  "messages": [
    {"role": "user", "content": "北京今天天气怎么样？"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"name": "get_weather", "arguments": {"city": "北京"}}
    ]},
    {"role": "tool", "content": "晴，25℃"},
    {"role": "assistant", "content": "北京今天晴，25℃。"}
  ]
}
```
#### 1.3.3 对话到训练序列

我们对话常使用JSON，但是模型训练时看到的不是JSON，而是一个**序列**。训练前要把它拼成一条字符串，在进行tokenize。

以ChatML风格为例：
```text
<|im_start|>system
你是一个简洁的助手。<|im_end|>
<|im_start|>user
什么是SFT？<|im_end|>
<|im_start|>assistant
SFT是监督微调……<|im_end|>
```

### 1.4  Loss Mask

我们上面提到 SFT 在计算 loss 时只计算 response 的loss，那么怎么做到的？
这就需要使用mask，把无关的token给 mask掉。

用 0/1 表示：
```text
<|im_start|> system 你是一个简洁的助手。<|im_end|>
<|im_start|> user 什么是SFT？<|im_end|>
<|im_start|> assistant SFT是监督微调……<|im_end|>

mask:  0 0 0 ... 0   0 0 0 ... 0   0 0 0   1 1 1 1 1 ... 1
```

- `0`= 不计算loss（system、user、角色标记、`<|im_end|>` 之前的标记）
- `1` = 算 loss（assistant 的 response token）

补充，`<|im_end|>`算不算？

通常有两种做法：

| 做法  | 说明           | 影响               |
| --- | ------------ | ---------------- |
| 算   | 让模型学会“什么时候停” | 推荐，尤其对生成 EOS 很重要 |
| 不算  | 只学内容         | 模型可能不会主动停        |
一般做法：**assistant 的结束标记要算 loss**，否则模型学不会停

代码实现：
```python
 # 拼接后的序列
input_ids = [...]
# 标记哪些位置属于 assistant response
labels = input_ids.clone()
labels[~response_mask] = -100   # -100 是 PyTorch CrossEntropy 的 ignore_index

loss = CrossEntropyLoss(ignore_index=-100)(logits, labels)
```
---
## 二、SFT的微调方法

### 2.1 全量微调和部分微调
| 维度           | 全量微调（Full FT）  | LoRA                      | QLoRA                         |
| ------------ | -------------- | ------------------------- | ----------------------------- |
| **更新参数**     | 全部参数（100%）     | 仅低秩矩阵 A,BA,B（<1%）         | 仅低秩矩阵 A,BA,B（<1%）             |
| **基座权重**     | 可训练，被更新        | 冻结，不变                     | 冻结，且量化为 4-bit                 |
| **基座精度**     | FP16 / BF16    | FP16 / BF16               | 4-bit（NF4）                    |
| **7B 显存门槛**  | 80GB+（多卡）      | 16-24GB                   | 6-10GB                        |
| **能否单卡**     | 否（通常需多卡）       | 24GB 单卡可跑                 | 8-12GB 单卡可跑                   |
| **训练速度**     | 慢（全参反向）        | 较快                        | 略慢（量化开销）                      |
| **数据需求**     | 万条以上           | 千~万条                      | 千~万条                          |
| **灾难性遗忘**    | 高              | 低                         | 低                             |
| **效果上限**     | 最高             | 接近全参                      | 略低于 LoRA                      |
| **多任务存储**    | 每任务一个完整模型      | 每任务仅存 LoRA 增量（几十 MB）      | 同 LoRA                        |
| **推理开销**     | 无额外            | 可合并，零额外                   | 可合并，需反量化                      |
| **典型学习率**    | 1e-5 ~ 5e-6    | 1e-4 ~ 2e-4               | 1e-4 ~ 2e-4                   |
| **典型 epoch** | 2-3            | 3-5                       | 3-5                           |
| **关键超参**     | 学习率、batch size | rank、alpha、target_modules | rank、alpha、target_modules     |
| **适用场景**     | 任务差异大、资源充足     | 显存中等、追求效果                 | 显存有限、训大模型                     |
| *典型实现*       | `Trainer` 全参   | `peft.LoraConfig`         | `LoraConfig` + `bitsandbytes` |
### 2.2 方法解释

- **全量微调**：更新模型全部参数，效果上限最高，但显存开销大（7B 需 80GB+）、易发生灾难性遗忘、每个任务都要存一份完整模型，存储成本高。

- **LoRA**：冻结基座权重，只训练注入的低秩矩阵 A,BA,B，可训练参数不到 1%。显存省、训练后可合并回原权重（推理零开销）、多任务只需切换几十 MB 的增量，效果接近全参。

- **QLoRA**：在 LoRA 基础上把基座量化到 4-bit，显存再砍一半，让 8-12GB 的消费级显卡也能微调 7B/13B 模型，代价是训练略慢、效果略低于 LoRA。

---
## 三、SFT训练流程
### 3.1 完整流程
```python
# ============================================================
# SFT 监督微调完整流程
# 数据 → 基座 → 微调方法 → 训练参数 → 训练 → 评估 → 部署
# ============================================================

# ---------------------- 步骤1：准备数据 ----------------------
# 说明：把原始数据整理成模型能学的"指令-回答"格式
from datasets import load_dataset

# 从本地 JSON 文件加载数据（每行一条：instruction / input / output）
raw_dataset = load_dataset('json', data_files='sft_data.json', split='train')

# 把 instruction、input、output 拼成一条完整的训练文本
# 这里使用 ChatML 风格模板，方便模型区分"问题"和"回答"
def build_prompt(sample):
    user_content = sample['instruction']
    if sample.get('input', '').strip():           # 如果有补充输入就拼进去
        user_content += f"\n{sample['input']}"
    # ChatML 模板：<|im_start|>角色 ... <|im_end|>
    text = (
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['output']}<|im_end|>"
    )
    return {"text": text}

# 对整个数据集逐条应用模板，并删掉原始字段
dataset = raw_dataset.map(build_prompt, remove_columns=raw_dataset.column_names)

# ---------------------- 步骤2：选择基座模型 ----------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 这里选用 Qwen2.5-1.5B-Instruct
base_model = "Qwen/Qwen2.5-1.5B-Instruct"

# 加载分词器，补齐 pad_token
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"                  # SFT 常用右侧 padding

# 加载基座模型，用 bfloat16 省显存
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",                            # 自动分配到可用设备
    trust_remote_code=True,
)
model.config.use_cache = False                    # 与梯度检查点配合

# ---------------------- 步骤3：选择微调方法 ----------------------
# 这里使用 QLoRA：4bit 量化 + LoRA，显存占用最低
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# 4bit 量化配置（QLoRA 的核心）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                            # 权重用 4bit 存储
    bnb_4bit_quant_type="nf4",                    # nf4 量化类型，效果更稳
    bnb_4bit_compute_dtype=torch.bfloat16,        # 计算时回到 bf16
    bnb_4bit_use_double_quant=True,               # 双重量化，进一步省显存
)

# 重新加载模型（带上量化配置）
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# QLoRA 训练前必须做的一步：冻结原权重、准备梯度
model = prepare_model_for_kbit_training(model)

# LoRA 配置：只训练少量低秩矩阵，不动原始权重
lora_config = LoraConfig(
    r=8,                                          # 秩，8 已足够，越大越吃显存
    lora_alpha=16,                                # 缩放系数，一般取 r 的 2 倍
    lora_dropout=0.1,                             # 稍大一点防过拟合
    bias="none",
    task_type="CAUSAL_LM",
    # 对注意力 + MLP 全部线性层加 LoRA，比只加 q/v 效果更好
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# 把 LoRA 挂到模型上
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()                # 打印可训练参数量（通常 <1%）

# ---------------------- 步骤4：设置训练参数 ----------------------
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./sft_output",                    # 输出目录
    num_train_epochs=3,                           # 训练轮数
    per_device_train_batch_size=4,                # 单卡 batch size
    gradient_accumulation_steps=4,                # 梯度累积，等效 batch=16
    learning_rate=1e-4,                           # LoRA 学习率通常比全参大
    lr_scheduler_type="cosine",                   # 余弦衰减
    warmup_ratio=0.05,                            # 前 5% 步数做 warmup
    bf16=True,                                    # bfloat16 混合精度
    gradient_checkpointing=True,                  # 梯度检查点，省显存
    optim="paged_adamw_8bit",                     # 8bit 优化器，省显存
    logging_steps=20,                             # 每 20 步打印日志
    save_strategy="epoch",                        # 每个 epoch 保存
    save_total_limit=2,                           # 最多保留 2 个检查点
    report_to="none",                             # 不上报 wandb
    seed=42,
)

# ---------------------- 步骤5：开始训练 ----------------------
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# 让 collator 只对 assistant 的回答部分计算 loss
# 这样模型专注学"怎么回答"，而不是学"怎么复述问题"
response_template_ids = tokenizer.encode(
    "<|im_start|>assistant\n", add_special_tokens=False
)
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tokenizer,
)

# 创建 SFT 训练器
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,                          # 新版 trl 中改名为 processing_class
    data_collator=collator,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,                                # 不把多条样本拼在一起
)

trainer.train()                                   # 启动训练
trainer.save_model("./sft_model")                 # 保存 LoRA adapter
tokenizer.save_pretrained("./sft_model")          # 分词器一起保存
print("✅ 微调完成，LoRA adapter 已保存")

# ---------------------- 步骤5.5：合并 LoRA 权重（可选） ----------------------
# 推理时把 LoRA 合并回基座，得到一个独立可用的模型
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./sft_model_merged")
tokenizer.save_pretrained("./sft_model_merged")
print("✅ 合并后的模型已保存到 ./sft_model_merged")


# ---------------------- 步骤6：评估效果 ----------------------
# 定量评估思路：
# 1. 用 BLEU / ROUGE / BERTScore 对比参考答案
# 2. 用 LLM-as-Judge（让强模型打分）
# 3. 人工抽样，看格式遵循、事实正确性、语言流畅度


# ---------------------- 步骤7：部署使用 ----------------------
```
### 3.2 参数解释

| 参数                            | 作用                | 常用值                   |
| ----------------------------- | ----------------- | --------------------- |
| `load_in_4bit`                | 4bit 量化省显存（QLoRA） | `True`                |
| `r`                           | LoRA 低秩维度         | `8` / `16`            |
| `learning_rate`               | 学习率               | LoRA `1e-4`，全参 `2e-5` |
| `num_train_epochs`            | 训练轮数              | `2` ~ `3`             |
| `per_device_train_batch_size` | 单卡 batch          | `2` ~ `8`             |
| `gradient_accumulation_steps` | 梯度累积省显存           | `4` ~ `16`            |
| `bf16`                        | 混合精度省显存           | `True`                |
| `gradient_checkpointing`      | 梯度检查点省显存          | `True`                |
| `max_seq_length`              | 最大序列长度            | `1024` / `2048`       |
### 3.3 Loss Mask 流程代码实现

```python
# ============================================================
# Loss Mask 手工实现（理解原理用）
# ============================================================
from transformers import AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------- 原始样本 ----------
instruction = "请介绍成都"
response = "成都是四川省的省会，以美食和大熊猫闻名。"

# ---------- 拼接成完整文本 ----------
prompt_text = f"### Instruction:\n{instruction}\n### Response:\n"
full_text = prompt_text + response
print("完整文本:\n", full_text)

# ---------- 步骤1：分词 ----------
# 分开 tokenize，方便我们找到 prompt 的 token 长度
prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
# 末尾加上 eos，让模型学会"什么时候停"
response_ids = response_ids + [tokenizer.eos_token_id]

input_ids = prompt_ids + response_ids

# ---------- 步骤2 & 3：构造 labels 并做 mask ----------
# 关键：labels 初始等于 input_ids，然后把 prompt 部分全部置为 -100
labels = input_ids.copy()
labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

print("\nprompt 长度:", len(prompt_ids))
print("response 长度:", len(response_ids))
print("总长度:", len(input_ids))

# ---------- 可视化对比 ----------
print("\nToken 与 Label 对照：")
print(f"{'位置':<6}{'Token':<20}{'Label':<10}{'是否计算loss'}")
for i, (tok_id, lab) in enumerate(zip(input_ids, labels)):
    tok = tokenizer.decode([tok_id]).replace("\n", "\\n")
    flag = "✅ 计算" if lab != -100 else "❌ 忽略"
    print(f"{i:<6}{tok:<20}{lab:<10}{flag}")
```

output:
```text
位置    Token               Label     是否计算loss
0      ###                -100      ❌ 忽略
1      ␣Instruction       -100      ❌ 忽略
2      :                  -100      ❌ 忽略
3      \n                 -100      ❌ 忽略
4      请                 -100      ❌ 忽略
5      介绍               -100      ❌ 忽略
6      成都               -100      ❌ 忽略
7      \n                 -100      ❌ 忽略
8      ###                -100      ❌ 忽略
9      ␣Response          -100      ❌ 忽略
10     :                  -100      ❌ 忽略
11     \n                 -100      ❌ 忽略
12     成都               12345     ✅ 计算
13     是                 23456     ✅ 计算
...
25     <|im_end|>         151645    ✅ 计算
```
---

## 四、常见面试问题

**Q1：SFT 的损失函数是什么？为什么只在 response 上算 loss？**

SFT 的损失与预训练同构 —— 都是逐个 token 的交叉熵，区别只在数据与计算范围（展开过程见 1.2）：

$$\mathcal{L} = - \mathbb{E}_{(x,y) \sim \mathcal{D}_{SFT}}\left[ \sum^T_{t=1} log \pi_{\theta}\left( y_t | x, y_{<t}\right)\right]$$

两处关键改动：

- 数据从"一整段文本"换成 $(x, y) = (\text{prompt}, \text{response})$ 对；
- 求和只覆盖 response 的 token，prompt 部分被 mask，不产生梯度。

为什么 prompt 不算 loss：

1. **prompt 不是要学的内容**：它是输入条件。在 prompt 上算 loss，等于训练模型"复述问题"，梯度花在了抄写输入上；
2. **信号更聚焦**：同一份数据，梯度全部来自"该怎么回答"，学指令遵循更高效；
3. **prompt 上的 loss 很快就能压低**：基座模型本来就会续写，这部分 loss 对"学会回答"没有增量贡献，反而会分散优化目标。

---

**Q2：Loss Mask 是怎么实现的？`<|im_end|>` 这类结束标记要不要算 loss？**

实现只有三行（完整版与可视化见 3.3）：labels 与 input_ids 等长，把不属于 response 的位置统一改成 -100 —— 交叉熵的 `ignore_index` 会在这些位置上跳过计算、不产生梯度：

```python
labels = input_ids.clone()
labels[~response_mask] = -100     # -100 是 PyTorch CrossEntropy 的 ignore_index
loss = CrossEntropyLoss(ignore_index=-100)(logits, labels)
```

结束标记**要算**。`<|im_end|>`（或 EOS）教模型"什么时候停"：若把它 mask 掉，模型没有建模过停止行为，推理时容易停不下来。所以 mask 的准确范围是：**assistant 回复的全部 token（含结尾标记）算 loss，system、user、角色标记等其余位置一律不算**。

多轮对话按同样规则处理：每个 assistant 回合的内容与结束标记都算，user 与 system 的全部内容不算。

---

**Q3：全量微调、LoRA、QLoRA 有什么区别？实际怎么选？**

核心区别（完整对比见 2.1）：

| 方法 | 训练什么 | 基座精度 | 7B 显存门槛 | 效果上限 |
|------|---------|---------|------------|---------|
| 全量微调 | 全部参数（100%） | FP16 / BF16 | 80GB+（多卡） | 最高 |
| LoRA | 低秩矩阵 A、B（<1%） | FP16 / BF16 | 16-24GB | 接近全参 |
| QLoRA | 低秩矩阵 A、B（<1%） | 4-bit（NF4） | 6-10GB | 略低于 LoRA |

LoRA 的原理：冻结原权重 $W_0$，用低秩分解近似权重增量 $\Delta W = BA$，只训练 $A$、$B$（秩 $r$ 远小于权重矩阵的维度）；推理前把 $BA$ 加回 $W_0$ 即可合并，零额外开销。

选型经验（代价与回报见 2.2）：

- 任务与基座分布差异大、多卡资源充足 → 全量微调；
- 单卡 16-24GB、想要接近全参的效果 → LoRA；
- 只有 8-12GB 消费级显卡、要微调 7B/13B → QLoRA，用一点训练速度与效果换显存。

---

**Q4：SFT 数据怎么准备？多轮对话里 loss 怎么算？**

数据准备的三个要点：

1. **形态**：本质是 $(x, y)$ 对，常见写法有 instruction/input/response、messages、tool_calls 三种（见 1.3.2）；喂给模型前要先拼成一条 ChatML 风格的序列再 tokenize（见 1.3.3）；
2. **质量优先于数量**：SFT 是模仿学习，示范里的口吻、格式、错误都会被原样学走；数据量达到千~万条（见 2.1）即可见效，重点是把答案写对、格式写整齐；
3. **多样性**：覆盖目标任务的各类指令，同时避免与评测集内容重复，否则评测数字虚高、真实效果打折。

多轮对话的 loss：**只有 assistant 的回合算 loss**，system、user、历史轮次全部 mask；function calling 场景中，tool 的返回内容是环境给模型的输入，同样不算。实现上仍是 3.3 的 mask 逻辑 —— 按角色标出每个 assistant 回复区间，区间内置 1，其余置 0。

---

**Q5：既然有了 SFT，为什么还需要 RLHF / DPO？**

因为 SFT 是**模仿学习**，它只做一件事：抬高示范答案的概率 —— 这带来四个天然局限：

1. **只有正例，没有负例**：数据里没有"哪个回答更差"的信号，模型学不会区分好坏，更不会主动避开坏答案；
2. **效果被数据质量封顶**：人类示范的数量与质量都有限，天花板由数据决定；
3. **单点示范压缩多样性**：同一个问题可以有多种同样好的回答，但示范只给了一种，模型会把概率集中到这一种上，输出变得保守、单一；
4. **训练与推理错位**：训练时每一步都看到真实前缀（teacher forcing），推理时只能吃自己生成的 token，一步错、步步错，误差会累积。

RLHF / DPO 用偏好数据补上这块：不再告诉模型"应该输出什么"，而是告诉它"哪个输出更好"（RLHF 让模型自己生成、由奖励模型打分后优化；DPO 直接在偏好数据上优化，绕开奖励模型与 PPO）。所以后训练的标准分工是：SFT 给出一个"能用"的初始策略（见 1.1 第 3 点），RLHF / DPO 在此之上完成偏好对齐。

---

## 五、总结

### 5.1 速查表

| 概念 | 核心要点 |
|------|---------|
| SFT 是什么 | 在 $(x, y)$ 对（prompt, response）上做监督微调，把"续写器"基座改造成会回答、懂对话格式的模型 |
| 损失函数 | $\mathcal{L} = - \mathbb{E}_{(x,y)}\left[\sum_t log \pi_{\theta}(y_t \mid x, y_{<t})\right]$，与预训练同构 |
| 计算范围 | 只算 response：assistant 各回合内容与结束标记都算，prompt / system / user / tool 一律 mask |
| Loss Mask 实现 | labels 中非 response 位置置 -100（`ignore_index`），交叉熵跳过该位置、不产生梯度 |
| 数据格式 | 单轮指令 / 带 input / 多轮 messages / tool_calls；训练前拼接为 ChatML 序列 |
| 全量微调 | 全参更新，效果上限最高；7B 需 80GB+、易灾难性遗忘、每任务存一份完整模型 |
| LoRA | 冻结基座，只训低秩增量 $\Delta W = BA$（<1%）；16-24GB 可跑、可合并、推理零开销 |
| QLoRA | LoRA + 基座 4-bit（NF4）量化；6-10GB 可跑，训练略慢、效果略低于 LoRA |
| 关键超参 | LoRA 学习率 1e-4 ~ 2e-4、epoch 2-3、r 取 8/16、max_seq_length 1024/2048 |
| 训练流程 | 数据 → 基座 → 微调方法 → 训练参数 → 训练 → 评估 → 部署 |
| SFT 的局限 | 只有正例、单点示范、训练与推理错位 → 需要 RLHF/DPO 做偏好对齐 |

---

## 相关笔记

- [[【相关笔记1.md】]] — 【它讲什么、与本文什么关系】
- [[【相关笔记2.md】]] — 【…】
