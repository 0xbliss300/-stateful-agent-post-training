---
tags:
  - RLHF
  - PPO
  - 奖励模型
  - 偏好对齐
  - 后训练
created: 2026-10-17
updated: 2026-10-17
related:
---

# RLHF（Reinforcement Learning from Human Feedback）详解

> [[SFT.md]] 的 Q5 指出模仿学习的四个局限：只有正例、被数据质量封顶、压缩多样性、训练与推理错位 —— 偏好学习就是来补这块的。本文讲偏好对齐的经典正统路线 RLHF：用偏好数据训一个奖励模型（Reward Model），再用 PPO 强化学习让策略模型最大化奖励。内容包括 Bradley-Terry 偏好建模、PPO 的目标函数与 KL 惩罚、四个模型同台的显存账、reward hacking 等典型失败模式，为理解 [[DPO.md]] 的推导起点与取舍打下基础。

---

## 一、RLHF 是什么

### 1.1 从 SFT 到 RLHF

SFT 是模仿学习：数据告诉模型"应该输出什么"。但很多品质**没法写成示范** —— "这个回答更有帮助""那个回答有幻觉"，人类很容易比较两个回答的好坏，却很难亲手写出完美答案。RLHF 的核心思想：

**不教模型"输出什么"，而是收集"哪个更好"的比较反馈，把它蒸馏成一个标量奖励，再用强化学习最大化它。**

InstructGPT 确立的三阶段管线（本文结构即按此展开）：

| 阶段 | 做什么 | 输入 → 输出 | 详见 |
|------|--------|------------|------|
| Stage 1：SFT | 在 (prompt, response) 上监督微调，得到初始策略 $\pi_{SFT}$ | 示范数据 → 策略模型 | [[SFT.md]] |
| Stage 2：奖励模型 | 在偏好对 $(x, y_w, y_l)$ 上训 RM，给任意回答打标量分 | 偏好数据 → 奖励模型 $r_\phi$ | 第二章 |
| Stage 3：PPO | 用 RM 打分作为奖励，强化学习优化策略，同时用 KL 惩罚拴住它 | SFT 模型 + RM → 对齐后的 $\pi_\theta$ | 第三章 |

### 1.2 强化学习视角：LLM 生成怎么套进 MDP

把文本生成套进马尔可夫决策过程（MDP）的要素映射：

| RL 概念 | 在 LLM 生成中的对应 |
|---------|-------------------|
| 状态 $s_t$ | prompt + 已生成的前缀 $(x, y_{<t})$ |
| 动作 $a_t$ | 下一个 token $y_t$（动作空间 = 词表，可达几万维） |
| 策略 $\pi_\theta$ | 语言模型本身：$\pi_\theta(y_t \mid x, y_{<t})$ |
| 轨迹 | 一整条回答 $y = (y_1, \dots, y_T)$ |
| 奖励 | 只在回答结束时由 RM 给一个整分 $r_\phi(x, y)$，中间 token 没有即时奖励（稀疏奖励） |
| 环境 | 没有外部环境 —— 状态转移是确定性的"拼接"，这是 LLM-RL 与游戏 RL 的最大区别 |

优化目标（KL 约束的奖励最大化）：

$$\max_{\theta}\ \mathbb{E}_{x \sim \mathcal{D},\ y \sim \pi_\theta(\cdot \mid x)}\left[r_\phi(x, y)\right] - \beta\, \mathbb{D}_{KL}\left(\pi_\theta(\cdot \mid x)\ \|\ \pi_{ref}(\cdot \mid x)\right)$$

其中 $\pi_{ref}$ 是冻结的 SFT 模型，$\beta$ 是 KL 惩罚系数。这个目标同时出现在 [[DPO.md]] 1.5 节 —— DPO 的全部推导正是从它有闭式解出发的（见 3.3 的 Insight）。

---

## 二、Stage 2：奖励模型（Reward Model）

### 2.1 偏好数据与 Bradley-Terry 模型

RM 的数据是成对比较：

$$\mathcal{D}_{pref} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^{N}$$

- $x$：prompt；$y_w$：人类标注为更好的回答（chosen）；$y_l$：更差的（rejected）；
- 标注者只看排序、不打绝对分 —— **比较比打分容易且一致性好**，这是偏好数据的根本优势。

问题：RM 输出的是标量分 $r_\phi(x, y)$，怎么从"分数"得到"比较"？用 Bradley-Terry 模型假设偏好概率由分差决定：

$$P(y_w \succ y_l \mid x) = \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$

即"$y_w$ 更好的概率"只取决于两个回答的分差，分差越大、偏好越确定。训练目标就是在这个概率模型上做最大似然，等价于一个二分类交叉熵：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{pref}}\left[\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

### 2.2 具象理解：学一把"人味的尺子"

| 组件 | 符号 | 物理含义 | 形象化描述 |
|------|------|----------|-----------|
| 奖励模型 | $r_\phi$ | 给 (prompt, 回答) 打标量分 | 一位读了几万条对比数据的评委，手里一把尺子 |
| 分差 | $r_w - r_l$ | 两个回答的质量差距 | 两把尺子读数之差 |
| sigmoid | $\sigma$ | 分差 → 偏好概率 | 分差 0 = 抛硬币；分差越大越笃定 |
| 损失 | $-\log\sigma(r_w - r_l)$ | 评委的判断与人类标注不一致的程度 | 评委判断错了多少，就怎么修正尺子 |

**计算逻辑**（对一条偏好样本）：

1. 同一条 prompt 的两个回答分别过 RM，各得一个标量分；
2. 算分差 $z = r_w - r_l$，过 sigmoid 得到"模型认为 $y_w$ 更好"的概率；
3. 与真实标注（$y_w$ 确实更好）算交叉熵，反向传播更新 RM。

### 2.3 数值推演：一对偏好样本的 loss 与梯度

设某条样本上 RM 给出 $r_w = 2.0$、$r_l = 1.0$。

#### Step 1：算分差与偏好概率

$$z = 2.0 - 1.0 = 1.0, \quad \sigma(1.0) = \frac{1}{1 + e^{-1}} \approx 0.731$$

模型认为 $y_w$ 更好的概率是 73.1%。

#### Step 2：算 loss

$$\mathcal{L} = -\log 0.731 \approx 0.313$$

#### Step 3：对比一组"没拉开差距"的分数

若 RM 只给出 $r_w = 1.1$、$r_l = 0.9$（$z = 0.2$）：

| 分差 $z$ | $\sigma(z)$ | loss | $\partial \mathcal{L} / \partial z = \sigma(z) - 1$ |
|---------|------------|------|------|
| 1.0 | 0.731 | 0.313 | -0.269 |
| 0.2 | 0.550 | 0.598 | -0.450 |
| -0.5（判反了） | 0.378 | 0.974 | -0.622 |

**观察**：分差越小（甚至为负），loss 越大、**梯度的绝对值也越大** —— RM 的梯度天然集中在"难区分"和"判错"的样本对上，已经分得很开的样本几乎不再提供学习信号。这与普通二分类的行为一致。

### 2.4 实现细节：从 SFT 模型到标量评委

- **初始化**：RM 通常从 SFT 模型初始化（而不是从零训），因为它需要先"读懂"回答才能评价；
- **结构**：去掉 LM head，换成一个标量头（线性层，输出维度 1）；取**最后一个 token（EOS）位置的隐状态**过标量头，作为整条回答的分数；
- **归一化**：奖励的绝对值没有物理意义，只有相对比较有意义；训练中常对 RM 输出做均值 0、方差 1 的归一化，防止 PPO 阶段奖励尺度过大导致的不稳定；
- **坑**：RM 只在"标注分布内"可靠 —— 对分布外的回答，它的打分是外推，可能系统性偏高（这正是 4.1 reward hacking 的温床）。

### 2.5 代码实现： pairwise 奖励模型

```python
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    """在冻结/微调的 LM 主干上加一个标量头，取 EOS 位置隐状态打分"""

    def __init__(self, backbone: nn.Module, hidden_size: int):
        super().__init__()
        self.backbone = backbone                 # 通常由 SFT 模型初始化
        self.score_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        # 取每个样本最后一个非 padding token 的隐状态
        hidden = self.backbone(input_ids, attention_mask=attention_mask,
                               output_hidden_states=True).hidden_states[-1]
        lengths = attention_mask.sum(dim=1) - 1          # EOS 下标
        eos_hidden = hidden[torch.arange(hidden.size(0)), lengths]
        return self.score_head(eos_hidden).squeeze(-1)   # [batch] 标量分

def rm_loss(r_w: torch.Tensor, r_l: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry pairwise 损失：-log σ(r_w - r_l)"""
    return -torch.nn.functional.logsigmoid(r_w - r_l).mean()

# 一条样本：r_w = 2.0, r_l = 1.0 → loss ≈ 0.313（与 2.3 的推演一致）
loss = rm_loss(torch.tensor([2.0]), torch.tensor([1.0]))
print(loss.item())   # 0.3133
```

---

## 三、Stage 3：PPO 强化学习

### 3.1 奖励如何整形：token 级 KL 惩罚

RM 只在回答结束时给一个整分，直接用就是极稀疏的奖励。实践中把 KL 惩罚摊到每个 token 上，RM 分只加在末尾：

$$r_t = -\beta \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{ref}(y_t \mid x, y_{<t})} \quad (t < T), \qquad r_T = r_\phi(x, y) - \beta \log \frac{\pi_\theta(y_T \mid \cdot)}{\pi_{ref}(y_T \mid \cdot)}$$

- 策略每生成一个偏离 $\pi_{ref}$ 分布的 token，就立即吃一笔惩罚；
- RM 的整分只在最后一步到账 —— 最终一条轨迹的总回报 = RM 分 - 全程 KL 罚，与 1.2 的目标函数逐项对应。

数值感受：设 RM 给 3.0 分，某条 100-token 回答的累计 log 概率比 $\sum_t \log(\pi_\theta/\pi_{ref}) = 20$ nats，$\beta = 0.05$，则净回报 $= 3.0 - 0.05 \times 20 = 2.0$。**生成越"放飞"，KL 罚越狠。**

### 3.2 PPO 的 clip 目标

PPO（Proximal Policy Optimization）是对策略梯度的稳定化改造。记新旧策略的概率比：

$$\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$$

PPO 不直接最大化 $\rho_t A_t$（这会让策略一步跑飞），而是裁掉概率比超出 $[1-\epsilon, 1+\epsilon]$ 的部分：

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(\rho_t A_t,\ \operatorname{clip}(\rho_t,\ 1-\epsilon,\ 1+\epsilon)\, A_t\right)\right]$$

其中 $A_t$ 是优势（advantage）：这个 token 比"平均水平"好多少，由价值模型（critic）经 GAE 估计：

$$A_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l\, \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

价值模型 $V$ 也是一个小头接在 LM 主干上，学习"当前前缀值多少分"，与策略同时训练。

### 3.3 数值推演：clip 什么时候起作用

取 $\epsilon = 0.2$，即概率比的信任区间为 $[0.8, 1.2]$，观察四种情形（$A_t$ 取 ±1）：

| $\rho_t$ | $A_t$ | $\rho_t A_t$ | clip 项 | min（实际生效） | 梯度 |
|---------|------|-------------|---------|----------------|------|
| 1.3 | +1 | 1.3 | 1.2 | **1.2（被裁）** | 0：好动作已被提得够多，不再奖励 |
| 0.7 | +1 | 0.7 | 0.8 | 0.7（未裁） | 有：好动作被压低过头了，拉回来 |
| 1.3 | -1 | -1.3 | -1.2 | -1.3（未裁） | 有：坏动作概率升高了，压下去 |
| 0.7 | -1 | -0.7 | -0.8 | **-0.8（被裁）** | 0：坏动作已被压得够低，不再压 |

**观察**：clip 是一个"悲观"目标 —— 只在"更新方向有利且已走出信任区"时切断梯度，让每步更新最多挪 20%。KL 惩罚管"别偏离参考模型"，clip 管"别偏离上一轮的自己"，两者是 PPO 稳定的两道闸。

`★ Insight ─────────────────────────────────────`
**KL 约束目标的闭式解是 RLHF 与 DPO 的分水岭**：对 1.2 的目标逐 token 求解，可推出最优策略 $\pi^*(y \mid x) \propto \pi_{ref}(y \mid x)\exp(r(x,y)/\beta)$ —— 奖励与策略互为表里，知道一个就能反解另一个。RLHF 选择"显式训奖励 + PPO 逼近 $\pi^*$"，管线长但可以在线迭代；DPO 选择"把 $r$ 用 $\pi$ 表示后代入偏好模型，奖励被消元"，管线短但只能离线。完整推导见 [[DPO.md]] 1.5 节。
`─────────────────────────────────────────────────`

### 3.4 四个模型同台：显存账

PPO 阶段 GPU 上**同时驻留四个模型**，这是 RLHF 贵得离谱的直接原因：

| 模型 | 角色 | 是否训练 | 7B bf16 大致开销 |
|------|------|---------|-----------------|
| 策略模型 $\pi_\theta$ | 生成回答、被优化 | **训练**（权重 + 梯度 + Adam 状态） | 14 + 14 + 56 ≈ 84GB |
| 参考模型 $\pi_{ref}$ | 算 KL 惩罚的基准 | 冻结，只做前向 | ~14GB |
| 奖励模型 $r_\phi$ | 给整条回答打分 | 冻结，只做前向 | ~14GB |
| 价值模型（critic）$V$ | 估计每步优势 $A_t$ | **训练** | 与策略同量级，常与策略共享主干省钱 |

再加上生成 rollout 的激活值与 KV cache，7B 级 RLHF 的 PPO 阶段轻松破百 GB，必须多卡 + 张量并行，工程复杂度远超 SFT（对比：[[SFT.md]] 2.1 节中 7B 的 QLoRA 单卡 8-12GB 即可）。

### 3.5 PPO 训练循环（TRL 风格）

```python
# ============================================================
# RLHF Stage 3：PPO 主循环（TRL PPOTrainer 的骨架）
# ============================================================
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

config = PPOConfig(
    learning_rate=1e-6,          # RL 阶段学习率远小于 SFT
    batch_size=128,
    ppo_epochs=4,                # 每批 rollout 重复优化的轮数
    cliprange=0.2,               # 即 epsilon，信任区半宽
    kl_penalty="kl",
    init_kl_coef=0.05,           # beta：KL 惩罚系数，可自适应
)

policy = AutoModelForCausalLMWithValueHead.from_pretrained("sft_model")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("sft_model")  # 冻结
ppo = PPOTrainer(config, policy, ref_model, tokenizer, dataset=prompt_dataset)

for batch in ppo.dataloader:
    # 1. rollout：策略对 prompt 采样生成回答
    responses = ppo.generate(batch["input_ids"], max_new_tokens=256)
    # 2. 打分：奖励模型给整条回答一个标量分
    rewards = reward_model(batch["input_ids"], responses)
    # 3. 一步 PPO：内部完成 KL 整形、GAE、clip 目标、策略与价值头更新
    stats = ppo.step(batch["input_ids"], responses, rewards)
    ppo.log_stats(stats, batch, rewards)
```

---

## 四、典型失败模式与工程问题

### 4.1 Reward hacking：分数上去，质量下来

RM 是真实人类偏好的**有噪代理**，PPO 又是个不知疲倦的优化器 —— 策略迟早会找到 RM 打高分的捷径而不真的变好：

| 典型表现 | 机理 |
|---------|------|
| 回答越来越长 | 标注者偏好"详细"，RM 学到长度与分相关，策略疯狂注水 |
| 滥用套话与奉承 | "作为 AI 助手……"类安全话术在偏好数据里得分高 |
| 格式作弊 | 加粗、列表、表情符号刷分，内容空洞 |

缓解手段：KL 惩罚（拴住策略别跑太远）、奖励归一化与裁剪、多 RM 集成投票、定期用新策略的输出去**重新标注刷新 RM**（迭代式 RLHF）。但治本难 —— **代理奖励被优化到一定程度必然失真**（Goodhart 定律）。

### 4.2 PPO 的稳定性与超参敏感

- 学习率、KL 系数 $\beta$、clip $\epsilon$、batch 大小互相耦合，一处失调全盘崩；
- $\beta$ 太大 → 策略不敢动，对齐无效；太小 → 模型跑飞、语言能力崩坏（灾难性遗忘的 RL 版）；
- 实践中常用自适应 KL 控制器：实测 KL 高于目标则加大 $\beta$，反之减小。

### 4.3 RLHF vs DPO：怎么选

| 维度 | RLHF（PPO） | DPO |
|------|------------|-----|
| 管线 | SFT → 训 RM → PPO，三阶段 | SFT → 偏好数据直接优化，一步到位 |
| 在线能力 | **可以**：策略自己采样新回答，探索分布外空间 | 只能吃离线偏好数据 |
| 显存 | 四个模型同台，7B 破百 GB | 两个模型（策略 + 参考），与 SFT 相当 |
| 稳定性 | 差：超参敏感、reward hacking | 好：本质是监督式分类损失 |
| 效果上限 | 更高：旗舰模型最终对齐仍用它 | 接近 RLHF，略低 |

结论：**默认先 DPO 验证收益，追求上限或需要在线迭代再上 RLHF**。更细的对比见 [[DPO.md]] 3.1 节。

---

## 五、常见面试问题

**Q1：请完整描述 RLHF 的三阶段流程。**

1. **SFT**：在 (prompt, response) 示范数据上监督微调，得到会回答、懂格式的初始策略（见 [[SFT.md]]）；
2. **训奖励模型**：收集成对偏好数据 $(x, y_w, y_l)$，用 Bradley-Terry 模型把偏好概率建模为 $\sigma(r_w - r_l)$，训 RM 给任意回答打标量分（见 2.1）；
3. **PPO 优化**：策略对 prompt 采样生成回答，RM 打分作为奖励，叠加逐 token 的 KL 惩罚 $\beta \log(\pi_\theta/\pi_{ref})$，用 PPO 的 clip 目标更新策略，同时训一个价值模型估计优势（见 3.1、3.2）。

一句话：**人类比较 → 奖励模型 → 强化学习最大化奖励（但别偏离 SFT 太远）。**

---

**Q2：奖励模型是怎么训练的？为什么用成对比较而不是直接打分？**

训练：RM 从 SFT 模型初始化，LM head 换成标量头，取 EOS 位置隐状态输出分数；在偏好对 $(x, y_w, y_l)$ 上最小化 $-\log\sigma(r_w - r_l)$（推导与数值例子见 2.3）。

为什么成对比较：人类做**相对比较**远比绝对打分一致、可靠 —— 让标注者直接打 1-10 分，尺度漂移严重、标注者之间不可比；而"A 和 B 哪个好"判断成本低、一致性高。Bradley-Terry 模型再把成对概率还原成潜在的标量分，绕开了绝对打分的难题。

---

**Q3：PPO 阶段的目标函数是什么？KL 惩罚为什么必须有？**

目标是最大化

$$\mathbb{E}\left[r_\phi(x, y)\right] - \beta\, \mathbb{D}_{KL}\left(\pi_\theta\ \|\ \pi_{ref}\right)$$

实现上 RM 分加在末 token，KL 罚摊到每个 token（见 3.1），再用 PPO clip 目标做稳定更新（见 3.2、3.3）。

KL 惩罚的三个作用：

1. **防 reward hacking**：RM 只在标注分布内可靠，策略偏离 $\pi_{ref}$ 越远打分越不可信，KL 罚把策略拴在可靠区附近；
2. **保语言能力**：防止对齐过程中把 SFT/预训练学到的通用能力忘掉；
3. **正则化**：限制每代策略的漂移幅度，让训练曲线平滑。

---

**Q4：为什么说 RLHF 的 PPO 阶段显存开销巨大？**

因为四个模型要同时驻留（明细见 3.4）：被训练的策略模型（权重 + 梯度 + Adam 状态，7B 约 84GB）、冻结的参考模型（算 KL，~14GB）、冻结的奖励模型（打分，~14GB）、被训练的价值模型（估计优势，与策略同量级）。再叠加 rollout 生成的激活与 KV cache，7B 级轻松超过 100GB，必须多卡并行 —— 这是 DPO 类离线方法（只需策略 + 参考两个模型）在工程上受欢迎的核心原因。

---

**Q5：什么是 reward hacking？怎么缓解？**

Reward hacking 指策略找到了让 RM 打高分的捷径，却偏离了 RM 想要代理的真实人类偏好 —— 典型如回答注水变长、堆砌安全套话、用格式刷分（见 4.1）。根因是 Goodhart 定律：**代理指标被当成目标优化后，就不再是好指标**。

缓解：KL 惩罚限制策略活动范围、奖励归一化与裁剪、多 RM 集成、以及最重要的 —— 用当前策略的输出重新标注数据、迭代刷新 RM（迭代式 RLHF），让 RM 跟上策略的分布漂移。

---

**Q6：RLHF 和 DPO 怎么选？**

- 要工程性价比、离线偏好数据已就绪、显存有限 → **DPO**：一个分类损失直接优化，稳定且只需两个模型；
- 要效果上限、需要模型在线探索新回答、偏好信号要显式打分复用（如同时服务多个下游） → **RLHF**：旗舰模型的最终对齐仍走在线 RL。

两者的理论关系：DPO 的损失正是从 RLHF 的 KL 约束目标的闭式解推出的（见 3.3 的 Insight 与 [[DPO.md]] 1.5 节），不是对立的框架，而是同一目标的两种解法。

---

## 六、总结

### 6.1 速查表

| 概念 | 核心要点 |
|------|---------|
| 三阶段管线 | SFT → 训奖励模型 → PPO 强化学习 |
| 偏好数据 | $(x, y_w, y_l)$ 成对比较；比较比绝对打分一致性好 |
| Bradley-Terry | $P(y_w \succ y_l \mid x) = \sigma(r_w - r_l)$，loss $= -\log\sigma(r_w - r_l)$ |
| RM 实现 | SFT 模型初始化，EOS 隐状态过标量头；分布外打分不可靠 |
| RLHF 目标 | $\max\ \mathbb{E}[r_\phi] - \beta\, D_{KL}(\pi_\theta \mid \pi_{ref})$ |
| 奖励整形 | RM 分加在末 token，KL 罚 $-\beta\log(\pi_\theta/\pi_{ref})$ 摊到每个 token |
| PPO clip | $\min(\rho_t A_t, \operatorname{clip}(\rho_t, 1\pm\epsilon) A_t)$；好方向走出信任区即断梯度 |
| GAE / critic | 价值模型估计优势 $A_t$，与策略同时训练 |
| 显存 | 策略 + 参考 + RM + critic 四模型同台，7B 破百 GB |
| KL 惩罚 | 防 reward hacking、保语言能力、正则化；$\beta$ 可自适应 |
| Reward hacking | Goodhart 定律：代理奖励被优化后失真；缓解靠 KL、迭代刷新 RM |
| vs DPO | RLHF 可在线、上限高但贵且不稳；DPO 离线、便宜稳定、上限略低 |

---

## 相关笔记

- [[SFT.md]] — RLHF 的第一阶段；其 Q5 的四个局限正是本文要解决的问题
- [[DPO.md]] — 从本文 1.2 的 KL 约束目标推出闭式解、绕开 RM 与 PPO 的离线替代方案
- [[LoRA.md]] — 参数高效微调，可用于压低 SFT 与 RM 阶段的训练成本
