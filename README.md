## 每周固定模板

```Markdown
# Week XX Review
- 实际投入：__ / 15 小时
- 本周 Git commit：
- 本周模型 / 数据版本：
- GPU / 显存峰值 / 时间：
- 通过的测试：
- 最关键实验：配置、seed、train/dev/test、结果
- 失败与排查（至少 1 条）：
- 本周学习笔记：
- 下周 3 个最高优先级任务：
- 本周岗位进展（岗位/日期/反馈/下一动作）：
```

## 仓库结构

```
stateful-agent-post-training/
├── configs/                 # sft / grpo / reward / eval 配置
├── data/                    # 任务模板、划分清单、样本；敏感信息不上传
├── src/
│   ├── foundations/         # attention.py / grpo_math.py
│   ├── agents/              # rollout.py / schema.py
│   ├── envs/                # retail_env.py / tools.py / verifier.py
│   ├── data/                # build_tasks.py / build_sft_data.py
│   ├── training/            # train_sft.py / train_grpo.py / agent_env_adapter.py
│   └── eval/                # evaluate.py / error_analysis.py
├── tests/                   # attention/env/verifier/adapter/data leakage
├── results/                 # 统一实验汇总与图，不强行提交大 checkpoint
├── notes/                   # 指定课程笔记 + 公式算例 + 面试口述
├── reports/                 # 每周复盘 + 最终技术报告
├── internship_tracker.md
└── README.md
```
