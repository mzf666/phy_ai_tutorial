# Tutorial on Physical Intelligence Series


## Overview

本目录提供 Physical Intelligence 系列关键技术的开源复现. 目的是帮助有 LLM / RL 背景的人快速了解 π0 ~ π0.x 训练、推理的实现细节, 并且有能力进行复现. 目前 cover 以下内容:

- π0: A Vision-Language-Action Flow Model for General Robot Control: https://arxiv.org/abs/2410.24164v1
  - 上游代码: [openpi](https://github.com/Physical-Intelligence/openpi) @ `215abfb217dbac7d5f1273282331b9b1866c0479`
  - 目录 `pi0/`, 分支 `topic/pi0`


## 关键技术栈

### π0: 奠定了 VLM pretrain + Flow-Matching Policy 的范式

按 module 拆分, 每个 module 一个目录, 按下面顺序开发:

| module | 关键技术 | 状态 |
|---|---|---|
| [`pi0/data`](pi0/data/README.md) | 数据组织与预处理: 相机槽位, delta action, 归一化, 缩放填黑, 分词, 维度填充, 增广 | 完成 |
| [`pi0/vlm`](pi0/vlm/README.md) | SigLIP So400m/14 + Gemma 2B 的最小实现 (PaliGemma 结构), checkpoint 视为给定 | 完成 |
| [`pi0/action_expert`](pi0/action_expert/README.md) | Gemma 300M expert, 与 VLM 共享 attention 的双 expert 机制, state / action / timestep 嵌入, blockwise attention mask | 完成 |
| [`pi0/flow_matching`](pi0/flow_matching/README.md) | Beta 时间步采样, 线性插值与目标速度, MSE loss, 10 步 Euler 采样与前缀 KV cache | 完成 |
| [`pi0/infer`](pi0/infer/README.md) | 端到端推理链路, 参数量表, 各环节延时表 | 完成 |
| [`pi0/train`](pi0/train/README.md) | 优化器与 schedule, 冻结 / LoRA, 预训练 → post-training 两阶段, cost 表 | 完成 |

**推理**
- VLM 模型结构, 输入输出 protocol: 图像 → SigLIP → 线性投影到 2048 宽, 与指令 token 拼接进 Gemma 2B (→ `vlm`).
- Flow-Matching Policy 模型结构, 输入输出 protocol: state 1 个 token, 50 个 noisy action token, timestep 用 sin-cos 嵌入后与 action 拼接过 MLP; 三块 blockwise 因果 mask (→ `action_expert`, `flow_matching`).
- 去噪步数与积分方式 (10 步 Euler), action horizon 50 与实际执行长度 (25 或 16 步), 前缀 KV cache 只算一次 (→ `flow_matching`, `infer`).
- 实际部署的细节, 模型大小, 链路环节, 延时 (→ `infer`).

**训练**
- 数据组织形式与预处理 (→ `data`).
- 时间步采样分布 Beta(1.5, 1) 截断到 0.999, 图像增广参数, 归一化统计量来源, action padding 与 delta 规则, 数据混合权重 (→ `data`, `flow_matching`).
- 优化器, 学习率 schedule, 冻结策略, 预训练与 post-training 的 curriculum (→ `train`).

**评测**
- 评测环境接口 (LIBERO 观测键 / 动作维度 / 初始状态 / 步数上限), 论文任务与 rubric, 二值成功率与归一化得分的定义与聚合, 带重规划的 episode 循环 (→ `infer/eval.py`); 训练侧的验证 (→ `train`).

**背景事实 (只陈述, 不写代码)**
- PaliGemma checkpoint 的选取, 其预训练 recipe 与模型细节 (→ `vlm` README).
- π 预训练混合数据的构成 (→ `data` README).
