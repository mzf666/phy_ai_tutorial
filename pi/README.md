# Tutorial on Physical Intelligence Series


## Overview

本目录提供 Physical Intelligence 系列关键技术的开源复现. 目的是帮助有 LLM / RL 背景的人快速了解 π0 ~ π0.x 训练、推理的实现细节, 并且有能力进行复现. 目前 cover 以下内容:

- π0: A Vision-Language-Action Flow Model for General Robot Control: https://arxiv.org/abs/2410.24164v1
  - 上游代码: [openpi](https://github.com/Physical-Intelligence/openpi) @ `215abfb217dbac7d5f1273282331b9b1866c0479`
  - 目录 `pi0/`, 分支 `topic/pi0`
- FAST: Efficient Action Tokenization for Vision-Language-Action Models: https://arxiv.org/abs/2501.09747v1
  - 上游代码: [openpi](https://github.com/Physical-Intelligence/openpi) @ `215abfb217dbac7d5f1273282331b9b1866c0479` (`pi0_fast.py`, `gemma_fast.py`, `tokenizer.py`), tokenizer 源码与发布权重: [physical-intelligence/fast](https://huggingface.co/physical-intelligence/fast) (HuggingFace, 2026-09-13 访问)
  - 目录 `fast/`, 分支 `topic/fast`


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


### FAST: 用 DCT + BPE 把 action chunk 压成离散 token, 让 VLM 直接做 next-token prediction

π0-FAST 与 π0 共享 PaliGemma 底座、数据组织、评测环境. 本 topic 只写 **增量**: action 的离散化, 序列格式, 自回归解码, CE 训练. 其余一律 `from pi.pi0.<module> import ...`.

| module | 关键技术 (增量) | 复用 π0 | 状态 |
|---|---|---|---|
| [`fast/tokenizer`](fast/tokenizer/README.md) | quantile 归一化, 逐维 DCT-II, γ 缩放取整, 低频优先展平, 字节级 BPE 的 fit / encode / decode | `pi0/data` 的 chunk 截取 | 完成 |
| [`fast/data`](fast/data/README.md) | prompt + 256 分箱 state + action token 拼成一条序列; input / ar / loss 三个 mask; 映射进 PaliGemma 词表尾部; FAST 的相机槽位与不 mask 规则 | `pi0/data` 图像预处理, delta action, 维度 padding | 完成 |
| [`fast/model`](fast/model/README.md) | prefix-LM 三块 mask [images \| prompt+state \| action]; 右对齐 padding; 定长 KV cache (prefill + 256 步); greedy / temperature 采样; EOS 早停; tied logits head; 增长式 cache 与上游定长 cache 的等价性; 解码 position 偏 1 的上游行为 | `pi0/vlm` 的 SigLIP, Gemma, `make_attn_mask` | 完成 |
| [`fast/train`](fast/train/README.md) | 仅 postfix 的 next-token CE, 按有效 token 数归一; 只对 target 位置算 logits; warmup 1k → 常数 5e-5, AdamW(.9, .95) 无 wd, clip 1, EMA 0.999; 逐 head 的 LoRA rank 16 与只冻结 llm 的规则 | `pi0/train` 的 `EMA`, `clip_and_step`, `select_trainable` | 完成 |
| [`fast/infer`](fast/infer/README.md) | 端到端 `infer(raw) → actions`, 解码步数与延时; `eval.py` 补 DROID 接口与 16 任务 rubric, LIBERO 复用 | `pi0/infer` 的 `Pi0Policy`, `RobotSpec`, `Env`, `run_episode` | 完成 |

**推理**
- action token → DCT 系数 → 连续 action chunk 的逆变换; 解码失败时的上游行为 (→ `tokenizer`, `infer`).
- 自回归解码: prefix 一次 prefill, 之后每步一个 token, 直到 EOS 或 256 步; greedy, 双臂任务 temperature 0.7 (→ `model`).
- 没有 action expert: logits 头是 embedding 的转置, 总参数 2,923,335,408 (→ `model`).

**训练**
- FAST 的两个超参 (scale γ = 10, BPE vocab 1024 / 发布的 FAST+ 为 2048) 与 BPE 训练细节 (→ `tokenizer`).
- 序列格式与 loss mask: 只在 `Action: … |` 段算 CE (→ `data`, `train`).
- 优化器与 schedule (论文 Appendix C), LIBERO 40k 步 / DROID 240k 步 @ 256 的规模 (→ `train`).

**评测**
- tokenizer 自身: 压缩率 (tokens / chunk) 与重建误差随 γ 的 trade-off (→ `tokenizer/eval.py`).
- 策略: LIBERO 四套二值成功率; DROID 16 任务 44 trial 的进度 rubric; 实机任务的进度百分比 (→ `infer/eval.py`).

**背景事实 (只陈述, 不写代码)**
- FAST+ 的训练混合 (约 1M 个 1 秒 chunk, 论文 Appendix A) 与泛化评测数据集 (Table III) (→ `tokenizer` README).
- FSQ / naive 分箱基线与 OpenVLA + FAST 的消融结论 (→ `tokenizer` README).
