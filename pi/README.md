# Tutorial on Physical Intelligence Series


## Overview

本目录提供 Physical Intelligence 系列关键技术的开源复现. 目的是帮助有 LLM / RL 背景的人快速了解 π0 ~ π0.x 训练、推理的实现细节, 并且有能力进行复现. 目前 cover 以下内容:

- π0: A Vision-Language-Action Flow Model for General Robot Control: https://arxiv.org/abs/2410.24164v1
  - 上游代码: [openpi](https://github.com/Physical-Intelligence/openpi) @ `215abfb217dbac7d5f1273282331b9b1866c0479`
  - 目录 `pi0/`, 分支 `topic/pi0`
- FAST: Efficient Action Tokenization for Vision-Language-Action Models: https://arxiv.org/abs/2501.09747v1
  - 上游代码: [openpi](https://github.com/Physical-Intelligence/openpi) @ `215abfb217dbac7d5f1273282331b9b1866c0479` (`pi0_fast.py`, `gemma_fast.py`, `tokenizer.py`), tokenizer 源码与发布权重: [physical-intelligence/fast](https://huggingface.co/physical-intelligence/fast) (HuggingFace, 2026-09-13 访问)
  - 目录 `fast/`, 分支 `topic/fast`
- π0.5: a Vision-Language-Action Model with Open-World Generalization: https://arxiv.org/abs/2504.16054v1
  - 上游代码: [openpi](https://github.com/Physical-Intelligence/openpi) @ `215abfb217dbac7d5f1273282331b9b1866c0479` (`pi0_config.py` 的 `pi05` 开关, `pi0.py` 的 adaRMSNorm 路径, `gemma.py` `RMSNorm(x, cond)`, `tokenizer.py` L22-L29 的离散 state prompt, `config.py` 的 `pi05_*` 训练配置). 上游只开源了 post-training 之后的 flow-matching 推理与微调; 联合目标 (FAST token + flow), 两阶段 curriculum, 高层 subtask 推理只有论文, 本仓库按论文写并进 ledger
  - 同系列: Hi Robot: Open-Ended Instruction Following with Hierarchical VLA Models: https://arxiv.org/abs/2502.19417v2 (两级推理的来源: 高层 VLM 出 subtask 文本, 低层 π0 出动作; 合成用户指令数据; 高层训练超参). 无开源代码
  - 目录 `pi05/`, 分支 `topic/pi05`
- π0.6* (RECAP): a VLA That Learns From Experience: https://arxiv.org/abs/2511.14759v2 (arXiv:2511.14759v2); 底座 π0.6 的模型卡: [PI06_model_card.pdf](https://website.pi-asset.com/pi06star/PI06_model_card.pdf) (2025-11-17, 2026-09-18 访问)
  - 上游代码: **没有**. openpi @ `215abfb217dbac7d5f1273282331b9b1866c0479` (2026-08-24 HEAD) 不含 π0.6 / RECAP / Gemma 3 (issue #789, #791 仍开). 本 topic 全部按论文 + 模型卡写, 每处推断进 ledger
  - 同系列: Knowledge Insulation: https://arxiv.org/abs/2505.23705v1 (arXiv:2505.23705v1, π0.6 的训练 recipe: 单阶段联合目标 + attention 内 stop-gradient + web 数据 co-training); Gemma 3 technical report: https://arxiv.org/abs/2503.19786v1 (arXiv:2503.19786v1, 新底座), 结构定义取 [google-deepmind/gemma](https://github.com/google-deepmind/gemma) @ `0513283af5afffa27390b6ede2facc35d0f16e08` (`gemma/gm/nn/_gemma.py` `Gemma3_4B` L221-L246, `Gemma3_1B` L169-L193; `_modules.py` `create_sliding_mask` L36-L52, QK-norm L160-L161 / L192-L194, sliding mask L258-L267, Block L400-L490; `vision/_vision.py` `VisionExit` L202-L231 avg-pool 到 256 token; `math/_positional_embeddings.py` `apply_rope` L23-L75 的 scale_factor)
  - 目录 `pi06/`, 分支 `topic/pi06`


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


### π0.5: 先用 FAST token 做离散预训练, 再加一个 adaRMSNorm 的 flow-matching expert 做 post-training; 同一个模型先出 subtask 文本再出动作

π0.5 (arXiv:2504.16054v1) 相对 π0 与 FAST 的增量只有四处, 其余全部 `from pi.pi0 / pi.fast import ...`:

1. **输入侧**: proprioceptive state 不再是 action expert 的一个连续 token, 而是分 256 箱后写进 prompt 文本 (`tokenizer.py` L22-L29: `Task: <prompt>, State: <bins>;\nAction: `), `max_token_len` 48 → 200 (`pi0_config.py` L39); 归一化改 quantile (`config.py` L187); prompt 里加 `<control mode> joint/end effector <control mode>` 标签 (论文 Sec. IV-C); 动作维度 padding 到 32 (`config.py` L868).
2. **action expert**: timestep 不再与 noisy action 拼接过 MLP, 而是单独过 `swish(W2 swish(W1 φ(τ)))`, 然后在 expert 的每一层每个 RMSNorm 处做 adaptive RMSNorm (零初始化的 Dense(w → 3w) 给出 scale / shift / gate, 残差乘 gate) (论文 Appendix E; `pi0.py` L93-L95, L162-L169; `gemma.py` L113-L131, L303-L311, L453-L459).
3. **训练目标**: 一条序列同时带 FAST 动作 token 与 expert 的连续动作 token, loss = 文本 + FAST token 的 CE + α · flow-matching MSE (论文 Eq. 1); attention 上 expert token 看 prefix 与彼此, 不看 FAST token, FAST token 也不看 expert (论文 Appendix E, Fig. 18). 两阶段: 280k 步 α = 0 (纯离散, 没有 expert), 再 80k 步 α = 10 (expert 随机初始化) (论文 Sec. IV-D). 上游没有这段代码.
4. **两级推理**: 同一个模型先以高层 prompt (如 "clean the kitchen") 自回归解码一句 subtask (如 "pick up the pillow"), 再以 subtask 为 prompt 走 10 步 flow matching 出 action chunk (论文 Sec. IV-A, IV-B; Fig. 3). 这条机制来自 Hi Robot (arXiv:2502.19417v2): 高层每 1 s 或用户插话时重跑, 口头回复从命令里剥离后再送低层 (Hi Robot Sec. 4.1-4.2). 上游没有这段代码.

| module | 关键技术 (增量) | 复用 π0 / FAST | 状态 |
|---|---|---|---|
| [`pi05/data`](pi05/data/README.md) | 离散 state 进 prompt 的序列格式 (LL 推理 / FAST 训练 / HL 文本目标 三种 postfix), `max_token_len` 200, control-mode 标签, quantile 归一化, 32 维 padding; HL 样本 (高层 prompt → `Subtask: …`, 可带 `<locXXXX>` 框); 高层用 4 个相机、低层用 3 个的槽位规则; Hi Robot 合成标注的格式 (背景) | `pi0/data` 图像 / delta / padding / 增广; `fast/data` 分箱, 词表尾部映射, FAST 编码; `fast/tokenizer` quantile | 完成 |
| [`pi05/expert`](pi05/expert/README.md) | adaRMSNorm action expert: 时间 MLP, 每层两个 norm + final norm 的 (scale, shift, gate), gated residual, 零初始化 ⇒ 初始时 expert 是恒等映射; 去掉 state token 与 `state_proj`; 参数量增量 | `pi0/action_expert` `MoEBlock` / `MoEGemma` / `posemb_sincos`; `pi0/vlm` attention | 完成 |
| [`pi05/hier`](pi05/hier/README.md) | 一个模型两级推理: HL = prefix-LM 逐 token 解码 subtask 文本 (greedy, EOS 停), LL = 以 subtask 为 prompt 的 10 步 Euler; Hi Robot 的调度 (1 s 或插话重跑 HL, `respond:` 口头回复剥离, 插话完成后回到原命令) | `fast/model` 的右对齐 / cache / 解码循环, tied head; `pi0/flow_matching` 采样器 | 完成 |
| [`pi05/infer`](pi05/infer/README.md) | 端到端 `Pi05Policy.infer(raw)` (openpi 的 flow-only 推理: prompt + state 分箱 + `Action: ` 前缀), 移动操作机器人 spec (18 / 19 维, 50 Hz, 4 相机), 延时表 (Hi Robot App. B.3); `eval.py`: mock-home 四任务 rubric (App. B), 语言跟随两指标 (App. C), Hi Robot 的 IA / TP, 带高层节奏的 episode 循环 | `pi0/infer` `Env` / `run_episode`; `fast/infer` rubric 聚合 | 完成 |
| [`pi05/train`](pi05/train/README.md) | 联合目标 Eq. 1 与 Fig. 18 三块 mask; 两阶段 curriculum 与每阶段的数据混合 (MM / ME / CE / HL / WD / VI); openpi `pi05_*` 微调超参; Hi Robot 高层策略超参 (App. C.2); cost 表 | `pi0/train` 优化器 / EMA / clip; `fast/train` CE; `pi0/flow_matching/train` 时间步 / 插值 | 完成 |

**推理**
- state 分箱进 prompt, 序列 `Task: …, State: …;\nAction: `, 200 token 上限; LL 推理时 FAST token 段为空 (→ `data`).
- adaRMSNorm 的时间注入: `φ(τ)` → 时间 MLP → 每层 (scale, shift, gate); 动作 token 直接线性投影; 51 → 50 个 suffix token (→ `expert`).
- 高层解码: 4 相机 + 高层 prompt → subtask 文本 (greedy, EOS); 低层: 3 相机 + subtask + state → 10 步 flow (→ `hier`).
- 高低层节奏: HL 每 1 s 或用户插话; 口头回复 `respond: …` 的剥离; LL 的 chunk 长度 50 与执行长度 (未披露 → ledger) (→ `hier`, `infer`).
- 部署: 18 / 19 维目标 (关节 + 夹爪 + 底盘速度 + 升降), 50 Hz PD 跟踪, 延时 (Hi Robot App. B.3: 4090 上 LL 73 ms, HL prefill 47 ms + 13.2 ms / token) (→ `infer`).

**训练**
- 三种 postfix 的序列构造与三个 mask (input / ar / loss), HL 样本与 bbox 目标, control-mode 标签, quantile 统计量, 32 维 padding, 增广参数 (论文 App. E 与 π0 相同) (→ `data`).
- Eq. 1: CE (文本 + FAST) + α · MSE, α = 0 → 10; Fig. 18 mask; Beta(1.5, 1) 时间步 s = 0.999 与 π0 相同 (→ `train`).
- 两阶段: 280k 离散预训练 → 80k post-training, expert 随机初始化, post-training 去掉 CE 数据, 加 VI; 预训练 97.6% 样本非 MM; VI 占 HL-MM 样本 11% (→ `train`).
- 优化器: 论文未披露 (→ ledger); openpi `pi05_libero` (warmup 10k, 5e-5 常数, batch 256, EMA 0.999, 30k) 与 `pi05_full_droid_finetune` (warmup 1k, 5e-5, batch 256, 100k); Hi Robot 高层 AdamW(.9, .95) 无 wd, clip 1, EMA 0.999, warmup 1k → 1e-5, batch 512, 8 × H100 约 2 小时 (→ `train`).

**评测**
- mock-home 四任务 rubric (Dishes in Sink 8 分, Items in Drawer 4 分, Laundry Basket 3 分, Make Bed 5 分), 每策略每任务 10 次, 3 个 mock + 3 个真实家庭; 语言跟随: 2 场景 × 5 物体, language following rate 与 success rate, 随机基线 20%; Hi Robot: Instruction Accuracy 与 Task Progress, 每任务每方法 20 次 (→ `infer/eval.py`).
- 高层策略的替代基线 (implicit HL / no HL / GPT-4 / human HL) 只陈述 (→ `hier` README).

**背景事实 (只陈述, 不写代码)**
- 数据混合的构成: MM 约 400 小时 / 约 100 个家庭, ME, CE (含 OXE), HL 人工标注, WD (CapsFusion, COCO, Cambrian-7M, PixMo, VQAv2, 室内 bbox), VI 语言遥操作 (→ `data`, `train` README).
- Hi Robot 的合成数据生成 (用大 VLM 给 (观测, skill 标签) 反推用户 prompt 与机器人回复, 按场景 / 回复类型分类) (→ `data` README).
- 消融与对比结论 (Fig. 8-13, 15-17): 环境数 scaling, ME / CE / WD / VI / HL 各自的贡献, π0-FAST+Flow 基线 (→ `train`, `hier` README).
- 机器人平台: 两种移动操作臂, 4 相机, 2 × 6 DoF 臂 + 夹爪, 全向底盘, 1-2 DoF 升降 (→ `infer` README).

### π0.6*: 用价值函数给每条数据打 advantage 标签, 让 VLA 从自己的部署经验里学 (RECAP)

π0.6* = π0.6 底座 + RECAP. 相对 π0.5, 底座换成 SigLIP 400M + Gemma 3 4B 加一个同深度 (34 层) 的 860M action expert, 文本 token 之间改为因果 attention, 训练改用 Knowledge Insulation 的单阶段联合目标 (FAST CE + flow MSE 同时训, expert 的梯度不回传底座, α = 1) (模型卡 §2; KI Sec. 5). RECAP 在此之上加三件事 (论文 Sec. IV-V): (1) 一个分布式价值函数 V(o, ℓ) 预测 "到成功还差多少步" (201 bin 的 CE, 回报按任务最大长度归一到 (−1, 0)); (2) 用 V 算每条样本的 advantage, 按任务分位数阈值二值化成 `Advantage: positive / negative` 文本 token, 放在子任务 ℓ̂ 之后、动作之前, 训练时 30% 概率省略; (3) 部署 → 人工标成功 / 失败 + 可选纠正 → 重训 V → 重训策略 (每轮都从预训练 checkpoint 出发) 的循环. 本 topic 只写增量, 其余 `from pi.pi05 / pi.pi0 / pi.fast import ...`.

| module | 关键技术 (增量) | 复用 π0.5 / π0 / FAST | 状态 |
|---|---|---|---|
| [`pi06/data`](pi06/data/README.md) | 序列: `Task: … State: …;\n` + `Subtask: ℓ̂\n` + `Advantage: positive/negative\n` + `Action: <FAST> \|` 的段落 id (图像 / 文本前缀 / 子任务 / advantage / FAST), 文本段因果; advantage token 的 30% dropout; metadata 段; 4 × 448×448 相机槽位; KI 三类样本 (仅 VLM / 仅动作 / 动作 + 子任务) 的 loss mask; Eq. 5 奖励 → 按任务归一的回报 → 201 bin 目标; episode 标签 (成功 / 纠正段) | `pi05/data` state 分箱前缀, HL 文本, quantile; `fast/data` FAST postfix; `pi0/data` resize / 增广 / delta / padding | 完成 |
| [`pi06/backbone`](pi06/backbone/README.md) | Gemma 3 最小实现: GQA 8 q / 4 kv, QK-norm, 5 local : 1 global 交错 + sliding window 1024, RoPE 10k (local) / 1M × 1/8 缩放 (global), attn 与 FFN 的 post-norm, 262,144 词表; SigLIP 448 → 1024 patch → avg-pool 到 256 token → RMSNorm + 线性投影; 图像双向 / 文本因果的 prefix mask; 34 层 adaRMSNorm expert; `Pi06` 模型 (prefix cache, 5 步 flow, 子任务解码); 参数量 parity 对 Gemma 3 Table 1 | `pi0/vlm` SigLIP / attention 核 / GeGLU; `pi05/expert` adaRMSNorm / 投影; `pi05/hier` 解码循环 | 完成 |
| [`pi06/value`](pi06/value/README.md) | 分布式价值函数: 小 Gemma 3 底座 (670M) + 201 类 head, 期望读出 V ∈ (−1, 0); `train.py`: Eq. 1 CE, advantage 估计 (预训练 N = T, post-training N = 50), 分位数阈值 ε_ℓ (30% / 40% / 10%), 指示 I_t 与纠正段强制 True, SFT 阶段固定 True, web 数据 co-training 的 loss mask | `pi06/backbone` Gemma 3 + prefix; `fast/train` CE | 完成 |
| [`pi06/infer`](pi06/infer/README.md) | 端到端 `Pi06Policy.infer`: 子任务解码 (低频) → advantage token 进序列 → 5 步 Euler; CFG (Eq. 13) 条件 / 无条件双 prefix, β ∈ [1.5, 2.5]; 静态双臂 spec (2 × 6 关节 + 2 夹爪 = 14 维, 3 相机, 50 Hz); 延时表 (H100 63 ms / chunk); `eval.py`: throughput (成功 / 小时) 与人工标注成功率, 五个任务与时限, box 四阶段, 假环境跑通 | `pi05/infer` 逆变换 / `Env`; `pi0/flow_matching` Euler | 完成 |
| [`pi06/train`](pi06/train/README.md) | KI 联合目标: attention 内对 expert 行 detach K_b / V_b (Eq. 5-6), α = 1, M^ℓ / M^act 三类样本; RECAP Algorithm 1 循环 (聚合 D_ℓ, 先 V 后 π, 始终从预训练 checkpoint 微调, SFT 阶段 I = True), 每任务 episode 数; cost 表; AWR / PPO (SPO 信任域, Eq. 11) 只陈述 | `pi05/train` `make_joint_mask` / `joint_forward` 的骨架; `pi0/train` 优化器 / EMA / clip | 待做 |

**推理**
- 序列: 前缀 `[BOS] Task: p, State: s;\n` (含 metadata s), 然后 `Subtask: ℓ̂\n`, `Advantage: positive\n`, `Action: `; 文本 token 因果, 图像双向, action token 双向; expert 看图像 + 全部文本 (含子任务与 advantage), 不看 FAST token (→ `data`, `backbone`).
- Gemma 3 4B 的前向: 34 层, 每 6 层里 5 层 sliding-window 1024 (RoPE base 10k) + 1 层全局 (RoPE base 1M, 位置除以 8), GQA 8 / 4 头 × 256, q / k 各过 RMSNorm, attn 与 FFN 输出各过 post-norm (→ `backbone`).
- 图像: 4 路 448×448 → SigLIP 400M (patch 14, 1024 patch) → 2×2 avg-pool → 256 token → RMSNorm + 线性到 2560 (Gemma 3 的做法; π0.6 是否沿用未披露 → ledger) (→ `backbone`).
- 5 步 Euler (π0.5 是 10 步); CFG: v = v_uncond + β (v_cond − v_uncond), β ∈ [1.5, 2.5], 只作用 flow 分支; H100 上 3 相机 63 ms / chunk (→ `infer`).
- 子任务预测频率低于动作生成 (具体节奏未披露 → ledger, 沿用 Hi Robot 的 1 s) (→ `infer`).
- 价值函数推理: 同样的 prefix (无 advantage token) → 201 类 logits → 期望 (→ `value`).

**训练**
- Eq. 5 奖励 (0 / −C_fail / −1), 回报按任务最大 episode 长度归一到 (−1, 0), 201 bin (→ `data`); Eq. 1 CE 训练 V, 期望读出 (→ `value`).
- advantage: 预训练 A = R_t − V(o_t) (N = T, 单次前向, 在线算), post-training N = 50 步 lookahead; ε_ℓ 使预训练 30% 的示范 / 微调 40% 的 rollout / T 恤任务 10% 为正 (10k 样本估计); 纠正段强制 I = True; SFT 阶段固定 True (→ `value`).
- advantage token 30% dropout 代替 loss 权重 α (→ `data`); KI 联合目标 Eq. 4 (α = 1) 与 Eq. 5-6 的 stop-gradient (→ `train`).
- RECAP Algorithm 1: 预训练 (全部示范, V 与 π), 每任务 SFT (I = True), 然后 K 轮 {采集, 从 V_pre 微调 V, 从 π_pre 微调 π}; episode 数: T 恤 300 × 4 台 / 轮 (无纠正), diverse 450 + 287 纠正, strict T 恤 ~1000 + 280 + 378 纠正 (3 台), box 600 + 360 纠正 / 轮 (3 台), cafe 429 纠正 + 414 自主 (1 轮) (→ `train`).
- 优化器 / lr / batch / 步数 / 算力: 全部未披露 → ledger (→ `train`, `value`).

**评测**
- throughput = 成功次数 / 小时 (含速度与成功率), 成功率 = 人工多维打分聚合成的二值标签; 任务与时限: laundry (T 恤 / 短裤) 200 s, diverse laundry (11 类, 报告 button-up shirt) 500 s, strict T 恤 (领口朝上) 200 s, double espresso 200 s, box assembly 600 s (四阶段: 取板 / 折盒 / 贴标 / 入箱) (→ `infer/eval.py`).
- 基线: π0.5, π0.6 (SFT), RL 预训练 π0.6*, offline RL + SFT, AWR, PPO (SPO 信任域 η = 0.01); 结论 (Fig. 7-12): throughput 翻倍以上, 失败率减半, strict T 恤 97% (→ `train` README, 只陈述).

**背景事实 (只陈述, 不写代码)**
- π0.6 的预训练数据: π0.5 的配方 + 更多机器人平台; 模型卡 Fig. 2-4 的 out-of-the-box 对比 (→ `backbone` README).
- Gemma 3 的预训练 / 蒸馏 recipe, Pan & Scan (推理期裁剪, π0.6 未提), 128k 上下文 (→ `backbone` README).
- 静态双臂平台: 2 × 6 DoF + 平行夹爪, 50 Hz 关节位置, 3 相机 (中间基座 + 双腕) (→ `infer` README).
- PPO 变体 Eq. 10-11 与 AWR 的实现细节 (→ `train` README).
