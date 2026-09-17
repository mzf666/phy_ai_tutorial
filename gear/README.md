# Tutorial on NVIDIA GEAR Series

## Overview

本目录提供 NVIDIA GEAR lab 机器人基础模型系列关键技术的开源复现. 目的是帮助有 LLM / RL 背景的人快速了解 "用人类第一视角视频当主要监督信号" 这条路线的训练、推理实现细节, 并且有能力进行复现. 目前 cover 以下内容:

- EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data: https://arxiv.org/abs/2602.16710v1
  - 上游代码: **EgoScale 本身未开源**. 项目页 https://research.nvidia.com/labs/gear/egoscale/ 的 GitHub 链接标注 "Coming Soon!" (2026-09-17 访问). 论文 §2.3 与附录 D.1 明确本模型架构 "similar to GR00T N1", 本体适配器 "following the design of GR00T-N1 and N1.5", 因此本目录的架构类代码以 GR00T N1.5 的开源实现为上游对照: [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) @ `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` (tag `n1.5-release`, Apache-2.0)
  - backbone 论文: GR00T N1: An Open Foundation Model for Generalist Humanoid Robots: https://arxiv.org/abs/2503.14734
  - 数据来源论文: EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video: https://arxiv.org/abs/2505.11709 (Stage I 中 829 小时的高精度部分)
  - 目录 `egoscale/`, 分支 `topic/egoscale`
- GR00T N1 / N1.5 本身: 未开始.

与 [`pi/`](../pi/README.md) 的关系: π0 系列证明了 "VLM 预训练 + flow-matching policy" 这个范式, 数据来源是机器人遥操作数据. EgoScale 换掉的是**数据来源与动作空间**, 而不是范式: 它把 20,854 小时的人类第一视角视频当作预训练数据, 把 "相对腕部 SE(3) 运动 + 22 自由度重定向手部关节角" 当作跨本体的统一动作空间. 因此本目录里 flow matching 的数学部分只写与 [`pi/pi0/flow_matching`](../pi/pi0/flow_matching/README.md) 的差异表, 不重复推导.


## 关键技术栈

### EgoScale: 把人类第一视角视频当作可预测的监督信号

论文的三条结论: (1) 人类数据规模与人类动作预测验证损失呈对数线性关系 `L = 0.024 − 0.003·ln(D)`, `R² = 0.9983` (论文 §3.3 式 (1)); (2) 该验证损失与真机下游表现强相关, 平均任务完成分从 1k 小时的 0.30 升到 20k 小时的 0.71, 未见饱和 (论文 §3.3); (3) 大规模人类预训练 + 少量对齐的人机 mid-training, 相对无预训练基线把 22 自由度灵巧手的平均成功率提升 54% (论文摘要), 并涌现单演示 (one-shot) 迁移能力 (论文 §3.4).

按 module 拆分, 每个 module 一个目录, 按下面顺序开发:

| module | 关键技术 | 状态 |
|---|---|---|
| [`egoscale/action`](egoscale/action/README.md) | 人类动作表示: 相机位姿 × 21 手部关键点 → 世界系腕部位姿; chunk 内相对腕部 SE(3) 运动; URDF 正运动学 + 22 关节非线性规划的手部重定向 (关节限位、warm start、一阶指数滤波); 三种动作空间 (wrist-only / fingertip / full retargeted joints) 的并排实现与消融 | 完成 |
| [`egoscale/data`](egoscale/data/README.md) | 统一 human / robot 样本: 头部 + 双腕三相机槽位; 人类数据无本体感受时的可学习占位 token; 多本体 state / action padding 与 `embodiment_id`, `action_mask`; 归一化统计量; Stage I / Stage II 数据混合 | 未开始 |
| [`egoscale/backbone`](egoscale/backbone/README.md) | Eagle-2.5 视觉语言 backbone 的最小实现与中间层特征抽取, VL LayerNorm 与 VL self-attention; checkpoint 视为给定 | 未开始 |
| [`egoscale/dit`](egoscale/dit/README.md) | DiT action expert: cross-attention / self-attention 交替块, AdaLN 时间步条件, 本体专属 state encoder / action encoder / action decoder, `[state, future_tokens, action]` token 布局; flow matching 目标与 K 步 Euler 采样 (与 `pi/pi0/flow_matching` 的差异表) | 未开始 |
| [`egoscale/train`](egoscale/train/README.md) | 三阶段 curriculum (预训练 / 对齐 mid-training / post-training) 的 batch size、学习率与逐阶段冻结表; scaling law 拟合与外推; cost 表 | 未开始 |
| [`egoscale/infer`](egoscale/infer/README.md) | 端到端推理链路装配, 参数量表, 延时表; `eval.py`: 五个灵巧任务的 additive / progress-based rubric, one-shot 两任务, G1 跨本体两任务, 人类验证损失协议 | 未开始 |

**推理**
- 观测 `o_t = (I_t, l_t)` 经 Eagle-2.5 编码为视觉语言嵌入 `φ_t`, 取中间层而非最后一层 (→ `backbone`).
- `vlln` LayerNorm 与 VL self-attention 对 backbone 输出的后处理 (→ `backbone`, `dit`).
- token 拼接布局 `[state(1), future_tokens(32), action(H)]`, 人类样本的 state 位置换成可学习占位 token (→ `data`, `dit`).
- DiT 的 cross-attention 条件在 `φ_t` 上, self-attention 只在 state / action token 之间; AdaLN 注入离散化后的时间步 (→ `dit`).
- K 步前向 Euler 积分, 模型输出解释为速度场而非动作本身 (→ `dit`).
- 输出端本体专属 MLP 解码器: 22 自由度 Sharpa 手与 7 自由度 G1 三指手共享同一个 DiT, 只换 encoder / decoder (→ `dit`, `infer`).
- 相对腕部动作在机器人侧如何变回可执行的增量末端位姿命令 (→ `action`, `infer`).
- 参数量与端到端延时 (→ `infer`).

**训练**
- flow matching 加噪 `A^τ = τ·A + (1−τ)·ε`, 回归目标为 `A − ε`, 时间步分布 `Beta(1.5, 1)` 经 `s = 0.999` 变换后离散化到 1000 个桶 (→ `dit`).
- loss 是被 `action_mask` 加权的 MSE: 跨本体 padding 出来的维度不计入分母 (→ `dit`).
- 归一化统计量的来源与计算方式 (→ `data`).
- 动作 chunk 的切分、相对腕部运动的参考帧选取 (→ `action`, `data`).
- Stage I 人类预训练: 100k 步, 全局 batch 8,192, 学习率 5e-5, 全参数解冻 (→ `train`).
- Stage II 对齐 mid-training: 50k 步, batch 2,048, 学习率 3e-5, 冻结视觉语言 backbone, 只更新视觉编码器、DiT 与 state / action 编解码器 (→ `train`).
- Stage III post-training: 10k 步, batch 512, 学习率 3e-5, 是否冻结视觉编码器取决于有无 mid-training (→ `train`).
- Stage I / Stage II 的数据混合与采样权重 (→ `data`).

**评测**
- 人类验证损失协议: 2,000 条留出的第一视角 episode, 每条随机抽 20 个时刻, 每个时刻从 flow-matching 策略采 16 个样本、对预测的动作 chunk 取平均后与真值算 MSE (→ `infer/eval.py`).
- 五个 R1Pro 灵巧任务 (Shirt Rolling / Card Sorting / Tong Fruit Transfer / Bottle Cap Unscrewing / Syringe Liquid Transfer) 的任务完成分 rubric, 分 additive 与 progress-based 两类, 以及二值成功率的定义与聚合 (→ `infer/eval.py`).
- one-shot 两任务 (T-Shirt Folding / Bottle Cap Unscrewing) 与 G1 两任务 (Pen in Bin / Dish Handover in Rack) 的 rubric (→ `infer/eval.py`).
- 每个方法两个随机种子, 每个 checkpoint 10 次试验 (Bottle 为 4 个瓶子各 4 次共 16 次), 以图像叠加方式统一初始场景 (→ `infer/eval.py`).
- scaling law 的拟合方式与 `R²`, 以及在 1k / 2k / 4k / 10k / 20k 小时五个数据量上的验证损失曲线 (→ `train`).

**背景事实 (只陈述, 不写代码)**
- Eagle-2.5 视觉语言模型的构成与预训练 recipe; GR00T N1 用的 Eagle-2 由 SmolLM2 与 SigLIP-2 微调而来, 图像 224×224 经 pixel shuffle 得到每帧 64 个图像 token, 取第 12 层 LLM 表征 (→ `backbone` README).
- Stage I 数据构成: 20,854 小时, 其中 in-the-wild 部分覆盖 9,869 个场景、6,015 个任务、43,237 个物体, 30 FPS; 另含 EgoDex 829 小时 (194 个桌面任务, Apple Vision Pro 采集) (→ `data` README).
- 相机运动与手部姿态由 "off-the-shelf" 的 SLAM 与手部姿态估计 pipeline 恢复, 论文未指明具体方法 (→ `action` README).
- Stage II 对齐数据采集栈: 344 个桌面任务, 每任务约 30 条人类轨迹与 5 条机器人轨迹, 合计约 50 小时人类 + 4 小时机器人; Vive tracker 提供腕部位姿, Manus 手套记录 25 个关节 transform (→ `data` README).
- 机器人硬件: Galaxea R1Pro 双臂轮式人形 (固定底盘与躯干, 双 7 自由度臂, 相对末端位姿控制) 配 22 自由度 Sharpa Wave 手; Unitree G1 配 7 自由度三指手, 下肢平衡与运动由单独训练的 Homie 策略接管; 三路 RGB 相机为一个 OAK-D-Wide 头部相机与两个 OAK-1-Wide 腕部相机 (→ `infer` README).


## 修订记录

| 日期 | 改了什么 | 原因 |
|---|---|---|
| 2026-09-17 | 建立大纲, 六个 module | 初次拆分 |
