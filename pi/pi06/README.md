# π0.6* 一页总结: 相对 π0.5 的关键增量

论文 [π0.6*: a VLA That Learns From Experience](https://arxiv.org/abs/2511.14759v2) (arXiv:2511.14759v2); 底座 π0.6 的 [模型卡](https://website.pi-asset.com/pi06star/PI06_model_card.pdf) (2025-11-17); 训练配方 [Knowledge Insulation](https://arxiv.org/abs/2505.23705v1) (arXiv:2505.23705v1); 新底座 [Gemma 3](https://arxiv.org/abs/2503.19786v1) (arXiv:2503.19786v1), 结构定义取 [google-deepmind/gemma](https://github.com/google-deepmind/gemma) @ `0513283af5afffa27390b6ede2facc35d0f16e08`. 上游 [openpi](https://github.com/Physical-Intelligence/openpi) @ `215abfb217dbac7d5f1273282331b9b1866c0479` **没有** π0.6 / RECAP / Gemma 3 代码 (issue #789, #791), 本 topic 全部按论文 + 模型卡写, 每处推断进各 module 的 §8.

π0.6* = π0.6 + RECAP. 骨架仍是 "VLM 底座 + flow-matching action expert + 同一模型出子任务文本"; 变的是下面五件事, 其余 `from pi.pi05 / pi.pi0 / pi.fast import ...`.

## 五个增量

| # | 增量 | 一句话 | 解决什么 | module |
|---|---|---|---|---|
| 1 | **序列多两段: `Subtask: ℓ̂\n` 与 `Advantage: positive/negative\n`**, 文本全因果 | 五段序列 prefix → Subtask (CE) → Advantage (输入) → `Action: <FAST>` (CE); expert 读前三段, 不读 FAST; 指示 30% 随机省略; 图像 4 × 448; Eq. 5 奖励 → 按任务归一回报 → 201 bin | 把 "这条动作比行为策略好不好" 变成一个输入 token, 训练目标不变 (Eq. 3), 全部数据 (好的坏的) 都做监督学习, 推理只取 positive 支 | [`data`](data/README.md) |
| 2 | **Gemma 3 4B 底座 + 34 层 860M expert** | SigLIP 400M @ 448 → 2 × 2 pool → 256 token; GQA 8 / 4 × 256, QK-norm, 5 local (window 1024, RoPE 10k) : 1 global (RoPE 1M, 位置 / 8), post-norm, 262,144 词表; expert 同深度; 5 步 Euler | 更大更新的 VLM (out-of-the-box 就能折衣服、20% 完整组装纸盒, 模型卡 Fig. 2-4); H100 上 63 ms / chunk | [`backbone`](backbone/README.md) |
| 3 | **分布式价值函数** | 同架构、670M 小底座 + 201 类 head, CE 学 "归一化剩余步数", 期望读出 V ∈ (−1, 0); advantage = R − V (预训练) 或 50 步 lookahead; ε_ℓ 取分位数 (30% / 40% / 10% 为正); 纠正段 True, SFT 全 True | 只要每条 episode 一个成功标签, 就能给每一步打 "好 / 坏"; Monte Carlo 目标简单可靠 | [`value`](value/README.md) |
| 4 | **KI 单阶段联合目标 + stop-gradient** | CE (Subtask + FAST) + α · flow MSE 同时训, α = 1; expert 的 query 只看 sg(K_b), sg(V_b); web 数据 co-training | flow 的梯度不损伤 VLM (语言跟随), 训练又比纯 flow 快 7.5× (KI Fig. 4, 6); π0.5 的两阶段 α 0 → 10 被一个阶段取代 | [`train`](train/README.md) |
| 5 | **RECAP 循环** | 预训练 (V_pre + π_pre 带指示) → 每任务 SFT (I = True) → K 轮 {部署采集 (自主 + 可选纠正, 人工标签) → 从 V_pre 重训 V → 从 π_pre 重训 π}; 部署固定 I = True, 可选 CFG β ∈ [1.5, 2.5] | 让策略修正自己在部署中真正犯的错、变快: throughput 翻倍以上, 失败率减半, strict T 恤 97% (Fig. 7-12) | [`train`](train/README.md), [`infer`](infer/README.md) |

评测 (Sec. VI): 五个任务各有时限 (laundry 200 s / diverse 500 s / strict T 恤 200 s / espresso 200 s / box 600 s), 指标是 **throughput (成功 / 小时)** 与人工聚合的 **成功率**, box 分四阶段; 每条评测 episode 直接成为下一轮数据 (`infer/eval.py`).

## 没变的 (直接复用)

| 部件 | 来自 |
|---|---|
| `Task: …, State: <256 分箱>;\n` 前缀, `Subtask:` 措辞, quantile 归一化, FAST postfix 与逆变换, 缺相机 mask 规则, delta / 32 维 padding / 增广 | `pi/pi05/data`, `pi/fast`, `pi/pi0/data` |
| 双 expert 的 "只在 attention 相遇" 布局, adaRMSNorm expert 与投影, `posemb_sincos`, Euler 采样器, Beta(1.5, 1) 时间步 | `pi/pi0`, `pi/pi05/expert` |
| 右对齐 prefill 的逐 token 解码, tied head, 优化器 / EMA / clip, `Env` / episode 循环, 逆变换 | `pi/fast/model`, `pi/pi0/train`, `pi/pi0/infer`, `pi/pi05/infer` |
| 高层 1 s 的节奏 | Hi Robot (π0.6 自己的频率未披露) |

## 关键数字

| 项目 | 值 | 来源 |
|---|---|---|
| 底座 | Gemma 3 4B: 非 embedding 3,209,010,688 + embedding 671,088,640; SigLIP 416,277,360 | 报告 Table 1 (3,209M / 675M / 417M); `backbone` 闭式 |
| expert | "约 860M", 34 层 → 宽 1024 / mlp 4096 的 adaRMSNorm 块 = 859,082,752 (推断) | 模型卡 §2; `backbone` §8 |
| 总参数 | 约 5,155,459,440 (π0.5: 3,353,433,872) | 本仓库 |
| 价值函数 | 670M 底座 (替身 Gemma 3 1B: 697,896,064 + 301,989,888) + head 231,753 | Sec. V-C; 报告 Table 1 |
| 序列 | 4 × 448 图像 (各 256 token); 文本 200 (沿用); 201 bin; 30% 指示 dropout; ε_ℓ 30% / 40% / 10%, 10k 样本; N = 50 | 模型卡 §2; Sec. IV-A; App. F |
| 推理 | 5 步 Euler; 3 相机 63 ms / chunk, 1 × H100; β = 1 默认, CFG β ∈ [1.5, 2.5] | 模型卡 §2; App. E |
| 训练 | α = 1; KI 每步 +20% 算力, π0 需 7.5× 步数; 优化器 / lr / batch / 步数 / 算力未披露 | KI Sec. 5.2, 7, Fig. 6b |
| 经验数据 | 每轮: T 恤 300 (4 台) / diverse 450 + 287 纠正 / strict ~1000 + 280 + 378 (3 台) / box 600 + 360 (3 台) / cafe 414 + 429 (1 轮) | Sec. VI-C.2, App. F |
| 评测 | 时限 200 / 500 / 200 / 200 / 600 s; throughput 与成功率, 标准误; espresso 连续 13 h | Sec. VI-A, VI-C, Sec. I |

## 上游没有、本仓库按论文写的 (每条都在对应 README 的 §8)

- 段分隔符 (`\n`), metadata 拼法, `max_token_len` / H / 动作维数沿用 π0.5, C_fail, 归一化 clip, bin 网格 (`data`).
- expert 宽度 / 块结构, 448 的 token 数 (pool 到 256), 图像与文本的排布, 解码停止规则, FAST id → Gemma 3 词表 (`backbone`).
- 670M 底座配置, 读出位置, 预训练 advantage 的求和下标 (论文写 t' = 0), N-step 越界规则, 阈值两种说法 (`value`).
- 14 维排列, 子任务频率, 执行长度 k, CFG 的速度场实现, 打分聚合, T_max = 时限 × 50 Hz, prompt 措辞 (`infer`).
- α_η 权重, expert 位置 id, 纠正段是否进 CE, SFT 阶段是否 dropout, 全部优化器超参; AWR / PPO 只陈述 (`train`).

## 目录与运行

```
pi06/
  data/       五段序列, advantage token, 448 图像, Eq. 5 → 201 bin       uv run python -m pi.pi06.data.data
  backbone/   Gemma 3 4B, 视觉路径, mask, Pi06 模型                       uv run python -m pi.pi06.backbone.model
  value/      价值函数, Eq. 1, advantage / 阈值 / 指示                     uv run python -m pi.pi06.value.model / .train
  infer/      Pi06Policy (子任务 + Advantage + CFG + 5 步), eval.py         uv run python -m pi.pi06.infer.model / .eval
  train/      KI 联合目标 (sg, α = 1), 阶段指示, Algorithm 1               uv run python -m pi.pi06.train.train
```

```
uv run pytest pi/pi06 -q      # 35 tests, CPU 约 10 s
```

每个 module 的 README 顺序固定: TL;DR → `figs/pipeline.png` (逐步 shape 与真实 tiny 数值) → 论点图 → 九节 (I/O 契约, 范围, 推理, 训练, 评测, cost, reference 映射, gap ledger, 概念表).
