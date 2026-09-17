# π0.5 一页总结: 相对 π0 与 FAST 的关键增量

论文 [π0.5: a Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054v1) (arXiv:2504.16054v1); 两级推理来自 [Hi Robot](https://arxiv.org/abs/2502.19417v2) (arXiv:2502.19417v2). 上游 [openpi](https://github.com/Physical-Intelligence/openpi) @ `215abfb217dbac7d5f1273282331b9b1866c0479` 只开源了 post-training 之后的 flow-only 推理与微调 (`pi0_config.py` 的 `pi05` 开关); 联合目标、两阶段 curriculum、高层推理只有论文.

π0.5 的骨架不变: 还是 PaliGemma (SigLIP So400m/14 + Gemma 2B) 加一个 300M 级的 action expert, 还是 flow matching 出 50 步 chunk. 变的是下面四件事, 其余全部 `from pi.pi0 / pi.fast import ...`.

## 四个增量

| # | 增量 | 一句话 | 解决什么 | module |
|---|---|---|---|---|
| 1 | **state 分箱进 prompt** | proprioceptive state 分 256 箱写成文本 `Task: <prompt>, State: <ints>;\nAction: `, action expert 不再有 state token; `max_token_len` 48 → 200; quantile 归一化 | 让 state 与文本共用一个入口, 预训练时整个模型是一个标准 VLM, 文本 / 检测 / 动作三类数据同一格式 | [`data`](data/README.md) |
| 2 | **adaRMSNorm 的 action expert** | timestep 单独过 `swish(W₂ swish(W₁ φ(τ)))`, 在 expert 每层的两个 RMSNorm 与 final norm 处用零初始化的 Dense(w → 3w) 给出 (scale, shift, gate), 残差乘 gate | τ 逐层注入而不是只在入口混一次; 零初始化 ⇒ 刚加进来的 expert 是恒等映射, post-training 从 "VLM + 被动 expert" 平滑起步 | [`expert`](expert/README.md) |
| 3 | **联合目标 + 两阶段** | loss = CE(文本 + FAST 动作 token) + α · flow MSE (Eq. 1); 预训练 280k 步 α = 0 (纯离散, 没有 expert), post-training 80k 步 α = 10 (expert 随机初始化); Fig. 18 三块 mask: expert 看 prefix 与彼此, 不看 FAST token, FAST 也不看 expert | 训练用 FAST token (比 flow 训练省算力, 语言跟随更强), 推理用 flow (10 步, 不用几十步自回归); 两全 | [`train`](train/README.md) |
| 4 | **同一模型两级推理** | 高层: 4 相机 + 高层 prompt → 自回归解码一句 subtask; 低层: 3 相机 + subtask + state 分箱 → 10 步 flow. Hi Robot 的调度: 高层每 1 s 或用户插话时重跑, `respond:` 口头回复剥离后再给低层 | 长任务 (10-15 分钟清厨房) 要先决定 "现在做哪一步"; Fig. 13: 完整两级 > implicit HL > human HL > no HL > GPT-4 | [`hier`](hier/README.md), [`infer`](infer/README.md) |

配套的数据增量 (不是模型): 训练混合从 π0 的机器人动作数据扩到 MM / ME / CE (动作) + HL (subtask + bbox 标注) + WD (caption / VQA / 检测) + VI (语言遥操作), 预训练 97.6% 的样本不来自目标平台 (Sec. I); 消融 (Fig. 10-11, 13) 表明 ME / CE 决定操作能力, WD 决定 OOD 物体的语言跟随, VI 决定高层质量.

## 没变的 (直接复用)

| 部件 | 来自 |
|---|---|
| SigLIP, Gemma 块, RoPE, GQA attention, `make_attn_mask`, 图像 resize / 增广, delta action, 32 维 padding | `pi/pi0` |
| 双 expert 的 `MoEBlock` / `MoEGemma` 布局, `posemb_sincos`, Euler 采样器, Beta(1.5, 1) 时间步, 优化器 / EMA / clip, `Env` / episode 循环 | `pi/pi0` |
| 256 分箱 state 文本, FAST id ↔ 词表尾部映射, FAST 编码 / 解码, quantile 归一化, postfix-only CE, 右对齐 + KV cache 的逐 token 解码 | `pi/fast` |

## 关键数字

| 项目 | 值 | 来源 |
|---|---|---|
| 总参数 | 3,353,433,872 = VLM 2,923,335,408 + expert 427,932,672 + 投影 2,165,792 | `expert` §1.5 (meta device); 论文称 expert "300M" (不含 adaRMS) |
| 序列 | prompt 200 token; chunk 50 步 @ 50 Hz; 动作 32 维 padding; 移动臂 18 / 19 维 | `pi0_config.py` L26, L39; Sec. IV-E |
| 训练 | 280k 离散预训练 + 80k post-training, α = 10; 优化器 / batch / 算力未披露 | Sec. IV-D |
| 数据 | MM 约 400 h / 约 100 家; 其余规模未披露 | Sec. IV-C |
| 推理 | 高层每 1 s (Hi Robot); Hi Robot 在 RTX 4090 上: 低层 73 ms / chunk, 高层 prefill 47 ms + 13.2 ms / token; π0.5 自己未披露 | Hi Robot App. B.3 |
| 评测 | mock home 四任务 rubric (8 / 4 / 3 / 5 分), 每任务 10 次, 12 个地点; 语言跟随 2 场景 × 5 物体; Hi Robot IA / TP 每任务 20 次 | App. B, C; Hi Robot Sec. 5.2 |

## 上游没有、本仓库按论文写的 (每条都在对应 README 的 §8 ledger)

- 联合目标与 Fig. 18 mask (`train`); expert 的 position id 跳过 FAST token 是推断.
- 高层 subtask 解码、`respond:` 剥离、1 s / 插话调度、`resume()` (`hier`).
- HL 目标文本格式 `Bounding boxes: <loc…>label\nSubtask: …` 从 Fig. 4 抄; control-mode 标签的位置; 4 相机槽位名 (`data`).
- 移动臂 18 / 19 维的排列与底盘速度的 delta / 绝对处理; 执行长度 k (`infer`).
- 优化器、学习率、batch、算力 (`train`).

## 目录与运行

```
pi05/
  data/     序列格式 (四种 layout), 相机槽位, batch        uv run python -m pi.pi05.data.data
  expert/   adaRMSNorm expert                              uv run python -m pi.pi05.expert.model
  hier/     Pi05 模型, 高层解码 + 低层采样, 调度            uv run python -m pi.pi05.hier.model
  infer/    端到端策略, 移动臂 spec, eval.py                uv run python -m pi.pi05.infer.model / .eval
  train/    Eq. 1, Fig. 18 mask, curriculum, 配置           uv run python -m pi.pi05.train.train
```

```
uv run pytest pi/pi05 -q      # 31 tests, CPU 约 20 s
```

每个 module 的 README 顺序固定: TL;DR → `figs/pipeline.png` (逐步 shape 与真实 tiny 数值) → 论点图 → 九节 (I/O 契约, 范围, 推理, 训练, 评测, cost, reference 映射, gap ledger, 概念表).
