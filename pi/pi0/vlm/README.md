# π0 · vlm: SigLIP + Gemma 2B, 也就是 PaliGemma 的结构

本 module 复现 π0 的 VLM backbone: 三张图经 SigLIP So400m/14 变成 3×256 个 token, 与指令 token 拼成 "prefix", 过一遍 Gemma 2B, 得到隐状态和 KV cache. π0 不用 Gemma 生成文字, 只用它的隐状态和 KV cache 给 action expert 看 (见 `../action_expert`).

上游: [openpi](https://github.com/Physical-Intelligence/openpi) @ `215abfb217dbac7d5f1273282331b9b1866c0479` (`openpi@215abfb`) 的 `src/openpi/models/siglip.py`, `gemma.py`, `pi0.py`; 论文 π0 [arXiv:2410.24164v1](https://arxiv.org/abs/2410.24164v1) Appendix B, PaliGemma [arXiv:2407.07726v2](https://arxiv.org/abs/2407.07726v2), Gemma [arXiv:2403.08295v4](https://arxiv.org/abs/2403.08295v4). PyTorch 重写, 不 import openpi.

![architecture](figs/architecture.png)

## 1. I/O 契约

**`SigLIP(cfg)(images)`**

| 名称 | shape / dtype | 取值 | 说明 |
|---|---|---|---|
| 入 `images` | float32[B, 224, 224, 3] | [−1, 1] | `../data` 的输出, channels-last |
| 出 | float32[B, 256, 2048] | 任意 | 16×16 个 patch token, 已由 ViT head 投影到 Gemma 宽度. 无 pooling, 无 CLS |

**`Gemma(cfg).embed(tokens)`**: int64[B, 48] → float32[B, 48, 2048], 查表后乘 √2048.

**`Gemma(cfg)(x, positions, mask, kv_cache=None)`**

| 名称 | shape / dtype | 说明 |
|---|---|---|
| 入 `x` | float32[B, T, 2048] | 本次要算的 token 的嵌入 |
| 入 `positions` | int64[B, T] | RoPE 位置, 由 `cumsum(input_mask) − 1` 给出, 填充 token 不占位 |
| 入 `mask` | bool[B, T, S] | query i 能否看 key j; S = cache 长度 + T |
| 入 `kv_cache` | list[depth] of (k, v), 各 float32[B, S_cache, 1, 256] | 上一次前向留下的 key / value |
| 出 hidden | float32[B, T, 2048] | 过 final RMSNorm 后的隐状态 |
| 出 kv_cache | list[depth] of (k, v), 各 float32[B, S, 1, 256] | 供后续 token (action expert) attend |

**`PaliGemma(vit_cfg, gemma_cfg)(images, image_masks, tokens, token_mask)`**: 输入即 `../data` 的 `Observation` 的四个字段 (不含 state), 输出 hidden float32[B, 816, 2048] 和 KV cache; 816 = 3×256 + 48. `embed_prefix` 另外返回 `input_mask` bool[B, 816] 与 `ar_mask` bool[816] (全 False: prefix 内全双向).

常量: SigLIP So400m/14 = {width 1152, depth 27, mlp 4304, heads 16, patch 14} (`openpi@215abfb siglip.py` L308-L370); Gemma 2B = {width 2048, depth 18, mlp 16384, heads 8, kv heads 1, head_dim 256, vocab 257152} (`gemma.py` L41, L79-L87).

### 1.1 `make_attn_mask` 的逻辑与动机

一个一维向量 `mask_ar` 描述整张二维 mask. `mask_ar[j] = True` 表示 token j 开一个新块, 它之前的 token 不能看它; `cumsum(mask_ar)` 给每个 token 一个块编号; query i 能看 key j 当且仅当 `块(j) ≤ 块(i)` 且两者都是有效 token (`input_mask`). 也就是块之间因果, 块内双向.

计算例子, 6 个 token: 两个图像、一个文本、一个 state、两个 action:

```
token      : img0  img1  txt   state  act0  act1
mask_ar    :  0     0     0      1     1     0
cumsum(块) :  0     0     0      1     2     2

            img0 img1 txt state act0 act1      (行 query, 列 key)
img0   块0    1    1   1    0    0    0
img1   块0    1    1   1    0    0    0
txt    块0    1    1   1    0    0    0
state  块1    1    1   1    1    0    0
act0   块2    1    1   1    1    1    1
act1   块2    1    1   1    1    1    1
```

前缀三个 token 互相全看但看不到 state 和 action; state 看前缀和自己; 两个 action 看全部且彼此双向. 这正是 π0 论文 Appendix B 的三块结构. 若 `txt` 是 padding (`input_mask` False), 第 2 行和第 2 列整体置 0.

只换 `mask_ar` 就得到所有常见结构: `[0 0 0 0 0 0]` 全双向 (编码器); `[1 1 1 1 1 1]` 纯因果 (普通 LLM); `[0 0 0 1 1 1]` 前缀双向后缀因果 (PaliGemma 的 prefix-LM). π0 真实的 `mask_ar` 长 867: 768 个图像 token 和 48 个文本 token 全 0, state 为 1, 第一个 action 为 1, 其余 49 个 action 为 0.

动机 (函数来自 big_vision, PaliGemma 训练 prefix-LM 就用它): 一是可组合, `embed_prefix` 和 `embed_suffix` 各返回自己的一维 `ar_mask`, `concat` 后调一次就是整张矩阵, 不用手写二维块; 二是块编号只依赖 `cumsum`, 推理时可以只为 suffix 的行建 mask、列上拼接已缓存的 prefix (`pi0.py` `sample_actions` 的 `full_attn_mask`), KV cache 和 mask 的语义天然一致.

## 2. 复现范围

| 有代码 | 只有事实 (背景) |
|---|---|
| SigLIP ViT 前向 (patch conv, 学习式位置嵌入, 27 层 pre-LN block, head 投影) | SigLIP 的对比预训练 (sigmoid loss, WebLI) |
| Gemma 2B 前向 (RMSNorm, GQA/MQA attention, RoPE, GeGLU, √width 嵌入缩放) | Gemma 2B 的预训练 (3T token, TPUv5e) |
| prefix 的拼接、attention mask、位置编号、KV cache | PaliGemma 三阶段预训练 recipe (见第 4.3 节) |
| `tiny` 配置整条前向 | checkpoint 的下载与权重映射 |

## 3. 推理侧

一次动作 chunk 的推理里, 本 module 只跑一次 (`pi0.py` L233-L237): 编码三张图, prefix 前向, 存下 18 层的 (k, v). 之后 10 步去噪只跑 action expert, 每步 attend 到这份 cache (见 `../flow_matching`, `../infer`).

延时 (论文 Table I, RTX 4090, 三张图): image encoders 14 ms, observation forward pass 32 ms. 两项合计 46 ms, 占端到端 73 ms 的大头; 这就是 KV cache 存在的理由.

参数量 (本仓库按 openpi 配置解析计算, `test_parity.py` 断言):

| 部件 | 参数量 | 对照 |
|---|---|---|
| SigLIP So400m/14 含 head | 414,803,696 | PaliGemma 论文称 "SigLIP: 400M Vision Model" |
| Gemma 2B 非嵌入 | 1,981,884,416 | Gemma 论文 Table 2, 精确相等 |
| Gemma 词表嵌入 | 526,647,296 = 257152 × 2048 | Gemma 论文为 524,550,144 (词表 256128); PaliGemma 多出 1024 个 `<loc>` + 128 个 `<seg>` token |
| 合计 VLM | 2,923,335,408 | PaliGemma "3B" |
| action expert (Gemma 300M, 无嵌入) | 311M | `gemma.py` L70 注释 "311M params" |

## 4. 训练侧

### 4.1 π0 训练时 backbone 怎么用

- 初始化: `PaliGemma.img` 与 `PaliGemma.llm` 从官方 PaliGemma 224px 预训练 checkpoint 加载 (`gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz`, `openpi@215abfb weight_loaders.py` L58-L73); 同名权重覆盖, action expert 权重不在 checkpoint 里, 保持随机初始化.
- 全部参与训练, 不冻结 (LoRA 变体除外, `pi0_config.py` L88-L117). 论文 Appendix B: "The images and language prompt are routed to the larger VLM backbone, which we initialize from PaliGemma."
- ViT 的 `train=False` 恒定 (`pi0.py` L114): SigLIP 里没有 dropout 起作用, 训练和推理前向一致.

### 4.2 数据预处理

全部在 `../data`. 本 module 只要求图像已是 [−1, 1] 的 224×224, 指令已是 48 个 token id.

### 4.3 背景事实: 三个预训练 (只陈述, 不复现)

π0 拿到的 backbone 经历了三次预训练: SigLIP 单独训图像塔, Gemma 2B 单独训语言模型, PaliGemma 把两者接起来再训. 每一层都写清 **任务 (loss)**, **数据组织形式**, **规模**.

**(a) SigLIP So400m: 图文对比预训练** ([arXiv:2303.15343v4](https://arxiv.org/abs/2303.15343v4))

| 项目 | 内容 | 来源 |
|---|---|---|
| 任务 | 图文对比学习. 每个 mini-batch 里 \|B\| 张图配 \|B\| 段文本, 匹配的 (I_i, T_i) 为正样本, 其余所有 (I_i, T_j≠i) 为负样本 | SigLIP Sec. 3 |
| loss | pairwise sigmoid, 不做 softmax 归一化, 把问题变成 \|B\|² 个独立的二分类: `logits = t · x yᵀ + b`, `labels = 2I − 1`, `L = −Σ log σ(labels · logits) / \|B\|`. x, y 是 L2 归一化后的图像 / 文本嵌入; 温度 t 与偏置 b 可学习, 初值 log 10 与 −10, 以抵消负样本远多于正样本的失衡 | SigLIP Sec. 3.2, Algorithm 1 |
| 为什么不用 softmax | 不需要跨设备 gather 全局归一化项, 内存省, batch 能开大; 小 batch (< 16k) 下也更好; 32k 之后收益饱和 | SigLIP Sec. 1, Sec. 4.2 |
| 数据组织 | WebLI 图文对, 仅英文; 图像 resize 到 224×224 (So400m 训练用 256 个 patch, 即 224/14 网格); 文本用 32k 词表的 sentencepiece, So400m 保留 64 个 token | SigLIP Sec. 4.1, Sec. 4.6 |
| 图像塔 / 文本塔 | ViT-So400m/14 与同尺寸的文本 Transformer | SigLIP Sec. 4.6 |
| 规模 | batch 32k, 400 亿样本; 再以 100 倍小的学习率、无 weight decay 在目标分辨率上多训 50 亿样本 | SigLIP Sec. 4.6 |
| 与 PaliGemma 的关系 | PaliGemma 只取图像塔, 丢掉文本塔; 论文引用的即此 "shape optimized" So400m | PaliGemma Sec. 3.1 |

**(b) Gemma 2B: 自回归语言模型预训练** ([arXiv:2403.08295v4](https://arxiv.org/abs/2403.08295v4))

| 项目 | 内容 | 来源 |
|---|---|---|
| 任务 | 标准 decoder-only 下一 token 预测, 因果 mask, 上下文 8192 | Gemma Sec. 2 |
| 数据组织 | 3T token, 以英文为主的网页文档、数学、代码. 不是多模态, 也未针对多语言优化 | Gemma Sec. 3 "Training Data" |
| 分词 | Gemini 的 SentencePiece 子集, 256k 词表; 切分数字, 不去多余空白, 未知 token 走 byte 级 | Gemma Sec. 3 |
| 过滤 | 启发式 + 模型分类器去除有害 / 低质内容与个人信息; 从预训练集中剔除所有评测集, 做污染分析 | Gemma Sec. 3 "Filtering" |
| 混合调度 | 训练分阶段改变语料混合, 末期提高高质量数据权重; 混合比例由 2B / 7B 上的消融决定 | Gemma Sec. 3 |
| 算力 | 512 个 TPUv5e (2 个 pod), 256 路数据并行, 优化器状态 ZeRO-3 式分片 | Gemma Sec. 3 "Training Infrastructure" |
| π0 用的版本 | PaliGemma 用 Gemma-2B v1.0 原始预训练 checkpoint, 不是指令微调版 | PaliGemma Sec. 3.1 |

**(c) PaliGemma: 多模态预训练** ([arXiv:2407.07726v2](https://arxiv.org/abs/2407.07726v2))

| 项目 | 内容 | 来源 |
|---|---|---|
| 连接方式 | 零初始化线性层把 SigLIP token 投到 Gemma 宽度; 试过 MLP 无收益 | PaliGemma Sec. 3.1 |
| 序列格式 | `[image tokens..., BOS, prefix tokens..., SEP, suffix tokens..., EOS, PAD...]`; 图像 + prefix 全双向, suffix 自回归 (prefix-LM). π0 的 `make_attn_mask` 就是这套 mask 的实现 | PaliGemma Sec. 3.1, Fig. 2 |
| 任务 (loss) | 只在 suffix 上算下一 token 预测 loss; 每个任务用唯一的文本前缀区分, 避免不同技能的信号冲突 | PaliGemma Sec. 3.1, Sec. 3.2.5 |
| 任务混合 | `caption {lang}`: WebLI 100+ 种语言与 CC3M-35L 的图像描述; `ocr`: 图上文字按阅读顺序拼接; `answer en {question}`: CC3M-35L 生成的 VQA 与 OpenImages 上的物体列举 / 存在 / 计数问答; `question {lang} {answer}`: 给答案生成问题; `detect {thing}; ...`: Pix2Seq 式多目标检测, 坐标离散为 1024 个 `<loc>` token; `segment {thing}; ...`: 实例分割, 128 个 `<seg>` token; `caption <ymin><xmin><ymax><xmax>`: 框内 grounded caption. 检测 / 分割数据由 OWL-ViTv2 与 SAM 伪标注 | PaliGemma Sec. 3.2.5 |
| 数据组织 | 一条样本 = 一张图 + 一段带任务前缀的文本 (prefix) + 目标文本 (suffix); Stage 1 为 224 px, N_img = 256, N_txt = 128 | PaliGemma Sec. 3.2.2 |
| 数据来源声明 | 没有任何任务的输出来自更大的商业 VLM (区别于 LLaVA 用 GPT-4 生成数据); 从预训练集中移除所有与下游评测集近重复的图像; 部分数据集不公开 | PaliGemma Sec. 3.2.5, Sec. 3.2.6 |
| Stage 0 | 单模态, 直接用 (a) (b) 的公开 checkpoint | PaliGemma Sec. 3.2.1 |
| Stage 1 | 224 px, 10 亿样本, **不冻结**图像编码器 (与 PaLI 惯例不同, 理由是 caption 等任务能给图像塔补上空间关系信号); 图像塔学习率先慢速 warm-up 以免被早期错位梯度破坏 | PaliGemma Sec. 3.2.2 |
| Stage 2 | 448 px 再 5000 万样本, 896 px 再 1000 万样本, 任务混合相同但上调高分辨率任务. π0 用的 224 checkpoint 未经此阶段 | PaliGemma Sec. 3.2.3 |
| 学习率 | 各阶段串成一条 rsqrt 的 "无限" 调度, 阶段间不衰减 | PaliGemma Sec. 3.2.6 |
| 算力与精度 | TPUv5e-256, Stage 1 略少于 3 天 (约 350B token), 每个 Stage 2 15 小时; MFU 55%, 5189 token/s/设备; 参数与优化器状态 float32, 推理 bfloat16 无损 | PaliGemma Sec. 3.4 |
| 图像预处理 | 预训练时随机化 resize 方法、JPEG 编码, 加很轻的 inception crop, 以免对框架差异敏感 | PaliGemma Sec. 3.4 |

π0 从 (c) 的 Stage 1 224 px checkpoint 出发 (`pt_224.npz`), 然后用机器人数据继续训练整个 backbone 加 action expert; 见 `../train`.

## 5. 评测

`test_parity.py` (CPU, 约 2 秒). 论文规模的模型在 `meta` 设备上构建, 只数参数不分配内存:

- Gemma 2B 非嵌入参数量与 Gemma 论文精确相等; SigLIP 与 300M expert 参数量与解析值 / openpi 注释一致;
- 输出 shape 与第 1 节契约一致, 256 token 网格, KV cache 形状与续算;
- 语义检查: RMSNorm 零初始化输出单位 RMS; RoPE 位置 0 为恒等; `make_attn_mask` 的块结构; 被 mask 的 token 不影响有效 token 的输出 (等价于把它们删掉), 这一条同时验证了 mask 与位置编号的配合.

```
uv run pytest pi/pi0/vlm -q
```

看一次前向怎么走 (tiny 配置, 逐步打印 shape):

```
uv run python -m pi.pi0.vlm.model
```

## 6. cost 信息

| 项目 | 值 | 来源 |
|---|---|---|
| VLM 参数量 | 约 2.92B (SigLIP 415M + Gemma 2.51B) | 本仓库解析计算; PaliGemma "3B" |
| 每张图 token 数 | 256 (224 / 14 = 16, 16²) | PaliGemma Sec. 3.1 |
| prefix 长度 | 816 = 3 × 256 + 48 | `pi0_config.py` L39; `model.py` L39-L47 |
| 推理延时 | image encoders 14 ms + observation forward 32 ms, RTX 4090 | π0 Table I |
| PaliGemma 预训练算力 | TPUv5e-256 × ~3 天 (Stage 1) | PaliGemma Sec. 3.4 |
| π0 训练 backbone 的 GPU 时间 | 未披露 | gap ledger |

## 7. reference 映射表

| 本仓库 | 上游 |
|---|---|
| `ViTConfig`, `SIGLIP_SO400M_14` | `openpi@215abfb src/openpi/models/siglip.py` L293-L370 (`decode_variant`), `pi0.py` L81-L87 (`num_classes=width, variant="So400m/14", pool_type="none"`) |
| `SigLIP.embedding / pos_embedding` | `siglip.py` L216-L229 |
| `ViTBlock` | `siglip.py` L53-L108 |
| `SigLIP.encoder_norm / head` | `siglip.py` L161, L284-L288 |
| `GemmaConfig`, `GEMMA_2B`, `GEMMA_300M` | `gemma.py` L41, L58-L109 |
| `RMSNorm` | `gemma.py` L113-L125 |
| `Embedder` | `gemma.py` L135-L154 |
| `ExpertAttnProj`, `attend` | `gemma.py` L158-L249 (单 expert 情形) |
| `apply_rope` | `gemma.py` L424-L440 |
| `GeGLU` | `gemma.py` L253-L280 |
| `GemmaBlock` | `gemma.py` L284-L333 |
| `Gemma` | `gemma.py` L340-L411 |
| `make_attn_mask` | `pi0.py` L19-L45 |
| `PaliGemma.embed_prefix` | `pi0.py` L106-L137 |
| `PaliGemma.forward` 与位置编号 | `pi0.py` L208, L233-L237 |
| checkpoint 来源 | `weight_loaders.py` L58-L73 |
| 论文 | π0 Appendix B; PaliGemma Sec. 3; Gemma Sec. 2, Table 1-2 |

## 8. gap ledger

| 未披露 / 未复现 | 说明 |
|---|---|
| 权重加载 | 不下载 `pt_224.npz`, 不做 JAX→PyTorch 参数映射; 属性名已按 openpi 命名以便日后映射 |
| So400m/14 的精确维度出处 | ViT-shape 论文 (arXiv:2305.13035) 只在图中给出; 本仓库以 openpi 的 variant 表为准 |
| π0 论文 Appendix B 写 Gemma 2B "num heads=18" | 与 Gemma 论文和 openpi 的 8 不符, 按 8 |
| SigLIP 的 dropout / stochastic depth | openpi 恒为 0 且 `train=False`, 未实现 |
| bfloat16 | openpi 默认 `dtype=bfloat16` 做矩阵乘, softmax 与 RMSNorm 统计在 float32; 本仓库全 float32, 精度策略属于可省的系统工程 |
| Gemma 的 `decode` (tied logits) | π0 不生成文字, 未实现 |
| 数值对齐 | 只对齐参数量与 shape (仓库原则 1) |

## 9. 机器人概念表

本 module 没有新的机器人特有概念; 与硬件的联系只有一条: 每张相机图固定占 256 个 token, 三个槽位固定占 768 个, 缺失相机的 256 个 token 由 `image_masks` 屏蔽 (见 `../data` README 第 9 节). 相机越多, prefix 越长, 第 3 节的 46 ms 大致按图数线性增长.
