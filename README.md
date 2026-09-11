# Tutorial on Physical Agents

## Overview

本 repo 提供 physical agent 著名工作的关键技术的代码复现. 目的是帮助有 LLM / RL 背景的人快速了解并复现 physical agent 训练、推理的实现细节.

本 repo 复现的是 **机制** (代码), 不承诺复现论文数字. 代码以阅读为第一目的, 在 CPU 上可以跑通 `tiny` 配置, 但设计目标是 NVIDIA H100 / B200.

目前 cover 以下内容:

- Physical Intelligence series: [pi/README.md](pi/README.md)
- Coming soon.

本 README 同时是 AI 辅助开发的协议: 后文每一条 "必须 / 禁止" 都是验收标准, 不是建议.


## 读者契约

**假设读者已懂**: Transformer, causal attention, 基础的 LLM 训练与推理流程, RL 基础.

**假设读者不懂**: 所有机器人特有概念. 每一个这类概念第一次出现时, 必须在 module README 或代码注释里讲清楚以下五点, 并举一个具体例子:

1. shape 与 dtype;
2. 物理含义 (单位, 坐标系, 取值范围);
3. 数据是怎么采集来的 (哪种硬件, 什么频率, 谁在操作);
4. 为什么这个模型需要它 (去掉会怎样);
5. 与硬件之间的联系 (哪个传感器 / 执行器产生或消费它).

典型需要讲清的概念: proprioceptive state, action chunk, delta action vs absolute action, action / state 归一化, gripper 的表示, 多相机与相机命名, 控制频率, 不同本体之间的 action 维度 padding.


## 复现原则

### 1. 忠于原文, 且可检查

- 每个 module 必须有 **对齐检查**: 参数量与 tensor shape 与上游实现或论文一致. 不要求数值精度对齐.
- 对齐检查放在 `test_parity.py`, 在 CPU 上几分钟内跑完. 不要把它做复杂.
- 每个 module 的 README 必须有一张 **reference 映射表**: 本仓库的类 / 函数 → 论文章节或上游代码位置.

### 2. 引用格式

- 代码引用: `repo URL` + `commit hash` + `文件路径` + `行号范围`. 必须 pin commit, 因为行号会漂移.
- 论文引用: arXiv 编号 + 版本号 (例如 `2410.24164v1`). 不同版本细节可能不同, 引用哪个版本就以哪个版本为准.
- 博客 / 官方文档引用: URL + 访问日期.

### 3. 什么可以省, 什么不能省

**可以省** (系统工程, 不改变数学):

- 分片 / FSDP / tensor parallel;
- 混合精度技巧, 自定义 kernel, 编译优化;
- dataloader 并行, 预取, 缓存;
- config 框架, logging, checkpoint 管理;
- 大仓库里为通用性引入的抽象层.

**不能省** (任何改变数学或数据语义的东西):

- attention mask 的结构 (例如 prefix / suffix 之间谁能看谁);
- 归一化统计量的来源与计算方式;
- timestep / noise 的采样分布;
- loss 的形式与加权;
- tokenization 与 action 的离散化 / 连续化方式;
- action chunk 长度, 执行长度, 控制频率;
- 数据预处理的每一步 (见下文 "数据预处理");
- 所有超参数.

如果一个细节不确定属于哪一类, 按 "不能省" 处理.

### 4. 框架与依赖

- 默认 **PyTorch**. 上游是 JAX 的工作也用 PyTorch 复现.
- 实现代码 **禁止** `import` 上游仓库. 测试代码允许 `import` 上游仓库作为对照.
- 依赖保持最少. 新增一个依赖需要在 module README 里说明原因.

### 5. 超参数与 gap ledger

- 所有超参数必须使用论文 / 上游代码披露的精确值, 并且 **逐值** 标注引用来源.
- 未披露的值 **禁止猜测填入**. 放进 module README 的 **gap ledger** 表, 标明 "未披露", 代码里对应位置用显式的占位参数并加注释指向 ledger.

### 6. 运行配置

每个 module 提供两档配置:

- `tiny`: 单卡消费级 GPU 或 CPU, 几分钟内跑通完整代码路径. 用于阅读与验证.
- `paper`: 论文原始规模. 只保证语义正确, 不要求实际跑.

代码按 GPU 写法组织 (device 可切换), 以 H100 / B200 为设计目标, 但不引入只有特定硬件才能运行的代码路径.

### 7. 复现代码, 记录 cost

- 本 repo 以复现代码为主, 不要求复现论文结果.
- 但每个 module README 必须记录从 information source 中搜集到的 **cost 信息**, 逐项引用来源, 未披露的进 gap ledger:
  - 训练 GPU 时间 (卡型 × 卡数 × 时长);
  - 训练数据规模 (小时数 / 轨迹数 / episode 数 / 来源数据集);
  - token 数 (如披露);
  - 模型大小 (总参数量, 各子模块参数量);
  - 推理各模块延时 (VLM 前向, policy 去噪, 端到端), 以及测量硬件.

### 8. 署名与许可

- 遵循上游仓库的 license, 在复现代码文件头部注明来源与许可.
- **禁止出现任何 AI 的署名**: 代码, 文档, commit message 中均不得包含 AI 生成 / 协作的署名或标记.


## Module 的定义与交付物

一个 module 对应一个 **关键技术** (例如 "π0 的 flow-matching action expert"), 不是一篇论文. 一篇论文通常拆成多个 module.

每个 module 目录固定包含:

```
<module>/
  README.md        # 见下文模板
  model.py         # 模型结构
  data.py          # 数据组织与预处理
  train.py         # 训练 loop 与 objective
  infer.py         # 推理链路
  test_parity.py   # 参数量与 shape 对齐检查
  figs/            # I/O 图, 由脚本生成
```

倾向单文件可读: 一个文件能从上读到下, 不需要跳转.

`model.py` 只写推理路径 (模型结构, 前向, KV cache); 训练专用的前向 (例如 prefix + suffix 一次过的 joint forward), loss, 采样分布, 优化器一律放 `train.py`, 在推理代码稳定后再加. 训练和推理分开写, 读者先看懂一次推理再看训练.

主文件 (`model.py`, 或 `data.py` 这类没有模型的 module) 必须带一个 `main()` 与 `if __name__ == "__main__"`: 用 `tiny` 配置走一遍完整前向, 逐步打印每个中间张量的 shape 和关键标量, 让读者不看测试也能看到一次 forward 是怎么走的. 运行方式写在 README 第 5 节.


## Module README 模板

顺序固定, **I/O 先行**:

1. **I/O 契约**: 每个对外 API 的输入与输出表 (名称, shape, dtype, 取值范围, 物理含义). 这是 README 的第一节, 先于任何结构描述.
2. **复现范围**: 哪些部分 **有代码** (复现), 哪些部分 **只有事实** (背景, 例如 VLM backbone 的预训练 recipe). 背景部分只做事实陈述加引用, 不写代码.
3. **推理**: 模型结构, 推理链路, 各环节延时.
4. **训练**: 数据组织形式, **数据预处理** (逐步写清: 归一化, 增广, 多相机处理, action padding, chunk 切分, 混合权重), training objective, curriculum.
5. **评测**: benchmark, 指标, 与论文对齐到什么程度.
6. **cost 信息表**: 见原则 7.
7. **reference 映射表**: 见原则 1.
8. **gap ledger**: 见原则 5.
9. **机器人概念表**: 本 module 引入的机器人特有概念, 每个按读者契约的五点说明.


## AI 协作工作流

1. **主人枚举技术栈**. 每个技术栈的大纲由主人手动设立 (例如 pi/README.md), 一次做一个 topic.
2. **开工前先 refine 大纲**. 每次主人提出开始一个新 topic, AI 必须先对该 topic 的复现结构提出建议: 哪些属于可复现范围, 哪些是背景; 大纲缺了哪些 "不能省" 的细节; 建议的 module 拆分. 主人确认后再动手.
3. **先读源再写码**. 写任何代码前必须先读对应的论文段落与上游代码, 并在 reference 映射表里登记.
4. **不确定就问, 禁止编造**. 任何不确定的细节标 `TODO` 并向主人提问, 或进 gap ledger.
5. **一次一个 module**. 一个 module 的交付物齐全 (README, 代码, parity test 通过) 才算完成, 才开始下一个.
6. **完成的定义**: `test_parity.py` 在 CPU 上通过; `tiny` 配置的 train / infer 跑通; README 九节齐全; 无 AI 署名.


## Git 执行规范

目标: 从 git 历史上能直接看出每个技术栈是哪个阶段加进来的, 以及它内部是怎么一步步开发出来的.

**分支模型**

- `main` 只接收 merge commit, 禁止在 `main` 上直接提交 (仓库初始化除外).
- 每个技术栈一个分支, 命名 `topic/<name>`, 例如 `topic/pi0`, `topic/pi05`.
- 一个技术栈内的多个 module 都在同一个 topic 分支上按顺序提交, 每个 module 至少一个 commit, 建议按 "README → data → model → infer → train → test" 的顺序小步提交, 让开发过程可读.
- 技术栈完成 (满足上文 "完成的定义") 后, 以 **no-ff** 方式合回 `main`, 并打 tag:

  ```
  git checkout main
  git merge --no-ff topic/<name> -m "merge topic/<name>: <一句话说明该技术栈>"
  git tag <name>
  ```

- 禁止 squash, 禁止 rebase 已经推送的 topic 分支. 历史只增不改.
- 合并后 topic 分支保留, 不删除.

**commit message**

- 格式: `<topic>/<module>: <做了什么>`, 例如 `pi0/action-expert: add flow-matching head with adaRMSNorm`.
- 仅涉及 README 的提交用 `<topic>/docs: ...`; 仓库级改动用 `repo: ...`.
- 每个 commit 对应一个可描述的步骤, 不提交 "wip".
- 不得出现任何 AI 署名或标记 (见原则 8).

**个人信息**

- README 与代码中不出现真实姓名, 邮箱, 单位.

**阅读历史的方式**

```
git log --first-parent main     # 只看各技术栈的合入节点
git log main..topic/<name>      # 看某个技术栈内部的开发过程
```
