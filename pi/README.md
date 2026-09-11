# Tutorial on Physical Intelligence Series


## Overview

本 repo 是旨在提供 physical intelligence 的关键技术的开源复现. 目的是帮助有 LLM / RL 背景的人快速了解 pi-0 ~ pi-0.7 训练、推理的实现细节, 并且有能力进行复现. 目前 cover 以下内容:

- π0: A Vision-Language-Action Flow Model for
General Robot Control: https://arxiv.org/pdf/2410.24164v1
- 


## 关键技术栈

### π0: 奠定了 VLM pretrain + Flow-Matching Policy 的范式

- 推理：
    - VLM 模型结构，输入输出 protocol.
    - Flow-Matching Policy 模型结构，输入输出 protocol.
    - 实际部署的细节, 模型大小, 图例链路环节, 延时.
- 训练：
    - VLM 预训练数据组织形式 & training objective and task / curriculum.
    - VLM checkpoint 选取. 该 checkpoint 的训练 reciepe 和模型细节.
    - VLA 训练数据组织形式 & training objective and task / curriculum.
