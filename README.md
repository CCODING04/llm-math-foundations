# 🧮 LLM 数学基础：从零理解大语言模型背后的数学

> 学无止境，数学是通往深度理解的阶梯。🌱

## 项目简介

这是一个面向实践者的 **LLM 数学基础教程**——专为想要真正理解大语言模型（LLM）工作原理的开发者和学习者编写。

市面上的 LLM 教程大多从代码出发，告诉你「怎么用」，却很少解释「为什么」。为什么 Transformer 用矩阵乘法？注意力机制的公式到底在算什么？缩放定律背后的数学原理是什么？

**不懂原理，调参只能靠运气；懂了原理，调参才有依据。**

本仓库用清晰的讲解、直观的可视化、可运行的代码，带你一步步建立理解 LLM 所需的数学直觉。不需要你是数学专业——只需要高中数学基础和一颗好奇心。

---

## 为什么需要数学基础？

很多人问：我只是想用 LLM / 微调模型，为什么要学数学？

简单来说——**数学是 LLM 的语言**。大语言模型的每一步计算，背后都有明确的数学含义：

| 你看到的概念 | 背后的数学 |
|:---|:---|
| 词向量（Embedding） | 线性代数：向量空间、矩阵乘法 |
| 反向传播（Backpropagation） | 微积分：链式法则、梯度 |
| 语言模型输出概率 | 概率论：条件概率、贝叶斯 |
| 模型训练收敛 | 优化理论：梯度下降、学习率 |
| 注意力机制（Attention） | 矩阵运算 + Softmax + 点积相似度 |
| 生成质量评估 | 信息论：交叉熵、KL 散度 |
| Scaling Law | 幂律、统计估计 |
| RLHF / PPO | 强化学习 + 概率进阶 |

**理解这些数学，你就能：**
- 读懂论文里的公式，不再跳过数学部分
- 理解超参数的意义，做出有依据的调参决策
- 遇到问题能定位原因，而不是盲目试错
- 具备阅读前沿研究的能力

---

## 📖 学习路线图

本仓库按 **三个阶梯 + 预备章** 组织，由浅入深。每个阶梯包含三个章节，循序渐进。

```
预备章                阶梯一（基础）          阶梯二（进阶）          阶梯三（前沿）
━━━━━━━━━           ━━━━━━━━━━━━━         ━━━━━━━━━━━━━         ━━━━━━━━━━━━━
Ch0 数学预备  →      Ch1 线性代数    →      Ch4 矩阵深度    →     Ch7 注意力机制数学
                     Ch2 微积分      →      Ch5 优化理论    →     Ch8 概率进阶与采样
                     Ch3 概率统计    →      Ch6 信息论      →     Ch9 缩放定律
                                                                                 ↓
                                                                          Ch10 RL基础与对齐
━━━━━━━━━           ━━━━━━━━━━━━━         ━━━━━━━━━━━━━         ━━━━━━━━━━━━━
扫清障碍              打地基阶段             建框架阶段             理解前沿阶段
```

### 🔧 预备章：扫清障碍

| 章节 | 主题 | 你将学到 |
|:---:|:---|:---|
| [Ch0](chapter00-math-prep/) | 数学预备 | Σ/Π 求和符号、log/exp 规则、函数复合、NumPy 极简入门——扫清后续章节的符号障碍 |

### 🏗️ 第一阶：打地基（基础数学）

| 章节 | 主题 | 你将学到 |
|:---:|:---|:---|
| [Ch1](chapter01-linear-algebra/) | 线性代数基础 | 向量、矩阵运算、点积、范数——理解 Embedding 和模型参数的基础 |
| [Ch2](chapter02-calculus/) | 微积分核心 | 导数、偏导数、链式法则、梯度——理解反向传播的数学本质 |
| [Ch3](chapter03-probability/) | 概率统计入门 | 概率分布、条件概率、最大似然估计——理解语言模型如何预测下一个词 |

### 🏢 第二阶：建框架（进阶工具）

| 章节 | 主题 | 你将学到 |
|:---:|:---|:---|
| [Ch4](chapter04-matrix-deep/) | 矩阵深度 | 特征值、SVD、正交矩阵、行列式、矩阵求导——Transformer 计算的核心数学 |
| [Ch5](chapter05-optimization/) | 优化理论 | 梯度下降变体、Adam 优化器、学习率调度——理解模型训练如何收敛 |
| [Ch6](chapter06-information-theory/) | 信息论 | 熵、交叉熵、KL 散度、互信息——理解损失函数和模型评估的标准 |

### 🚀 第三阶：理解前沿（LLM 核心数学）

| 章节 | 主题 | 你将学到 |
|:---:|:---|:---|
| [Ch7](chapter07-attention-math/) | 注意力机制数学 | QKV 计算、多头注意力、因果掩码——逐行拆解 Transformer |
| [Ch8](chapter08-probability-advanced/) | 概率进阶与采样 | 温度、Top-K、Top-P、MLE、NLL——理解生成过程和解码策略 |
| [Ch9](chapter09-scaling-rl/) | 缩放定律 | Scaling Law、Chinchilla、幂律拟合——理解大模型的训练范式 |
| [Ch10](chapter10-rl-alignment/) | RL 基础与对齐 | Policy Gradient、PPO、DPO、GRPO——理解 RLHF 对齐算法 |

### 🛠️ 阶段项目：从"能看懂"到"能做出来"

每完成一个阶梯，用阶段项目把知识串起来：

| 项目 | 覆盖章节 | 你将做到 |
|:---:|:---|:---|
| [项目 1](project01-linear-regression/) | Ch1-Ch3 | 从零实现线性回归：手写前向传播、梯度下降、验证 MSE = MLE |
| [项目 2](project02-two-layer-nn/) | Ch4-Ch6 | 从零实现两层神经网络：手写反向传播、交叉熵、Adam 优化器 |
| [项目 3](project03-mini-attention/) | Ch7-Ch8 | 手写多头注意力 + 解码采样器：QKV、因果掩码、温度/Top-K/Top-P |

---

## 如何使用本仓库

### 推荐学习方式

1. **按顺序学习**：章节之间有依赖关系，建议从 Ch0 开始，依次推进。如果数学基础扎实，可以跳过 Ch0 直接从 Ch1 开始
2. **动手运行**：每章的 `scripts/` 目录包含可视化代码，运行它们来建立直觉
3. **做好笔记**：数学需要反复理解，建议边学边记
4. **不跳过推导**：看懂推导过程比记住结论更重要

### 前置要求

- **数学**：高中数学水平即可起步（会代数运算和基本函数）
- **编程**：基础的 Python 能力（能读懂简单脚本）
- **工具**：Python 3.8+、NumPy、Matplotlib（用于运行可视化脚本）

### 快速开始

```bash
# 克隆仓库
git clone https://github.com/your-username/llm-math-foundations.git
cd llm-math-foundations

# 安装依赖（可选，用于运行可视化脚本）
pip install -r requirements.txt

# 从第一章开始
cd chapter01-linear-algebra
```

---

## 📂 仓库结构

```
llm-math-foundations/
├── README.md                          # 你正在看的这个文件
├── requirements.txt                   # 依赖列表
├── chapter00-math-prep/               # 🔧 预备章：数学预备
│   └── README.md
├── chapter01-linear-algebra/          # 🏗️ 第一阶：线性代数基础
│   ├── README.md                      # 章节正文
│   ├── images/                        # 配图和示意图
│   └── scripts/                       # 可视化代码
├── chapter02-calculus/                # 🏗️ 第一阶：微积分核心
│   ├── README.md
│   ├── images/
│   └── scripts/
├── chapter03-probability/             # 🏗️ 第一阶：概率统计入门
│   ├── README.md
│   ├── images/
│   └── scripts/
├── chapter04-matrix-deep/             # 🏢 第二阶：矩阵深度
│   ├── README.md
│   ├── images/
│   └── scripts/
├── chapter05-optimization/            # 🏢 第二阶：优化理论
│   ├── README.md
│   ├── images/
│   └── scripts/
├── chapter06-information-theory/      # 🏢 第二阶：信息论
│   ├── README.md
│   ├── images/
│   └── scripts/
├── chapter07-attention-math/          # 🚀 第三阶：注意力机制数学
│   ├── README.md
│   ├── images/
│   └── scripts/
├── chapter08-probability-advanced/    # 🚀 第三阶：概率进阶与采样
│   ├── README.md
│   ├── images/
│   └── scripts/
├── chapter09-scaling-rl/              # 🚀 第三阶：缩放定律
│   ├── README.md
│   ├── images/
│   └── scripts/
├── chapter10-rl-alignment/            # 🚀 第三阶：RL 基础与对齐
│   ├── README.md
│   ├── images/
│   └── scripts/
├── project01-linear-regression/       # 🛠️ 阶段项目 1：线性回归
│   ├── README.md
│   ├── images/
│   └── scripts/
├── project02-two-layer-nn/            # 🛠️ 阶段项目 2：两层神经网络
│   ├── README.md
│   └── scripts/
├── project03-mini-attention/          # 🛠️ 阶段项目 3：Mini Attention + 采样器
│   ├── README.md
│   └── scripts/
└── resources/                         # 📚 推荐资源汇总
    └── further-reading.md
```

每个章节包含三个部分：
- **README.md** — 章节正文，概念讲解 + 数学推导 + 直观解释
- **images/** — 配图、示意图、公式图解
- **scripts/** — Python 可视化脚本，帮助建立数学直觉

---

## 📚 章节目录

- [Ch0 - 数学预备](chapter00-math-prep/)
- [Ch1 - 线性代数基础](chapter01-linear-algebra/)
- [Ch2 - 微积分核心](chapter02-calculus/)
- [Ch3 - 概率统计入门](chapter03-probability/)
- [Ch4 - 矩阵深度](chapter04-matrix-deep/)
- [Ch5 - 优化理论](chapter05-optimization/)
- [Ch6 - 信息论](chapter06-information-theory/)
- [Ch7 - 注意力机制数学](chapter07-attention-math/)
- [Ch8 - 概率进阶与采样](chapter08-probability-advanced/)
- [Ch9 - 缩放定律](chapter09-scaling-rl/)
- [Ch10 - RL 基础与对齐](chapter10-rl-alignment/)

**阶段项目**：
- [🛠️ 项目 1 - 从零实现线性回归](project01-linear-regression/)（覆盖 Ch1-Ch3）
- [🛠️ 项目 2 - 从零实现两层神经网络](project02-two-layer-nn/)（覆盖 Ch4-Ch6）
- [🛠️ 项目 3 - Mini Attention + 解码采样器](project03-mini-attention/)（覆盖 Ch7-Ch8）

- [推荐资源](resources/)

---

## 学习建议

> 「数学不是看懂的，是算懂的。」

几个过来人的建议：

- **别怕慢**：第一章看一周不丢人，扎实比速度重要
- **动手算**：看到公式，自己推一遍。光看等于没看
- **跑代码**：`scripts/` 里的代码不是装饰，运行、修改、实验
- **多问为什么**：每个公式背后都有一个「为什么是这个形式」的故事
- **关联已有知识**：如果你有 ML 实践经验，把数学对应到你用过的工具上

---

## 致谢

本仓库的编写参考了大量优秀的教材、论文和开源项目，详见 [resources/](resources/)。

---

## License

MIT License — 自由使用、分享、修改。

---

<p align="center">
  <i>🌱 数学是通往理解的阶梯，每一步都算数。</i>
</p>
