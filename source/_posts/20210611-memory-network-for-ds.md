---
title: Memory Network 在对话系统中的应用
date: 2021-06-11 14:58:16
tags:
    - 对话系统
    - Memory Network
categories:
    - 对话系统
---


本文是南洋理工大学2021年5月份的对话系统综述文章 [Recent Advances in Deep Learning Based Dialogue Systems: A Systematic Survey](http://arxiv.org/abs/2105.04387) 2.4 节 **Memory Network** 的理解和展开.


# Memory Network 概述

Memory Network 是 FAIR 的 Jason Weston 等人于 2014 年在论文 [Memory Networks](http://arxiv.org/abs/1410.3916) 提出的一种神经网络框架, 它在通用的神经网络模型中增加了一个 Memory 模块, 用来记忆神经网络中需要经常使用(但又经常被遗忘)的信息. 对照起来看, Memory 之于神经网络, 类似于 海马体(hippocampus) 之于 大脑, 内存和硬盘 之于 计算机.

在此之前的神经网络中, 不管是 MLP, 还是 CNN 和 RNN, 都没有显示的定义记忆体. 尽管模型的隐层参数(如 RNN 的 hidden state, CNN 的卷积核) 能够记忆一些长期记忆, 但通常来说其记忆能力有限.

Memory Network 通过显示引入大小不限(当然受限于硬件环境)的记忆体, 显示定义记忆体中存储的内容, 并在模型训练和预测过程中(选择性)动态更新记忆, 增强了模型对重要信息的记忆和利用能力. 从作者在 QA (Question Answer) 任务上的实验结果来看, 记忆体可以较大地提升模型效果.


## MemNN

正式地, Memory Network 包括如下五个部分:

* Memory $M$: 记忆体. 可以认为是一个包含多个记忆单元的数组, 记忆单元的具体内容和形式, 则取决于具体的任务和设计. 为了行文方便, 下文中记忆体用符号 $M$ 表示, 其记忆单元用符号 $m_{i}$ 表示, 即 $M = [m_0, m_1, \cdots, m_i, \cdots, m_{N-1}]$.
* Input Map $I$: 输入特征提取器. 设原始输入为 $x$, 则输入 $f = I(x)$ 可以提取对应的输入数据特征 $f$
* Generalization $G$: 记忆生成器. 基于新的输入特征, 获取新的记忆表示, 插入记忆体或更新相关的记忆单元. 数学表示为 $M = G(f, M)$, 或 
$$
m_i = G(m_i, f, M), \forall i \in \{0, 1, \cdots\}
$$
* Output Map $O$: 输出特征提取器. 数学表示为 $o = O(f, M)$. 注意, 这里的记忆体 M 已经被更新过.
* Response $R$: 响应器. 数学表示为 $r = R(o)$. 这个与具体的任务有关.

Memory Network 是一种模型框架, 其各个部分均可以自由设计. 在 Jason Weston 2014 年的原始论文中, 作者设计了一个 MemNN 模型, 用来解决 QA 问题. 

* MemNN 的使用了只插入不更新的记忆体, 记忆单元保存原始输入 $x$, 可以是一个描述事实的句子 (如 *"Joe travelled to the office."*), 也可以是对应的问题 (如 *"Where was Joe before the office?"*). 记忆生成器 $G$ 的工作模式为

$$
m_{k+1} = x
$$

其中 $k$ 为当前已经使用的记忆单元个数.

* MemNN 的特征提取器 $I$ 使用了 BoW 特征.
* MemNN 的输出特征提取器 $O$ 和 响应器 $R$ 均使用了相似度排序模型. 

其输出特征提取器 $O$ 会从记忆体中提取两个最相关的记忆单元:

$$
\begin{aligned}
o_1 &= \argmax_{m_i \in M} S_{O} (x, m_i) \\
o_2 &= \argmax_{m_i \in M} S_{O} ([x, o_1], m_i)
\end{aligned}
$$

响应器采用了类似的方式.


MemNN 最主要的问题是, 模型使用的 $\argmax$ 是不可导的, 因此不能采用端到端的训练方法.



## MemN2N

FAIR 在 2015 年的论文 [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895) 提出了 MemN2N 模型, 将 MemNN 改造为 End2End 方式.

相对于 MemNN, MemN2N 的主要改动在输出特征提取器 $O$ 的设计上. 与 MemNN 使用 $\argmax$ 不同, MemN2N 使用了可导的 Softmax 函数, 其输出计算如下:

$$
\begin{aligned}
p_i &= \text{Softmax}(\mu^{T} m_i) \\
o &= \sum_{i} p_i c_i
\end{aligned}
$$

其中记忆单元 $m_i$ 是对应输入 $x_i$ (一个描述事实的句子) 的稠密向量表示, $\mu$ 对应当前输入的问题 $q$ 的向量表示 (与 x 的编码矩阵可能不同), $c_i$ 又是 $x_i$ 对应的另一个稠密向量表示. 这里使用了三个不同的嵌入矩阵, 讲起来有点绕. 事实上, 作者也有尝试过上面三个向量表示方法使用同一个嵌入矩阵的情形.

对应的响应器 $R$ 也不再使用排序的方法, 而改成了分类器的方式 (作者简化了 QA 问题, 假设答案只包括一个单词), 数学表示如下:

$$
r = \text{Softmax}(W (o + u))
$$


## MemN2N 与 Attention 的关系

我们从上面的输出特征提取过程可以很容易联想到 Attention, 因为两者的计算过程实在太相似了.

以 Dzmitry Bahdanau 等人 2014 年在论文 [Neural machine translation by jointly learning to align and translate](http://arxiv.org/abs/1409.0473) 给出的 Attention 计算过程作为对比:

$$
\begin{aligned}
e_{ij} &= a(s_{i-1}, h_j) \\
\alpha_{ij} &= \text{Softmax}(e_{ij}) \\
c_i &= \sum{j=1}^{T_x} \alpha_{ij} h_j
\end{aligned}
$$


无论是 MemN2N 还是 Attention, 都是通过概率分布的方式找到与当前关注点最相关的信息. MemN2N 的当前关注点是 Query, 相关信息是 Query 之前的事实描述. Attention 中的当前关注点是隐状态 $s_{i-1}$, 而相关信息是编码器输出的序列矩阵表示.


事实上, 还有一些 Transformer 的后续研究 (应该是之前在知乎上看到的介绍, 但是现在找不到具体的知乎帖子及对应的原始论文了, 后续补上) 称, 将 Transformer 中 Self-Attention 对应的 Q 和 K, 或者其计算结果替换成一个随机初始化的矩阵变量 (还是直接将系数 $\alpha$ 作为变量学习, 记不清楚了), 也能够达到 Self-Attention 类似的效果. 这里想表达的是, 不管是显示定义的 Memory, 还是通过 Q 和 K 计算得到的 Score, 甚至是随机初始化的 Score 向量, 都为模型提供了额外的记忆信息, 都能给模型带来效果提升, Memory 和 Attention 没有本质的区别.


# Memory Network 在对话系统中的应用

这里先把 [Recent Advances in Deep Learning Based Dialogue Systems: A Systematic Survey](http://arxiv.org/abs/2105.04387) 罗列的一些研究工作简单梳理一下, 后续再逐一展开.

* [Chen et al. (2019c): The JDDC corpus: A large-scale multi-turn chinese dialogue dataset for e-commerce customer service](https://arxiv.org/abs/1911.09969)
作者认为, 将对话历史和知识库混合在同一个 Memory 中可能影响答复的质量, 因此作者设计了包含三个记忆体的 TOD 系统: 两个长期记忆体, 分别存储对话历史 和 知识库; 一个工作记忆体记忆两个概率分布并控制 Response 生成.

* [He et al. (2020a): Amalgamating knowledge from two teachers for task-oriented dialogue system with adversarial training](https://www.aclweb.org/anthology/2020.emnlp-main.281/)
作者提出了一种 "Two-Teacher-One-Student" TOD 系统, 该框架的学生模型使用了 Memory Network. 作者首先使用强化学习方法训练两个 Teacher 模型, 然后使用 GAN 训练方法来训练学生模型.

* [Kim et al. (2019): Efficient dialogue state tracking by selectively overwriting memory](https://arxiv.org/abs/1911.03906)
论文作者主要关注 DST (Dialog State Tracking) 问题. 作者使用 Memory 来记忆对话状态, 在更新记忆时, 作者会(通过模型)主动选择需要被更新的记忆单元(对话状态).

* [Dai et al. (2020): Learning low-resource end-to-end goal-oriented dialog for fast and reliable system deployment](https://www.aclweb.org/anthology/2020.acl-main.57/)
作者使用了 MemN2N 模型作为 Utterance 编码器, 其记忆体主要用于记录对话历史和已经存在的响应, 然后使用 **MAML(Model-Agnostic Meta-Learning, 模型无关的元学习)** 方法训练模型在 Few-shot 情形下找到正确的响应.


* [Tian et al. (2019): Learning to abstract for memory-augmented conversational response generation](https://www.aclweb.org/anthology/P19-1371/)
作者使用 Memory Network 记忆 Query-Response 对, 在生成答复时同时参考输入 Query 和 Memory.

* [Xu et al. (2019): Neural response generation with metawords](https://arxiv.org/abs/1906.06050)
作者在 Memory 中记录 Meta-Words (用于描述响应的短语), 生成答复时同时考虑 User Utterance 和 Meta-Words.

* [Gan et al. (2019): Multi-step reasoning via recurrent dual attention for visual dialog](https://arxiv.org/abs/1902.00579)
论文使用了 对话历史记忆体 和 视觉记忆体, 可以执行多步推理 (Multi-Step Reasoning).

* [Han et al. (2019): Episodic memory reader: Learning what to remember for question answering from streaming data](https://arxiv.org/abs/1903.06164)
当 Memory 被占满时, 论文提出了一种强化学习方法来选择需要被替换的记忆单元, 这解决了 Memory Network 的扩展性问题.

* [Gao et al. (2020c): Explicit memory tracker with coarse-to-fine reasoning for conversational machine reading](https://arxiv.org/abs/2005.12484)
作者是为了解决 Memory Network 的扩展性问题, 他们使用了 EMT (Explicit Memory Tracker) 来判断记忆是否足够(不再扩张), 并使用了 coarse-to-fine 策略提出澄清问题(clarification questions) 以获得更多的信息来完善推理过程.
