---
title: 通用变分推断算法 SVGD 简介
date: 2023-04-22 13:29:50
tags: 
    - SVGD
    - 变分推断
    - 贝叶斯推断
    - RKHS
    - 泛函分析

categories:
    - 强化学习
    - 贝叶斯推断
    - 泛函分析
---


> 学习 SVGD 的动机链路如下:
> 
> * 首先, 以 ChatGPT 为代表的大模型技术在业界和外界都造成了极大的轰动, 但 ChatGPT 在基础模型层面还是保持了 [GPT-3](https://arxiv.org/abs/2005.14165), 其相对于 GPT-3 的主要提升来自于 [RLHF](https://github.com/opendilab/awesome-RLHF), RLHF 使用强化学习的方法将语言模型的输出对齐人类偏好
> 
> * ChatGPT 的 RLHF 的思想主要是在论文 [Ouyang et al., 2022 (InstructGPT)](http://arxiv.org/abs/2203.02155) 中描述的, 其具体的强化学习算法是 [PPO](http://arxiv.org/abs/1707.06347), PPO 本质是一种 Actor-Critic 策略梯度算法
> 
> * 由于对 PPO / Actor-Critic 及其它强化学习概念和算法不甚了解, 又进一步找到强化学习书籍 [Sutton et al., 2017 - Reinforcement Learning: An Introduction (2nd Ediction)](http://www.incompleteideas.net/book/the-book-2nd.html) 和 [张伟楠 et al., 2022 - 动手学强化学习](https://hrl.boyuai.com/)
> 
> * 通过 [动手学强化学习](https://hrl.boyuai.com/) 了解到, 策略梯度算法训练不稳定, 因此发展出 Actor-Critic 算法; Actor-Critic 训练还是不足够稳定, 又出现了 [TRPO (Schulman et al., 2015)](http://arxiv.org/abs/1502.05477) 和 [PPO (Schulman et al., 2017)](http://arxiv.org/abs/1707.06347) 算法
> 
> * 还是在 [动手学强化学习](https://hrl.boyuai.com/) 中提到, [Soft Q-Learning (Haarnoja et al., 2017)](http://arxiv.org/abs/1702.08165) 通过在 Actor-Critic 框架的目标函数中增加 熵正则化项, 能够学习到更鲁棒的策略; 而 [SAC (Soft Actor-Critic, Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290) 算法则基于 Soft Q-Learning 提出了一种 离线 (Off-policy) 的 Actor-Critic 学习算法
> 
> * 通过学习 [Soft Q-Learning 论文](http://arxiv.org/abs/1702.08165), Soft Q-Learning 需要一个高效的近似采样网络对策略进行采样, 该网络的实现使用了 [(Amortized) SVGD (Liu & Wang, 2016)](http://arxiv.org/abs/1608.04471) 算法
>
>
> [Liu Qiang](http://arxiv.org/abs/1608.04471) 是 UT Austin (University of Texas at Austin) 的助理教授, 其 [研究团队 (UT Statistical Learning & AI Group)](https://www.cs.utexas.edu/~qlearning/index.html) 发表了数偏关于 [Stein 方法](https://en.wikipedia.org/wiki/Stein%27s_method) 的理论和应用文章 (不完全列表):
>
> * Liu, Qiang, Jason D. Lee, and Michael I. Jordan. 2016. “A Kernelized Stein Discrepancy for Goodness-of-Fit Tests and Model Evaluation.” arXiv. http://arxiv.org/abs/1602.03253.
>
> * Liu, Qiang, and Dilin Wang. 2016. “Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm.” arXiv. http://arxiv.org/abs/1608.04471.
> 
> * Wang, Dilin, and Qiang Liu. 2016. “Learning to Draw Samples: With Application to Amortized MLE for Generative Adversarial Learning.” arXiv. http://arxiv.org/abs/1611.01722.
>
> * Liu, Yang, Prajit Ramachandran, Qiang Liu, and Jian Peng. 2017. “Stein Variational Policy Gradient.” arXiv. http://arxiv.org/abs/1704.02399.
>
> * Liu, Qiang. 2017. “Stein Variational Gradient Descent as Gradient Flow.” In Advances in Neural Information Processing Systems. Vol. 30. Curran Associates, Inc. https://arxiv.org/abs/1704.07520
> 
> * Gong, Chengyue, Jian Peng, and Qiang Liu. 2019. “Quantile Stein Variational Gradient Descent for Batch Bayesian Optimization.” In Proceedings of the 36th International Conference on Machine Learning, 2347–56. PMLR. https://proceedings.mlr.press/v97/gong19b.html
>
> 其中 Kernelized Stein Discrepancy 文章算是奠基之作, 有大量的关于 Stein 方法的证明和推导, 要想理解 SVGD 算法的详细推导过程, 至少需要详细阅读前 2 篇论文



我们开始进入正题, SVGD 论文 [Liu & Wang, 2016](http://arxiv.org/abs/1608.04471) 第一句话就提到:
> We propose a general purpose variational inference algorithm that forms a natural counterpart of gradient descent for optimization.

这句话有两个关键点:

* SVGD 算法是一种通用的 [变分推断 (Variational Inference, VI)](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) 算法
* SVGD 之于变分推断犹如 SGD 之于最优化


要想理解 SVGD, 我们需要先了解变分推断.


# 背景 - 变分推断

变分推断 是 [贝叶斯推断](https://en.wikipedia.org/wiki/Bayesian_inference) 的一种近似解法 (另一种常用解法是采样法, 典型算法是 [MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)).


为什么贝叶斯推断需要近似解法? 这是因为贝叶斯推断的配分函数需要对完整的参数空间求积分, 这常常是一个难解问题, 特别是在高维参数空间的情况下.

我们先简单介绍一下贝叶斯推断.


假设给定了一个观察数据集 $D = \{ D_k\}, k = 1, 2, \cdots, N$, 共有 N 个独立同分布的的观察样本, 现在我们想要基于此数据集推断出生成该数据集的分布, 假设该分布我们用 $p(x | D)$ 表示. 则根据贝叶斯公式, 有:

$$
\begin{aligned}
P(x | D) &= \frac{P(x) \cdot P(D | x)}{P(D)} \\
    &= \frac{P(x) \cdot P(D | \theta)}{\textcolor{red}{\int_{x} P(D | x) \mathrm{d} x}}
\end{aligned}
$$

公式中分母 $P(D) = \int_{x} P(D | x) \mathrm{d} x$ 需要对整个高维空间求积分, 这在计算上是难解的.


变分推断则引入一个新的简单的参数化分布 $q_{\theta}$, 只需要通过优化该分布的参数 $\theta$, 让分布 $q_{\theta}$ 和 待求解的分布 $p$ 足够接近, 就可以使用 $q_{\theta}$ 近似替代原始分布 $p$. 两个概率分布的近似程度, 可以使用 [KL 散度](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) 进行度量: KL 散度越小, 两个分布越接近. 因此变分推断的数学表达如下:

$$
q_{\theta}^{*} = \argmin_{q_{\theta} \in \mathcal{Q}} \text{KL} (q_{\theta} \| p)
$$

变分推断的优势是: 通过引入新的带参近似分布, 将原始的概率推断问题转换为了 最优化问题, 可以使用最优化工具 (如 SGD 等) 对问题进行求解.

变分推断可以使用 [SGD (Stochastic Gradient Descent)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) 方法求解, 其核心步骤是

$$
\theta \leftarrow \theta + \eta \nabla_{\theta} \text{KL} (q_{\theta} \| p)
$$

主要就是计算 KL 散度相对于概率参数的梯度.


不过, 按照 SVGD 论文的讲法是: 候选分布及其参数 $p_{\theta}$ 的选择对 VI 至关重要, 候选分布过于简单, 则其表达力不足, 可能无法拟合复杂情况下的目标概率分布; 过于复杂, 则又会导致整个后续计算也较复杂. 因此 VI 常常是特定问题特定解法, 没有一套通用的工具可以应用于所有问题.

# SVGD (Stein Variational Gradient Descent)


<!-- 不同于上述变分推断采用 SGD 求解, SVGD 则采取了一种不同的思路, 简化了 VI 的计算, 同时能够动态调整分布的复杂度, 是一种高效且通用的 变分推断 算法.

SVGD 核心包括两点:

* 对一组粒子 (particles) 迭代进行微小扰动, 使其分布逐步接近于目标分布
* 将扰动函数限制在一个 [RKHS (Reproducing Kernel Hilbert Space)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) 的球上, 便于使用 RKHS 的理论和工具进行求解 -->

不同于 SGD 需要逐步迭代参数求解 VI 问题, SVGD 利用 RKHS 的相关理论, 通过对一组粒子进行微小扰动, 逐步驱使粒子的分布接近真实的目标分布.  


## 微小扰动

首先, 我们讨论 微小扰动 是如何改变粒子的概率分布的.

数学上, 对于粒子 $x \in \R ^ {d}$ 其扰动可以通过函数 $z = T(x)$ 来表示. 分了便于讨论, 我们要求改扰动函数 是光滑的、连续的、可微分的一一映射, 用数学语言表述就是: 处处可导的可逆函数.

假设原始粒子 x 服从 p 分布, 则扰动后的粒子服从的分布满足如下关系:

$$
p_{[T]} (z) = p( T^{-1} (z)) \cdot \left | \det (\nabla_{z} T^{-1} (z)) \right |
$$

其中 $T^{-1} (\cdot)$ 表示扰动变换的逆变换, $\nabla_{z} T^{-1} (z)$ 是逆变换的 Jacobian 矩阵, 而 $\det (\cdot)$ 表示矩阵的 [行列式 (Determinant)](https://en.wikipedia.org/wiki/Determinant) 求值.  整个 $\det (\nabla_{z} T^{-1} (z))$ 还有一个专门的数学名字 [雅可比行列式 (Jacobian Determinant)](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant#Jacobian_determinant).

当 $x \in \R$, 且 扰动变换 $T$ 是严格单调递增或递减递减函数时, 上述结论是比较容易证明的. 现考虑 T 是单调递增函数时, 证明如下:

$$
\begin{aligned}
F_{T}(z) &= P(T(x) \le z) \\
&= P(x \le T^{-1} (z)) & (\text{单调递增函数}) \\
&= F_{x} (T^{-1} (z))
\end{aligned}
$$

其中 $F (\cdot)$ 表示累积密度函数 (Cumulative Distribution Function, CDF).

利用链式法则, 则概率密度函数可推导如下:

$$
\begin{aligned}
p_{T}(z) &= \nabla_{z} F_{T} (z)  \\
&= \nabla_{z} F_{x} ( T^{-1} (z))  \\
&= p(T^{-1} (z)) \cdot \nabla_{z} T^{-1} (z)
\end{aligned}
$$

当 T 是单调递减变换, 则累计概率密度的推导结果是 $F_{T}(z) = P(T(x) \le z) = P(x \ge T^{-1} (z)) = 1 - F_{x} (T^{-1} (z))$. 其概率密度函数相对于递增变换时多了一个负号. 而此时 $\nabla_{z} T^{-1} (z) \lt 0$.  因此这两种情况可以整合为:

$$
p_{T}(z) = p(T^{-1} (z)) \cdot | \nabla_{z} T^{-1} (z) |
$$

当 $x \in \R^{d}$ 是高维向量时, 该推导过程需要 [参考相关资料](https://en.wikibooks.org/wiki/Probability/Transformation_of_Probability_Densities).


SVGD 论文中, 为了保证扰动变换的 Jacobian 行列式的存在 (变换函数光滑/可微/可逆), 对变换函数进行了进一步的限制:

$$
z = T(x) = T_{\epsilon} (x) = x + \epsilon \phi (x)
$$

其中 $\phi(x)$ 用于控制扰动方向, $\epsilon$ 用于控制扰动幅度. 只要扰动幅度 $\epsilon$ 足够小, 扰动变换 $T(x)$ 的 Jacobian 矩阵就可以无限接近单位矩阵 $I$.

但是, 扰动方向 $\phi(x)$ 还是自由度太高, SVGD 进一步将其限制在 RKHS 上的一个单位球中, 这样就可以利用 RKHS 的相关理论直接求得最优扰动方向的闭式解.

下面 我们先介绍 RKHS 的简单知识, 然后给出在 RKHS 单位球上的最优扰动方向的闭式解.

## RKHS 简介 (可略过不看)

[RKHS (Reproducing Kernel Hilbert Space)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space), 一般翻译为 可再生核希尔伯特空间, 它是泛函分析所研究的内容之一, 包括如下关键词

* **Space**: 指的是向量空间

* **Hilbert Space**: 希尔伯特空间, 它是一类特殊的向量空间, 可以看作 欧式空间 在无限维上面的推广. 它是以德国数学家 [大卫.希尔伯特](https://en.wikipedia.org/wiki/David_Hilbert) (1862 ~ 1943) 命名的

* **Kernel**: 核函数, 具体可以参考 [Kernel method](https://en.wikipedia.org/wiki/Kernel_method) 和 [Kernel function](https://en.wikipedia.org/wiki/Positive-definite_kernel).

> 网上没有找到具体的 核函数 得名由来, 这个概念可能是法国人提出的, 法语词为 noyau; 希尔伯特在其德语论文中使用了该概念, 对应的德语词汇为 kern; 后来翻译到英语中, 则使用了 kernel 一词 (参考自 [知乎](https://www.zhihu.com/question/56961198)). 该概念所表达的意思 (可能是) "the central or most important part of something" (参考自 [Quara](https://www.quora.com/Why-is-Kernel-called-a-kernel-in-machine-learning-not-the-OS)). 其具体操作, 与其说是函数, 称之为 kernel tricks 更合适, 它是计算高维空间两个向量的内积的一种技巧

* **Reproducing Kernel**: 指的是 RKHS 的核函数满足 可再生特性 (reproducing property)

更多信息可以查看相关资料.

根据 RKHS 相关理论 (参考 [Moore–Aronszajn theorem](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space#Moore%E2%80%93Aronszajn_theorem)), 对于一个给定的 满足可再生特性的 核函数 $k(x, x'): X \times X \to \R$, 总是存在唯一一个与之对应的 RKHS $\mathcal{H}$:

$$
\mathcal{H} = \left \{f: f(x) = \sum_{i=1}^{m} a_i k(x, x_i), ~ a_i \in \R, x \in X \subset \R^{d} \right \}
$$

该 RKHS 的内积定义如下:

$$
\langle f, g \rangle_{\mathcal{H}} = \sum_{i, j =1}^{m} a_i b_j k(x_i, x_j)  \in \R
$$

其中 $g(\cdot) = \sum_{i=1}^{m} b_i k(\cdot, x_i)$ .

同样, 还可以定义范数, 这里主要是 2-范数, 如下:

$$
\| f \|_{\mathcal{H}} ^ {2} = \langle f, f \rangle_{\mathcal{H}}
= \sum_{i, j =1}^{m} a_i a_j k(x_i, x_j)
$$

$\mathcal{H}$ 上的 d 个函数可以组成函数向量 $\phi = [ \phi_{1}, \cdots, \phi_{d}]$, 所有 d 维函数向量组成的向量空间记作 $\mathcal{H} ^ {d}$.


## Kernelized Stein Discrepancy

前面介绍了 RKHS. 但是 RKHS 在变分推断中怎么使用呢? 这需要我们进一步介绍 Kernelized Stein Discrepancy (一般简写为 KSD).

在 KSD 中, K (Kernelized) 对应 [核函数](https://en.wikipedia.org/wiki/Positive-definite_kernel), 一个核函数存在唯一一个与之对应的 RKHS ([Moore Aronszajn 定理](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space#Moore%E2%80%93Aronszajn_theorem)).

[Discrepancy](https://dictionary.cambridge.org/dictionary/english/discrepancy) 表示差异, 指的是 KSD 是用来衡量两个概率的差异的, 与 KL 散度类似.

> discrepancy / divergence / deviation / deviance 傻傻分不清楚, 参考 [Wikipedia: Discrepancy](https://en.wikipedia.org/wiki/Discrepancy).

Stein 则特指 [Stein 方法](https://en.wikipedia.org/wiki/Stein%27s_method). Stein 方法由 美国数理统计学家 Charles M. Stein 于 1972 年发表, 主要用于衡量两个概率分布的距离.

> [Charles M. Stein](https://en.wikipedia.org/wiki/Charles_M._Stein), 1920 ~ 2016, 是美国斯坦福大学教授、美国国家科学院院士、著名数理统计学家, 其主要贡献有 Stein 悖论、James-Stein Estimator、Stein 引理、Stein 方法 等。


要讲清楚 KSD, 我们需要用到 Stein 方法中的如下概念:

* Stein 算子 (Stein Operator)
* Stein 等式 (Stein Identity)
* Stein 类 (Stein class of p)
* Stein Discrepancy

我们使用数学语言来描述这几个概念.

### Stein 算子

对于一个 $\R^{d}$ 上的光滑可微概率分布 $p(x)$, 及 d 维函数向量 $\phi(x) = [\phi_{1}(x), \cdots, \phi_{d} (x)]$, Stein 算子 $\mathcal{A}_{p}$的数学描述如下:

$$
\mathcal{A}_{p} \phi (x) = \phi(x) \nabla_{x} \log p(x) ^{\top} + \nabla_{x} \phi (x)
$$

Stein 算子 $\mathcal{A}_{p}$ 的计算结果是一个 $d \times d$ 维度的矩阵.

### Stein 等式

Stein 等式描述的内容是, 对于一些特殊的 函数向量 (sufficiently regular) $\phi (x)$, 可以满足如下等式:

$$
\mathbb{E}_{x \sim p} [ \mathcal{A}_{p} \phi(x)] = 0
$$

这个等式就称为 Stein 等式 (Stein Identity).

这样的 $\phi (x)$ 应该满足哪些条件呢? 我们可以简单推导如下:

$$
\begin{aligned}
\mathbb{E}_{x \sim p} [ \mathcal{A}_{p} \phi(x)] &= \int_{- \infin} ^{\infin} p(x) \mathcal{A}_{p} \phi(x) \mathrm{d} x \\
&= \int_{- \infin} ^{\infin} p(x) ( \phi(x) \nabla_{x} \log p(x) ^{\top} + \nabla_{x} \phi (x) )\mathrm{d} x  & \text{(Stein 算子展开)} \\
&= \int_{- \infin} ^{\infin} p(x) \left ( \phi(x) \frac{1}{p(x)} \nabla_{x} p(x) + \nabla_{x} \phi (x) \right ) \mathrm{d} x  \\
&= \int_{- \infin} ^{\infin} ( \phi(x) \nabla_{x} p(x) + p(x) \nabla_{x} \phi(x) )  \mathrm{d} x \\
&= \int_{- \infin} ^{\infin} \nabla_{x} (p(x) \cdot \phi(x)) \mathrm{d} x \\
&= p(x) \phi(x) | _{- \infin} ^ {\infin}
\end{aligned}
$$

即 Stein 等式要求 $\phi(x)$ 满足的以下任一条件即可:

* $p(x) \phi(x) = 0, ~~ \forall x \in \mathcal{X} \subset \R^{d}$
* $\lim _{\| x \| \to \infin} p(x) \phi(x) = 0$

其中条件 2 是比较容易满足的, 只要限制 $p(x)$ 或 $\phi(x)$ 在一定范围外取值维 0 即可.

### Stein 类

对于概率分布 $p(x)$, 满足 Stein 等式的函数向量组成的集合称之为 概率 p 的 Stein 类.


### Stein Discrepancy

我们注意到, 对于 p 的 Stein 类中的函数 $\phi (x)$, 满足 Stein 等式的条件, 即

$$
\mathbb{E}_{x \sim p} [ \mathcal{A}_{p} \phi(x)] = 0
$$

如果我们对另一个分布 $x \sim q(x)$ 求期望, $\mathbb{E}_{x \sim q} [ \mathcal{A}_{p} \phi(x)] \ne 0$. 我们可以基于此, 定义一个 差异度量 (discrepancy measure), 称之为 Stein Discrepancy, 如下:

$$
\mathbb{S} (q, p) = \max_{\phi \in \mathcal{F}} \left \{ (\mathbb{E}_{x \sim q} [  \text{tr} (\mathcal{A}_{p} \phi(x)]) ) ^ {2}\right \}
$$

其中 $\text{tr} (A)$ 表示求方阵 $A$ 的 [迹 (trace)](https://en.wikipedia.org/wiki/Trace_(linear_algebra)). $\mathcal{F}$ 可以是 p 的 Stein 类的任意子集.


Stein Discrepancy 需要对整个 $\mathcal{F}$ 求最值, 这具有较大的计算难度 (intractable).  KSD 通过将 $\mathcal{F}$ 限制在某一核函数对应的 RKHS 的单位球上, 利用 RKHS 的相关理论, 直接给出最优函数向量 $\phi ^ {*}$ 的闭式解, 绕过了对 $\mathcal{F}$ 求最值的难解问题.


### KSD

前面讲了, 直接求解 Stein Discrepancy, 需要在 $\mathcal{F}$ 对 $(\mathbb{E}_{x \sim q} [  \text{tr} (\mathcal{A}_{p} \phi(x)]) ) ^ {2}$ 求最值, 这通常是难解的.

但是, 如果我们存在一个正定核函数 $k(x, x'): \mathcal{X} \times \mathcal{X} \to \R$, 总是存在一个与之对应的 RKHS $\mathcal{H}$ (Moore Aronszajn 定理). 我们将 $\mathcal{F}$ 限定为 $\mathcal{H} ^ {d}$ 上的单位球, 即:

$$
\mathcal{F} = \{ \phi : \phi \in \mathcal{H} ^ {d}, \| \phi \| _{\mathcal{H} ^ {d}} \le 1 \}
$$

则前面的 Stein Discrepancy 变成了 Kernelied Stein Discrepancy, 如下:

$$
\mathbb{S}_{\text{K}} (q, p) = \max_{\phi \in \mathcal{H} ^ {d}} \left \{ (\mathbb{E}_{x \sim q} [  \text{tr} (\mathcal{A}_{p} \phi(x)]) ) ^ {2} ~~s.t.~~  \| \phi \| _{\mathcal{H} ^ {d}} \le 1 \right \}
$$


根据 RKHS 相关理论, 当 $\phi(x) = \frac{\phi_{p, q}^{*} (x)}{ \| \phi_{p, q} ^ {*}\| _{\mathcal{H}^{d}}}$ 时, 可以求得 KSD 的结果, 且 $\mathbb{S}_{K} (q, p) = \| \phi_{p, q} ^ {*}\| _{\mathcal{H}^{d}} ^ {2}$. 其中

$$
\phi_{p, q} ^ {*} (\cdot) = \mathbb{E}_{x \sim q} [ A_{p} k(x, \cdot)]
$$


## SVGD

好了, 现在我们要回过头来求解我们的变分推断问题.

前面讲了, SVGD 的核心是, 给定初始分布 $q_{0}$, 采样一组粒子 (particles) $\{ x_{i} \in \mathcal{X} \subset \R^{d} \}_{i=1}^{n}$; 对这一组粒子逐步施以微小扰动 $T_{\epsilon} (x) = x + \epsilon \phi(x)$, 驱使其分布逼近我们求解的目标分布 $p$.

在每次迭代中, 我们只需要确定扰动函数的 幅度系数 $\epsilon$ 和 方向 $\phi (x)$ 即可.

SVGD 的论文给出了一个重要的定理 (参考论文定理 3.1 和 引理 3.2), 该定理将 KL 散度的导数 与 KSD 联系起来, 并表明: 满足 KSD 取最值条件的函数 $\phi_{p, q} ^ {*}$ 也正是满足 KL 散度最小化的 最优扰动方向.


### 定理 3.1

给定 $x \in \mathcal{X} \subset \R ^ {d}$ 服从分布 $q$, 经扰动函数 $z = T_{\epsilon} (x) = x + \epsilon \phi(x)$ 变换后的分布为 $q_{T} (z)$, 则有:

$$
\nabla_{\epsilon} \text{KL} (q_{T} \| p) | _{\epsilon = 0} = - \mathbb{E}_{x \sim q} [ \text{tr}( \mathcal{A}_{p} \phi (x) )]
$$

其证明过程如下:

* 首先, 前面介绍过, 扰动变换带来的分布变化如下:

$$
q_{T} (z) = q(T^{-1} (z)) \cdot | \det ( \nabla_{z} T^{-1} (z) ) |
$$

类似地, 如果已知变换后的概率分布 $p(z)$, 则其变换前的分布为

$$
p_{[T^{-1}]}(x) = p(T(x)) \cdot  | \det ( \nabla_{x} T(x) ) |
$$

实际上就是对 z 施加了 $x = T^{-1} (z)$ 变换.

* 然后, 我们对 $\text{KL} (q_{T} \| p)$ 可以进行变量变换如下:

$$
\begin{aligned}
\text{KL} (q_{T} \| p) &= \mathbb{E}_{z \sim q_{T}} \left [ \log \frac{q_{T} (z)}{p(z)} \right ]  \\
&= \mathbb{E}_{x \sim q} \left [ \log \frac{q(x)}{p_{[T^{-1}]}(x)} \right ]  \\
&= \text{KL} (q \| p_{[T^{-1}]})
\end{aligned}
$$

* 分布 q 与参数 $\epsilon$ 无关, 因此

$$
\begin{aligned}
\nabla_{\epsilon} \text{KL} (q_{T} \| p) &= \nabla_{\epsilon} \text{KL} (q \| p_{[T^{-1}]}) \\
&= \nabla_{\epsilon} \mathbb{E}_{x \sim q} \left [ \log \frac{q(x)}{p_{[T^{-1}]}(x)} \right ] \\
&= - \mathbb{E}_{x \sim q} [ \nabla_{\epsilon} \log p_{[T^{-1}]}(x)]  \\
&= - \mathbb{E}_{x \sim q} [ \nabla_{\epsilon} \log ( p(T(x)) \cdot  | \det ( \nabla_{x} T(x) )| ) ]
\end{aligned}
$$

* 对 $p_{[T^{-1}]} (x)$ 展开, 继续求导有:

$$
\begin{aligned}
\nabla_{\epsilon} \log p_{[T^{-1}]}(x) &= \nabla_{\epsilon} \log ( p(T(x)) \cdot  | \det ( \nabla_{x} T(x) )| ) \\
&= \nabla_{\epsilon} \log p(T(x)) + \nabla_{\epsilon} \log |\det ( \nabla_{x} T(x) )|  \\
&= \nabla_{T} \log p(T(x)) ^ {\top} \nabla_{\epsilon} T(x) + \text{tr} ( (\nabla_{x} T(x)) ^ {-1} \cdot \nabla_{\epsilon} \nabla_{x} T(x))
\end{aligned}
$$

其中, 关于 Jacobian 行列式求值的过程可以参考 [Wikipedia: Jacobi_formula](https://en.wikipedia.org/wiki/Jacobi%27s_formula).

* 因为我们只需要对 $\epsilon = 0$ 的时候求导数的值, 则有如下已知条件

$$
\begin{aligned}
T(x) &= x + \epsilon \phi (x) = x \\
\nabla_{\epsilon} T(x) &= \phi(x) \\
\nabla_{x} T(x) &= I \\
\nabla_{\epsilon} \nabla_{x} T(x) &= \nabla_{x} \phi(x)
\end{aligned}
$$

* 带入上述已知条件, 有:

$$
\begin{aligned}
\nabla_{\epsilon} \text{KL} (q_{T} \| p) | _{\epsilon = 0} &= \nabla_{\epsilon} \text{KL} (q \| p_{[T^{-1}]}) | _{\epsilon = 0}  \\
&= - \mathbb{E}_{x \sim q} [ \nabla_{\epsilon} \log ( p(T(x)) \cdot  | \det ( \nabla_{x} T(x) )| ) ] |  _{\epsilon = 0} \\
&= - \mathbb{E}_{x \sim q} [ \nabla_{T} \log p(T(x)) ^ {\top} \nabla_{\epsilon} T(x) + \text{tr} ( (\nabla_{x} T(x)) ^ {-1} \cdot \nabla_{\epsilon} \nabla_{x} T(x)) ] | _{\epsilon = 0} \\
&= - \mathbb{E}_{x \sim q} [ \nabla_{x} \log p(x) ^ {\top} \phi(x) + \text{tr} ( \nabla_{x} \phi (x)) ]  \\
&= - \mathbb{E}_{x \sim q} [ \text{tr} ( \nabla_{x} \log p(x) \phi (x) + \nabla_{x} \phi(x) )]  \\
&= - \mathbb{E}_{x \sim q} [ \text{tr} ( \mathcal{A}_{p} \phi(x)) ]
\end{aligned}
$$

定理 3.1 得证.


### 引理 3.2

根据定理 3.1 可知:
$$
\nabla_{\epsilon} \text{KL} (q_{T} \| p) | _{\epsilon = 0} =  - \mathbb{E}_{x \sim q} [ \text{tr} ( \mathcal{A}_{p} \phi(x)) ]
$$


而我们前面也给出了 KSD 的定义:

$$
\mathbb{S}_{\text{K}} (q, p) = \max_{\phi \in \mathcal{H} ^ {d}} \left \{ (\mathbb{E}_{x \sim q} [  \text{tr} (\mathcal{A}_{p} \phi(x)]) ) ^ {2} ~~s.t.~~  \| \phi \| _{\mathcal{H} ^ {d}} \le 1 \right \}
$$

两者是极为相似的.

事实上, 当我们将扰动方向限制在 RKHS 的球 $\mathcal{B} \equiv \{ \phi \in \mathcal{H} ^ {d}: \| \phi \|_{\mathcal{H} ^ {d}} ^ {2} \le \mathbb{S}_{K} (q, p) \}$ 上时, 有:

$$
\min_{\phi \in \mathcal{B}} \{ \nabla_{\epsilon} \text{KL} (q_{T} \| p) | _{\epsilon = 0} \} = - \mathbb{S}_{\text{K}} (q, p) 
$$

且 $\phi(x) = \phi_{p, q} ^ {*} (x)$ 取到最小值.


### 定理 3.3

如果采用泛函微分的思想, 定理 3.1 和 引理 3.2 可以合并表述如下:

$$
\nabla_{\phi} \text{KL} (q_{T} \| p) | _{\phi = 0} = - \phi_{p, q} ^ {*}
$$

其中 $\phi \in \mathcal{H} ^ {d}$ 是 d 维函数向量, $\text{KL} (q_{T} \| p)$ 是一个关于 $\phi$ 泛函.


### SVGD 迭代过程

有了上述理论基础, 我们可以给出 SVGD 的迭代逼近过程了.

根据初始分布 $q_{0} (x)$ 采样一组粒子 $\{ x_i \}_{i=1}^{n}$. 

在第 $t$ 轮迭代中, 粒子的分布为 $q_{t} (x)$.

* 我们先求粒子的最优扰动方向 $\phi_{q_{t}, p} ^ {*} = \mathbb{E}_{x \sim q_{t}} [ \mathcal{A}_{p} k(x, \cdot)]$;

* 我们对粒子施以变换 $T^{*} (x) = x + \epsilon_{t} \phi_{q_{t}, p} ^ {*} (x)$, 则变换后粒子分布为 $q_{t+1} = q_{T^{*}}$

重复上述过程, 直至达到迭代上线 或 过程收敛.


上述迭代过程的最优扰动方向涉及到求期望, 这个我们需要使用 经验期望来代替, 即:

$$
\begin{aligned}
\phi_{q_{t}, p} ^ {*} &= \mathbb{E}_{x \sim q_{t}} [ \mathcal{A}_{p} k(x, \cdot)] \\
&\approx \frac{1}{n} \sum_{i=1}^{n} A_{p} k(x_{i}, \cdot) &= \hat{\phi}_{q_{t}, p} ^ {*}
\end{aligned}
$$

而且我们也无须求出 $\phi_{q_{t}, p} ^ {*}$ 的显式结果, 只需要求其在每个粒子上的取值 $\phi_{q_{t}, p} ^ {*} (x_{i})$ 即可.


该过程还保留了未知的核函数, 这给了该过程较大的自由度, 可以灵活选择合适的核函数.
