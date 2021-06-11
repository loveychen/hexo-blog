---
title: "机器学习数学基础: Fisher Information"
date: 2021-04-25 16:19:42
tags:
    - 数学基础
    - Fisher Information
    - Information
categories:
    - Mathematics
    - Statistics
---

> 主要参考 [Wikipedia: Fisher Information](https://en.wikipedia.org/wiki/Fisher_information).



# 简介

在统计学中, Fisher Information 有时也直接成为 Information, 它是一个用于衡量 随机观察变量 $X$ 所携带的关于未知参数 $\theta$ 的信息的统计量. 其中 $\theta$ 是关于 X 的分布的模型的参数, 即

$$
X \sim f(X; \theta)
$$


Fisher Information 可以看作是 Score 的方差, 也可以看作是 观察信息 (the observed information) 的期望.

> [Score](https://en.wikipedia.org/wiki/Score_(statistics))
> 在统计学中, score (或称为 informant) 是 对数似然函数的关于参数向量的梯度, 即:
> $$
> s(\theta) \equiv \frac{\partial \log \cal{L} (\theta)}{\partial \theta}
> $$
> 其中, $\cal{L}(\theta)$ 是参数 $\theta$ 的似然函数.
> 
> [Observed Information](https://en.wikipedia.org/wiki/Observed_information)
> 在统计学中, Observed Information (或称为 Observed Fisher Information) 是对数似然函数的二阶导数 (Hessian Matrix), 它是 Fisher Information 的 Sample 版本. 
> $$
> \begin{aligned}
> \cal{J}(\theta^{*}) &= \nabla \nabla ^ {T} \ell (\theta) \\
> &= -\left.\left(\begin{array}{cccc}
> \frac{\partial^{2}}{\partial \theta_{1}^{2}} & \frac{\partial^{2}}{\partial_{1} \partial \theta_{2}} & \cdots & \frac{\partial^{2}}{\partial \theta_{1} \partial \theta_{p}} \\
> \frac{\partial^{2}}{\partial \theta_{2} \partial \theta_{1}} & \frac{\partial^{2}}{\partial \theta_{2}^{2}} & \cdots & \frac{\partial^{2}}{\partial \theta_{2} \partial \theta_{p}} \\
> \vdots & \vdots & \ddots & \vdots \\
> \frac{\partial^{2}}{\partial \theta_{p} \partial \theta_{1}} & \frac{\partial^{2}}{\partial \partial_{p} \partial \theta_{2}} & \cdots & \frac{\partial^{2}}{\partial \Phi_{p}^{2}}
> \end{array}\right) \quad \ell(\theta) \right|_{\theta=\theta^{*}}
> \end{aligned}
> $$


在贝叶斯统计中, 

* 后验分布 的 [渐进分布 (asymptotic distribution)](https://en.wikipedia.org/wiki/Asymptotic_distribution) 依赖于 Fisher Information 而非先验分布
* Fisher Information 可用于计算 [Jeffeys prior](https://en.wikipedia.org/wiki/Jeffreys_prior)
* Fisher Information 可用于在 [极大似然估计 (Maximum-Likelihood Estimation)](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) 中计算 方差矩阵


# 定义

正式地, 假设随机变量 $X$ 的概率密度函数记为 $f(X; \theta)$. 对数似然函数关于参数 $\theta$ 的偏导数称为 Score, Score 的方差就是 Fisher Information, 即:

$$
\begin{aligned}
\mathcal{I}(\theta) &= \mathrm{E}\left[\left(\frac{\partial}{\partial \theta} \log f(X ; \theta)\right)^{2} \mid \theta\right] \\
&= \int\left(\frac{\partial}{\partial \theta} \log f(x ; \theta)\right)^{2} f(x ; \theta) dx \\
& \geq 0
\end{aligned}
$$

其中 $\frac{\partial}{\partial \theta} \log f(x ; \theta)$ 称为 Score. 


如果 $\theta$ 为 X 分布的真实参数, 即 $X \sim f(X; \theta)$, 则 Score 的期望为 0, 即:

$$
\begin{aligned}
\mathbf{E} (s(\theta)) &= \mathbf{E}\left[\frac{\partial}{\partial \theta} \log f(X ; \theta) \mid \theta\right] \\
=& \int \frac{\frac{\partial}{\partial \theta} f(x ; \theta)}{f(x ; \theta)} f(x ; \theta) d x \\
=& \frac{\partial}{\partial \theta} \int f(x ; \theta) d x \\
=& \frac{\partial}{\partial \theta} 1=0
\end{aligned}
$$

## 从二阶导数角度看 Fisher Information

如果分布 $f(X; \theta)$ 对于 $\theta$ 二阶可导, Fisher Information 还可以解释如下:

$$
\begin{aligned}
\mathcal{I}(\theta) &= \mathrm{E}\left[\left(\frac{\partial}{\partial \theta} \log f(X ; \theta)\right)^{2} \mid \theta\right] \\
&= \int\left(\frac{\partial}{\partial \theta} \log f(x ; \theta)\right)^{2} f(x ; \theta) dx
\end{aligned}
$$

其证明过程如下:

$$
\begin{aligned}
\frac{\partial^{2}}{\partial \theta^{2}} \log f(X ; \theta) &= \frac{\frac{\partial^{2}}{\partial \theta^{2}} f(X ; \theta)}{f(X ; \theta)}-\left(\frac{\frac{\partial}{\partial \theta} f(X ; \theta)}{f(X ; \theta)}\right)^{2} \\
&= \frac{\frac{\partial^{2}}{\partial \theta^{2}} f(X ; \theta)}{f(X ; \theta)}-\left(\frac{\partial}{\partial \theta} \log f(X ; \theta)\right)^{2}
\end{aligned}
$$

而右边的第一项的期望为 0, 证明如下:

$$
\mathbf{E}\left[\frac{\frac{\partial^{2}}{\partial \theta^{2}} f(X ; \theta)}{f(X ; \theta)} \mid \theta\right]=\frac{\partial^{2}}{\partial \theta^{2}} \int f(x ; \theta) d x=0
$$

从这个视角来看, Fisher Information 可以解释为 [支持曲线](https://en.wikipedia.org/wiki/Support_curve) (support curve, 即 对数似然函数的曲线) 的 曲率 (valcature).  

在 MLE 问题上, Fisher Information 低意味着 
