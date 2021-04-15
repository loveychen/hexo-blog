---
title: ConvLab2 介绍(一) User Simulator
date: 2021-04-14 15:29:43
tags: 
    - TOD
    - ConvLab2
    - User Simulator
    - 更新中
---



为了自动评估 ConvLab 对话系统的能力, ConvLab 提供了 User Simulator, 用来模拟用户.


User Simulator 中最重要的是 User Policy. User Policy 以预设的 User Goal 和 当前的系统答复 System Dialogue Acts 为输入, 最终输出 User Dialogue Acts.

ConvLab2 提供了两种类型的 User Policy:

* Agenda-based model
* Neural Netword based model, 如 HUS 及其变种.


# Agenda-based User Model

**Agenda-based User Simulator** 最早可以追溯到 2007 年 Jost Schatzmann / Blaise Thomson / Karl Weilhammer / Hui Ye / Steve Young 等人的论文 [Agenda-based user simulation for bootstrapping a POMDP dialogue system](https://www.aclweb.org/anthology/N07-2038/).


> Steve Young 是 Conversational AI 的先驱之一.

而使用 Agenda-based 方法来进行对话管理, 则可以追溯到 1999 年 Alexander Rudnicky / Wei Xu 的论文 [An agenda-based dialog management architecture for spoken language systems](http://www.cs.cmu.edu/~xw/asru99-agenda.pdf).


> [agenda](https://www.collinsdictionary.com/dictionary/english/agenda) "议事日程; 议程表; 记事册"
> 1. [Countable Noun] You can refer to the **political issues which are important** at a particular time as an agenda
> 2. [Countable Noun] An agenda is **a list of the items** that have to be discussed at a meeting.


> **User Agenda** 
>
> 在 Jost Schatzmann 等人的工作中, Agenda 是一个栈式结构, 内部元素为正在处理或还未处理的 User Dialogue Acts (以达到 User Goal).
>
> " **The user agenda A** is a **stack-like structure** containing
the **pending user dialogue acts** that are needed to elicit the information specified in the goal"
> 
> [pend](https://www.collinsdictionary.com/dictionary/english/pend), "悬吊, 悬挂; 悬而未决, 待决"
> 1. [verb] to await judgement of settlement
> 2. [verb, dialect] to hang; depend
> 3. [noun, Scottish] an archway or vaulted passage (拱道, 拱廊)
>
> [elicit](https://www.collinsdictionary.com/dictionary/english/elicit) "引出, 探出, 诱出"
> 1. [verb] If you elicit a response or a reaction, you do or say something which makes other people respond or react.
> 2. [verb] If you elicit a piece of information, you get it by asking the right questions.

Rudnicky 等人使用 Agenda 方法来进行对话管理, 属于系统层面的研究 (也是当前较为火热的研究方向, 最新技术一般是 `DST + POL` 来实现 DM 或者 直接使用 E2E 方式建模系统); 

Jost Schatzmann 则将 Agenda 的方法应用到 User Simulator 之上, 用以辅助 Dialogue System 的学习和评估.


Agenda-based User Simulator 是一个 模拟用户行为 的概率模型, 它主要依赖于两项输入:

* (a compact representation of the) **User Goal**
* (a stack-like) **User Agenda**

User Goal 和 User Agenda 组成了 User Simulator 的 **User Dialogue State**, 即:



$$
S = <A, G> \\
G = <C, R>
$$

其中 S 表示 State, A 表示 Agenda, G 表示 User Goal, C 表示 Constraints, R 表示 Requests.

> User Goal 可以划分为两部分: Constraints 和 Requests, 这个在 MultiWOZ 中一般表示为 Inform Slots 和 Request Slots.



而 Dialogue State 与 Dialogue Act 组成了一个 Dialogue, 即:

$$
S_t \xrightarrow{a_u} S_t^{'} \xrightarrow{a_m} S_{t+1} 
$$

基于此, 用户建模可以划分为三个部分:

* Action Selection: $P(a_u | S_t)$, 用户模型在对话状态 $S_t$ 下, 选择 User Action $a_u$ 的概率.
* System State Transition: $P(S_t^{'} | a_u, S_t)$, 系统模型在对话状态 $S_t$ 及 User Action $a_u$ 下, 转移到新状态 $S_t^{'}$ 的概率
* ~~$P(a_m | S_t^{'})$ 系统模型, 不属于用户模拟器的功能~~
* User State Transition: $P(S_{t+1} | a_m, S_t^{'})$, 用户模型在新的对话状态 $S_t^{'}$ 及 System Action $a_m$ 下转移到新的状态 $S_{t+1}$ 的概率


> **注意**
> 这里的 System State Transition 和 User State Transition 是我命名的, 原始论文没有给这两个状态转移模型命名.
> 说一下这样命名的理由
> 1. 系统状态转移 $P(S_t^{'} | a_u, S_t)$ 是由 User Action $a_u$ 触发的, 一般地我们认为 User Action 会引起系统状态的变化, 因此命名为 **系统状态转移**;
> 2. 同理, 用户状态转移  $P(S_{t+1} | a_m, S_t^{'})$ 是由 System Action $a_m$ 触发的, 造成了用户状态的变化, 因此命名为 **用户状态转移**


Agenda-based User Simulator 的工作过程如下:

1. 在对话开始时, Simulator 会随机设置 User Goal. 而 User Goal 中的 Constraints 则转换成 **Inform Acts** 压入 Agenda 中, User Goal 中的 Requests 则转换为 **Request Acts** 压入 Agenda 中.

2. 在对话过程中, User Goal 和 Agenda 都会动态更新. Agenda 的顶部元素会用来生成 User Act $a_u$; 而系统响应 System Act $a_m$ 到来时, 新的 User Acts 被压入 Agenda 中, 而不再需要的 User Acts 则从 Agenda 中移除.


关于 Simulator 的工作过程, 建议参考文章 [Schatzmann et al, 2017, Agenda-based user simulation for bootstrapping a POMDP dialogue system](https://www.aclweb.org/anthology/N07-2038/) 中的 Figure 1 给出的示例.


## User Action Selection Model

User Simulator 的第一个问题是 **User Action Selection**, 即:  $P(a_u | S_t)$, 在对话状态 $S_t$ 下, 选择 User Action $a_u$ 的概率. 

如前所述, $S = <A, G>$, 可以将 User Action Selection Model 进行拆分:

$$ 
P(a_u | S_t) \xlongequal{S_t = <A_t, G_t>} P(a_u | <A_t, G_t>)
$$

又因为当前 User Action 的选择与当前的 User Goal 没有直接关系 (User Goal 中的信息已经在 Agenda 中体现了), 因此, 问题可以简化为:

$$
\begin{aligned}
P(a_u | S_t) & = P(a_u | <A_t, G_t>) \\
    & = P(a_u | A_t)
\end{aligned}
$$


我们只使用了 Agenda 中最新的 n 个元素参与预测  User Action $a_u$, 则问题建模可做如下简化:

$$
\begin{aligned}
P(a_u | S_t) & = P(a_u | <A_t, G_t>) \\
    & = P(a_u | A_t) \\
    &= P(a_u | A_t[-n:]) \\
    &= \delta (a_u, A_t [-n:])
\end{aligned}
$$


## System State Transition Model 

User Simulator 的第二个问题是 **System State Transition**, 即: $P(S_t^{'} | a_u, S_t)$, 系统模型在对话状态 $S_t$ 及 User Action $a_u$ 下, 转移到新状态 $S_t^{'}$ 的概率.

同样, 我们需要根据 $S = <A, G>$ 对模型进行拆分, 因此:

$$
P(S_t^{'} | a_u, S_t)  \xlongequal{S_t = <A_t, G_t>} P(<A_t^{'}, G_t^{'}> | a_u, <A_t, G_t>)
$$


同时, 我们假设 Simulator 在执行 User Act 时, 其 User Goal 是不变化的, 即 

$$
G_t^{'}  \nleftrightarrow a_u
$$

同时, 根据前述的 User Action Selection 模型可知, User Action $a_u$ 是一个关于 $A_t [-n:]$ 的函数,

因此 System State Transition 可以拆分为:

$$
\begin{aligned}
P(S_t^{'} | a_u, S_t) &= P(<A_t^{'}, G_t^{'}> | a_u, <A_t, G_t>) \\
&= g(A_t^{'}, A_t[-n:]) * f(G_t^{'}, G_t)
\end{aligned}
$$



## User State Transition Model

与 System State Transition Model 类似, 遵循如下几个原则

* $A_t^{'}$ 与 $G_t^{'}$ 没有其它限制
* 新的 User Goal $G_{t+1}$ 条件独立于 Agenda $A_t{'}$
* 链式条件概率法则

User State Transition Model 可以如下分解

$$
\begin{aligned}
P(S_{t+1} | a_m, S_t^{'}) &= P( <A_{t+1}, G_{t+1}> | a_u, <A_t^{'}, G_t^{'}>) \\
&= \underbrace{P(A_{t+1} | a_m, A_t^{'}, G_{t+1})}_{\text{agenda update}} * \underbrace{P(G_{t+1} | a_m, G_t^{'})}_{\text{goal update}}
\end{aligned}
$$

即 User State Transition Model 可以拆分为 Goal Update Model 和 Agenda Update Model 两部分.


**Goal Update Model**

$$
\begin{aligned}
P(G_{t+1} | a_m, G_t^{'}) &= P(<C_{t+1}, R_{t+1}> | a_m, <C_t^{'}, R_t^{'}>) \\
    &= P(R_{t+1} | a_m, R_t^{'}, C_{t+1}) * P(C_{t+1} | a_m, R_t^{'},  C_t^{'})
\end{aligned}
$$


**Agenda Update Model**

$$
\begin{aligned}
P(A_{t+1} | a_m, A_t^{'}, G_{t+1}) &= P(A_{t+1}[-n:] | a_m, A_t [-n: ], G_{t+1}) \\
 &= P(A_{t+1}[-n:] | a_m, G_{t+1}) * h(A_{t+1}[-n:], A_t[-n:])
\end{aligned}
$$
