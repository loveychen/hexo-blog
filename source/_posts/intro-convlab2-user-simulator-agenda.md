---
title: ConvLab2 01. Agenda-based User Simulator
date: 2021-04-14 15:29:43
tags: 
    - TOD
    - ConvLab2
    - User Simulator
    - Agenda
categories:
    - 对话系统
    - ConvLab2
---



为了自动评估 ConvLab 对话系统的能力, ConvLab 提供了 User Simulator, 用来模拟用户.


User Simulator 中最重要的是 User Policy. User Policy 以预设的 User Goal 和 当前的系统答复 System Dialogue Acts 为输入, 最终输出 User Dialogue Acts.

ConvLab2 提供了两种类型的 User Policy:

* Agenda-based model
* Neural Netword based model, 如 HUS 及其变种.


# Agenda-based User Simulator

**Agenda-based User Simulator** 最早可以追溯到 2007 年 Jost Schatzmann / Blaise Thomson / Karl Weilhammer / Hui Ye / Steve Young 等人的论文 [Agenda-based user simulation for bootstrapping a POMDP dialogue system](https://www.aclweb.org/anthology/N07-2038/).


> Steve Young 是 Conversational AI 的先驱之一.

而使用 Agenda-based 方法来进行对话管理, 则可以追溯到 1999 年 Alexander Rudnicky / Wei Xu 的论文 [An agenda-based dialog management architecture for spoken language systems](http://www.cs.cmu.edu/~xw/asru99-agenda.pdf).


> [agenda](https://www.collinsdictionary.com/dictionary/english/agenda) "议事日程; 议程表; 记事册"
> 1. [Countable Noun] You can refer to the **political issues which are important** at a particular time as an agenda
> 2. [Countable Noun] An agenda is **a list of the items** that have to be discussed at a meeting.


> **User Agenda** 
>
> 在 Jost Schatzmann 等人的工作中, Agenda 是一个栈式结构, 内部元素为正在处理或还未处理的 User Dialogue Acts (以完成 User Goal).
>
> " **The user agenda A** is a **stack-like structure** containing the **pending user dialogue acts** that are needed to elicit the information specified in the goal"
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


----


Jost Schatzmann 则将 Agenda 的方法应用到 User Simulator 之上, 用以辅助 Dialogue System 的学习和评估.

Agenda-based User Simulator 是一个 模拟用户行为 的概率模型, 它主要依赖于两项输入:

* (a compact representation of the) **User Goal**. User Goal 包括两部分内容: Constraints 与 Requests. 其中 Constraints 是用户指定的限制条件, 而 Requests 是在对话过程中, 用户想要获取的信息.
* (a stack-like) **User Agenda**. User Agenda 是一个栈式结构, 其元素是挂起的 User Act, 这些 User Act 用于在后续的对话中获取 (elicit) User Goal 中指定的需要获取的信息.

举例来讲, 假设用户目标是在指定区域 `area=central` 搜索提供 `drinks=beer` 的酒吧 `type=bar`, 这些都是用户指定的限制条件, 即:

$$
C_0 = \begin{bmatrix}
    type &= & bar \\
    drinks &= & beer \\
    area &= & central
\end{bmatrix}
$$

用户需要在对话过程中, 获取酒吧名字 `name`, 具体地址 `addr` 及 电话 `phone`, 这三个则是用户目标中的 Request 部分, 即:

$$
R_0 = \begin{bmatrix}
    name &= & ? \\
    addr &= & ? \\
    phone &= & ?
\end{bmatrix}
$$


User Agenda 是 Pending User Action 组成的栈式结构, 示例如下:

$$
A_0 = \begin{bmatrix}
    inform(type = bar) \\
    inform(drinks = beer) \\
    inform(area = central) \\
    request(name) \\
    request(addr) \\
    request(phone) \\
    bye()
\end{bmatrix}
$$


User Goal 和 User Agenda 组成了 User Simulator 的 **User Dialogue State**, 即:


$$
S = <A, G> \\
G = <C, R>
$$

其中 S 表示 State, A 表示 Agenda, G 表示 User Goal, C 表示 Constraints, R 表示 Requests.

> **重点**
> User Dialogue State 由 User Agenda 和 User Goal 组成. 
> User Goal 可以分为 Constraints 和 Requests 两部分, 分别表示用户的限制条件 和 需要在对话中获取的信息. 
> User Agenda 则是由 User Dialogue Act 组成的栈式结构. 
> Dialogue Act 在多领域对话系统中常常表示为 $<Domain, Intent, Slot, Value>$ 四元组组成的集合. 在单领域对话系统中, 可以省去 Domain 元素, 简化为 $<Intent, Slot, Value>$ 三元组. 


----


而 Dialogue State 与 Dialogue Act 组成了一个 Dialogue, 即:

$$
\cdots \rightarrow S_t \xrightarrow{a_u} S_t^{'} \xrightarrow{a_m} S_{t+1} \rightarrow \cdots
$$

即在时刻 t, 用户处于状态 $S_t$, 此时产生 User Action $a_u$, 用户进入临时状态 $S_{t}^{'}$. (*系统响应用户输入动作, 产生 System Action $a_m$*). 用户接受系统响应 $a_m$, 进入到下一时刻状态 $S_{t+1}$.


基于上述用户与系统的交互流程, 用户建模可以划分为三个部分:

* Action Selection: $P(a_u | S_t)$, 用户在对话状态 $S_t$ 下, 产生 User Action $a_u$ 的概率.
* User action based State Transition: $P(S_t^{'} | a_u, S_t)$, 用户在对话状态 $S_t$ 执行 User Action $a_u$, 转移到临时状态 $S_t^{'}$ 的概率
* System action based State Transition: $P(S_{t+1} | a_m, S_t^{'})$, 用户在临时状态 $S_t^{'}$ 下接收 System Action $a_m$ 转移到新的状态 $S_{t+1}$ 的概率


----

Agenda-based User Simulator 的工作过程如下:

1. 在对话开始时, Simulator 会随机设置 User Goal. 而 User Goal 中的 Constraints 则转换成 **Inform Acts** 压入 Agenda 中, User Goal 中的 Requests 则转换为 **Request Acts** 压入 Agenda 中.

2. 在对话过程中, User Goal 和 Agenda 都会动态更新. Agenda 的顶部元素会用来生成 User Act $a_u$; 而系统响应 System Act $a_m$ 到来时, 新的 User Acts 被压入 Agenda 中, 而不再需要的 User Acts 则从 Agenda 中移除. $a_m$ 中携带的槽位信息会用于更新 User Goal (主要跟新 User Goal 中的 Requests 部分).


关于 Simulator 的工作过程, 建议参考文章 [Schatzmann et al, 2017, Agenda-based user simulation for bootstrapping a POMDP dialogue system](https://www.aclweb.org/anthology/N07-2038/) 中的 Figure 1 给出的示例.


## User Action Selection Model

User Simulator 的第一个问题是 **User Action Selection**  $P(a_u | S_t)$, 即用户在对话状态 $S_t$ 下, 生成 User Action $a_u$ 的概率. 

如前所述, 对话状态由 User Agenda 和 User Goal 两部分组成,  $S = <A, G>$,  因此可以将 User Action Selection Model 进行拆分:

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


我们只使用了 Agenda 中优先级最高的顶部 n 个元素 $A[-n:]$ 参与预测  User Action $a_u$, 则问题建模可做如下简化:

$$
\begin{aligned}
P(a_u | S_t) & = P(a_u | <A_t, G_t>) \\
    & = P(a_u | A_t) \\
    &= P(a_u | A_t[-n:]) \\
    &= \delta (a_u, A_t [-n:])
\end{aligned}
$$


## User action based State Transition Model 

User Simulator 的第二个问题是 **User action based State Transition**, 即: $P(S_t^{'} | a_u, S_t)$, 用户在状态 $S_t$ 下执行 User Action $a_u$, 转移到临时状态 $S_t^{'}$ 的概率.

同样, 我们需要根据 $S = <A, G>$ 对模型进行拆分, 因此:

$$
P(S_t^{'} | a_u, S_t)  \xlongequal{S_t = <A_t, G_t>} P(<A_t^{'}, G_t^{'}> | a_u, <A_t, G_t>)
$$


其中 $A_t^{'}, G_t^{'}$ 分别代表执行 User Action $a_u$ 后, 临时的 User Agenda 和 User Goal.


同时, 我们假设 Simulator 在执行 User Act 时, 其 User Goal 是不变化的, 即 

$$
G_t^{'}  \nleftrightarrow a_u
$$

同时, 根据前述的 User Action Selection 模型可知, User Action $a_u$ 是一个关于 $A_t [-n:]$ 的函数,

因此 System State Transition 可以拆分为:

$$
\begin{aligned}
P(S_t^{'} | a_u, S_t) &= P(<A_t^{'}, G_t^{'}> | a_u, <A_t, G_t>) \\
&= g(A_t^{'}, A_t) * f(G_t^{'}, G_t)
\end{aligned}
$$



## System action based State Transition Model

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

所谓 Goal Update Model, 即在给定 System Action $a_m$ 时, 将临时状态中的 Constraints $C_t'$ 及 Requests $R_t'$ 更新至新状态 $C_{t+1}$ 和 $R_{t+1}$.

假设新的 Requests 与旧的 Constraints 无关, 即 $R_{t+1} \nleftrightarrow C_t'$, 则:

$$
\begin{aligned}
P(G_{t+1} | a_m, G_t^{'}) &= P(<C_{t+1}, R_{t+1}> | a_m, <C_t^{'}, R_t^{'}>) \\
    & \xlongequal{R_{t+1} \nleftrightarrow C_t'} P(R_{t+1} | a_m, R_t^{'}, C_{t+1}) * P(C_{t+1} | a_m, R_t^{'},  C_t^{'})
\end{aligned}
$$

同时假设 Requests 槽位之间相互独立, 并定义 Match 函数 $M(a_m, C)$, 则:

$$
P(R_{t+1} | a_m, R_t^{'}, C_{t+1}) = \prod_{s \in a_m} P( R_{t+1}[s] | s, R_t' [s], M(a_m, C_{t+1}) )
$$


$P(C_{t+1} | a_m, R_t^{'},  C_t^{'})$ 在给定 $a_m$ 下更新 $C_t'$ 主要有如下情形:

* 将某些 Constraints 槽位更新为一个新值 (如 `dontcare` 表示不再限制该槽位, 或 `type=wine` 表示将类型由 `beer` 调整为 `wine` 等)
* 维持不变


**Agenda Update Model**

Agenda 更新 $P(A_{t+1} | a_m, A_t', G_{t+1})$ 可以看作 一组入栈操作 和 一个 **clean-up** 过程. 

清除过程主要用于处理 重复的 Dialogue Act / NULL Act 及 非必要的 Requests Act (如所需信息已得到), 这些可以从 Agenda 中清除掉.


下面主要看入栈操作, 假设 $a_m$ 中的每个槽位入栈相互独立, 则

$$
P(A_{t+1} | a_m, A_t', G_{t+1}) = \prod_{s \in a_m} P(A_{t+1} [s] | s, G_{t+1})
$$


# ConvLab2 实现


在 ConvLab2 中, 在 MultiWOZ 数据集上的 Agenda-based User Simulator 实现在 [convlab2/policy/rule/multiwoz/policy_agenda_multiwoz.py](https://github.com/thu-coai/ConvLab-2/blob/master/convlab2/policy/rule/multiwoz/policy_agenda_multiwoz.py) 中.

具体地, 该文件定义了一个 `UserPolicyAgendaMultiWoz` 类, 该类主体架构如下:

```python
class UserPolicyAgendaMultiWoz(Policy):

    def __init__(self):
        # 一些限制条件
        self.max_turn = 40
        self.max_initiative = 4

        # 目标生成器, 用于初始化 Simulator
        self.goal_generator = GoalGenerator()

        self.__turn = 0

        # 主要成员 User Goal 和 User Agenda
        self.goal = None
        self.agenda = None

        Policy.__init__(self)

    def init_session(self, init_goal=None):
        """初始化 User Goal 和 User Agenda"""
        self.reset_turn()
        if not ini_goal:
            self.goal = Goal(self.goal_generator)
        else:
            self.goal = ini_goal
        self.domain_goals = self.goal.domain_goals
        self.agenda = Agenda(self.goal)

    def predict(self, sys_dialog_act):
        """给定 System Action, 生成 User Action
        
        该函数会完成 Agenda-based User Simulator 的三个过程:
        1. System Action based State Transition
        2. User Action Selection
        3. User Action based State Transition
        """
        ...
        sys_action = self._transform_sysact_in(sys_action)

        ...
        self.agenda.update(sys_action, self.goal)
        
        ...
        action = self.agenda.get_action(self.max_initiative)
        action = self._transform_usract_out(action)
```

Agenda-based Simulator 中最重要的两个元素是 User Goal 和 Agenda, 在 ConvLab2 中分别使用了 `Goal` 和 `Agenda` 两个类实现.

## Goal 实现

ConvLab2 中 Goal 类也在 `policy_agenda_multiwoz.py` 文件中实现.

```python
class Goal(object):
    def __init__(self, goal_generator: GoalGenerator):
        pass
    def set_user_goal(self, user_goal: dict):
        pass
    def task_complete(self) -> bool:
        pass
    def next_domain_incomplete(self):
        pass
```

Goal 接收一个 `GoalGenerator` 初始化参数. GoalGenerator 可以随机生成特定数据集的 User Goal, 示例如下:

```json
{
    "train":{
        "info":{
            "arriveBy":"12:15",
            "day":"tuesday",
            "departure":"cambridge",
            "destination":"peterborough"
        },
        "reqt":[
            "trainID"
        ]
    },
    "attraction":{
        "info":{
            "area":"east",
            "type":"museum"
        },
        "reqt":[
            "entrance fee",
            "phone"
        ]
    },
    "domain_ordering":[
        "train",
        "attraction"
    ]
}
```

> **说明**
> MultiWOZ 是一个 multi-domain 数据集, 因此, 生成的 User Goal 也包括多个 Domain, 并通过 `domain_ordering` 字段控制 Domain 的优先顺序.
> 
> 具体到每个 Domain, 其 User Goal 主要包括两项内容: `info` 与 `reqt`, 这对应 User Goal 中的 Constraints 和 Requests.


## Agenda

ConvLab2 中 Agenda 类也在 `policy_agenda_multiwoz.py` 文件中实现.

```python
class Agenda(object):

    def __init__(self, goal: Goal):
        ...
        self.__stack = []

        self.__push(self.CLOSE_ACT)        
        for idx in ran
        
        ge(len(goal.domains) - 1, -1, -1):
            ...
            self.__push(domain + '-inform', "none", "none")

        self.cur_domain = None

    def update(self, sys_action, goal: Goal):
        pass

    def get_action(self, initiative=1):
        pass
```

Agenda 的主要目的是 存储尚需进一步处理的 User Action 以及 生成新的 User Action.

存储 User Action 使用了 `self.__stack = []`. 鉴于 Stack 是后进先出的, 在初始化时, 首先压入了 `CLOSE_ACT` (即 `generate_bye`), 作为对话结束时的 User Action.
然后, 根据生成的 User Goal 中各个 Domain 的先后顺序, 将优先级靠后的 Domain 中的槽位构造为 `<{domain}-inform, {slot}, {value}>` Action 入栈. 这可以确保优先级较高的 User Action 先于优先级较低的 User Action 出栈.

Agenda 提供了 `update(self, sys_action, goal: Goal)` 方法, 用于在新的 System Action 到来之时更新 Agenda.

Agenda 的 `get_ation(self, initiative=1)` 方法用于生成新的 User Action, 以便 Simulator 可以与 System 持续交互.
