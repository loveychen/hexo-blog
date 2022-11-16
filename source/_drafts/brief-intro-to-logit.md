---
title: "logit、logic、logistic 傻傻分不清楚"
tags: 
    - Probability
    - Mathematics
    - logit
    - 概率论
categories:
    - Mathematics
    - Probability
---

在看 逻辑回归 (Logistic Regression, LR) 时, 对 "逻辑" 一词不甚理解, 该词的中文翻译也词不达意.

事实上, `logistic` 与 `logic` 及 `logit` 是同根词, 而 `logit` 是 LR 中的重要概念.

**TODO: 这里需要搜集和整理资料, 给一个让人深刻的引入和简介**



# Logistic 函数

此处主要参考 [Wikipedia: Logistic function](https://en.wikipedia.org/wiki/Logistic_function).

Logistic 函数是比利时数学家 [Pierre François Verhulst](https://en.wikipedia.org/wiki/Pierre_Fran%C3%A7ois_Verhulst)
在 1938 ~ 1947 的三篇论文中提出的, 其目的是通过调整指数增长模型, 对人口增长函数进行建模.

作者在 30 年代设计了该函数, 并与 1938 年发表了一个简短说明; 1944年进一步分析了该函数, 并给出正式命名 (对应论文则是 1845 年发表的).

> **Logistic 名字来源**
> Verhulst 在1945年发表的论文中将该曲线命名为 **logistic**, 但是并没有解释原因. 这大概是一个数学领域的不解之谜吧.
> 参考 [Quora: How did "logistic equation" get its name?](https://www.quora.com/How-did-a-%E2%80%9Clogistic-equation%E2%80%9D-get-its-name-Does-logistic-mean-logical-or-logistic-work)


Logistic 函数的数学表达式如下

$$
f(x) = \frac{L}{1 + e ^ {-k (x - x_0)}}
$$

其中, 
- $x_0$ 是 S 曲线的中点 (mid-point)
- $L$ 是曲线的最大值
- $k$ 为曲线的增长率 (或成为陡度)

当 $x_0 = 0, L = 1, k = 1$ 时, 称之为 **标准逻辑函数** (Standard Logistic Function), 其表达式及函数图像如下:

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

![标准逻辑回归函数](brief-intro-to-logit/Logistic-curve.svg.png)


参照 Logistic 函数的函数图像, 人口增长可以划分为如下阶段:
- 指数增长阶段: 在初始阶段, 人口呈现几何增长趋势
- 线性增长阶段: 随着逐渐饱和, 增长曲线放缓至线性增长
- 滞胀阶段: 在成熟阶段, 人口停止增长


## logistic 词源探秘: 一些有趣的小知识

### logistic / logistical

logistic, 也可以写作 logistical, 是一个形容词, 其含义是 "relating to the **careful** organization of a **complicated** activity", 中文解释为 "后勤的".

logistic 从词源上看,
> * "pertaining to logic," 1620s, 
> * from Medieval Latin **logisticus**, 
> * from Greek **logistikos** "skilled in calculating; endued with reason," from logistes "a calculator," 
> * from **logos** "calculation, proportion" (see Logos). 
> 
> Related: Logistical (1560s); logistically. 
> 
> **Logistics**, from this word, in the sense "art of arithmetical calculation" is from 1650s.


logistic 从构词上看, `log- + -istic`, 由词根 `log-` (表示 "说话", 引申为 "学说") 和 形容词后缀 `-istic` (表示 "...的").  `-istic` 是一个符合后缀, 由 `-ist + -ic` 构成.


### `log-` 词根

`log-` 词根表示 "说话, 推理, 计算", 源自希腊词汇 log (means "word"), 引申为 "学说" (常用词根 `-logy` 表示 (means "study of"), 如 biology, mythology).

源于该词根的单词示例如下:
* log: book of "words", 日记, 日志
* catalog: listing of "words", 目录
* dialogue: "words" between people, 对话
* monologue: "words" of one person, 独白
* prologue: "words" beforehand, 前言, 开场白, 引子, 序
* epilogue: after "words", 收场白, 后记, 跋
* logophile: "words" lover, 爱好词语的人
* logprrhea: "word" diarrhea, 多言症, 速语癖
* biology: "study" of life, 生物学
* zoology: "study" of animals, 动物学
* etymology: "study" of the origin of words, 词源学
* genealogy: "study" of one's family history, 家谱(学), 宗谱(学)


### logic

柯林斯词典对 logic 给出了三个解释:

*  **Logic** is a method of reasoning that involves **a series of statements**, each of which must be true if the statement before it is true. 逻辑 (学)
*  The **logic** of a conclusion or an argument is **its quality of being correct and reasonable**. (结论或观点的) 逻辑
   *  eg. I don't follow the logic of your argument.
*  A particular kind of **logic** is the way of thinking and reasoning about things that is characteristic of a particular type of person or particular field of activity. (某种人或某行为领域的) 逻辑
   *  eg. The plan was based on sound commercial logic.  (商业逻辑)


logic 虽然与 logistic 属于同根词, 但是其相对于词根 `log-` 的本意已经相去甚远了.

logic 从词源上看,
> * mid-14c., **logike**, "branch of philosophy that treats of forms of thinking; the science of distinction of true from false reasoning," 
> * from Old French **logique** (13c.), 
> * from Latin **(ars)logica** "logic," 
> * from Greek **(he)logike (techne)**  "(the) reasoning (art)," 
> * from fem. of **logikos** "pertaining to speaking or reasoning" (also "of or pertaining to speech"), 
> * from **logos** "reason, idea, word" (see Logos). 
> * Formerly also **logick**. 
> 
> Sometimes formerly plural, as in ethics, but this is not usual. 
> * Meaning "logical argumentation" is from c. 1600. Contemptuous logic-chopper "sophist, person who uses subtle distinctions in argument" is from 1846.


### logit

logit 是 Joseph Berkson 在 1944 年创造的一个词, 它是由 logistic + unit 构成的一个.

作者的原话如下:
>  “I use this term **logit** for $\displaystyle \ln p/q$ following Bliss, who called the analogous function which is linear on $\displaystyle x$ for the normal curve ‘probit.’”


logit 是类比 probit 创造出的一个词语. 如果非要给 logit 一个定义
* (mathematics) the logit function is the inverse of the sigmoid (logistic) function. 
$$
\text{logit} (p) = \log (\frac{p}{1-p})
$$


在统计学上, $\frac{p}{1-p}$ 也有专门的术语表示, 称之为 **odds**, 一般翻译为 "发生比, 胜算, 几率", 所以 logit 有时也翻译成 **对数比** 或 **对数几率**.


> **probit**
> probit 是 Bliss 在 1934 年的论文 "The method if probits" 中创造的词语, 它有两个含义:
> * (statistics) A unit, derived from a standard distribution, used in measuring the response to doses
> * The probit function is the inverse of the cumulative distribution function


### Logistics, 物流 / 后勤

logistics 是形容词 logistic 的复数名词形式, 其英文含义是 "the careful organization of a complicated activity so that it happens in a successful and effective way", 中文翻译为 "后勤, 物流; 物流学, 后勤学". 中文翻译来自日本语.

logistics 从词源上看
> * "art of moving, quartering, and supplying troops," 1846, 
> * from French **(l'art)logistique** "(art) of quartering troops,", which apparently(据说) is 
>   - from **logis** "lodging" 
>       - from Old French **logeiz** "shelter for an army, encampment," from loge
>   - Greek-derived suffix **-istique** (see -istic). 
> * The form in French was influenced by **logistique**, from the Latin source of English logistic. 


以下内容来自 [英文维基百科: Logistics](https://en.wikipedia.org/wiki/Logistics) 和 [中文维基百科: 物流](https://zh.wikipedia.org/wiki/%E7%89%A9%E6%B5%81).


**物流（英语：Logistics）**，是军事领域后勤概念的民间用语。

在西方该词语源于希腊语：λογιστικός, Logistikos，意为“计数科学”或“精于算计”。

“物流”是一套通过计算、策划来控制原材料、制成品、产成品或信息在供、需、仓储不同部之间转运的管理系统。“物流”或也可详称为其最终目的之“策略性物流运输”或“策运”。物质资料从供给者到需求者的物理运动，是创造时间价值、场所价值和一定的加工价值的活动。物流是指物质实体从供应者向需求者的物理移动，它由一系列创造时间价值和空间价值的经济活动组成，包括运输、保管、配送、包装、装卸、流通加工及处理等多项基本活动，是这些活动的统一。

相关概念最早出现于军事行政组织，在中国古代一直被称为辎重，后来在近代被逐渐改为后勤。

现代的“物流”概念最早可能是以在二战中，围绕战争物资供应，美军创建的后勤理论为原型的。当时的「后勤」是指将战时物资生产、采购、运输、配给等活动作为一个整体进行统一布置，以求战略物资补给的费用更低、速度更快、服务更好。后来，将“后勤”体系移植到现代经济生活中，才逐步演变为今天的物流。物流系统也可像互联网般，促进全球化。在贸易上，若要更进一步与世界连系，就得靠良好的物流管理系统。

市场上的商品很多是「游历」各国后才来到的。原料可能来自马来西亚和泰国，加工可能在新加坡，生产却在中国，最后才入口到美国。产品的「游历」路线就是由物流师计划、组织、指挥、协调、控制和监督，使各项物流活动实现最佳的协调与配合，以实现产品物流的目标，目标可能是：降低物流成本（cost），提高物流效率及质量（efficiency & quality），或提高物流的供应满足性（availability）。目标可能会有取舍和侧重。


### "逻辑" 词源

中文词 "逻辑" 是一个外来词, 由中国清末翻译家严复先生所创, 它是对英文词 **logic** 的音译. 

"逻辑"一词后由中国传入日本, ロジック；论理（ろんり）, 但在日语中则注明只是对Logic的注音，Logic在日语中的正式汉语翻译词为“论理”。

logic 还有一个很好的意译："理则"，这是由孙中山先生所译。孙中山先生[《建国方略.以作文为证》](http://www.sunyat-sen.org/index.php?m=content&c=index&a=show&catid=46&id=6662)：“然则逻辑究为何物？当译以何名而后妥……吾以为当译之为‘理则’者也。”孙中山《建国方略·以作文为证》：“学者之对于理则之学，则大都如陶渊明之读书，不求甚解而已。”

"逻辑" 有四种常见含义:
1. 指客观事物的规律。例如：“历史的逻辑决定了人类社会讲一直向前发展”
2. 指某种特殊的理论、观点或看问题的方法。例如：“侵略者奉行的是强盗逻辑”
3. 指思维的规律、规则。例如：“写文章要讲逻辑”
4. 指逻辑学这门科学。例如：“大学生要学点逻辑”


# Logistic 分布

当我们限定 $L = 1$ 时, Logistic 函数是一条值域为 `(0, 1)` 的单调递增 S 曲线. 它实际上是 Logistic 分布的 累积分布函数 (Cumulative Distribution Function, CDF). 

Logistic 分布的 累积分布函数 及 概率密度函数 的数学表达式 和 函数图像吐下:

$$
\begin{aligned}
F(x; \mu, s) &= \frac{1}{1 + e^{-(x - \mu) / s}} \\
f(x; 0, 1) &= \frac{e^{-x}}{(1 + e^{-x})^2} \\
           &= F(x; 0, 1) [1 - F(x; 0, 1)]
\end{aligned}
$$

![累积分布函数](brief-intro-to-logit/Logistic_cdf.svg.png)

![概率密度函数](brief-intro-to-logit/Logistic_pdf.svg.png)

Logistic 分布的概率密度函数图像与正态分布相似, 但长尾分布更明显.
