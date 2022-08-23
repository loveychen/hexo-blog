---
title: 关于 Python 你需要知道 - packaging
date: 2022-08-23 18:15:02
tags:
    - Python
categories:
    - Python
---


# Python 包组织结构

package 是一种常见的代码组织模块, 在很多语言中都有 package 的概念, 比如 [Python packages](https://docs.python.org/3/tutorial/modules.html#packages) / [Java pacakges](https://docs.oracle.com/javase/specs/jls/se18/html/jls-7.html).


在 Python 中, 一个 package 由一个或多个 sub-package 和 [module](https://docs.python.org/3/tutorial/modules.html) 组成. 
对 module 最通俗的理解就是一个 `.py` 文件. module 内部可以定义 变量 / 函数 (在 Python 里也是一种变量), 也可以执行其它 Python 语句. 

module 是 Python 代码复用的最小物理单元, 可以通过 Python 的 import 机制添加对其它模块的使用.
