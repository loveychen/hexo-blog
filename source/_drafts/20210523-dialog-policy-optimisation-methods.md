---
title: 对话策略优化算法学习笔记
date: 2021-05-24 16:20:38
tags:
    - 对话策略
    - 对话系统
    - POMDP
    - Phd Thesis
categories:
    - 对话系统
---

该笔记主要参考剑桥大学 2018 届博士毕业生 [蘇培豪](https://eddy0613.github.io/) 的博士毕业论文 [Reinforcement Learning and Reward Estimation for Dialogue Policy Optimisation](https://www.repository.cam.ac.uk/handle/1810/275649) 第 3.1 节 "Reinforcement Learning for Dialogue Policy Optimisation". 

蘇培豪的博士导师是鼎鼎大名的 CUED (Cambridge University Engineering Departmen) 的 [Steve Young](http://mi.eng.cam.ac.uk/~sjy/index.html). Steve Young 是统计对话系统 (SDS, Statistical Dialogue Systems) 的先驱.



# 对话系统概述

人机对话可以看做是一个决策任务. 在每一轮对话中, SDS 首先通过 ASR 和 SLU 模块分析并理解用户表述; 然后使用 DST 模块编码对话历史, 称为对话状态; 基于对话状态, 对话策略模块选择一个合适的 Action; Action 会被 NLG 和 TTS 模块转换成语音信号, 传递给用户.


# 对话策略

对话策略模块是 SDS 的核心模块.

对话策略
