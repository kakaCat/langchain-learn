---
title: "LangChain 入门教程：Basic Reflection 代理实现详解"
description: "从理念到实战全面解析 Basic Reflection（先给初稿再反思修订）的代理模式，结合 LangChain/LangGraph 实现与运行指南，掌握低开销的自我反思增强技巧。"
keywords:
  - LangChain
  - Reflection
  - Agent
  - LangGraph
  - Prompt Engineering
  - Tool Use
  - Self-critique
  - Planning
  - Workflow
tags:
  - Tutorial
  - Agent
  - Reflection
  - LLM
author: "langchain-learn"
date: "2025-10-29"
lang: "zh-CN"
canonical: "/blog/langchain-basic-reflection-agent-tutorial"
audience: "中级 / 具备Python基础和LLM应用经验的AI工程师"
difficulty: "intermediate"
estimated_read_time: "10-15min"
topics:
  - LangChain Agents
  - Reflection Pattern
  - LangGraph Workflow
  - Prompt Engineering
entities:
  - LangChain
  - LangGraph
  - Basic Reflection
  - Agent
qa_intents:
  - "什么是 Basic Reflection？为什么实用？"
  - "如何用 LangChain/LangGraph 实现反思闭环？"
  - "提示词如何设计以得到更好修订？"
  - "如何配置与切换模型（OpenAI/DeepSeek/Ollama）？"
  - "工作流图生成失败如何处理？"
  - "反思何时停止？如何避免过度迭代？"
---

# LangChain 入门教程：Basic Reflection Agent 实现详解

## 本页快捷跳转

* 目录：
  * [引言：为什么要 Basic Reflection？](#intro)
  * [理念与模式：反思—修订的闭环](#concept)
  * [代码示例：LangChain/LangGraph 实现](#code-example)
  * [运行指南](#run-guide)
  * [模型兼容性与切换](#model-compat)
  * [缺点与局限性](#limitations)
  * [FAQ 常见问题](#faq)
  * [总结](#summary)

***

<a id="tldr" data-alt="TLDR 要点 速览 总结"></a>

### 本页摘要

- 模式定义：先生成初稿答案，再由“教师”角色进行批改与建议，随后根据建议产出修订版，提升质量与一致性。
- 低复杂度增益：无需复杂工具编排，仅通过两节点闭环即可显著提升输出质量，适合写作、摘要、解释类任务。
- 三要素设计：Generation（初稿）→ Reflection（批改建议）→ 再次 Generation（修订版）；通过条件边控制停止时机。
- 环境与依赖：统一 `pip install -r requirements.txt`；`.env` 包含 `OPENAI_*`（可对接 DeepSeek 或 OpenAI）。
- 模型切换：云端（OpenAI/DeepSeek）或本地（Ollama），按需替换 `ChatOpenAI`。
- 输出产物：终端日志与 `blog/plan.png` 工作流图，最后打印“最终结果”。

<a id="intro" data-alt="introduction 引言 概述 reflection"></a>

## 引言

在复杂任务中，往往难以一次到位，通常需要自查与他评来提升质量。Basic Reflection 以最小闭环实现自我审校，尤其适用于写作与解释类任务。

写作场景示例：

- 生成初稿。
- 以“教师”视角批改，给出具体建议（长度、结构、风格、细节）。
- 按建议修订，产出更优版本。

核心特征：

- 自洽性：先批改后修订，确保输出与目标、约束一致。
- 迭代性：每轮反思驱动改进，直至达到质量门槛。

### Agent 的实现步骤

**Basic Reflection（先初稿后反思再修订）** 通过两个节点和一条条件边构成闭环：

1.  初稿生成（Generate）：根据用户需求输出最佳初稿。
2.  批改反思（Reflect）：针对初稿给出具体改进建议（长度、深度、风格、结构）。
3.  再次生成（Generate）：依据建议产出修订版。满足质量条件则停止，否则继续迭代。

该模式的价值：

- 质量提升：通过明确标准与建议进行二次修订。
- 可解释性：建议与修订形成“理由—修改”的链条，便于审阅与质控。
- 实施简洁：无需工具与复杂状态，仅用消息流即可完成闭环。

***

## 论文与理念：反思\思考的理论基础 [Reflexion论文](https://arxiv.org/abs/2303.11366)

Plan-and-Solve 属于“先规划后执行”的通用工程范式，灵感来源于经典 AI 的规划理论与现代 LLM 的链式推理：

*   先规划能提升全局最优性与步骤完整性，避免漏项与循环依赖。
*   执行中的观察反馈支撑动态调整，使得代理在复杂环境中更稳健。
*   与 ReAct 的差异：ReAct更强调“思考-行动-观察”的交替；Plan-and-Solve先拿到全局计划再执行，二者可互补。

该范式在检索问答、工具组合、任务编排等场景表现出良好可解释性与稳定性，利于调试和审计。

***
<a id="concept" data-alt="理念 模式 反思 修订"></a>

## 理念与模式：反思—修订的闭环

- 角色分工：生成节点扮演“写作者”，反思节点扮演“教师”；二者使用不同的 system 提示以明确目标与标准。
- 闭环逻辑：Generate → Reflect → Generate；通过条件边 `should_continue` 决定是否停止迭代。
- 停止条件：示例中采用“消息条数阈值”（如超过 3 次往返）简化控制；生产环境可替换为“评分门槛”“关键词校验”“冗余检测”等。



<a id="code-example" data-alt="代码示例 实现 langchain langgraph"></a>

## 代码示例：LangChain/LangGraph 实现

示例文件路径：`10-agent-examples/05_basic_reflection_demo.py`

架构如图所示：

![plan.png](./plan.png)

核心结构（节选）：

```python
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START, add_messages
from typing_extensions import TypedDict

class BASIC_REFLECTION(TypedDict):
    messages: Annotated[list, add_messages]

def generate():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an essay assistant tasked with writing excellent 5-paragraph essays..."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    return prompt | get_llm()

def reflect():
    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a teacher grading an essay submission. Generate critique and recommendations..."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    return reflection_prompt | get_llm()

def generation_node(state: BASIC_REFLECTION) -> BASIC_REFLECTION:
    msg = generate().invoke({"messages": state["messages"]})
    state["messages"] = [msg]
    return state

def reflection_node(state: BASIC_REFLECTION) -> BASIC_REFLECTION:
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    translated = [state["messages"][0]] + [cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]]
    res = reflect().invoke({"messages": translated})
    state["messages"] = [HumanMessage(content=res.content)]
    return state

def should_continue(state: BASIC_REFLECTION):
    return END if len(state.get("messages", [])) > 3 else "reflect"

def create_workflow():
    workflow = StateGraph(BASIC_REFLECTION)
    workflow.add_node("generate", generation_node)
    workflow.add_node("reflect", reflection_node)
    workflow.add_edge(START, "generate")
    workflow.add_conditional_edges("generate", should_continue)
    workflow.add_edge("reflect", "generate")
    return workflow.compile()
```

实现要点：

- 不同角色提示：生成节点与反思节点采用不同的 system 指令，分别聚焦“产出”和“批改”。
- 消息翻转技巧：在反思节点中，将历史 `AI/Human` 角色映射翻转，以便教师角色从“用户视角”进行评估与建议。
- 停止条件：示例采用“消息数量阈值”作演示，生产中建议使用更稳健的质量度量（评分/关键词/格式校验）。
- 可视化：通过 `app.get_graph().draw_mermaid_png(...)` 生成 `blog/plan.png` 工作流图。

***

<a id="run-guide" data-alt="运行 安装 指南 依赖 env"></a>

## 运行指南

- 安装依赖：`pip install -r requirements.txt`
- 配置环境：在项目根或脚本同目录创建 `.env`，设置 `OPENAI_*`（或 DeepSeek）参数（注意不要在键值前后留空格）。
- 运行示例：`python 10-agent-examples/05_basic_reflection_demo.py`
- 预期输出：终端打印生成与反思过程日志，并在 `blog/plan.png` 生成工作流图，最后输出“最终结果”。
- 图形依赖与排错：若工作流图生成失败，请按系统提示安装图形渲染依赖（如 Graphviz/Mermaid 渲染链），或暂时仅查看终端日志与最终结果。

示例 `.env`（DeepSeek 对接）：

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
OPENAI_TEMPERATURE=0
OPENAI_MAX_TOKENS=512
```

示例 `.env`（OpenAI 官方）：

```text
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0
OPENAI_MAX_TOKENS=512
```

***

<a id="model-compat" data-alt="模型 兼容 切换 openai deepseek ollama"></a>

## 模型兼容性与切换

- 云端提供商：
  - 使用 `ChatOpenAI`，将 `OPENAI_BASE_URL` 指向对应服务地址，`OPENAI_MODEL` 为该服务支持的模型名。
- 本地 Ollama：
  - 替换为 `ChatOllama` 使用本地模型：
  ```python
  from langchain_ollama import ChatOllama
  def get_llm():
      return ChatOllama(model="qwen2.5:7b", temperature=0)
  ```
- 使用建议：
  - 写作/结构化输出建议用低温度（`temperature=0`），保证一致性与可预测性。

***

<a id="limitations" data-alt="缺点 局限 适用场景"></a>

## 缺点与局限性

- 建议质量依赖模型能力：反思节点的“教师”水平决定修订成效，弱模型可能提出泛化建议。
- 停止条件需调优：简单的计数阈值可能过早/过晚停止，建议结合评分或目标达成判定。
- 成本与时延：相比一次生成，多一步反思与再生成会增加调用次数与延迟。
- 无工具模式限制：该示例不使用外部工具，不适合需要检索或操作性强的任务（可与 ReAct/Plan-and-Solve 组合）。

***

<a id="faq" data-alt="FAQ 常见问题"></a>

## FAQ 常见问题

- 反思节点的提示词如何写得更有效？
  - 明确评估维度：长度、结构、风格、事实一致性、读者对象；要求“具体且可操作”的建议。
- 为什么在反思节点要翻转 `AI/Human` 角色？
  - 让教师从用户视角评估上一轮输出，避免上下文混淆，使建议更贴近用户目标。
- 何时停止迭代？
  - 示例用“消息条数阈值”。实际可用评分门槛、关键词覆盖率、格式校验、可读性评分等更稳健策略。
- 工作流图生成失败怎么办？
  - 安装 Graphviz 或开启 Mermaid 渲染链；或跳过图，仅查看日志与最终结果。
- 能否与其他模式组合？
  - 可以。写作任务用 Reflection 提升质量，检索/工具调用用 ReAct；复杂任务先用 Plan-and-Solve 定全局，再在关键环节用 Reflection 做质控。

***

<a id="summary" data-alt="总结 结语"></a>

## 总结

- Basic Reflection 用最小闭环实现“先产出后批改再修订”，对写作与解释任务尤为有效。
- 示例以两个节点（Generate/Reflect）与条件边构建闭环，易于扩展到评分停止或多轮细化。
- 生产使用建议：
  - 强化反思提示词，明确评估维度；
  - 使用低温度与结构化要求；
  - 将停止条件换成评分门槛或质量判定；
  - 与 ReAct/Plan-and-Solve 组合应对更复杂的检索与多步任务。

对应代码：`10-agent-examples/05_basic_reflection_demo.py`