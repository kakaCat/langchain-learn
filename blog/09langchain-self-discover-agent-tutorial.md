---
title: "LangChain Self-Discover 自我发现代理模式详解"
description: "深入解析 Self-Discover 自我发现代理模式，通过选择、适配、结构化和推理四步法提升复杂问题解决能力"
keywords: ["LangChain", "Self-Discover", "自我发现", "代理模式", "AI 代理", "LangGraph", "复杂问题解决"]
tags: ["langchain", "agents", "self-discover", "ai", "llm"]
author: "AI Assistant"
date: "2024-12-19"
audience: ["AI 开发者", "机器学习工程师", "LangChain 用户"]
difficulty: "中级"
read_time: "15 分钟"
topics: ["Self-Discover", "代理模式", "问题解决", "LangGraph"]
entities: ["LangChain", "LangGraph", "OpenAI", "DeepSeek", "Ollama"]
qa_intent: ["Self-Discover 是什么", "如何实现自我发现代理", "Self-Discover 与 Plan-and-Solve 的区别", "自我发现四步法"]
---

# LangChain Self-Discover 自我发现代理模式详解

## 引言

Self-Discover（自我发现）是一种先进的代理模式，它通过系统性的推理过程来提升复杂问题解决能力。与传统的 Plan-and-Solve 模式不同，Self-Discover 强调模型自身的认知能力，通过选择、适配、结构化和推理四个关键步骤，让模型能够"思考如何思考"，从而更有效地解决复杂问题。

Self-Discover 的核心思想是：让模型首先识别问题类型，然后选择合适的推理框架，接着将问题适配到该框架中，最后执行推理过程。这种"元认知"方法显著提升了模型在复杂任务中的表现。

## 理论基础

### Self-Discover 四步法

Self-Discover 模式包含四个核心步骤：

1. **选择（Select）**：从预定义的推理模块中选择最相关的一个或多个模块
2. **适配（Adapt）**：将选定的推理模块适配到当前具体问题
3. **结构化（Structure）**：基于适配后的模块构建推理链
4. **推理（Reason）**：执行构建的推理链来解决问题

### 与 Plan-and-Solve 的对比

| 特性 | Plan-and-Solve | Self-Discover |
|------|----------------|---------------|
| 核心思想 | 先规划再执行 | 元认知推理 |
| 步骤数量 | 2-3 步 | 4 步 |
| 灵活性 | 中等 | 高 |
| 适用场景 | 多步骤任务 | 复杂推理任务 |
| 认知层次 | 任务级 | 元认知级 |

## 代码实现

### 环境准备

首先创建 `requirements.txt` 文件：

```txt
langchain-core>=0.2.0
langchain-openai>=0.1.0
langgraph>=0.1.0
python-dotenv>=1.0.0
pydantic>=2.0.0
typing-extensions>=4.0.0
```

创建 `.env` 配置文件：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
OPENAI_TEMPERATURE=0
OPENAI_MAX_TOKENS=2048
```

### 核心代码实现

以下是 Self-Discover 代理的完整实现：

```python
import os
from typing import List, Tuple, Annotated, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# 加载环境变量
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

class SelfDiscoverState(TypedDict):
    """Self-Discover 状态定义"""
    input: str
    selected_modules: List[str]
    adapted_modules: Dict[str, str]
    reasoning_structure: List[str]
    reasoning_steps: Annotated[List[Tuple], operator.add]
    final_answer: str

def get_llm() -> ChatOpenAI:
    """创建并返回语言模型实例"""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    base_url = os.getenv("OPENAI_BASE_URL")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2048"))

    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 120,
        "max_retries": 3,
        "request_timeout": 120,
        "verbose": True,
        "base_url": base_url
    }

    return ChatOpenAI(**kwargs)

class ReasoningModule(BaseModel):
    """推理模块定义"""
    name: str = Field(description="模块名称")
    description: str = Field(description="模块描述")
    application: str = Field(description="适用场景")

class ModuleSelection(BaseModel):
    """模块选择结果"""
    selected_modules: List[str] = Field(description="选中的模块名称列表")

class ModuleAdaptation(BaseModel):
    """模块适配结果"""
    adapted_modules: Dict[str, str] = Field(description="适配后的模块内容")

class ReasoningStructure(BaseModel):
    """推理结构"""
    reasoning_steps: List[str] = Field(description="推理步骤列表")

class FinalAnswer(BaseModel):
    """最终答案"""
    answer: str = Field(description="最终答案")

def get_module_selection_agent():
    """模块选择代理"""
    from langchain_core.prompts import ChatPromptTemplate
    
    selection_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的推理模块选择器。请根据用户的问题，从以下推理模块中选择最相关的一个或多个模块：

可用推理模块：
1. 分解法 - 将复杂问题分解为更小的子问题
2. 类比法 - 寻找类似问题的解决方案
3. 假设检验法 - 提出并验证假设
4. 逆向推理法 - 从目标反推步骤
5. 模式识别法 - 识别问题中的模式和规律
6. 约束满足法 - 识别并满足约束条件
7. 多视角分析法 - 从不同角度分析问题
8. 渐进逼近法 - 逐步逼近最终答案

请返回一个 JSON 对象，包含字段 `selected_modules`，值为选中的模块名称列表。
只返回 JSON，不要包含其他文本。"""),
        ("user", "问题：{input}")
    ])
    
    llm = get_llm()
    return selection_prompt | llm.with_structured_output(ModuleSelection, method="function_calling")

def get_module_adaptation_agent():
    """模块适配代理"""
    from langchain_core.prompts import ChatPromptTemplate
    
    adaptation_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个推理模块适配专家。请将选定的推理模块适配到具体问题中。

对于每个选中的模块，请提供：
- 如何将该模块应用到当前问题
- 具体的应用步骤和方法

返回一个 JSON 对象，包含字段 `adapted_modules`，值为字典，键为模块名称，值为适配后的具体应用说明。
只返回 JSON，不要包含其他文本。"""),
        ("user", "问题：{input}\n选中的模块：{selected_modules}")
    ])
    
    llm = get_llm()
    return adaptation_prompt | llm.with_structured_output(ModuleAdaptation, method="function_calling")

def get_structure_agent():
    """推理结构构建代理"""
    from langchain_core.prompts import ChatPromptTemplate
    
    structure_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个推理结构设计师。请基于适配后的推理模块，构建一个清晰的推理步骤链。

请返回一个 JSON 对象，包含字段 `reasoning_steps`，值为推理步骤的字符串列表。
每个步骤应该清晰、具体、可执行。
只返回 JSON，不要包含其他文本。"""),
        ("user", "问题：{input}\n适配后的模块：{adapted_modules}")
    ])
    
    llm = get_llm()
    return structure_prompt | llm.with_structured_output(ReasoningStructure, method="function_calling")

def get_reasoning_agent():
    """推理执行代理"""
    from langchain_core.prompts import ChatPromptTemplate
    
    reasoning_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个推理专家。请按照给定的推理步骤链来解决问题。

请仔细执行每个推理步骤，确保推理过程逻辑严密、步骤清晰。
最后给出问题的最终答案。

返回一个 JSON 对象，包含字段 `answer`，值为最终答案。
只返回 JSON，不要包含其他文本。"""),
        ("user", "问题：{input}\n推理步骤：{reasoning_steps}")
    ])
    
    llm = get_llm()
    return reasoning_prompt | llm.with_structured_output(FinalAnswer, method="function_calling")

def select_step(state: SelfDiscoverState):
    """选择步骤：选择合适的推理模块"""
    selector = get_module_selection_agent()
    result = selector.invoke({"input": state["input"]})
    print(f"选择步骤 - 选中的模块: {result.selected_modules}")
    state["selected_modules"] = result.selected_modules
    return state

def adapt_step(state: SelfDiscoverState):
    """适配步骤：将模块适配到具体问题"""
    adapter = get_module_adaptation_agent()
    result = adapter.invoke({
        "input": state["input"],
        "selected_modules": state["selected_modules"]
    })
    print(f"适配步骤 - 适配后的模块: {result.adapted_modules}")
    state["adapted_modules"] = result.adapted_modules
    return state

def structure_step(state: SelfDiscoverState):
    """结构化步骤：构建推理步骤链"""
    structurer = get_structure_agent()
    result = structurer.invoke({
        "input": state["input"],
        "adapted_modules": state["adapted_modules"]
    })
    print(f"结构化步骤 - 推理步骤: {result.reasoning_steps}")
    state["reasoning_structure"] = result.reasoning_steps
    return state

def reason_step(state: SelfDiscoverState):
    """推理步骤：执行推理并生成最终答案"""
    reasoner = get_reasoning_agent()
    result = reasoner.invoke({
        "input": state["input"],
        "reasoning_steps": state["reasoning_structure"]
    })
    print(f"推理步骤 - 最终答案: {result.answer}")
    state["final_answer"] = result.answer
    
    # 记录推理步骤
    reasoning_steps = state.get("reasoning_steps") or []
    for i, step in enumerate(state["reasoning_structure"]):
        reasoning_steps.append((f"step_{i+1}", step))
    reasoning_steps.append(("final_answer", result.answer))
    state["reasoning_steps"] = reasoning_steps
    
    return state

def create_self_discover_workflow():
    """构建 Self-Discover 工作流"""
    workflow = StateGraph(SelfDiscoverState)
    
    # 添加节点
    workflow.add_node("select", select_step)
    workflow.add_node("adapt", adapt_step)
    workflow.add_node("structure", structure_step)
    workflow.add_node("reason", reason_step)
    
    # 构建工作流
    workflow.add_edge(START, "select")
    workflow.add_edge("select", "adapt")
    workflow.add_edge("adapt", "structure")
    workflow.add_edge("structure", "reason")
    workflow.add_edge("reason", END)
    
    app = workflow.compile()
    
    # 生成工作流图
    import os
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "blog" / "self_discover_workflow.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    app.get_graph().draw_mermaid_png(output_file_path=str(output_path))
    
    return app

def main() -> None:
    """运行 Self-Discover 工作流"""
    app = create_self_discover_workflow()
    
    # 测试用例
    test_cases = [
        "如何设计一个可持续的城市交通系统？",
        "解释量子计算的基本原理及其潜在应用",
        "分析气候变化对全球经济的影响"
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n=== 测试用例 {i+1}: {test_case} ===")
        initial_state = {"input": test_case}
        final_state = app.invoke(initial_state)
        print(f"最终答案: {final_state.get('final_answer')}")

if __name__ == "__main__":
    main()
```

<a id="run-guide" data-alt="运行 安装 指南 依赖 env"></a>

## 运行指南

*   安装依赖：`pip install -r requirements.txt`
*   配置环境：在项目根或脚本同目录创建 `.env`，正确设置 `OPENAI_*` 环境变量
*   运行示例：`python 10-agent-examples/08_self_discover_demo.py`
*   预期输出：终端打印四个步骤的执行过程，包括模块选择、适配、结构化和推理结果
*   工作流图：在 `blog/self_discover_workflow.png` 生成工作流图

<a id="model-compat" data-alt="模型 兼容 切换 openai deepseek ollama"></a>

## 模型兼容性与切换

Self-Discover 模式支持多种语言模型：

*   **OpenAI 示例**（官方接口）：
    ```env
    OPENAI_API_KEY=your_openai_api_key
    OPENAI_BASE_URL=https://api.openai.com
    OPENAI_MODEL=gpt-4
    OPENAI_TEMPERATURE=0
    OPENAI_MAX_TOKENS=2048
    ```

*   **DeepSeek 示例**：
    ```env
    OPENAI_API_KEY=your_api_key_here
    OPENAI_BASE_URL=https://api.deepseek.com
    OPENAI_MODEL=deepseek-chat
    OPENAI_TEMPERATURE=0
    OPENAI_MAX_TOKENS=2048
    ```

*   **Ollama 示例**（本地部署）：
    ```python
    from langchain_ollama import ChatOllama
    def get_llm():
        return ChatOllama(model="qwen2.5:7b", temperature=0)
    ```

**使用建议**：
- 对于复杂推理任务，建议使用 GPT-4 或 DeepSeek 等更强大的模型
- 确保 `OPENAI_MAX_TOKENS` 设置足够大以容纳完整的推理过程
- 本地模型可能需要调整提示词以适应模型能力

<a id="limitations" data-alt="缺点 局限 适用场景"></a>

## 缺点与局限性

*   **模型依赖性强**：推理质量高度依赖底层语言模型的能力
*   **计算成本较高**：四步法需要多次模型调用，增加了计算开销
*   **模块选择偏差**：如果初始模块选择不当，可能影响后续推理效果
*   **结构化约束**：严格的四步流程可能不适合所有类型的问题
*   **提示词敏感性**：每个步骤的提示词设计对结果影响显著

**适用场景**：
- 复杂的推理和分析问题
- 需要多角度思考的决策任务
- 学术研究和深度分析
- 创意性问题的解决方案

**不适用场景**：
- 简单的信息查询
- 实时性要求高的任务
- 计算资源受限的环境

<a id="faq" data-alt="FAQ 常见 问题 排错"></a>

## FAQ 常见问题

### Self-Discover 与 Plan-and-Solve 的主要区别是什么？

Self-Discover 强调"元认知"，让模型思考如何思考；Plan-and-Solve 更注重任务执行流程。Self-Discover 通过模块选择和适配实现更灵活的推理，而 Plan-and-Solve 更结构化地执行预定计划。

### 如何选择合适的推理模块？

模块选择基于问题类型：
- 复杂问题：分解法 + 多视角分析法
- 创新问题：类比法 + 假设检验法
- 逻辑问题：逆向推理法 + 约束满足法
- 模式识别问题：模式识别法 + 渐进逼近法

### 为什么需要四个步骤而不是直接推理？

四步法通过系统性的"元认知"过程：
1. 识别问题本质（选择）
2. 定制解决方案（适配）
3. 构建执行路径（结构化）
4. 执行推理（推理）

这种方法显著提升了复杂问题的解决质量。

### 如何处理模块选择错误的情况？

可以通过以下方式改进：
1. 在提示词中提供更详细的模块描述
2. 增加模块选择的置信度评估
3. 实现多轮选择机制
4. 添加人工干预选项

### 如何优化 Self-Discover 的性能？

*   缓存模块选择结果
*   并行执行部分步骤
*   使用更高效的模型
*   优化提示词设计
*   实现增量推理

### 常见错误如何排查？

*   **模块选择失败**：检查提示词设计，确保模块描述清晰
*   **适配效果差**：验证问题与模块的匹配度
*   **推理链断裂**：确保推理步骤的逻辑连贯性
*   **答案质量低**：可能需要调整模型或增加推理深度

### 如何扩展自定义推理模块？

在 `get_module_selection_agent()` 中添加新的模块定义：
```python
# 在系统提示中添加新模块
"""
9. 成本效益分析法 - 分析方案的投入产出比
10. 风险评估法 - 识别和评估潜在风险
11. 时间序列法 - 基于时间维度的分析
"""
```

**成功检查清单**

*   终端输出显示四个步骤的完整执行
*   模块选择合理且符合问题类型
*   适配后的模块能够有效指导推理
*   推理步骤清晰且逻辑连贯
*   最终答案准确且有深度
*   工作流图成功生成

<a id="links"></a>

## 官方链接的内容

*   LangChain 文档：<https://python.langchain.com/docs>
*   LangChain Agents 指南：<https://python.langchain.com/docs/use_cases/agents>
*   LangChain 结构化输出：<https://python.langchain.com/docs/guides/structured_output>
*   LangGraph 文档主页：<https://langchain-ai.github.io/langgraph/>
*   LangGraph 状态图概念：<https://langchain-ai.github.io/langgraph/concepts/>
*   Self-Discover 论文（arXiv）：<https://arxiv.org/abs/2402.03620>
*   OpenAI API 文档：<https://platform.openai.com/docs/>
*   DeepSeek API 文档：<https://api-docs.deepseek.com/>
*   Ollama 本地模型文档：<https://ollama.com/docs>
*   Pydantic 文档：<https://docs.pydantic.dev/latest/>

<a id="summary" data-alt="总结 收尾 建议"></a>

## 总结

Self-Discover 自我发现代理模式通过系统性的四步法——选择、适配、结构化和推理，实现了更高级的"元认知"推理能力。这种模式特别适合处理复杂的、需要深度思考的问题。

**核心优势**：
- 提升模型在复杂任务中的表现
- 实现更灵活的推理过程
- 支持多角度问题分析
- 增强解决方案的创造性

**实践建议**：
1. 从简单问题开始，逐步增加复杂度
2. 精心设计每个步骤的提示词
3. 根据问题类型选择合适的推理模块
4. 监控和评估推理过程的质量
5. 持续优化模块库和适配策略

通过 LangChain 和 LangGraph 的强大组合，Self-Discover 模式为构建智能代理系统提供了新的思路和方法论，推动了 AI 代理在复杂问题解决领域的发展。