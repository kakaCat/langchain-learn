---
title: "LangChain 入门教程：学习提示词模块"
description: "基于模块 02-prompt-templates 的实战介绍，含环境准备、依赖、快速上手与进阶示例索引。"
keywords:
  - LangChain
  - PromptTemplate
  - ChatPromptTemplate
  - FewShot
  - 结构化输出
  - JSON
  - Pydantic
  - JsonOutputParser
  - PydanticOutputParser
  - System Prompt
  - Human Prompt
  - Few-shot
  - 模板组合
tags:
  - Tutorial
  - Prompt
  - LLM
author: "langchain-learn"
date: "2025-10-12"
lang: "zh-CN"
canonical: "/blog/langchain-prompt-templates-tutorial"
audience: "初学者 / 具备Python基础的LLM工程师"
difficulty: "beginner-intermediate"
estimated_read_time: "12-18min"
topics:
  - LangChain Core
  - PromptTemplate
  - FewShotPromptTemplate
  - ChatPromptTemplate
  - JsonOutputParser
  - PydanticOutputParser
entities:
  - LangChain
  - OpenAI
  - Pydantic
  - dotenv
qa_intents:
  - "PromptTemplate 是什么？如何快速上手？"
  - "FewShotPromptTemplate 怎么用？有哪些最佳实践？"
  - "ChatPromptTemplate 中 System/Human 如何协同？"
  - "如何让 AI 输出稳定 JSON？"
  - "为什么选用 PydanticOutputParser？字段如何设计？"
  - "format_instructions 写不清会怎样？如何修复？"
  - "结构化输出失败的常见原因与排查步骤？"
  - "多模板组合如何避免上下文冲突？"
  - "Few-shot 示例数量如何把握？"
  - "如何安全管理环境变量与切换模型？"
---


# LangChain 入门教程：学习提示词模块

## 本页快捷跳转
- 目录：
  - [引言](#intro)
  - [PromptTemplate 是什么？如何快速上手？](#prompttemplate-basic)
  - [Few-shot 提示词是什么](#fewshot-basics)
    - [langchain 的 FewShotPromptTemplate 如何实现](#fewwhotprompt)
  - [System 与 User 提示词如何配合？](#system-user)
    - [ChatPromptTemplate 在 LangChain 中如何使用？](#chatprompt)
  - [控制 AI 输出特定格式](#structured)
    - [AI回答返回JSON格式](#json)
    - [AI回答返回指定模型](#pydantic)
  - [LangChain 的不同模板的组合使用](#composition)
  - [LangChain 的模板类总结](#templates-overview)
  - [常见错误与快速排查 (Q/A)](#qa)
  - [总结](#summary)
  - [术语与别名（便于检索与问法对齐）](#glossary)

---

<a id="intro" data-alt="introduction 引言 概述 目标 受众"></a>
## 引言
本教程面向具备 Python 基础的 LLM 工程师，围绕 LangChain 的提示词模块，帮助你在工程实践中构建可复用、可维护、可组合的提示词体系。你将快速掌握：
- PromptTemplate：通过变量插值实现提示词的标准化管理与复用
- FewShotPromptTemplate：用示例稳定输出风格与结构，减少歧义与漂移
- ChatPromptTemplate：System/Human 协同的多轮对话提示设计
- 结构化输出：JsonOutputParser 与 PydanticOutputParser 的约束与解析

<a id="intro-case" data-alt="变量插值 动机 收益 示例"></a>
### 真实案例：变量插值的动机与收益
我们经常需要让 AI 根据商品信息生成一段吸引人的广告文案。最直接的做法是把具体产品和风格硬编码在提示词里：

```python
template = """请写一篇关于智能手机的科幻风格的广告文案。"""
```
这种写法存在几个问题：
- 不可复用：每换产品/风格都要手改提示词
- 易出错：不同人写法不一致，输出风格不稳定
- 难维护：无法统一结构、长度、语气等约束

因此，我们把产品名与风格抽象成变量，占位在模板中：

```python
template = """请写一篇关于{product}的{style}风格的广告文案。"""
```
在实际工程中，借助 LangChain 的 PromptTemplate 可以对这种变量插值进行标准化管理、复用与校验，并与 FewShotPromptTemplate、ChatPromptTemplate、结构化输出等能力自然组合。详见 [PromptTemplate 快速上手](#prompttemplate-basic)。

<a id="prompttemplate-basic" data-alt="PromptTemplate 代码实现"></a>
## PromptTemplate 是什么？如何快速上手？

项目配置与依赖安装请参见下文的 [环境准备](#setup) 章节；如需完整步骤，请参考另一篇教程：[LangChain 入门教程：构建你的第一个聊天机器人](https://juejin.cn/post/7559428036514709554)。

```python
from langchain_core.prompts import PromptTemplate

template = """请写一篇关于{product}的{style}风格的广告文案。"""
prompt = PromptTemplate.from_template(template)
final_prompt = prompt.format(product="智能手机", style="科幻")
print(final_prompt)
# 组装后的提示词：请写一篇关于智能手机的科幻风格的广告文案。
```

通过 LangChain 的模板方式，保证了提示词的可复用性、一致性、可维护性、标准化。

#### 参数与输出预期（PromptTemplate）
- 输入变量：
  - product：string（产品名称，如“智能手机”）
  - style：string（写作风格，如“科幻”）
- 关键参数：
  - template：字符串模板，包含占位符 {product}、{style}
  - prompt.format(product, style)：进行变量插值，返回最终提示词文本
- 期望输出形状：
  - 文本字符串（1–2 段落的广告文案，语言为中文，风格由 style 指定）
- 稳定性小贴士：
  - 模板中使用明确动词与结构（如“请写…包含…字数约…”）
  - 对长度、风格与语气进行显式约束，避免泛化输出


<a id="fewshot-basics" data-alt="few-shot fewshot 少样本 示例提示词"></a>
## Few-shot 提示词是什么

**Few-shot 提示词**（少样本提示词）是一种通过提供少量示例来引导语言模型生成特定格式输出的技术。它让模型通过观察示例来学习任务模式，而不是仅仅依赖指令描述。

```text
请写一篇关于{product}的{style}风格的广告文案。
请根据以下示例，生成指定产品的广告文案：

示例1：
产品：智能手机
风格：科幻
文案："穿越时空的智能体验！量子芯片带来毫秒级运算速度，全息投影让沟通触手可及。这不是手机，这是通往未来的钥匙。"

示例2：
产品：智能手表
风格：科技
文案："24小时健康守护者！精准监测心率血氧，AI算法预测健康风险。你的私人健康管家，时刻守护你的每一刻。"

示例3：
产品：电动汽车
风格：环保
文案："零排放，零妥协！清洁能源驱动未来，智能续航消除里程焦虑。为地球减负，为生活加速。"

现在请生成：
产品：智能家居系统
风格：温馨
文案：
```
1. 解决风格理解偏差 ：通过具体示例让模型准确理解"科幻"、"科技"、"温馨"等抽象风格概念
2. 确保输出格式一致性 ：示例展示了完整的文案结构（产品-风格-文案），避免模型输出不完整的回答
3. 提升内容质量 ：示例提供了高质量文案的标准，引导模型生成同等水平的输出
4. 减少歧义 ：相比单纯的指令"写一篇温馨风格的广告文案"，Few-shot示例更清晰地定义了什么是"温馨风格"
5. 支持复杂任务 ：对于需要特定知识或专业术语的任务，Few-shot示例可以包含必要的专业内容
这个示例展示了Few-shot提示词如何通过提供具体示例来解决纯指令提示词在复杂任务中的局限性，确保模型输出更准确、更符合预期的结果。

<a id="fewwhotprompt" data-alt="FewShotPromptTemplate 代码实现"></a>
### langchain 的 FewShotPromptTemplate 如何实现？

```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

def main() -> None:
    # 1. 创建广告文案例子
    examples = [
        {
            "product": "智能手机",
            "style": "科幻",
            "copy": "穿越时空的智能体验！量子芯片带来毫秒级运算速度，全息投影让沟通触手可及。这不是手机，这是通往未来的钥匙。"
        },
        {
            "product": "智能手表", 
            "style": "科技",
            "copy": "24小时健康守护者！精准监测心率血氧，AI算法预测健康风险。你的私人健康管家，时刻守护你的每一刻。"
        },
        {
            "product": "电动汽车",
            "style": "环保", 
            "copy": "零排放，零妥协！清洁能源驱动未来，智能续航消除里程焦虑。为地球减负，为生活加速。"
        }
    ]
    # 2. 指定如何格式化每个例子
    example_formatter_template = "示例：\n产品：{product}\n风格：{style}\n文案：{copy}"
    example_prompt = PromptTemplate.from_template(example_formatter_template)
    # 3. 创建 FewShotPromptTemplate
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="请根据以下示例，生成指定产品的广告文案：",
        suffix="现在请生成：\n产品：{product}\n风格：{style}\n文案：",
        input_variables=["product", "style"]
    )
    final_messages = few_shot_prompt.format(product="智能家居系统", style="温馨")
    print(final_messages)

if __name__ == "__main__":
    main()
```

#### 参数与输出预期（FewShotPromptTemplate）
- 输入变量：
  - product：string（产品名称）
  - style：string（写作风格）
- 关键参数：
  - examples：示例列表，每项含 product/style/copy 字段
  - example_prompt：示例渲染模板（如“示例：产品：{product}…”）
  - prefix/suffix：示例前后缀文本，用于明确任务与输出位置
  - input_variables：允许插值的变量集合
- 期望输出形状：
  - 与示例一致的“文案”字段，结构与风格统一
- 稳定性小贴士：
  - 示例数量 2–5 个为宜；过多会稀释任务指令
  - 确保示例覆盖常见风格，避免偏向某单一风格


<a id="system-user" data-alt="system user 提示词 消息 协同 用法"></a>
## System 与 User 提示词如何配合？

系统提示词和用户提示词是聊天模型中的两种核心提示词类型，它们分别承担不同的角色和功能，共同构建完整的对话交互。

**系统提示词（System Prompt）**是聊天模型中用于设定AI角色身份、行为准则和知识范围的全局性指导提示，它为整个对话会话提供背景和约束条件。

**用户提示词（User Prompt）**是聊天模型中用户提出的具体请求或问题，构成对话的主体内容，针对当前轮次的特定需求。

```text
system：你是一个专业的广告文案专家，擅长为不同产品创作各种风格的广告文案。

human：请为"智能手机"创作一篇"科幻"风格的广告文案。
```

| 系统提示词（System Prompt） | 用户提示词（User Prompt） |
| --- | --- |
| 角色定义：设定AI的专家身份、行为准则和知识范围 | 具体任务：提出具体的请求或问题 |
| 全局指导：为整个对话会话提供背景和约束条件 | 会话内容：构成对话的主体内容 |
| 风格控制：决定AI的回应风格、专业程度和语气 | 上下文依赖：可能依赖于之前的对话历史 |
| 持续影响：在整个对话过程中持续发挥作用 | 即时响应：针对当前轮次的特定需求 |

1. **明确角色分工** ：系统提示词定义"你是谁"，用户提示词定义"你要做什么"
2. **分离关注点** ：将身份设定与具体任务解耦，提高提示词的复用性
3. **提升对话质量** ：系统提示词确保AI保持一致的专家身份和回应风格
4. **支持多轮对话** ：系统提示词为整个对话会话提供稳定的上下文背景
5. **灵活任务切换** ：在保持专家身份的同时，可以处理不同类型的用户请求

这个示例展示了系统提示词和用户提示词如何协同工作，系统提示词建立AI的专业身份和行为准则，用户提示词则提出具体的创作需求，两者结合确保AI能够以专业的方式完成特定任务。

<a id="chatprompt" data-alt="ChatPromptTemplate 代码实现"></a>
### ChatPromptTemplate 在 LangChain 中如何使用？

```python
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

# 从当前模块目录加载 .env
def load_environment() -> None:
    """加载环境变量配置"""
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)


def get_llm() -> ChatOpenAI:
    """
    创建并配置语言模型实例

    配置参数详解：
    - model: 指定使用的语言模型名称
    - api_key: API 认证密钥
    - temperature: 控制回答随机性 (0-1)
    - max_tokens: 限制回答最大长度
    - timeout: 请求超时时间（秒）
    - max_retries: 失败重试次数
    - request_timeout: 单次请求超时
    - base_url: API 服务端点
    - verbose: 是否启用详细日志

    返回：
    ChatOpenAI 实例 - 配置完整的语言模型对象
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    base_url = os.getenv("OPENAI_BASE_URL")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "512"))
    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 120,
        "max_retries": 3,
        "request_timeout": 120,
        "verbose": False,
        "base_url": base_url
    }

    return ChatOpenAI(**kwargs)


def main() -> None:
    """聊天模板示例：广告文案生成"""
    print("🔄 正在加载环境配置...")
    load_environment()
    print("=== 聊天模板示例：广告文案生成 ===")

    # 系统消息：定义角色和任务
    system_template = "你是一位专业的广告文案专家，擅长创作各种风格的广告文案。"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    # 用户消息：提供具体需求
    human_template = "请写一篇关于{product}的{style}风格的广告文案。"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # 创建聊天提示词模板
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    
    # 格式化提示词
    final_messages = chat_prompt.format_prompt(
        product="智能手机", 
        style="科幻"
    ).to_messages()
    
    print("生成的聊天消息：")
    for message in final_messages:
        print(f"{message.type}: {message.content}")
    print()

    # 演示不同产品的广告文案生成
    products_styles = [
        ("智能手表", "科技"),
        ("电动汽车", "环保"),
        ("智能家居系统", "温馨")
    ]
    
    print("=== 不同产品的广告文案生成示例 ===")
    for product, style in products_styles:
        messages = chat_prompt.format_prompt(product=product, style=style).to_messages()
        print(f"\n产品：{product}，风格：{style}")
        print(f"用户消息：{messages[1].content}")

        # 调用语言模型生成响应
        try:
            llm = get_llm()
            response = llm.invoke(messages)
            print(f"模型回复：{response.content}")
        except Exception as e:
            print("❌ 调用失败，请检查 .env 配置（OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_MODEL）或网络。")
            print(f"错误详情：{e}")

if __name__ == "__main__":
    main()
```

#### 参数与输出预期（ChatPromptTemplate）
- 输入变量：
  - product：string（用户消息中的产品变量）
  - style：string（用户消息中的风格变量）
- 关键参数：
  - SystemMessagePromptTemplate：系统消息模板
  - HumanMessagePromptTemplate：用户消息模板
  - ChatPromptTemplate：组合并格式化多轮消息
- 期望输出形状：
  - 模型回复文本，遵循系统设定的语气与专业性
- 稳定性小贴士：
  - 将长期规则写入 system，短期任务写入 human
  - 为安全与合规在 system 中加入边界与拒绝策略

<a id="structured" data-alt="结构化 输出 控制 格式 json pydantic"></a>
## 控制 AI 输出特定格式

<a id="json" data-alt="json 输出 structured output 格式化 format_instructions jsonoutputparser"></a>

### 1、AI回答返回JSON格式

```python
#!/usr/bin/env python3
"""
JSON 模式输出示例：广告文案生成
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# 从当前模块目录加载 .env
def load_environment() -> None:
    """加载环境变量配置"""
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)


def get_llm() -> ChatOpenAI:
    """
    创建并配置语言模型实例

    配置参数详解：
    - model: 指定使用的语言模型名称
    - api_key: API 认证密钥
    - temperature: 控制回答随机性 (0-1)
    - max_tokens: 限制回答最大长度
    - timeout: 请求超时时间（秒）
    - max_retries: 失败重试次数
    - request_timeout: 单次请求超时
    - base_url: API 服务端点
    - verbose: 是否启用详细日志

    返回：
    ChatOpenAI 实例 - 配置完整的语言模型对象
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    base_url = os.getenv("OPENAI_BASE_URL")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "512"))
    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 120,
        "max_retries": 3,
        "request_timeout": 120,
        "verbose": False,
        "base_url": base_url
    }

    return ChatOpenAI(**kwargs)


def main() -> None:
    """JSON 模式输出示例：广告文案生成"""
    print("🔄 正在加载环境配置...")
    load_environment()
    print("=== JSON 模式输出示例：广告文案生成 ===")

    # 创建 JSON 输出解析器
    parser = JsonOutputParser()

    # 创建提示词模板
    template = """
    请为指定产品生成广告文案，并以 JSON 格式输出：

    产品：{product}
    风格：{style}

    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["product", "style"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 示例产品
    product = "智能手机"
    style = "科幻"

    # 格式化提示词
    formatted_prompt = prompt.format(product=product, style=style)
    print("提示词：")
    print(formatted_prompt)
    print("\n" + "=" * 50 + "\n")

    # 调用模型并解析输出
    try:
        response = get_llm().invoke(formatted_prompt)
        print("模型回复：")
        print(response.content)
        print("\n" + "=" * 50 + "\n")
        ad_copy = parser.parse(response.content)
        print("解析后的广告文案：")
        print(ad_copy)
    except Exception as e:
        print("❌ 调用或解析失败，请检查：OPENAI_API_KEY、OPENAI_MODEL、OPENAI_BASE_URL 以及输出格式。")
        print(f"错误详情：{e}")
        print("原始响应内容:", response.content if 'response' in locals() else "<no-response>")

if __name__ == "__main__":
    main()
```

#### 参数与期望输出形状（JSON 示例)
- 输入变量：product（字符串）、style（字符串）
- 关键参数：temperature=0–0.3、max_tokens≥256、包含 parser.get_format_instructions()
- 期望输出：合法 JSON，示例字段建议：{"product":string, "style":string, "headline":string, "description":string, "call_to_action":string}
- 稳定性提示：若字段缺失或结构漂移，降低 temperature、增补 format_instructions、增强示例约束

<a id="pydantic" data-alt="pydantic 输出 parser 验证 schema pydanticoutputparser"></a>

### 2、AI回答返回指定模型

```python
#!/usr/bin/env python3
"""
Pydantic 模型约束输出示例：广告文案生成
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

# 从当前模块目录加载 .env
def load_environment() -> None:
    """加载环境变量配置"""
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)


def get_llm() -> ChatOpenAI:
    """
    创建并配置语言模型实例

    配置参数详解：
    - model: 指定使用的语言模型名称
    - api_key: API 认证密钥
    - temperature: 控制回答随机性 (0-1)
    - max_tokens: 限制回答最大长度
    - timeout: 请求超时时间（秒）
    - max_retries: 失败重试次数
    - request_timeout: 单次请求超时
    - base_url: API 服务端点
    - verbose: 是否启用详细日志

    返回：
    ChatOpenAI 实例 - 配置完整的语言模型对象
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    base_url = os.getenv("OPENAI_BASE_URL")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "512"))
    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 120,
        "max_retries": 3,
        "request_timeout": 120,
        "verbose": False,
        "base_url": base_url
    }

    return ChatOpenAI(**kwargs)

class AdCopy(BaseModel):
    """广告文案模型"""
    product: str = Field(description="产品名称")
    style: str = Field(description="文案风格")
    headline: str = Field(description="广告标题")
    description: str = Field(description="广告描述")
    call_to_action: str = Field(description="行动号召")

    # Pydantic v2 配置
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "product": "智能手机",
                "style": "科幻",
                "headline": "穿越时空的智能体验",
                "description": "量子芯片带来毫秒级运算速度，全息投影让沟通触手可及。这不是手机，这是通往未来的钥匙。",
                "call_to_action": "立即预订，开启未来之旅",
            }
        }
    )


def main() -> None:
    """Pydantic 模型约束输出示例：广告文案生成"""
    print("🔄 正在加载环境配置...")
    load_environment()
    print("=== Pydantic 模型约束输出示例：广告文案生成 ===")

    # 创建 Pydantic 输出解析器
    parser = PydanticOutputParser(pydantic_object=AdCopy)

    # 创建提示词模板
    template = """
    请为指定产品生成广告文案，并按照指定格式输出：

    产品：{product}
    风格：{style}

    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["product", "style"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 示例产品
    product = "智能手机"
    style = "科幻"

    # 格式化提示词
    formatted_prompt = prompt.format(product=product, style=style)
    print("提示词：")
    print(formatted_prompt)
    print("-" * 30)

    # 调用模型
    try:
        response = get_llm().invoke(formatted_prompt)
        print("模型回复：")
        print(response.content)
    except Exception as e:
        print("❌ 调用失败，请检查 .env 配置（OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_MODEL）或网络。")
        print(f"错误详情：{e}")
        return
    # 解析输出
    try:
        ad_copy = parser.parse(response.content)
        print("\n解析后的广告文案：")
        print(ad_copy)
    except Exception as e:
        print("Pydantic 解析错误:", e)
        print("原始响应内容:", response.content)
        # 提供更友好的错误信息
        print("\n提示：如果模型未返回有效格式，请检查：")
        print("1. 模型是否理解Pydantic格式要求")
        print("2. 提示词中的format_instructions是否清晰")
        print("3. 尝试调整temperature参数降低随机性")
        print("4. 检查模型是否支持结构化输出")

if __name__ == "__main__":
    main()
```

#### 参数与期望输出形状（Pydantic 示例）
- 输入变量：product（字符串）、style（字符串）
- 关键参数：temperature=0–0.3、max_tokens≥256、parser.get_format_instructions() 必须包含
- 期望输出：满足 AdCopy 模型的 JSON（含 product、style、headline、description、call_to_action）
- 稳定性提示：字段不全或类型不符时，降低 temperature、增加示例、明确字段描述与示例 sample

<a id="composition" data-alt="模板 组合 组合使用 PromptTemplate ChatPromptTemplate"></a>
## LangChain 的不同模板的组合使用

我们实际使用的时候，会有很多不同的场景，所以需要选择不同的模板进行组合使用，来拼装我们的提示词，使得 AI 能够根据不同的场景生成不同的结果。

### 1、基础模板的组合使用

下面例子是系统角色+全局指导（约束）的一个例子

```python
from langchain_core.prompts import PromptTemplate


def main() -> None:
    """基础模板组合示例：广告文案生成"""
    print("=== 基础模板组合示例：广告文案生成 ===")

    # 创建基础模板
    base_template = PromptTemplate.from_template(
        """你是一个{role}。请为{product}创作{style}风格的广告文案。"""
    )

    # 创建具体任务模板
    task_template = PromptTemplate.from_template(
        """{base_prompt}

        创作要求：
        - {requirement1}
        - {requirement2}
        - {requirement3}
        """
    )

    # 组合模板
    role = "专业广告文案创作师"
    product = "智能手机"
    style = "科幻"
    base_prompt = base_template.format(role=role, product=product, style=style)

    requirements = {
        "requirement1": "突出产品的科技感",
        "requirement2": "语言富有想象力和感染力",
        "requirement3": "长度控制在50-100字之间"
    }

    final_prompt = task_template.format(
        base_prompt=base_prompt,
        **requirements
    )

    print("组合后的提示词：")
    print(final_prompt)
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
``` 

### 2、对话模板的组合使用

下面例子是**系统提示词（System Prompt）**+**用户提示词（User Prompt）**的一个例子

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

def main() -> None:
    """聊天模板组合示例：广告文案生成"""
    print("=== 聊天模板组合示例：广告文案生成 ===")

    # 创建系统消息模板
    system_template = PromptTemplate.from_template(
        "你是一个{role}，擅长{expertise}。"
    )

    # 创建用户消息模板
    user_template = PromptTemplate.from_template(
        "请为{product}创作{style}风格的广告文案。"
    )

    # 组合聊天模板
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template.format(role="专业广告文案创作师", expertise="各种风格的广告文案创作")),
        ("human", user_template.format(product="智能手机", style="科幻"))
    ])

    print("聊天提示词：")
    messages = chat_prompt.format_prompt().to_messages()
    for m in messages:
        print(f"{m.type}: {m.content}")
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
```


### 3、模板继承的组合使用

下面例子是模板继承的组合使用，我们可以在基础模板的基础上，扩展出不同的功能模板，来满足不同的场景需求。

```python

from langchain_core.prompts import PromptTemplate

def main() -> None:
    """模板继承示例"""
    print("=== 模板继承示例 ===")

    # 基础模板
    base_template = """
    请为{product}写一篇{style}风格的{content_type}。

    要求：
    - 语言生动有趣
    - 突出产品特点
    - 吸引目标用户
    """

    # 扩展模板：广告文案
    ad_template = base_template + """

    广告文案特点：
    - 包含行动号召
    - 强调优惠信息
    - 长度控制在200字以内
    """

    # 扩展模板：产品介绍
    intro_template = base_template + """

    产品介绍特点：
    - 详细说明功能（例如：{main_features}）
    - 包含使用场景（例如：{use_cases}）
    - 目标用户：{target_audience}
    - 长度控制在500字以内
    """

    base_prompt = PromptTemplate.from_template(base_template)
    ad_prompt = PromptTemplate.from_template(ad_template)
    intro_prompt = PromptTemplate.from_template(intro_template)

    print("基础模板：")
    print(base_prompt.format(product="智能手机", style="科技", content_type="内容"))
    print("\n广告模板：")
    print(ad_prompt.format(product="智能手机", style="科技", content_type="广告文案"))
    print("\n介绍模板：")
    print(intro_prompt.format(product="智能手机", style="科技", content_type="产品介绍", main_features="AI相机、120Hz屏幕、5000mAh电池", use_cases="差旅、摄影、移动办公", target_audience="年轻职场人群"))
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
```
### 4、模板分步的组合使用

下面例子是模板分步的组合使用，我们可以将一个复杂的任务，分解成多个简单的任务，每个任务对应一个模板，最后将这些模板组合起来，形成一个完整的任务。

```python

from langchain_core.prompts import PromptTemplate

def main() -> None:
    """多步骤模板组合示例：广告文案创作流程"""
    print("=== 多步骤模板组合示例：广告文案创作流程 ===")

    # 步骤1：产品分析
    analysis_template = PromptTemplate.from_template(
        """请分析以下产品特点：
        产品：{product}
        目标风格：{style}

        分析要点：
        - 产品核心卖点
        - 目标用户群体
        - 风格适配要点
        """
    )

    # 步骤2：文案创作
    creation_template = PromptTemplate.from_template(
        """{analysis_result}

        请基于以上分析，创作符合要求的广告文案：
        """
    )

    # 步骤3：优化建议
    optimization_template = PromptTemplate.from_template(
        """{ad_copy}

        请提供优化建议：
        - 语言表达
        - 情感共鸣
        - 营销效果
        """
    )

    # 执行多步骤流程
    product = "智能手机"
    style = "科幻"

    print("步骤1：产品分析")
    analysis_prompt = analysis_template.format(
        product=product,
        style=style
    )
    print(analysis_prompt)
    print("\n" + "-"*30 + "\n")

    print("步骤2：文案创作")
    creation_prompt = creation_template.format(
        analysis_result="[分析结果占位]"
    )
    print(creation_prompt)
    print("\n" + "-"*30 + "\n")

    print("步骤3：优化建议")
    optimization_prompt = optimization_template.format(
        ad_copy="[广告文案占位]"
    )
    print(optimization_prompt)
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
```


<a id="templates-overview" data-alt="模板 总结 概览"></a>
## LangChain 的模板类总结

### 核心模板类继承关系

```
BasePromptTemplate
├── PromptTemplate (基础字符串模板)
├── FewShotPromptTemplate (少样本模板)
└── ChatPromptTemplate (聊天模板)
    ├── MessagesPlaceholder (消息占位符)
    ├── SystemMessagePromptTemplate (系统消息模板)
    ├── HumanMessagePromptTemplate (用户消息模板)
    └── AIMessagePromptTemplate (AI消息模板)
```

### 模板类功能对比

| 模板类 | 主要用途 | 输入格式 | 输出格式 | 适用场景 |
|--------|----------|----------|----------|----------|
| `PromptTemplate` | 基础字符串模板 | 变量插值 | 字符串 | 简单提示词、指令式任务 |
| `FewShotPromptTemplate` | 少样本学习 | 示例+变量 | 字符串 | 复杂任务、格式控制 |
| `ChatPromptTemplate` | 多角色对话 | 消息列表 | 消息列表 | 聊天应用、多轮对话 |
| `SystemMessagePromptTemplate` | 系统角色定义 | 变量插值 | SystemMessage | 角色设定、行为约束 |
| `HumanMessagePromptTemplate` | 用户输入 | 变量插值 | HumanMessage | 用户请求、问题输入 |

### 模板选择指南

1. **简单指令任务** → `PromptTemplate`
   - 单一指令，无需对话历史
   - 变量插值简单
   - 输出为纯文本

2. **复杂格式控制** → `FewShotPromptTemplate`
   - 需要示例引导
   - 输出格式严格
   - 任务模式复杂

3. **聊天应用** → `ChatPromptTemplate`
   - 多角色对话
   - 需要对话历史
   - 系统角色定义

4. **结构化输出** → `ChatPromptTemplate` + 输出解析器
   - JSON/Pydantic 输出
   - 数据验证
   - 类型安全

### 最佳实践

1. **模板复用**：将常用模板保存为可复用的组件
2. **模板组合**：使用模板继承和组合构建复杂提示词
3. **模板验证**：在运行时验证模板变量和格式
4. **性能优化**：预编译模板，避免重复格式化
5. **国际化**：为不同语言创建对应的模板版本

### 结构化输出策略与工具补充

- JsonOutputParser：用于指导模型输出严格的 JSON，并负责解析与错误提示
- PydanticOutputParser：使用 Pydantic 模型定义字段与约束，让解析更安全可控


建议：根据场景选择结构化策略，简单结构用 JSON，约束严格用 Pydantic。



<a id="qa" data-alt="常见错误 快速排查 QA"></a>
## 常见错误与快速排查 (Q/A)

### 快速排查清单（原子步骤）
1. 检查环境：OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL 是否存在且正确。
2. 检查提示词：是否包含 parser.get_format_instructions()；占位符与传入变量是否一致。
3. 收敛参数：temperature ≤ 0.3；max_tokens ≥ 256；必要时增加示例数量。
4. 解析失败：打印原始响应片段；尝试正则清洗；逐步放宽字段约束再逐步收紧。
5. 网络与速率：降低并发，设置 timeout/max_retries；对重复请求做缓存。

### 常见问答（FAQ）
- PromptTemplate、FewShotPromptTemplate、ChatPromptTemplate 的差异与选型？
  - 单轮/参数化→PromptTemplate；需示例引导→FewShotPromptTemplate；多轮与角色→ChatPromptTemplate。
- 如何让 AI 产出稳定、可解析的 JSON？
  - 写入清晰的 format_instructions；降低 temperature；必要时用 PydanticOutputParser 强约束。
- PydanticOutputParser 为什么更稳？字段如何设计？
  - 定义明确字段与描述；用示例 json_schema_extra 提供参考；解析严格校验可快速暴露问题。
- format_instructions 写不清会怎样？如何修复？
  - 模型易发散；补充字段名/类型/示例；减少修辞，使用命令式语言。
- 多模板组合的原则是什么？如何避免上下文冲突？
  - 明确角色与输入边界；避免重复约束；分层组织 prefix/suffix 与系统/用户消息。
- FewShot 示例数量怎么把握？
  - 2–5 个为宜，覆盖边界案例；示例字段顺序与输出格式保持一致。
- 出现中英混输如何保持格式一致？
  - 在 format_instructions 指定语言与字符集；必要时加入语言选择提示。
- 部署时如何保证环境变量安全与模型可切换？
  - .env 管理，避免在代码中硬编码；将模型名与端点做成可配置项。

<a id="links"></a>
### 详细代码和文档
- 完整代码：查看 [GitHub 仓库](https://github.com/kakaCat/langchain-learn/tree/main/02-prompt-templates)
- 项目结构：参考仓库中的 `README.md`
- LangChain 文档：https://python.langchain.com/
- LangChain OpenAI 集成（Python API 索引）：https://api.python.langchain.com/
- DeepSeek API 文档：https://api-docs.deepseek.com/
- OpenAI API 文档：https://platform.openai.com/docs/api-reference
- Azure OpenAI 文档：https://learn.microsoft.com/azure/ai-services/openai/
- Ollama 文档：https://ollama.com/docs

<a id="summary" data-alt="总结 归纳"></a>
## 总结

通过本指南，你已经掌握了提示词工程的核心能力：从模板化与 few-shot，到聊天模板与结构化输出。建议在真实项目中将模板作为“可测试、可复用”的一等资产进行管理，并结合验证与性能优化策略，持续提升模型输出的稳定性与可控性。

<a id="glossary" data-alt="术语 名词 定义"></a>
## 术语与别名（便于检索与问法对齐）
- PromptTemplate：提示词模板；别名→ 模板、Prompt 模板
- Few-shot：少样本；别名→ FewShot、示例提示、样例引导
- System Prompt：系统提示词；别名→ 系统消息、角色设定
- User Prompt：用户提示词；别名→ 用户消息、任务请求
- Structured Output：结构化输出；别名→ JSON 输出、Pydantic 约束