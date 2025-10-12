---
title: "LangChain 提示词模块应用指南：从 PromptTemplate 到结构化输出"
description: "基于模块 02-prompt-templates 的实战介绍，含环境准备、依赖、快速上手与进阶示例索引。"
keywords:
  - LangChain
  - PromptTemplate
  - ChatPromptTemplate
  - Few-shot
  - 结构化输出
  - JSON
  - Pydantic
  - 国际化
tags:
  - Tutorial
  - Prompt
  - LLM
author: "langchain-learn"
date: "2025-10-10"
lang: "zh-CN"
canonical: "/blog/langchain-prompt-templates-tutorial"
---

#  LangChain 提示词模块应用指南

## 本页快捷跳转

- 直达： [引言](#intro) | [环境准备](#setup) | [核心概念](#concepts) | [快速上手](#quickstart) | [进阶示例索引](#demos) | [常见错误与快速排查 (Q/A)](#qa) | [官方链接](#links) | [总结](#summary)

---

<a id="intro"></a>
## 引言

在构建 LLM 应用时，提示词（Prompt）是质量的"第一因子"。LangChain 的提示词模块提供了从基础模板到多角色聊天模板、从 few-shot 示例到结构化输出的完整能力，让你能稳定、可复用地"喂给模型正确的指令"。

### 真实的一个案例
我们希望ai根据商品的信息，生成一个吸引人的描述。

```python
template = """请写一篇关于智能手机的科幻风格的广告文案。"""
```
但是我们会发现，我们需要生成不同种产品的描述，而不是只生成智能手表的描述。
这时候我们就需要使用变量插值，将商品名称、品牌、特点和目标用户等变量插入到模板中。

```python
template = """请写一篇关于{product}的{style}风格的广告文案。"""
```
### langchain的基础模板代码实现

项目配置请看 https://juejin.cn/post/7559428036514709554

```python
from langchain_core.prompts import PromptTemplate

template = """请写一篇关于{product}的{style}风格的广告文案。"""
prompt = PromptTemplate.from_template(template)
final_prompt = prompt.format(product="智能手机", style="科幻")
print(final_prompt)
# 组装后的提示词：请写一篇关于智能手机的科幻风格的广告文案。
```

通过langchain的模板方式,保证了提示词的可复用性、一致性、可维护性、标准化。

## Few-shot 提示词

**Few-shot 提示词**（少样本提示词）是一种通过提供少量示例来引导语言模型生成特定格式输出的技术。它让模型通过观察示例来学习任务模式，而不是仅仅依赖指令描述。

```text
请写一篇关于{product}的{style}风格的广告文案。
请根据以下示例，生成指定产品的广告文案：

示例1：
产品：智能手机
风格：科幻
文案："穿越时空的智能体验！量子芯片带来前所未有的运算速度，全息投影让沟通触手可及。这不是手机，这是通往未来的钥匙。"

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

### langchain的Few-shot模板代码实现

```python

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

def main() -> None:
    # 1. 创建广告文案例子
    examples = [
        {
            "product": "智能手机",
            "style": "科幻",
            "copy": "穿越时空的智能体验！量子芯片带来前所未有的运算速度，全息投影让沟通触手可及。这不是手机，这是通往未来的钥匙。"
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


## 系统提示词和用户提示词

系统提示词和用户提示词是聊天模型中的两种核心提示词类型，它们分别承担不同的角色和功能，共同构建完整的对话交互。

**系统提示词（System Prompt）**是聊天模型中用于设定AI角色身份、行为准则和知识范围的全局性指导提示，它为整个对话会话提供背景和约束条件。

**用户提示词（User Prompt）**是聊天模型中用户提出的具体请求或问题，构成对话的主体内容，针对当前轮次的特定需求。

```text
系统提示词：你是一个专业的广告文案专家，擅长为不同产品创作各种风格的广告文案。

用户提示词：请为"智能手机"创作一篇"科幻"风格的广告文案。
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

### langchain的ChatPromptTemplate代码实现

```python

from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

def main() -> None:
    """聊天模板示例：广告文案生成"""
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
        print("-" * 50)

if __name__ == "__main__":
    main()
```

## 控制ai输出特定格式

### 1、AI回答返回JSON格式

```python

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# 从当前模块目录加载 .env
def load_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)


def get_llm() -> ChatOpenAI:
    """创建并配置语言模型实例"""

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
    load_environment()
    """JSON 模式输出示例：广告文案生成"""
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
    response = get_llm().invoke(formatted_prompt)
    print("模型回复：")
    print(response.content)
    print("\n" + "=" * 50 + "\n")
    # 解析 JSON 输出
    try:
        ad_copy = parser.parse(response.content)
        print("解析后的广告文案：")
        print(ad_copy)
    except Exception as e:
        print("JSON 解析错误:", e)
if __name__ == "__main__":
    main()
```

### 2、AI回答返回指定模型

```python

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

# 从当前模块目录加载 .env
def load_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)


def get_llm() -> ChatOpenAI:

    """创建并配置语言模型实例"""
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
                "description": "量子芯片带来前所未有的运算速度，全息投影让沟通触手可及。这不是手机，这是通往未来的钥匙。",
                "call_to_action": "立即预订，开启未来之旅",
            }
        }
    )


def main() -> None:
    load_environment()
    """Pydantic 模型约束输出示例：广告文案生成"""
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
    response =  get_llm().invoke(formatted_prompt)
    print("模型回复：")
    print(response.content)
    # 解析输出
    ad_copy = parser.parse(response.content)
    print("\n解析后的广告文案：")
    print(ad_copy)

if __name__ == "__main__":
    main()
```

### 3、AI回答调用工具

```python

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 从当前模块目录加载 .env
def load_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)


def get_llm() -> ChatOpenAI:

    """创建并配置语言模型实例"""
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
    load_environment()
    """函数调用模板示例：广告文案生成"""
    print("=== 函数调用模板示例：广告文案生成 ===")

    # 定义函数调用模板
    template = """
    根据用户需求，调用相应的广告文案生成函数：

    需求：{query}

    可用的函数：
    - generate_ad_copy(product: str, style: str): 生成指定产品和风格的广告文案
    - analyze_market_trend(keyword: str): 分析市场趋势
    - optimize_ad_performance(copy: str): 优化广告文案效果

    请选择最合适的函数并返回函数调用信息。
    """

    prompt = PromptTemplate.from_template(template)

    # 示例查询
    queries = [
        "请为智能手机生成科幻风格的广告文案",
        "分析当前智能手表的市场趋势",
        "优化这段电动汽车广告文案的效果",
    ]

    for query in queries:
        formatted_prompt = prompt.format(query=query)
        print(f"需求：{query}")
        print("提示词：")
        print(formatted_prompt)
        print("-" * 30)
        response = get_llm().invoke(formatted_prompt)
        print("AI回复：")
        print(response.content)
        print("-" * 30)

if __name__ == "__main__":
    main()
```

## langchain的不同模板的组合使用

我们实际使用的时候，会很多不同的场景，所有需要选择不同的模板进行组合使用，来拼装我们的提示词，使得AI能够根据不同的场景生成不同的结果。

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
        "requirement1": "突出产品的科技感和未来感",
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
from langchain_core.prompts import ChatPromptTemplate

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
    print(chat_prompt.format())
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

def maim() -> None:
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
    maim()
```

## 总结

通过本指南，你已经掌握了提示词工程的核心能力：从模板化与 few-shot，到聊天模板与结构化输出。建议在真实项目中将模板作为“可测试、可复用”的一等资产进行管理，并结合验证与性能优化策略，持续提升模型输出的稳定性与可控性。