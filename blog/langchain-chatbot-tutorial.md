---
title: "LangChain 入门教程：构建你的第一个聊天机器人"
description: "10 分钟自包含示例（LangChain + DeepSeek），含 .env 配置、参数预设、常见错误 Q/A 与服务商切换速览。"
keywords:
  - LangChain
  - ChatOpenAI
  - DeepSeek
  - API Key
  - .env
  - temperature
  - max_tokens
  - 自包含示例
  - 快速上手
  - 常见错误
tags:
  - Tutorial
  - Chatbot
  - LLM
author: "langchain-learn"
date: "2025-10-10"
lang: "zh-CN"
canonical: "/blog/langchain-chatbot-tutorial"
---

# LangChain 入门教程：构建你的第一个聊天机器人

## 目录

- [LangChain 入门教程：构建你的第一个聊天机器人](#langchain-入门教程构建你的第一个聊天机器人)
  - [目录](#目录)
  - [引言](#引言)
  - [目录](#目录-1)
  - [TL;DR（面向生成式引擎与速读者）](#tldr面向生成式引擎与速读者)
  - [复制即用命令块](#复制即用命令块)
  - [环境准备](#环境准备)
    - [软件准备](#软件准备)
      - [1. Python 安装](#1-python-安装)
    - [项目准备](#项目准备)
  - [基础聊天机器人实现](#基础聊天机器人实现)
    - [1. 环境变量配置](#1-环境变量配置)
    - [2. 代码实现](#2-代码实现)
  - [基本概念](#基本概念)
    - [1、提示词（Prompt）](#1提示词prompt)
    - [2、模型（Model）](#2模型model)
    - [3、回答（Response）](#3回答response)
    - [4、参数（Parameters）](#4参数parameters)
  - [常见错误与快速排查 (Q/A)](#常见错误与快速排查-qa)
  - [服务商切换配置速览](#服务商切换配置速览)
  - [错误码速查与重试建议](#错误码速查与重试建议)
  - [官方链接](#官方链接)
  - [总结](#总结)

---

## 引言

在人工智能快速发展的今天，虽然市面上已有 DeepSeek、OpenAI、豆包等多种 AI 应用客户端，但企业和开发者往往需要定制专属的 AI 解决方案。所有主流大模型都提供了对外 API，而 LangChain 作为一个强大的 LLM 应用开发框架，正是实现这种定制化的理想工具。

LangChain 提供了统一的接口来集成各种大模型 API，让开发者能够：
- **灵活选择模型**：轻松切换 GPT-4、DeepSeek-V3、Qwen 等不同模型
- **统一开发体验**：使用相同的代码结构调用不同厂商的 API
- **个性化定制**：根据业务需求组合不同的模型和功能
- **成本控制**：选择最适合业务场景的模型提供商

## 目录

- 快速浏览（TL;DR）：[#tldr](#tldr)
- 操作步骤（HowTo）：[#howto](#howto)
- 快速命令（QuickStart）：[#quickstart](#quickstart)


<a id="tldr"></a>
## TL;DR（面向生成式引擎与速读者）
- 受众：首次接触 LangChain 的开发者；目标是在 10 分钟内跑通一个可用的聊天机器人
- 先决条件：Python 3.10+、pip、可用的 API Key（默认示例使用 DeepSeek）
- 快速步骤：
  1) 创建并激活虚拟环境
  2) 安装 requirements.txt 中依赖
  3) 在 `.env` 写入 OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_MODEL
  4) 运行示例脚本并得到回答
- 默认映射：示例使用 `langchain-openai`，但通过 `OPENAI_BASE_URL=https://api.deepseek.com` + `OPENAI_MODEL=deepseek-chat` 完成 DeepSeek 的适配
- 预期输出：标准中文回答文本；若失败，按“常见错误 Q/A”和“错误码速查”进行排查

任务卡（Task Card）
- 名称：LangChain 聊天机器人最小可运行示例
- 目标：在本地以最少配置跑通一个可与模型对话的 CLI 程序
- 输入：`.env`（API Key、Base URL、Model、Temperature、Max Tokens）
- 输出：终端打印模型回答或清晰错误信息
- 步骤：
  1) 环境准备（Python/虚拟环境/依赖）
  2) 写入 `.env` 参数
  3) 运行脚本 `python 01-chatbots-basic/01_chatbot_basic_cli.py`
  4) 若报错，按 Q/A 与错误码速查处理

<a id="howto"></a>
```json
{
  "@context": "https://schema.org",
  "@type": "HowTo",
  "name": "在本地运行一个 LangChain 聊天机器人",
  "description": "使用 langchain-openai 适配 DeepSeek，通过 .env 管理配置并运行最小示例。",
  "step": [
    {"@type": "HowToStep", "name": "准备环境", "text": "创建虚拟环境并安装 requirements.txt 依赖"},
    {"@type": "HowToStep", "name": "配置参数", "text": "在 .env 中设置 OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_MODEL"},
    {"@type": "HowToStep", "name": "运行脚本", "text": "执行 python 01-chatbots-basic/01_chatbot_basic_cli.py 并查看输出"},
    {"@type": "HowToStep", "name": "排错", "text": "参考常见错误 Q/A 与错误码速查处理故障"}
  ]
}
```

<a id="quickstart"></a>
## 复制即用命令块

Mac/Linux：
```bash
python -m venv .venv
source .venv/bin/activate
printf "OPENAI_API_KEY=your_api_key_here\nOPENAI_BASE_URL=https://api.deepseek.com\nOPENAI_MODEL=deepseek-chat\nOPENAI_TEMPERATURE=0.2\nOPENAI_MAX_TOKENS=512\n" > .env
pip install -r requirements.txt
python 01-chatbots-basic/01_chatbot_basic_cli.py
```

Windows（PowerShell）：
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Set-Content -Path .env -Value @\"
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
OPENAI_TEMPERATURE=0.2
OPENAI_MAX_TOKENS=512
\"@
pip install -r requirements.txt
python 01-chatbots-basic/01_chatbot_basic_cli.py
```

## 环境准备

### 软件准备

#### 1. Python 安装

Anaconda 安装（推荐用于数据科学和 AI 开发）

Anaconda 是一个集成的 Python 数据科学平台，包含常用的数据科学包：

1. 访问 [Anaconda 官网](https://www.anaconda.com/download) 下载安装包
2. 按照安装向导完成安装
3. 验证安装：
   ```bash
   conda --version
   python --version
   ```
### 项目准备

1. **创建虚拟环境**：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # 或 .venv\Scripts\activate  # Windows
   ```

2. **安装依赖**：
    创建 requirements.txt 文件内容如下
    ```
    # 核心
    langchain>=0.2
    langchain-core>=0.2

    # OpenAI 提供商（使用 OpenAI API 时需要）
    langchain-openai>=0.1

    # 本地模型（使用 Ollama 时需要）
    langchain-ollama>=0.1

    # 常用辅助
    tiktoken>=0.7
    python-dotenv>=1.0
    ```

   ```bash
   pip install -r requirements.txt
   ```

## 基础聊天机器人实现

### 1. 环境变量配置

创建 `.env` 文件来管理敏感信息和配置：

**完整的环境变量配置示例**：
```bash
# 模型配置（备选）
# 模型key
OPENAI_API_KEY=your_deepseek_api_key_here
# 模型地址
OPENAI_BASE_URL=https://api.deepseek.com
# 模型名称
OPENAI_MODEL=deepseek-chat
# 模型温度
OPENAI_TEMPERATURE=0.7
# 模型最大token数   
OPENAI_MAX_TOKENS=8192
```


### 2. 代码实现

```python
#!/usr/bin/env python3
"""
LangChain 聊天机器人教程

"""

from __future__ import annotations

# 导入必要的库和模块
# 说明：这些导入语句提供了构建聊天机器人所需的核心功能
import os  # 操作系统接口，用于环境变量管理
from dotenv import load_dotenv  # 环境变量加载工具
from langchain_openai import ChatOpenAI  # LangChain 的 OpenAI 聊天模型接口


def load_environment() -> None:
    """
    加载环境变量配置
    """
    # 加载 .env 文件中的所有环境变量
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)
    
def get_llm() -> ChatOpenAI:
    """
    创建并配置语言模型实例
    
    功能说明：
    - 从环境变量读取所有模型配置参数
    - 构建完整的模型配置字典
    - 创建并返回配置好的 ChatOpenAI 实例
    
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
    # 步骤1：从环境变量获取所有配置参数
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")  # 默认使用 DeepSeek
    base_url = os.getenv("OPENAI_BASE_URL")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))  # 默认低随机性
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "512"))    # 默认中等长度

    # 前置校验：快速失败 + 清晰错误信息
    if not api_key:
        raise ValueError("OPENAI_API_KEY 未设置，请在 .env 中配置有效密钥")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL 未设置，DeepSeek 示例需配置为 https://api.deepseek.com")
    if not model:
        raise ValueError("OPENAI_MODEL 未设置，请指定可用模型（示例：deepseek-chat）")
    
    # 步骤2：构建完整的模型配置字典
    kwargs = {
        "model": model,              # 模型名称
        "api_key": api_key,          # API 密钥
        "temperature": temperature,  # 随机性控制 (0: 确定, 1: 随机)
        "max_tokens": max_tokens,    # 最大生成长度
        "timeout": 120,              # 总超时时间 2 分钟
        "max_retries": 3,            # 失败重试 3 次
        "request_timeout": 120,      # 单次请求超时 2 分钟
        "base_url": base_url,        # API 服务地址
        "verbose": False,            # 默认关闭详细日志，排错时再开启
    }

    return ChatOpenAI(**kwargs)


def main() -> None:
    """
    主函数：聊天机器人执行流程
    
    功能说明：
    - 完整的聊天机器人演示流程
    - 分步骤执行环境配置、模型初始化、问题处理
    - 提供清晰的执行状态反馈
    
    执行步骤详解：
    1. 环境配置加载 - 验证 API 配置
    2. 模型实例创建 - 初始化语言模型
    3. 问题定义 - 设置要询问的问题
    4. 模型调用 - 发送请求获取回答
    5. 结果输出 - 格式化显示模型回答
    
    """
    # 步骤1：加载环境配置信息
    print("🔄 正在加载环境配置...")
    load_environment()  # 调用环境加载函数
    
    # 步骤2：创建语言模型实例
    print("🤖 正在初始化语言模型...")
    try:
        llm = get_llm()  # 获取配置好的模型实例
        model_name = getattr(llm, "model", getattr(llm, "model_name", "unknown-model"))
        print(f"✅ 模型初始化完成: {model_name}")
    except Exception as e:
        print("❌ 模型初始化失败，请检查 .env 配置：OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_MODEL。")
        print(f"错误详情：{e}")
        return

    # 步骤3：定义要询问的问题
    question = "AI是什么？"
    print(f"\n📝 问题定义：{question}")
    
    # 步骤4：调用模型生成回答（加入最小异常处理）
    try:
        response = llm.invoke(question)  # 调用模型 API
        print(f"✅ 答案：{response.content}")
    except Exception as e:
        print("❌ 调用失败，请检查：OPENAI_API_KEY、OPENAI_MODEL、OPENAI_BASE_URL 以及网络。")
        print(f"错误详情：{e}")
       
    
if __name__ == "__main__":
    # 执行主程序逻辑
    main()
```

## 基本概念

### 1、提示词（Prompt）

**定义**：提示词是用户输入给模型的文本指令，用于引导模型生成期望的回答。它是人机交互的桥梁，直接影响模型输出的质量和相关性。

**示例与解释**：
```python
# 基础提示词示例
question = "AI是什么？"
```
### 2、模型（Model）

**定义**：模型是指大语言模型的具体实现，如 GPT-4、DeepSeek-V3、Qwen 等。每个模型都有独特的训练数据、参数规模和能力特征。

### 3、回答（Response）

**定义**：回答是模型根据提示词生成的文本输出，是 AI 系统对用户输入的响应。回答的质量、相关性和准确性直接影响用户体验。


### 4、参数（Parameters）

**定义**：参数是控制模型生成行为的可调节设置，通过调整这些参数可以优化回答的质量、风格和特性。

**参数调优指南**：

| 参数 | 功能 | 调优建议 | 示例 | 适用场景 |
|------|------|----------|------|----------|
| Temperature | 控制回答的随机性 | RAG/事实任务：0.1-0.3<br>代码生成/严格写作：0.3-0.5<br>头脑风暴/创意写作：0.7-0.9 | `temperature = 0.7` | 需要平衡创意和准确性的场景 |
| Max Tokens | 限制回答的最大长度 | 短回答：256-512<br>标准回答：1024-2048<br>长文档：4096-8192 | `max_tokens = 8192` | 需要生成长篇内容的场景 |

**参数推荐组合（含新手预设）**：
| 场景 | Temperature(范围) | Max Tokens(范围) | 新手预设 |
|------|------------------|------------------|---------|
| 事实问答 | 0.1-0.3 | 256-512 | 0.2 / 512 |
| 代码/严谨写作 | 0.3-0.5 | 1024-2048 | 0.4 / 1024 |
| 创意写作/长文 | 0.7-0.9 | 4096-8192 | 0.8 / 4096 |

## 常见错误与快速排查 (Q/A)

- Q: 为什么提示 API Key 无效或 401？
  - A: 检查 `.env` 中 `OPENAI_API_KEY` 是否为空、是否有多余空格或换行；确保使用有效的密钥。
- Q: 请求超时或连接失败怎么办？
  - A: 确认 `OPENAI_BASE_URL` 可达；网络状况稳定后重试；必要时减少 `max_tokens` 或提高超时。
- Q: 提示模型不可用或 404？
  - A: 将 `OPENAI_MODEL` 设置为可用模型（例如 `deepseek-chat`），并参考服务商支持列表。
- Q: 代码读取到空配置/变量？
  - A: 确保 `.env` 与脚本同级目录，且运行命令在脚本所在目录执行；必要时在代码中打印关键变量进行自检。
- Q: 切换服务商后报错？
  - A: 同步更新 `OPENAI_BASE_URL`、`OPENAI_MODEL` 与鉴权方式；按“服务商切换配置速览”逐项检查。
- Q: 如何开启详细日志定位问题？
  - A: 将示例代码中的 `verbose` 设为 `True`，或使用更细粒度的异常日志打印。

## 服务商切换配置速览

为便于在不同模型提供商之间切换，建议通过统一的环境变量进行配置映射：

| 提供商 | OPENAI_BASE_URL | OPENAI_MODEL 示例 | 备注 |
|--------|------------------|-------------------|------|
| DeepSeek | https://api.deepseek.com | deepseek-chat | 与 `langchain-openai` 兼容，使用 `OPENAI_*` 命名即可 |
| OpenAI | https://api.openai.com/v1 | gpt-4o-mini | 使用官方 API Key 与默认 Base URL |
| Azure OpenAI | https://{your-resource}.openai.azure.com | {your-deployment-name} | 需使用 Azure 鉴权与部署名；API 版本请按官方要求配置 |
| Ollama（本地） | http://localhost:11434 | llama3:8b | 需切换到 `langchain-ollama` 客户端（ChatOllama），并非 `langchain-openai` |

JSON 结构化速览：
```json
{
  "providers": [
    {"name": "DeepSeek", "base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
    {"name": "OpenAI", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    {"name": "AzureOpenAI", "base_url": "https://{your-resource}.openai.azure.com", "model": "{your-deployment}", "note": "需要 Azure 鉴权与部署名"},
    {"name": "Ollama", "base_url": "http://localhost:11434", "model": "llama3:8b", "client": "langchain-ollama"}
  ]
}
```

## 错误码速查与重试建议

- 401 未授权：检查 `OPENAI_API_KEY` 是否有效、无多余空格；重新生成或刷新密钥；避免在代码或日志中泄露密钥。
- 403 禁止访问：确认账户有权限访问对应模型与区域；检查配额与账单状态。
- 404 未找到：校验 `OPENAI_MODEL` 与 `OPENAI_BASE_URL`，确保模型名与端点路径正确。
- 408/超时：网络波动或响应过长；可适当降低 `OPENAI_MAX_TOKENS`、提高超时、或重试请求。
- 429 速率限制：降低并发或请求频率，引入指数退避重试（`max_retries` + 退避间隔）。
- 500/503 服务端错误：服务端暂时不可用；重试并做好容灾（降级为轻量模型或缓存回答）。

## 官方链接

- 01-chatbots-basic： [01_chatbot_basic_cli.py 源码](https://github.com/kakaCat/langchain-learn/blob/main/01-chatbots-basic/01_chatbot_basic_cli.py)
- LangChain 文档：https://python.langchain.com/
- LangChain OpenAI 集成（Python API 索引）：https://api.python.langchain.com/
- DeepSeek API 文档：https://api-docs.deepseek.com/
- 本页快捷跳转：[#tldr](#tldr) | [#howto](#howto) | [#quickstart](#quickstart)
- OpenAI API 文档：https://platform.openai.com/docs/api-reference
- Azure OpenAI 文档：https://learn.microsoft.com/azure/ai-services/openai/
- Ollama 文档：https://ollama.com/docs

## 总结

🎉 **恭喜你完成了 LangChain 入门之旅！**

通过这个教程，你已经从一个 AI 新手成功迈出了构建智能聊天机器人的第一步。

