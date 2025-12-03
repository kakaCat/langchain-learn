# 03 - 语义识别与意图路由 Demo（Semantic Intent）

语义意图识别的核心是在**理解用户输入的含义后，为其匹配合适的处理流程**。典型应用包括 FAQ/搜索、任务分配、工具调度和多 Agent 协作等。本目录的 Demo 旨在给出最小可运行的骨架，并提示后续如何迭代成真实的语义路由器。本版本聚焦“纯 LLM” 方案，即利用大模型本身完成意图判断与路由，不依赖独立的 Embeddings 或向量数据库。

## 目标与要点
- 利用 LLM 少样本提示完成语义意图分类，无需额外向量检索。
- 在 Prompt 中显式定义意图、示例和输出格式，让模型生成结构化判断。
- 将分类结果映射到不同的处理链（问答、搜索、待办、计算等）。
- 记录置信度、置信阈值与兜底策略，确保对未知意图的回退体验。

## 前置准备
1. Python 3.10+，建议创建虚拟环境。
2. 安装依赖：
   ```bash
   cd 03-semantic-intent
   pip install -r requirements.txt
   ```
3. 若使用云模型（如 OpenAI、Moonshot 等），需在环境变量中配置 API Key；若使用本地部署 LLM，则需确保推理服务可访问。

## 项目结构
| 文件 | 说明 |
| --- | --- |
| `01_semantic_router_demo.py` | LLM Few-Shot 语义路由 Demo，支持交互与单次调用 |
| `requirements.txt` | Demo 所需依赖 |

## 实现思路
实现完整语义意图识别通常包含以下步骤：

1. **定义意图与 Few-Shot 示例**  
   - 例如：`问答`、`搜索`、`计算`、`待办`、`fallback` 等。  
   - 为每个意图准备 3~5 个典型用户输入示例，并标注目标动作。  
   - 样本可以动态更新（数据驱动），或写在配置文件中。

2. **构造 Prompt 与输出格式**  
   ```python
   from langchain.prompts import FewShotPromptTemplate
   from langchain_core.output_parsers import JsonOutputParser
   from langchain_openai import ChatOpenAI

   examples = [
       {"query": "帮我查下明天的天气", "intent": "搜索"},
       {"query": "2+3*4 等于多少", "intent": "计算"},
       {"query": "记一条待办，明天发周报", "intent": "待办"},
   ]

   example_prompt = """用户输入：{query}
   判定意图：{intent}"""
   few_shot = FewShotPromptTemplate(
       examples=examples,
       example_prompt=example_prompt,
       suffix="用户输入：{user_query}\n请只输出 JSON：{{\"intent\": \"...\", \"confidence\": 0-1, \"reason\": \"...\"}}",
       input_variables=["user_query"],
   )

   llm = ChatOpenAI(model="gpt-4o-mini")
   parser = JsonOutputParser()
   result = parser.parse(llm.invoke(few_shot.format(user_query=user_input)).content)
   ```
   - 使用 LLM 直接产出意图和信心（或理由）。  
   - 如果模型不支持结构化输出，可在上层解析 `intent: xxx` 形式的文本。

3. **阈值与兜底**  
   - 可以让 LLM 同时输出 `confidence`，当置信度低于阈值（如 0.6）时统一路由到 `fallback`。  
   - 或在 Prompt 中增加指令：“若不确定，请返回 `intent=fallback` 并给出原因”。  
   - 还可以二次调用 LLM，请其复核第一次结果，或结合规则做 sanity check（例如检测关键词）。

4. **路由到对应链路**  
   - `问答`：走知识库检索 + LLM 回答链。  
   - `搜索`：调用自定义搜索工具，返回摘要。  
   - `计算`：调用 Calculator Tool 或 Python REPL。  
   - `待办`：写入外部系统或保存到本地 `todo.json`。  
   - `fallback`：要求用户补充信息，或转人工处理。

5. **监控与反馈**  
   - 记录每次意图识别的输入、预测、置信度和最终动作。  
   - 对于低置信度或 fallback 的输入，收集真实标签，用于扩充 Few-Shot 示例，形成闭环迭代。

## 提示词设计建议
- **结构清晰**：Prompt 由三部分组成——意图定义、Few-Shot 示例、输出要求。示意：
  ```
  你是客服助手，需要判断用户属于下列意图之一：
  1. 搜索：需要联网查资料的问题
  2. 计算：涉及数学或代码计算
  3. 待办：让助手记录或提醒事项
  4. fallback：其他情形或不确定时

  示例：
  用户输入：帮我查一下 GPT 最新参数
  判定意图：搜索
  ...

  用户输入：{user_query}
  请输出 JSON，格式 {"intent": "...", "confidence": 0-1, "reason": "..."}
  若不确定，请设置 intent=fallback 并解释原因。
  ```
- **高亮约束**：用粗体/列表强调“只返回 JSON”“不确定时回退”等规则。
- **上下文可选**：若对话型应用，可拼接最近 N 轮对话摘要，让模型判断是否需要“追问/澄清”。
- **中文/英文一致性**：保持意图名称、示例语言和用户输入一致，避免模型切换语种导致误判。

## 代码实现示例
下面示例展示如何在 `01_semantic_router_demo.py` 中实现一个 LLM 路由器及简单的工具调度：
```python
#!/usr/bin/env python3
from typing import Any, Dict

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

INTENT_DEFS = "..."
EXAMPLES = [...]

example_prompt = PromptTemplate(
    input_variables=["query", "intent"],
    template="用户输入：{query}\n判定意图：{intent}",
)

prompt = FewShotPromptTemplate(
    prefix=INTENT_DEFS.strip(),
    examples=EXAMPLES,
    example_prompt=example_prompt,
    suffix=(
        "用户输入：{user_query}\n"
        "请输出 JSON：{\"intent\": \"...\", \"confidence\": 0-1, \"reason\": \"...\"}\n"
        "若不确定，请返回 intent=fallback，并说明理由。"
    ),
    input_variables=["user_query"],
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = JsonOutputParser()

def classify_intent(user_query: str) -> Dict[str, Any]:
    raw = llm.invoke(prompt.format(user_query=user_query))
    result = parser.parse(raw.content)
    if result["confidence"] < 0.6:
        result["intent"] = "fallback"
        result["reason"] = f"置信度过低：{result['confidence']}"
    return result

def handle_search(query: str, result: Dict[str, Any]) -> str:
    return f"[搜索工具] 模拟查询：{query}"

def handle_calculate(query: str, result: Dict[str, Any]) -> str:
    # 可替换成真实计算工具
    return f"[计算器] 暂未实现，原始输入：{query}"

TOOL_HANDLERS = {
    "搜索": handle_search,
    "计算": handle_calculate,
    "待办": lambda q, _: f"[待办] 保存：{q}",
    "fallback": lambda q, r: f"请澄清意图。模型输出：{r}",
}

def dispatch_intent(query: str, result: Dict[str, Any]) -> str:
    handler = TOOL_HANDLERS.get(result["intent"], TOOL_HANDLERS["fallback"])
    return handler(query, result)

if __name__ == "__main__":
    while True:
        text = input("User> ")
        if text.strip().lower() == "exit":
            break
        prediction = classify_intent(text)
        print("识别结果：", prediction)
        print(dispatch_intent(text, prediction))
```
- 将 `classify_intent` 的输出交给路由器即可调用不同工具链。
- 若需支持多轮对话，可在 `user_query` 中拼接上下文，或配置 `ChatPromptTemplate`。

> 在 Demo 脚本中，我们还实现了安全的简易算术解析器 `safe_math_eval` 来演示“计算”意图如何落地；真实项目可直接替换为更可靠的 Calculator Tool、Python REPL Tool 或外部服务。

## 快速运行 Demo
脚本已实现基于 LLM 的 Few-Shot 意图识别，可直接运行体验：
```bash
python 01_semantic_router_demo.py

# 单次调用
python 01_semantic_router_demo.py --oneshot "提醒我明天交周报"
```
你可以在此基础上逐步注入 Few-Shot 提示和 LLM 调度逻辑。推荐做法：
1. 在脚本中定义 `INTENTS = [...]`、示例列表以及 Prompt 模板。
2. 用 LangChain 的 LLM Chain 生成结构化意图判断，并解析为字典。
3. 将 LLM 输出的 `intent` 交给 `RouterChain` 或自定义调度器，执行对应工具链。
4. 写入日志，方便调试阈值和兜底策略。

按照上述步骤扩展后，该模块即可成为多 Agent/工作流的“语义入口”，在完整应用中承担“用户意图理解 + 工具调度”的职责。*** End Patch
