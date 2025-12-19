#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_basic_reflection_demo.py — LangChain v1 create_agent 版本（Basic Reflection）

统一工具与 CLI。该示例强调：先给出初稿答案，再基于简短反思进行修订，输出更优的最终结果。
https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/reflection/reflection.ipynb
"""

import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START, add_messages
from typing_extensions import TypedDict


def load_environment():
    """加载当前目录下的 .env 配置。"""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=env_path, override=True)  # 使用 override=True 确保覆盖系统环境变量
    print(f"✅ 已加载配置文件: {env_path}")
    print(f"   BASE_URL: {os.getenv('OPENAI_BASE_URL')}")
    print(f"   MODEL: {os.getenv('OPENAI_MODEL')}\n")

# 获取配置的语言模型
def get_llm() -> ChatOpenAI:
    """创建并返回语言模型实例。"""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
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
        "verbose": True,
        "base_url": base_url
    }

    return ChatOpenAI(**kwargs)

class BASIC_REFLECTION(TypedDict):
    messages: Annotated[list, add_messages]


def generate():
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an essay assistant tasked with writing excellent 5-paragraph essays."
            " Generate the best essay possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
    return prompt | get_llm()

def reflect():
    reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
            " Provide detailed recommendations, including requests for length, depth, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
    return reflection_prompt | get_llm()

def generation_node(state: BASIC_REFLECTION) -> BASIC_REFLECTION:
    # 使用可运行链实例并同步调用，传入 messages 映射
    # 调试打印：生成输入消息
    try:
        print("[Generate] Input messages:", [(m.type, getattr(m, "content", None)) for m in state["messages"]])
    except Exception:
        print("[Generate] Input messages (raw):", state["messages"])
    msg = generate().invoke({"messages": state["messages"]})
    # 调试打印：生成输出内容
    try:
        print("[Generate] Output:", getattr(msg, "content", str(msg)))
    except Exception:
        print("[Generate] Output (raw):", msg)

    state["messages"] = [msg]
    return state


def reflection_node(state: BASIC_REFLECTION) -> BASIC_REFLECTION:
    # Other messages we need to adjust
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    translated = [state["messages"][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]
    # 调试打印：反思输入消息
    try:
        print("[Reflect] Input messages:", [(m.type, getattr(m, "content", None)) for m in translated])
    except Exception:
        print("[Reflect] Input messages (raw):", translated)
    # 使用可运行链实例并同步调用，传入 messages 映射
    res = reflect().invoke({"messages": translated})
    # 调试打印：反思输出内容
    try:
        print("[Reflect] Output:", getattr(res, "content", str(res)))
    except Exception:
        print("[Reflect] Output (raw):", res)

    state["messages"] = [HumanMessage(content=res.content)]
    return state



def should_continue(state: BASIC_REFLECTION):
    msgs = state.get("messages", [])
    msg_len = len(msgs)
    try:
        last = msgs[-1]
        last_summary = (getattr(last, "type", "unknown"), getattr(last, "content", None))
    except Exception:
        last_summary = ("unknown", None)
    decision = END if msg_len > 3 else "reflect"
    print(f"[ShouldContinue] messages={msg_len}, last={last_summary}, decision={decision}")
    return decision

def create_workflow():
    """构建并编译工作流，生成可执行 `app` 与工作流图。"""

    workflow = StateGraph(BASIC_REFLECTION)
    workflow.add_node("generate", generation_node)
    workflow.add_node("reflect", reflection_node)
    workflow.add_edge(START, "generate")
    workflow.add_conditional_edges("generate", should_continue)
    workflow.add_edge("reflect", "generate")

    app = workflow.compile()

    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "blog" / "basic_reflection.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print("正在生成流程图...")
        img_bytes = app.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        print(f"✅ 流程图已保存至: {output_path}")
    except Exception as e:
        print(f"生成流程图图片失败: {e}")
        # Fallback: save mermaid code
        mermaid_path = output_path.with_suffix(".mmd")
        with open(mermaid_path, "w") as f:
            f.write(app.get_graph().draw_mermaid())
        print(f"⚠️ 已保存 Mermaid 代码至: {mermaid_path} (可使用 https://mermaid.live 查看)")
  

    return app


def main():
    load_environment()
    app = create_workflow()
    initial_state = {"messages":[
            HumanMessage(
                content="撰写一篇关于《小王子》的现实意义及其在现代生活中的启示的文章，要求使用中文"
            )
        ] }
    final_state = app.invoke(initial_state)
    print("最终结果:", final_state.get("messages"))


if __name__ == "__main__":
    main()
