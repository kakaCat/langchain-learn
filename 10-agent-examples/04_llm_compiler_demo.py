#!/usr/bin/env python3
"""
Module 5: LLMCompiler 示例（将用户任务编译为可执行指令并执行）

简化版 LLMCompiler：
- Compile：将用户任务编译为可执行的指令序列（program），每条包含 op 和 args
- Execute：逐条执行指令，调度内置工具（无外部调用）并收集输出
- Verify：评估最终答案是否满足用户意图，给出简短反馈
- Finalize：输出清理后的最终答案

特性：
- 使用 LangGraph 管理工作流与状态
- 无外部工具调用，部分操作使用 LLM（如 REWRITE/EXTRACT_POINTS）
- CLI 交互，兼容 dict/数据类两种返回形态
"""
import os
import sys
import json
import ast
import operator
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

# 终端中文输出
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# 环境

def load_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

# 模型

def get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    base_url = os.getenv("OPENAI_BASE_URL")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "768"))
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,
        max_retries=3,
        request_timeout=120,
        verbose=True,
    )

# 状态

@dataclass
class AgentState:
    messages: List[BaseMessage]
    question: str
    program: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None
    verified: bool = False
    feedback: Optional[str] = None

# ========= 工具集合（无外部调用） =========

def tool_outline(topic: str) -> str:
    topic = topic.strip() or "主题"
    outline = [
        f"一、{topic}的背景与目标",
        f"二、{topic}的关键要点",
        f"三、{topic}的步骤与实施",
        f"四、注意事项与常见误区",
        f"五、总结与下一步",
    ]
    return "\n".join(f"- {line}" for line in outline)


def tool_bullet(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    bullets = [f"- {l}" for l in lines]
    return "\n".join(bullets)


def tool_rewrite(text: str, style: Optional[str] = None) -> str:
    llm = get_llm()
    prompt = (
        "请将以下文本重写为更清晰、结构化、可执行的形式。\n"
        f"目标风格：{style or '简洁、要点明确、条理清晰'}\n\n"
        f"原文：\n{text}"
    )
    ai = llm.invoke([HumanMessage(content=prompt)])
    return ai.content


def tool_extract_points(text: str, n: int = 5) -> str:
    llm = get_llm()
    prompt = (
        "请从以下文本中提炼关键要点，以列表形式输出，数量约为"
        f"{n}，每条尽量简短。\n\n{text}"
    )
    ai = llm.invoke([HumanMessage(content=prompt)])
    return ai.content

# 简单安全计算器（仅支持 + - * / 和括号）
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _eval_expr(node):
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)
    if isinstance(node, ast.Num):  # py<3.8
        return node.n
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("仅允许数字常量")
    if isinstance(node, ast.BinOp):
        left = _eval_expr(node.left)
        right = _eval_expr(node.right)
        op_type = type(node.op)
        if op_type in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[op_type](left, right)
        raise ValueError("不支持的运算符")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_expr(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return +_eval_expr(node.operand)
    if isinstance(node, ast.Call):
        raise ValueError("不允许函数调用")
    raise ValueError("不支持的表达式")


def tool_calculate(expr: str) -> str:
    try:
        node = ast.parse(expr, mode="eval")
        value = _eval_expr(node)
        return f"{expr} = {value}"
    except Exception as e:
        return f"计算失败: {str(e)}"

# ========= 编译、执行、评估节点 =========

# Compile：将用户任务编译为 program（JSON 指令序列）

def compile_node(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = (
        "你是任务编译器。请将用户任务编译为可执行的指令序列(JSON)。\n"
        "可用操作：\n"
        "- OUTLINE: 生成主题的大纲，args: {\"topic\": string}\n"
        "- BULLET: 将文本转为要点列表，args: {\"text\": string}\n"
        "- REWRITE: 重写文本，args: {\"text\": string, \"style\": string?}\n"
        "- EXTRACT_POINTS: 提炼要点，args: {\"text\": string, \"n\": number?}\n"
        "- CALCULATE: 计算简单表达式，args: {\"expr\": string}\n\n"
        "严格输出JSON：{\"program\": [{\"op\": string, \"args\": object}, ...]}。\n"
        f"用户任务：{state.question}"
    )
    ai = llm.invoke([HumanMessage(content=prompt)])
    content = ai.content

    program: List[Dict[str, Any]] = []
    try:
        # 解析可能的代码块
        text = content
        if "```" in text:
            parts = text.split("```")
            # 找到可能的 JSON 片段
            candidates = [p for p in parts if "program" in p and "{" in p]
            if candidates:
                text = candidates[0]
        data = json.loads(text)
        program = data.get("program", []) if isinstance(data, dict) else []
    except Exception:
        # 兜底：构造一个简单的 REWRITE 指令
        program = [{"op": "REWRITE", "args": {"text": state.question, "style": "结构化、要点清晰"}}]

    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"Compile 阶段：\n{json.dumps({"program": program}, ensure_ascii=False, indent=2)}"))

    return AgentState(
        messages=new_messages,
        question=state.question,
        program=program,
        outputs=[],
        final_answer=None,
        verified=False,
        feedback=None,
    )

# Execute：执行 program

def execute_node(state: AgentState) -> AgentState:
    outputs: List[str] = []
    for instr in state.program:
        op = str(instr.get("op", "")).upper()
        args = instr.get("args", {}) or {}
        try:
            if op == "OUTLINE":
                res = tool_outline(str(args.get("topic", state.question)))
            elif op == "BULLET":
                res = tool_bullet(str(args.get("text", state.question)))
            elif op == "REWRITE":
                res = tool_rewrite(str(args.get("text", state.question)), style=args.get("style"))
            elif op == "EXTRACT_POINTS":
                res = tool_extract_points(str(args.get("text", state.question)), int(args.get("n", 5)))
            elif op == "CALCULATE":
                res = tool_calculate(str(args.get("expr", "0")))
            else:
                res = f"未知操作: {op}"
        except Exception as e:
            res = f"指令执行失败({op}): {str(e)}"
        outputs.append(f"[{op}]\n{res}")

    final_answer = assemble_final_answer_from_outputs(outputs)

    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"Execute 阶段：已执行 {len(state.program)} 条指令。"))
    new_messages.append(AIMessage(content=f"执行结果汇总：\n{final_answer}"))

    return AgentState(
        messages=new_messages,
        question=state.question,
        program=state.program,
        outputs=outputs,
        final_answer=final_answer,
        verified=False,
        feedback=None,
    )

# Verify：评估最终答案是否满足用户意图

def verify_node(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = (
        "你是评估器。判断最终答案是否满足用户意图，给出简短反馈。\n"
        "严格输出：\n"
        "结论: 满足 或 不满足\n"
        "理由: <一句话>\n\n"
        f"用户任务：{state.question}\n\n"
        f"最终答案：\n{state.final_answer or ''}"
    )
    ai = llm.invoke([HumanMessage(content=prompt)])
    feedback = ai.content
    verified = ("满足" in feedback) and ("不满足" not in feedback)

    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"Verify 阶段：\n{feedback}"))

    return AgentState(
        messages=new_messages,
        question=state.question,
        program=state.program,
        outputs=state.outputs,
        final_answer=state.final_answer,
        verified=verified,
        feedback=feedback,
    )

# Finalize：输出最终答案（可做一次清理）

def finalize_node(state: AgentState) -> AgentState:
    # 可选：用一次 REWRITE 清理格式
    cleaned = tool_rewrite(state.final_answer or assemble_final_answer_from_outputs(state.outputs), style="简洁、条理清晰")
    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"最终答案：\n{cleaned}"))
    return AgentState(
        messages=new_messages,
        question=state.question,
        program=state.program,
        outputs=state.outputs,
        final_answer=cleaned,
        verified=state.verified,
        feedback=state.feedback,
    )

# 汇总输出

def assemble_final_answer_from_outputs(outputs: List[str]) -> str:
    parts = []
    for i, out in enumerate(outputs, start=1):
        parts.append(f"步骤 {i}:\n{out}")
    return "\n\n".join(parts)

# 路由

def after_compile(state: AgentState) -> str:
    return "execute"

def after_execute(state: AgentState) -> str:
    return "verify"

def after_verify(state: AgentState) -> str:
    return "finalize"

# 构建工作流

def create_llmcompiler_workflow():
    graph = StateGraph(AgentState)
    graph.add_node("compile", compile_node)
    graph.add_node("execute", execute_node)
    graph.add_node("verify", verify_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("compile")

    graph.add_conditional_edges("compile", after_compile, {"execute": "execute"})
    graph.add_conditional_edges("execute", after_execute, {"verify": "verify"})
    graph.add_conditional_edges("verify", after_verify, {"finalize": "finalize"})

    graph.add_edge("finalize", END)

    app = graph.compile()
    return app

# CLI

def main():
    load_environment()
    app = create_llmcompiler_workflow()

    print("===== LLMCompiler 演示 ======")
    print("该代理会：Compile（编译任务）→ Execute（执行指令）→ Verify（评估）→ Finalize（输出最终答案）。")
    print("每次对话独立，部分指令使用 LLM 重写/提炼，无外部工具。输入 '退出' 结束对话。\n")
    print("示例任务（适合 LLMCompiler 的场景）：")
    print("1. 将下段文字整理为要点并做一个三层大纲：<你的文本>")
    print("2. 计算表达式并将结果放入总结：表达式 3*(5+2)-8，然后整理结论")
    print("3. 提炼这段文字的5条要点，并重写为行动清单：<你的文本>\n")

    while True:
        try:
            user_input = input("你: ")
            if user_input.lower() in ["退出", "exit", "quit", "q"]:
                print("代理: 再见！")
                break

            init_state = AgentState(
                messages=[HumanMessage(content=user_input)],
                question=user_input,
            )
            final_state = app.invoke(init_state)

            # 获取消息列表（兼容 dict / 数据类）
            messages = []
            if isinstance(final_state, dict):
                messages = final_state.get("messages", [])
            else:
                messages = getattr(final_state, "messages", [])

            # 输出最后一个 AI 回复（最终答案）
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    print(f"代理: {msg.content}")
                    break
            else:
                print("代理: 对不起，未能生成答案。")
        except KeyboardInterrupt:
            print("\n代理: 对话被中断，再见！")
            break
        except Exception as e:
            print(f"代理: 发生错误: {str(e)}")

if __name__ == "__main__":
    main()