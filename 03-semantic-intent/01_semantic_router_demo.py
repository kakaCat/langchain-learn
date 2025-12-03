#!/usr/bin/env python3
"""
Module 3: Semantic Router Demo
使用纯 LLM 的少样本提示来完成语义意图识别与路由。
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

INTENT_DEFS = """
你负责判断用户输入属于以下意图之一：
1. 搜索：需要联网查询的信息
2. 计算：数学、公式或代码计算
3. 待办：记录提醒或代办事项
4. fallback：不确定或超出范围
"""

EXAMPLES = [
    {"query": "帮我查下今天北京的天气", "intent": "搜索"},
    {"query": "12*8+5 等于多少", "intent": "计算"},
    {"query": "提醒我明天 9 点发日报", "intent": "待办"},
]

ROUTE_HINTS = {
    "搜索": "调用搜索/联网工具，返回最新信息。",
    "计算": "使用 Calculator Tool 或 Python REPL 计算结果。",
    "待办": "写入待办系统或保存到 todo.json 等外部存储。",
    "fallback": "请用户澄清需求，或转人工处理。",
}

TODO_IN_MEMORY: list[str] = []
SAFE_OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return float(node.n)
    raise ValueError("仅支持基础算术表达式。")


def safe_math_eval(expression: str) -> float:
    """安全解析基础算术表达式，避免直接 eval。"""
    tree = ast.parse(expression, mode="eval")
    return _eval_ast(tree.body)


def handle_search(user_query: str, result: IntentResult | None = None) -> str:
    return f"[搜索工具] 模拟联网查询：{user_query}"


def handle_calculate(user_query: str, result: IntentResult | None = None) -> str:
    try:
        value = safe_math_eval(user_query)
        return f"[计算器] 计算结果：{value}"
    except ValueError as err:
        return f"[计算器] 无法解析表达式：{err}"


def handle_todo(user_query: str, result: IntentResult | None = None) -> str:
    TODO_IN_MEMORY.append(user_query)
    return f"[待办] 已保存：{user_query}（当前共 {len(TODO_IN_MEMORY)} 条）"


def handle_fallback(user_query: str, result: IntentResult | None = None) -> str:
    reason = result.reason if result else "无"
    return f"[Fallback] 暂不处理，请澄清意图。模型解释：{reason}"


TOOL_HANDLERS = {
    "搜索": handle_search,
    "计算": handle_calculate,
    "待办": handle_todo,
    "fallback": handle_fallback,
}


@dataclass
class IntentResult:
    intent: str
    confidence: float
    reason: str

    def to_json(self) -> str:
        return json.dumps(
            {"intent": self.intent, "confidence": self.confidence, "reason": self.reason},
            ensure_ascii=False,
        )


class LLMIntentRouter:
    """LLM Few-Shot 语义路由器。"""

    def __init__(self, model_name: str, confidence_threshold: float = 0.6) -> None:
        self.confidence_threshold = confidence_threshold
        self.prompt = self._build_prompt()
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.parser = JsonOutputParser()

    @staticmethod
    def _build_prompt() -> FewShotPromptTemplate:
        example_prompt = PromptTemplate(
            input_variables=["query", "intent"],
            template="用户输入：{query}\n判定意图：{intent}",
        )
        return FewShotPromptTemplate(
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

    def classify(self, user_query: str) -> IntentResult:
        """调用 LLM 并解析意图。"""
        completion = self.llm.invoke(self.prompt.format(user_query=user_query))
        try:
            parsed: Dict[str, Any] = self.parser.parse(completion.content)
        except OutputParserException:
            # LLM 偶尔输出格式错误，做兜底。
            parsed = {"intent": "fallback", "confidence": 0.0, "reason": completion.content}

        intent = str(parsed.get("intent", "fallback")).strip()
        try:
            confidence = float(parsed.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        reason = str(parsed.get("reason", "")).strip() or "模型未提供解释。"

        if confidence < self.confidence_threshold and intent != "fallback":
            reason = f"置信度过低（{confidence:.2f}），转为 fallback。原因：{reason}"
            intent = "fallback"

        return IntentResult(intent=intent, confidence=confidence, reason=reason)


def dispatch_intent(user_query: str, result: IntentResult) -> str:
    handler = TOOL_HANDLERS.get(result.intent, handle_fallback)
    return handler(user_query, result)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM 语义路由 Demo")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="可用的聊天模型名称，需确保已配置对应 API Key。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="置信度阈值，低于该值将回退到 fallback。",
    )
    parser.add_argument(
        "--oneshot",
        help="直接传入一句用户输入，输出一次意图识别结果后退出。",
    )
    return parser.parse_args(argv)


def pretty_route(result: IntentResult) -> str:
    hint = ROUTE_HINTS.get(result.intent, "尚未配置的意图类型，可自定义处理逻辑。")
    return f"[{result.intent}] {hint}\nreason={result.reason}, confidence={result.confidence:.2f}"


def repl(router: LLMIntentRouter) -> None:
    print("请输入内容，输入 exit 退出：")
    while True:
        try:
            user_text = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("已退出。")
            return
        result = router.classify(user_text)
        action = dispatch_intent(user_text, result)
        print(pretty_route(result))
        print(action)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    router = LLMIntentRouter(model_name=args.model, confidence_threshold=args.threshold)
    if args.oneshot:
        result = router.classify(args.oneshot)
        print(result.to_json())
        print(pretty_route(result))
        print(dispatch_intent(args.oneshot, result))
        return
    repl(router)


if __name__ == "__main__":
    main(sys.argv[1:])
