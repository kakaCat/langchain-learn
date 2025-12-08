"""
Tool Registry and Tool Implementations for DeepSeek-R1 Agent V2

This module provides a centralized registry for tools that can be used
by the agent to enhance reasoning, especially for mathematical calculations.
"""

import os
from typing import List
from langchain_core.tools import Tool


class ToolRegistry:
    """工具注册中心 - 管理所有可用的工具"""

    @staticmethod
    def get_calculator_tool() -> Tool:
        """
        获取计算器工具（用于数学验证）

        Returns:
            Tool: 安全的数学计算器工具

        Examples:
            >>> calc = ToolRegistry.get_calculator_tool()
            >>> calc.func("2 + 2")
            '计算结果: 4'
        """

        def safe_eval(expr: str) -> str:
            """
            安全的数学表达式求值

            Args:
                expr: 数学表达式字符串，如 "2 + 2" 或 "(10 + 20) * 3"

            Returns:
                计算结果或错误信息
            """
            try:
                # 清理输入
                expr = expr.strip()

                # 仅允许基本数学运算和数字
                allowed = set('0123456789+-*/(). ')
                if not all(c in allowed for c in expr):
                    return f"错误：表达式包含非法字符。仅允许: 数字、+-*/()、空格"

                # 使用 eval 但限制全局和局部变量
                result = eval(expr, {"__builtins__": {}}, {})

                return f"计算结果: {result}"

            except ZeroDivisionError:
                return "错误：除数不能为零"
            except SyntaxError as e:
                return f"错误：表达式语法错误 - {str(e)}"
            except Exception as e:
                return f"计算错误: {str(e)}"

        return Tool(
            name="calculator",
            func=safe_eval,
            description=(
                "用于精确的数学计算。"
                "输入：数学表达式（如 '2 + 2' 或 '(10 + 20) * 3'）。"
                "输出：计算结果。"
                "支持运算：加(+)、减(-)、乘(*)、除(/)、括号()。"
            )
        )

    @staticmethod
    def get_python_tool() -> Tool:
        """
        获取 Python REPL 工具（用于复杂计算和数据处理）

        注意：此工具需要 langchain-experimental 包
        使用前请确保已安装：pip install langchain-experimental

        Returns:
            Tool: Python 代码执行工具
        """
        try:
            from langchain_experimental.utilities import PythonREPL

            python_repl = PythonREPL()

            return Tool(
                name="python_repl",
                func=python_repl.run,
                description=(
                    "执行 Python 代码。用于复杂计算、数据处理或需要多步计算的场景。"
                    "输入：Python 代码字符串。"
                    "输出：代码执行结果。"
                    "注意：仅用于数值计算和数据处理，不要用于系统操作。"
                )
            )
        except ImportError:
            # 如果未安装 langchain-experimental，返回一个占位工具
            def python_placeholder(code: str) -> str:
                return (
                    "错误：Python REPL 工具未安装。"
                    "请运行：pip install langchain-experimental"
                )

            return Tool(
                name="python_repl",
                func=python_placeholder,
                description="Python REPL（需要安装 langchain-experimental）"
            )

    @staticmethod
    def get_basic_tools() -> List[Tool]:
        """
        获取基础工具集（仅计算器）

        Returns:
            List[Tool]: 工具列表
        """
        return [ToolRegistry.get_calculator_tool()]

    @staticmethod
    def get_all_tools() -> List[Tool]:
        """
        获取所有可用工具

        Returns:
            List[Tool]: 工具列表
        """
        tools = [ToolRegistry.get_calculator_tool()]

        # 尝试添加 Python REPL
        try:
            from langchain_experimental.utilities import PythonREPL
            tools.append(ToolRegistry.get_python_tool())
        except ImportError:
            print("提示：Python REPL 工具不可用（需要 langchain-experimental）")

        return tools


if __name__ == "__main__":
    # 测试工具
    print("=== 测试计算器工具 ===")
    calculator = ToolRegistry.get_calculator_tool()

    test_cases = [
        "2 + 2",
        "(10 + 20) * 3",
        "100 / 4",
        "2 ** 3",  # 应该报错（不支持幂运算）
        "1 / 0",   # 除零错误
    ]

    for expr in test_cases:
        print(f"\n输入: {expr}")
        print(f"输出: {calculator.func(expr)}")

    print("\n=== 测试工具注册表 ===")
    basic_tools = ToolRegistry.get_basic_tools()
    print(f"基础工具数量: {len(basic_tools)}")
    for tool in basic_tools:
        print(f"  - {tool.name}: {tool.description[:50]}...")

    all_tools = ToolRegistry.get_all_tools()
    print(f"\n所有工具数量: {len(all_tools)}")
    for tool in all_tools:
        print(f"  - {tool.name}")
