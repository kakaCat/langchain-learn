#!/usr/bin/env python3
"""
14 - Claude Code Style Tool Usage Demo

演示 Claude Code 的核心工具使用能力：
1. 文件操作（Read, Write, Edit, Glob）
2. Bash 命令执行
3. Web 搜索
4. 工具组合使用

这是 Claude Code 最核心的能力 - 通过工具与环境交互。
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import END, StateGraph
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field


def load_environment() -> None:
    load_dotenv(
        dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True
    )


def get_llm(model: Optional[str] = None, temperature: float = 0) -> object:
    provider = os.getenv("LLM_PROVIDER", "").lower()
    use_ollama = provider in {"ollama", "local"} or not os.getenv("OPENAI_API_KEY")

    if use_ollama:
        model_name = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, base_url=base_url, temperature=temperature)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=2000,
        )


# ============================================================================
# Claude Code 风格的工具定义
# ============================================================================


class FileTools:
    """文件操作工具集"""

    @staticmethod
    def read_file(file_path: str) -> str:
        """读取文件内容"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return f"✓ 读取成功 ({len(content)} 字符):\n{content[:500]}..."
        except Exception as e:
            return f"✗ 读取失败: {str(e)}"

    @staticmethod
    def write_file(args_json: str) -> str:
        """写入文件

        Args:
            args_json: JSON 格式参数，包含 file_path 和 content
        """
        try:
            args = json.loads(args_json)
            file_path = args["file_path"]
            content = args["content"]

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"✓ 写入成功: {file_path} ({len(content)} 字符)"
        except Exception as e:
            return f"✗ 写入失败: {str(e)}"

    @staticmethod
    def glob_files(pattern: str) -> str:
        """使用 glob 模式查找文件"""
        try:
            from pathlib import Path
            import glob

            matches = glob.glob(pattern, recursive=True)
            if matches:
                result = f"✓ 找到 {len(matches)} 个文件:\n"
                for m in matches[:10]:  # 只显示前 10 个
                    result += f"  - {m}\n"
                if len(matches) > 10:
                    result += f"  ... 还有 {len(matches) - 10} 个文件"
                return result
            else:
                return f"✗ 未找到匹配 '{pattern}' 的文件"
        except Exception as e:
            return f"✗ 搜索失败: {str(e)}"


class BashTool:
    """Bash 命令执行工具"""

    @staticmethod
    def run_command(command: str) -> str:
        """执行 bash 命令"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

            output = f"✓ 命令执行完成\n"
            output += f"退出码: {result.returncode}\n"

            if result.stdout:
                output += f"\n标准输出:\n{result.stdout[:500]}"
            if result.stderr:
                output += f"\n标准错误:\n{result.stderr[:500]}"

            return output
        except subprocess.TimeoutExpired:
            return "✗ 命令超时（30秒）"
        except Exception as e:
            return f"✗ 执行失败: {str(e)}"


def create_tool_registry() -> List[Tool]:
    """创建工具注册表"""
    tools = []

    # 文件读取工具
    tools.append(
        Tool(
            name="read_file",
            description="读取文件内容。输入：文件路径（字符串）",
            func=FileTools.read_file,
        )
    )

    # 文件写入工具
    tools.append(
        Tool(
            name="write_file",
            description='写入文件。输入：JSON 字符串，格式 {"file_path": "路径", "content": "内容"}',
            func=FileTools.write_file,
        )
    )

    # 文件搜索工具
    tools.append(
        Tool(
            name="glob",
            description="使用 glob 模式搜索文件。输入：glob 模式（如 '*.py', '**/*.txt'）",
            func=FileTools.glob_files,
        )
    )

    # Bash 工具
    tools.append(
        Tool(
            name="bash",
            description="执行 bash 命令。输入：命令字符串（如 'ls -la', 'git status'）",
            func=BashTool.run_command,
        )
    )

    # Web 搜索工具
    try:
        search = DuckDuckGoSearchRun()
        tools.append(
            Tool(
                name="web_search",
                description="搜索互联网获取最新信息。输入：搜索关键词",
                func=search.run,
            )
        )
    except Exception as e:
        print(f"[警告] Web 搜索工具加载失败: {e}")

    return tools


# ============================================================================
# 示例任务
# ============================================================================


def demo_file_operations():
    """演示文件操作"""
    print("\n" + "=" * 80)
    print("示例 1: 文件操作工具")
    print("=" * 80)

    load_environment()
    tools = create_tool_registry()
    llm = get_llm()

    # 创建 ReAct Agent
    react_prompt = PromptTemplate.from_template(
        """你是一个文件操作助手。请使用提供的工具完成用户任务。

可用工具：
{tools}

工具名称：{tool_names}

使用以下格式：

Question: 用户的问题
Thought: 你的思考过程
Action: 要使用的工具名称
Action Input: 工具的输入
Observation: 工具的输出
... (可以重复 Thought/Action/Action Input/Observation)
Thought: 我现在知道最终答案了
Final Answer: 给用户的最终答案

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
    )

    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, max_iterations=5, handle_parsing_errors=True
    )

    # 任务：创建一个测试文件并读取
    task = "创建一个名为 'test_output.txt' 的文件，内容是 'Hello from Claude Code Demo!'，然后读取这个文件确认内容。"

    print(f"\n任务: {task}\n")

    try:
        result = agent_executor.invoke({"input": task})
        print(f"\n最终结果:\n{result['output']}")
    except Exception as e:
        print(f"\n执行出错: {e}")


def demo_bash_commands():
    """演示 Bash 命令执行"""
    print("\n" + "=" * 80)
    print("示例 2: Bash 命令工具")
    print("=" * 80)

    load_environment()
    tools = create_tool_registry()
    llm = get_llm()

    react_prompt = PromptTemplate.from_template(
        """你是一个系统管理助手。请使用 bash 工具完成用户任务。

可用工具：
{tools}

工具名称：{tool_names}

使用格式：
Question: 用户问题
Thought: 思考
Action: bash
Action Input: 命令
Observation: 输出
Final Answer: 答案

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
    )

    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, max_iterations=3, handle_parsing_errors=True
    )

    # 任务：检查当前目录和 Python 版本
    task = "检查当前工作目录，并获取 Python 版本信息"

    print(f"\n任务: {task}\n")

    try:
        result = agent_executor.invoke({"input": task})
        print(f"\n最终结果:\n{result['output']}")
    except Exception as e:
        print(f"\n执行出错: {e}")


def demo_web_search():
    """演示 Web 搜索"""
    print("\n" + "=" * 80)
    print("示例 3: Web 搜索工具")
    print("=" * 80)

    load_environment()
    tools = create_tool_registry()
    llm = get_llm()

    react_prompt = PromptTemplate.from_template(
        """你是一个研究助手。请使用 web_search 工具查找最新信息。

可用工具：
{tools}

工具名称：{tool_names}

使用格式：
Question: 用户问题
Thought: 思考
Action: web_search
Action Input: 搜索关键词
Observation: 搜索结果
Final Answer: 基于搜索结果的答案

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
    )

    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, max_iterations=3, handle_parsing_errors=True
    )

    # 任务：搜索最新的 AI 技术趋势
    task = "搜索 2025 年最新的 AI 技术趋势"

    print(f"\n任务: {task}\n")

    try:
        result = agent_executor.invoke({"input": task})
        print(f"\n最终结果:\n{result['output']}")
    except Exception as e:
        print(f"\n执行出错: {e}")


def demo_combined_tools():
    """演示工具组合使用"""
    print("\n" + "=" * 80)
    print("示例 4: 组合使用多种工具")
    print("=" * 80)

    load_environment()
    tools = create_tool_registry()
    llm = get_llm()

    react_prompt = PromptTemplate.from_template(
        """你是一个全能助手。请使用所有可用工具完成复杂任务。

可用工具：
{tools}

工具名称：{tool_names}

使用格式（ReAct）：
Question: 用户问题
Thought: 分析任务，决定使用哪个工具
Action: 工具名称
Action Input: 工具输入
Observation: 工具输出
... (重复直到完成)
Final Answer: 最终答案

重要提示：
1. 可以多次使用不同工具
2. 合理组合工具完成复杂任务
3. 确保每步操作都有意义

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
    )

    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, max_iterations=10, handle_parsing_errors=True
    )

    # 复杂任务：搜索信息 + 文件操作
    task = """完成以下任务：
1. 使用 glob 查找当前目录下的所有 Python 文件
2. 创建一个报告文件 'python_files_report.txt'，列出找到的文件
3. 使用 bash 统计找到了多少个文件"""

    print(f"\n任务: {task}\n")

    try:
        result = agent_executor.invoke({"input": task})
        print(f"\n最终结果:\n{result['output']}")
    except Exception as e:
        print(f"\n执行出错: {e}")


# ============================================================================
# 主入口
# ============================================================================


def main():
    """运行所有示例"""
    print("\n" + "=" * 80)
    print("Claude Code Style - 工具使用演示")
    print("=" * 80)
    print("\n这个演示展示了 Claude Code 的核心能力：通过工具与环境交互")
    print("包括：文件操作、Bash 命令、Web 搜索、工具组合\n")

    demos = [
        ("文件操作", demo_file_operations),
        ("Bash 命令", demo_bash_commands),
        ("Web 搜索", demo_web_search),
        ("工具组合", demo_combined_tools),
    ]

    print("选择要运行的示例：")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  {len(demos) + 1}. 运行所有示例")
    print("  0. 退出")

    try:
        choice = input("\n请输入选择 (0-5): ").strip()

        if choice == "0":
            print("退出演示")
            return
        elif choice == str(len(demos) + 1):
            for name, demo_func in demos:
                demo_func()
        elif choice.isdigit() and 1 <= int(choice) <= len(demos):
            demos[int(choice) - 1][1]()
        else:
            print("无效选择")
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
