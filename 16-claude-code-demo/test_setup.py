#!/usr/bin/env python3
"""
测试脚本 - 验证增强版 Agent 的环境配置

运行此脚本以检查：
1. 依赖包是否正确安装
2. 环境变量是否配置
3. LLM 连接是否正常
4. 工具是否可用
"""

import os
import sys
from typing import List, Tuple


def check_imports() -> List[Tuple[str, bool, str]]:
    """检查必需的包是否已安装"""
    results = []

    packages = [
        ("langchain", "LangChain 核心库"),
        ("langchain_core", "LangChain 核心组件"),
        ("langgraph", "LangGraph 工作流引擎"),
        ("pydantic", "数据验证库"),
        ("dotenv", "环境变量管理"),
        ("langchain_openai", "OpenAI 集成（如果使用 OpenAI）"),
        ("langchain_ollama", "Ollama 集成（如果使用本地模型）"),
        ("langchain_community", "社区工具集成"),
        ("duckduckgo_search", "Web 搜索工具"),
        ("langchain_experimental", "实验性工具（Python REPL）"),
    ]

    for package, description in packages:
        try:
            __import__(package)
            results.append((package, True, description))
        except ImportError:
            results.append((package, False, description))

    return results


def check_env_vars() -> List[Tuple[str, bool, str]]:
    """检查环境变量配置"""
    results = []

    # 检查 .env 文件
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    env_exists = os.path.exists(env_path)
    results.append((".env 文件", env_exists, "位置: 10-agent-examples/.env"))

    if env_exists:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path, override=False)

    # 检查 LLM 配置
    provider = os.getenv("LLM_PROVIDER", "").lower()
    results.append(
        ("LLM_PROVIDER", bool(provider), f"当前值: {provider or '（未设置）'}")
    )

    if provider in {"openai", ""}:
        openai_key = os.getenv("OPENAI_API_KEY")
        results.append(
            ("OPENAI_API_KEY", bool(openai_key), "OpenAI API 密钥" + (" ✓" if openai_key else ""))
        )

        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        results.append(("OPENAI_MODEL", True, f"模型: {openai_model}"))

    if provider in {"ollama", "local"}:
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        results.append(("OLLAMA_MODEL", True, f"模型: {ollama_model}"))

        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        results.append(("OLLAMA_BASE_URL", True, f"地址: {ollama_url}"))

    return results


def check_llm_connection() -> Tuple[bool, str]:
    """测试 LLM 连接"""
    try:
        from dotenv import load_dotenv

        load_dotenv(
            dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False
        )

        provider = os.getenv("LLM_PROVIDER", "").lower()
        use_ollama = provider in {"ollama", "local"} or not os.getenv("OPENAI_API_KEY")

        if use_ollama:
            from langchain_ollama import ChatOllama

            model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

            llm = ChatOllama(
                model=model_name, base_url=base_url, temperature=0, timeout=10
            )
        else:
            from langchain_openai import ChatOpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=0,
                max_tokens=50,
                timeout=10,
            )

        from langchain_core.messages import HumanMessage

        response = llm.invoke([HumanMessage(content="Say 'OK' if you can hear me.")])
        return True, f"连接成功！响应: {response.content[:50]}"

    except Exception as e:
        return False, f"连接失败: {str(e)}"


def check_tools() -> List[Tuple[str, bool, str]]:
    """检查工具可用性"""
    results = []

    # Web 搜索工具
    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        search = DuckDuckGoSearchRun()
        result = search.run("test")
        results.append(("Web Search (DuckDuckGo)", True, "搜索功能正常"))
    except Exception as e:
        results.append(("Web Search (DuckDuckGo)", False, f"失败: {str(e)[:50]}"))

    # Python REPL 工具
    try:
        from langchain_experimental.utilities import PythonREPL

        repl = PythonREPL()
        result = repl.run("1 + 1")
        is_ok = "2" in str(result)
        results.append(
            ("Python REPL", is_ok, "代码执行正常" if is_ok else "输出异常")
        )
    except Exception as e:
        results.append(("Python REPL", False, f"失败: {str(e)[:50]}"))

    # 文件读取工具
    try:
        test_file = __file__
        with open(test_file, "r") as f:
            content = f.read(100)
        results.append(("File Read", True, "文件读取正常"))
    except Exception as e:
        results.append(("File Read", False, f"失败: {str(e)[:50]}"))

    return results


def print_results(title: str, results: List[Tuple[str, bool, str]]):
    """打印检查结果"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

    for item, status, description in results:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {item:<30} {description}")


def main():
    print("\n" + "="*70)
    print("Claude Code Style Enhanced - 环境配置检查")
    print("="*70)

    # 1. 检查依赖包
    import_results = check_imports()
    print_results("1. 依赖包检查", import_results)

    # 统计失败的包
    failed_imports = [item for item, status, _ in import_results if not status]
    if failed_imports:
        print(f"\n⚠️  缺少 {len(failed_imports)} 个依赖包，请运行:")
        print(f"   pip install {' '.join(failed_imports)}")

    # 2. 检查环境变量
    env_results = check_env_vars()
    print_results("2. 环境变量检查", env_results)

    # 3. 检查 LLM 连接
    print(f"\n{'='*70}")
    print("3. LLM 连接测试")
    print(f"{'='*70}")

    if all(status for _, status, _ in import_results[:5]):  # 核心包都安装了
        llm_ok, llm_msg = check_llm_connection()
        status_icon = "✅" if llm_ok else "❌"
        print(f"{status_icon} LLM 连接: {llm_msg}")
    else:
        print("⏭️  跳过（缺少核心依赖）")

    # 4. 检查工具
    tool_results = check_tools()
    print_results("4. 工具可用性检查", tool_results)

    # 总结
    print(f"\n{'='*70}")
    print("总结")
    print(f"{'='*70}")

    all_checks = import_results + env_results + tool_results
    passed = sum(1 for _, status, _ in all_checks if status)
    total = len(all_checks)

    if llm_ok:
        passed += 1
        total += 1

    print(f"通过: {passed}/{total} 项检查")

    if passed == total:
        print("\n🎉 所有检查通过！你可以运行增强版 Agent 了：")
        print("   python 11_claude_code_style_enhanced.py")
    else:
        print("\n⚠️  部分检查未通过，请查看上述详情并修复问题。")
        print("\n常见问题解决：")
        print("1. 缺少依赖包 → pip install -r requirements.txt")
        print("2. 缺少 .env 文件 → 参考 README_enhanced.md 创建配置")
        print("3. LLM 连接失败 → 检查 API Key 或 Ollama 服务是否运行")
        print("4. Web 搜索失败 → 检查网络连接或安装 duckduckgo-search")


if __name__ == "__main__":
    main()
