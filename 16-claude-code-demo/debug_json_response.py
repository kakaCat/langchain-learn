#!/usr/bin/env python3
"""
诊断 JSON 解析失败的根本原因

这个脚本会：
1. 调用 LLM 获取原始响应
2. 打印响应的详细信息（字符、编码）
3. 分析为什么 json.loads() 失败
"""

import json
import os
import re
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


def load_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)


def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        max_tokens=500,
    )


def diagnose_json_response():
    """诊断 LLM 返回的 JSON 响应"""
    load_environment()
    llm = get_llm()

    # 测试原始 prompt（和代码中一样的）
    prompt = """
你是 Lead Researcher。刚刚收到子 Agent Researcher 对"学习资源调研"的结果：
总结：找到了官方文档和社区教程
引用：rust-lang.org, community forums

请评估：
1. 该结果是否可信并可纳入最终报告
2. 是否需要追加研究（True/False）
3. 如果需要，列出新的研究方面（最多2个）

输出 JSON：
{
  "accepted": true,
  "need_more_research": false,
  "new_aspects": ["..."],
  "comment": "..."
}
"""

    print("=" * 80)
    print("🔍 诊断 LLM JSON 响应")
    print("=" * 80)
    print(f"\n📤 发送 Prompt:\n{prompt}")
    print("\n" + "=" * 80)

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content

    print(f"\n📥 LLM 原始响应:\n{content}")
    print("\n" + "=" * 80)

    # 详细分析
    print("\n🔬 详细分析:")
    print(f"1. 响应类型: {type(content)}")
    print(f"2. 响应长度: {len(content)} 字符")
    print(f"3. 前50个字符: {repr(content[:50])}")
    print(f"4. 后50个字符: {repr(content[-50:])}")

    # 检查是否包含特殊字符
    print("\n5. 特殊字符检测:")
    if "```" in content:
        print("   ⚠️ 包含 Markdown 代码块标记 ```")
    if content.startswith(" ") or content.startswith("\n"):
        print(f"   ⚠️ 开头有空白字符: {repr(content[:5])}")
    if content.endswith(" ") or content.endswith("\n"):
        print(f"   ⚠️ 结尾有空白字符: {repr(content[-5:])}")

    # 尝试不同的解析方法
    print("\n" + "=" * 80)
    print("🧪 尝试不同的解析方法:")
    print("=" * 80)

    # 方法1: 直接解析
    print("\n方法1: 直接 json.loads()")
    try:
        data = json.loads(content)
        print(f"   ✅ 成功: {data}")
    except json.JSONDecodeError as e:
        print(f"   ❌ 失败: {e}")
        print(f"   错误位置: 第 {e.lineno} 行, 第 {e.colno} 列")
        print(f"   错误内容: {repr(content[max(0, e.pos-20):e.pos+20])}")

    # 方法2: 去除空白后解析
    print("\n方法2: 去除首尾空白后解析")
    try:
        data = json.loads(content.strip())
        print(f"   ✅ 成功: {data}")
    except json.JSONDecodeError as e:
        print(f"   ❌ 失败: {e}")

    # 方法3: 提取 Markdown 代码块
    print("\n方法3: 提取 Markdown 代码块中的 JSON")
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(code_block_pattern, content, re.DOTALL)
    if matches:
        for i, match in enumerate(matches, 1):
            print(f"   发现代码块 {i}: {repr(match[:50])}")
            try:
                data = json.loads(match.strip())
                print(f"   ✅ 成功: {data}")
            except json.JSONDecodeError as e:
                print(f"   ❌ 失败: {e}")
    else:
        print("   ℹ️ 未发现 Markdown 代码块")

    # 方法4: 查找 { ... } 块
    print("\n方法4: 查找第一个 JSON 对象 { ... }")
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, content, re.DOTALL)
    if matches:
        for i, match in enumerate(matches, 1):
            print(f"   发现 JSON 对象 {i}: {repr(match[:50])}")
            try:
                data = json.loads(match)
                print(f"   ✅ 成功: {data}")
                break
            except json.JSONDecodeError as e:
                print(f"   ❌ 失败: {e}")
    else:
        print("   ℹ️ 未发现 JSON 对象")

    # 方法5: 手动清理
    print("\n方法5: 手动清理（去除非 JSON 内容）")
    try:
        # 找到第一个 { 和最后一个 }
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            cleaned = content[start:end+1]
            print(f"   清理后内容: {repr(cleaned[:100])}")
            data = json.loads(cleaned)
            print(f"   ✅ 成功: {data}")
        else:
            print("   ℹ️ 未找到 JSON 对象的边界")
    except json.JSONDecodeError as e:
        print(f"   ❌ 失败: {e}")

    print("\n" + "=" * 80)
    print("📊 诊断总结")
    print("=" * 80)
    print("\n根据以上分析，JSON 解析失败的原因可能是:")
    print("1. LLM 返回了 Markdown 代码块（包含 ``` 标记）")
    print("2. JSON 前后有额外的文本或空白")
    print("3. JSON 格式本身有问题（缺少引号、逗号等）")
    print("\n建议的修复方法:")
    print("- 改进 Prompt，明确要求只返回纯 JSON")
    print("- 使用正则提取 JSON 内容")
    print("- 使用 with_structured_output() 强制格式")


if __name__ == "__main__":
    diagnose_json_response()
