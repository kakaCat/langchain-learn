"""
Think Tag Parser for DeepSeek-R1 Style Responses

This module provides utilities to parse <think> and <answer> tags
from LLM outputs, mimicking DeepSeek-R1 and OpenAI o1 style reasoning.
"""

import re
from typing import Dict


class ThinkTagParser:
    """解析 <think> 和 <answer> 标签"""

    @staticmethod
    def parse(text: str) -> Dict[str, str]:
        """
        从文本中提取 think 和 answer 内容

        Args:
            text: 包含 <think> 和 <answer> 标签的文本

        Returns:
            {"think": "...", "answer": "..."}
            如果没有标签，整个文本被视为 answer

        Examples:
            >>> parser = ThinkTagParser()
            >>> result = parser.parse("<think>推理过程</think><answer>42</answer>")
            >>> result["answer"]
            '42'
        """
        # 提取 <think>...</think>
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
        think = think_match.group(1).strip() if think_match else ""

        # 提取 <answer>...</answer>
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        answer = answer_match.group(1).strip() if answer_match else ""

        # 如果没有标签，整个文本视为 answer
        if not think and not answer_match:
            answer = text.strip()

        return {"think": think, "answer": answer}

    @staticmethod
    def has_think_tags(text: str) -> bool:
        """
        检查文本是否包含 think 标签

        Args:
            text: 要检查的文本

        Returns:
            True 如果包含 <think> 标签
        """
        return bool(re.search(r'<think>', text, re.IGNORECASE))

    @staticmethod
    def has_answer_tags(text: str) -> bool:
        """
        检查文本是否包含 answer 标签

        Args:
            text: 要检查的文本

        Returns:
            True 如果包含 <answer> 标签
        """
        return bool(re.search(r'<answer>', text, re.IGNORECASE))

    @staticmethod
    def extract_final_answer(text: str) -> str:
        """
        提取最终答案（优先从 answer 标签，否则取整个文本）

        Args:
            text: 包含答案的文本

        Returns:
            最终答案字符串
        """
        parsed = ThinkTagParser.parse(text)
        return parsed["answer"] if parsed["answer"] else text.strip()


if __name__ == "__main__":
    # 测试示例
    parser = ThinkTagParser()

    # 测试案例 1: 完整的 think + answer 标签
    test1 = """
    <think>
    让我计算一下：
    2 + 2 = 4
    </think>

    <answer>
    4
    </answer>
    """
    result1 = parser.parse(test1)
    print("测试 1 (完整标签):")
    print(f"  Think: {result1['think'][:50]}...")
    print(f"  Answer: {result1['answer']}")

    # 测试案例 2: 仅有 answer 标签
    test2 = "<answer>这是答案</answer>"
    result2 = parser.parse(test2)
    print("\n测试 2 (仅 answer):")
    print(f"  Answer: {result2['answer']}")

    # 测试案例 3: 没有标签
    test3 = "直接回答，没有标签"
    result3 = parser.parse(test3)
    print("\n测试 3 (无标签):")
    print(f"  Answer: {result3['answer']}")

    # 测试案例 4: 检测标签存在性
    print(f"\n测试 4 (标签检测):")
    print(f"  test1 有 think 标签: {parser.has_think_tags(test1)}")
    print(f"  test2 有 think 标签: {parser.has_think_tags(test2)}")
    print(f"  test1 有 answer 标签: {parser.has_answer_tags(test1)}")
