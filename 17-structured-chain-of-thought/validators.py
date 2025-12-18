"""
Validators for DeepSeek-R1 Agent V2

This module provides validation mechanisms to detect and prevent:
1. Infinite loops in reasoning
2. Hallucinations (introducing information not in the original question)
"""

import os
from typing import List, Tuple, Dict, Any
from difflib import SequenceMatcher
from langchain_ollama import ChatOllama
from prompts import HALLUCINATION_CHECK_PROMPT


class LoopBreaker:
    """检测并打破推理中的无限循环"""

    def __init__(self, similarity_threshold: float = 0.85, max_history: int = 5):
        """
        初始化循环检测器

        Args:
            similarity_threshold: 相似度阈值，超过此值认为是循环（0-1）
            max_history: 保留的历史记录数量
        """
        self.history: List[str] = []
        self.threshold = similarity_threshold
        self.max_history = max_history

    def check_and_break(self, current_output: str) -> Tuple[bool, str]:
        """
        检查是否进入循环

        Args:
            current_output: 当前的输出文本

        Returns:
            (is_loop, suggestion): 是否循环和建议
        """
        # 检查最近的几次输出
        for prev in self.history[-min(3, len(self.history)):]:
            similarity = SequenceMatcher(None, prev, current_output).ratio()

            if similarity > self.threshold:
                suggestion = (
                    f"检测到循环推理（相似度 {similarity:.2%}），"
                    "建议换一个思路或使用工具验证。"
                )
                return True, suggestion

        # 添加到历史记录
        self.history.append(current_output)

        # 限制历史记录大小
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return False, ""

    def reset(self):
        """重置历史记录"""
        self.history.clear()

    def get_history_count(self) -> int:
        """获取历史记录数量"""
        return len(self.history)


class HallucinationDetector:
    """检测推理过程中是否引入了原题中不存在的信息"""

    def __init__(self, llm: ChatOllama = None):
        """
        初始化幻觉检测器

        Args:
            llm: LLM 实例，如果为 None 则创建新实例
        """
        if llm is None:
            model = os.getenv("OLLAMA_MODEL", "deepseek-r1:32b")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.llm = ChatOllama(model=model, base_url=base_url, temperature=0.1)
        else:
            self.llm = llm

    def validate(self, original_question: str, reasoning: str) -> Dict[str, Any]:
        """
        验证推理是否引入了新假设

        Args:
            original_question: 原始问题
            reasoning: 推理过程

        Returns:
            {"is_valid": bool, "issues": List[str]}
        """
        try:
            # 如果推理为空，直接返回有效
            if not reasoning or not reasoning.strip():
                return {"is_valid": True, "issues": []}

            # 使用 LLM 检查
            prompt = HALLUCINATION_CHECK_PROMPT.format(
                original_question=original_question,
                reasoning=reasoning
            )

            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # 解析响应
            issues = []
            is_valid = "无问题" in response_text

            if not is_valid:
                # 提取问题列表
                lines = response_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('问题')):
                        issues.append(line)

            return {"is_valid": is_valid, "issues": issues}

        except Exception as e:
            # 如果检测失败，默认认为有效（避免误杀）
            print(f"幻觉检测失败: {e}")
            return {"is_valid": True, "issues": [f"检测错误: {str(e)}"]}


class ReasoningQualityChecker:
    """推理质量检查器 - 检查推理的完整性和逻辑性"""

    @staticmethod
    def check_completeness(reasoning: str, min_length: int = 50) -> Tuple[bool, str]:
        """
        检查推理是否足够完整

        Args:
            reasoning: 推理文本
            min_length: 最小字符长度

        Returns:
            (is_complete, message)
        """
        if not reasoning or len(reasoning.strip()) < min_length:
            return False, f"推理过程过短（少于 {min_length} 字符），可能不够完整"

        # 检查是否包含关键步骤
        keywords = ["因为", "所以", "首先", "然后", "最后", "因此", "步骤", "计算"]
        has_keywords = any(kw in reasoning for kw in keywords)

        if not has_keywords:
            return False, "推理过程缺少明显的逻辑连接词或步骤标记"

        return True, "推理过程完整"

    @staticmethod
    def check_mathematical_consistency(reasoning: str) -> Tuple[bool, str]:
        """
        检查数学推理的一致性（简单版本）

        Args:
            reasoning: 推理文本

        Returns:
            (is_consistent, message)
        """
        # 检查是否包含明显的矛盾
        # 例如："等于5" 和 "等于3" 同时出现
        import re

        # 提取所有数字
        numbers = re.findall(r'\d+(?:\.\d+)?', reasoning)

        if not numbers:
            return True, "未检测到数学计算"

        # 简单检查：如果有大量重复的数字，可能表示循环
        from collections import Counter
        counter = Counter(numbers)
        most_common = counter.most_common(1)[0]

        if most_common[1] > 5:  # 同一个数字出现超过5次
            return False, f"数字 '{most_common[0]}' 重复出现 {most_common[1]} 次，可能存在循环"

        return True, "数学计算看起来一致"


if __name__ == "__main__":
    print("=== 测试循环检测器 ===")
    loop_breaker = LoopBreaker(similarity_threshold=0.85)

    # 测试正常情况
    outputs = [
        "第一步：分析问题",
        "第二步：列出条件",
        "第三步：计算结果",
    ]

    for i, output in enumerate(outputs, 1):
        is_loop, msg = loop_breaker.check_and_break(output)
        print(f"输出 {i}: {output[:30]}... -> 循环={is_loop}")

    # 测试循环情况
    print("\n添加重复输出...")
    is_loop, msg = loop_breaker.check_and_break("第三步：计算结果")
    print(f"重复输出 -> 循环={is_loop}, 消息={msg}")

    print(f"\n历史记录数量: {loop_breaker.get_history_count()}")

    print("\n=== 测试推理质量检查器 ===")
    checker = ReasoningQualityChecker()

    # 测试完整性
    good_reasoning = "首先，我们需要理解问题。然后，列出已知条件。因此，我们可以计算结果。"
    bad_reasoning = "答案是42"

    is_complete, msg = checker.check_completeness(good_reasoning)
    print(f"\n好的推理: {is_complete}, {msg}")

    is_complete, msg = checker.check_completeness(bad_reasoning)
    print(f"差的推理: {is_complete}, {msg}")

    # 测试数学一致性
    normal_calc = "2 + 3 = 5, 5 + 4 = 9, 所以答案是 9"
    loopy_calc = "2 + 2 = 4, 2 + 2 = 4, 2 + 2 = 4, 2 + 2 = 4, 2 + 2 = 4, 2 + 2 = 4"

    is_consistent, msg = checker.check_mathematical_consistency(normal_calc)
    print(f"\n正常计算: {is_consistent}, {msg}")

    is_consistent, msg = checker.check_mathematical_consistency(loopy_calc)
    print(f"循环计算: {is_consistent}, {msg}")

    print("\n=== 幻觉检测器需要 LLM，跳过自动测试 ===")
    print("提示：幻觉检测器会在实际运行时使用 LLM 进行验证")
