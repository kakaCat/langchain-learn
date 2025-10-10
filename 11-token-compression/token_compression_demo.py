#!/usr/bin/env python3
"""
Module 9: Token Compression Demo
实现多种 Token 压缩策略与上下文管理技术
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CompressionResult:
    """压缩结果数据类"""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    compression_strategy: str


class TokenCompressor:
    """Token 压缩器基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    def compress(self, text: str) -> CompressionResult:
        """压缩文本（子类需要实现）"""
        raise NotImplementedError
    
    def estimate_tokens(self, text: str) -> int:
        """估算 token 数量（简化版）"""
        # 简单估算：英文单词数 + 标点符号
        words = re.findall(r'\b\w+\b', text)
        punctuation = len(re.findall(r'[^\w\s]', text))
        return len(words) + punctuation


class SummaryCompressor(TokenCompressor):
    """摘要压缩器"""
    
    def __init__(self, max_summary_length: int = 100):
        super().__init__("摘要压缩")
        self.max_summary_length = max_summary_length
    
    def compress(self, text: str) -> CompressionResult:
        original_tokens = self.estimate_tokens(text)
        
        # 简化版摘要生成：取前 N 个字符
        if len(text) > self.max_summary_length:
            compressed_text = text[:self.max_summary_length] + "..."
        else:
            compressed_text = text
        
        compressed_tokens = self.estimate_tokens(compressed_text)
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0
        
        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            compression_strategy=self.name
        )


class DeduplicationCompressor(TokenCompressor):
    """去重压缩器"""
    
    def __init__(self):
        super().__init__("去重压缩")
    
    def compress(self, text: str) -> CompressionResult:
        original_tokens = self.estimate_tokens(text)
        
        # 简单去重：移除连续的重复单词
        words = text.split()
        compressed_words = []
        prev_word = ""
        
        for word in words:
            if word != prev_word:
                compressed_words.append(word)
                prev_word = word
        
        compressed_text = " ".join(compressed_words)
        compressed_tokens = self.estimate_tokens(compressed_text)
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0
        
        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            compression_strategy=self.name
        )


class ContextMinimizer(TokenCompressor):
    """上下文最小化压缩器"""
    
    def __init__(self, max_context_length: int = 50):
        super().__init__("上下文最小化")
        self.max_context_length = max_context_length
    
    def compress(self, text: str) -> CompressionResult:
        original_tokens = self.estimate_tokens(text)
        
        # 保留最重要的部分（这里简单保留开头和结尾）
        words = text.split()
        if len(words) > self.max_context_length:
            half = self.max_context_length // 2
            compressed_words = words[:half] + ["..."] + words[-half:]
            compressed_text = " ".join(compressed_words)
        else:
            compressed_text = text
        
        compressed_tokens = self.estimate_tokens(compressed_text)
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0
        
        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            compression_strategy=self.name
        )


def demo_compression_strategies():
    """演示多种压缩策略"""
    
    # 测试文本
    test_text = """
    Hello world! This is a test message for token compression demonstration. 
    Hello world! This is a test message for token compression demonstration.
    We are testing different compression strategies to reduce token usage and costs.
    The goal is to maintain information quality while minimizing context length.
    """
    
    print("=== Token 压缩演示 ===")
    print(f"原始文本: {test_text[:100]}...")
    print()
    
    # 创建压缩器实例
    compressors = [
        SummaryCompressor(max_summary_length=80),
        DeduplicationCompressor(),
        ContextMinimizer(max_context_length=30)
    ]
    
    # 应用每种压缩策略
    for compressor in compressors:
        result = compressor.compress(test_text)
        
        print(f"策略: {result.compression_strategy}")
        print(f"原始 Token 数: {result.original_tokens}")
        print(f"压缩后 Token 数: {result.compressed_tokens}")
        print(f"压缩率: {result.compression_ratio:.2%}")
        print(f"压缩文本: {result.compressed_text[:80]}...")
        print("-" * 50)


def main() -> None:
    """主函数"""
    print("Module 9 - Token 压缩与上下文管理 Demo")
    print("=" * 60)
    
    demo_compression_strategies()
    
    print("\n用法说明:")
    print("- 摘要压缩: 生成文本摘要，保留核心信息")
    print("- 去重压缩: 移除重复内容，减少冗余")
    print("- 上下文最小化: 动态调整上下文窗口")
    print("- 压缩率: 压缩后 token 数 / 原始 token 数")


if __name__ == "__main__":
    main()