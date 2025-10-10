#!/usr/bin/env python3
"""
Token 压缩策略评估器
比较不同压缩算法的效果和性能
"""

import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from token_compression_demo import (
    TokenCompressor, SummaryCompressor, 
    DeduplicationCompressor, ContextMinimizer
)


@dataclass
class EvaluationResult:
    """评估结果数据类"""
    strategy_name: str
    compression_ratio: float
    execution_time: float
    quality_score: float
    original_tokens: int
    compressed_tokens: int


class CompressionEvaluator:
    """压缩策略评估器"""
    
    def __init__(self):
        self.test_cases = self._load_test_cases()
    
    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """加载测试用例"""
        return [
            {
                "name": "短文本",
                "text": "Hello world! This is a short test message.",
                "expected_quality": 0.9
            },
            {
                "name": "长文本",
                "text": """
                Natural language processing (NLP) is a subfield of linguistics, computer science, 
                and artificial intelligence concerned with the interactions between computers and human language.
                It focuses on how to program computers to process and analyze large amounts of natural language data.
                The goal is a computer capable of understanding the contents of documents, including the contextual nuances.
                """,
                "expected_quality": 0.7
            },
            {
                "name": "重复文本",
                "text": """
                Hello world! Hello world! Hello world!
                This is a test. This is a test. This is a test.
                Repeat message. Repeat message. Repeat message.
                """,
                "expected_quality": 0.8
            },
            {
                "name": "技术文档",
                "text": """
                Token compression is a technique used to reduce the number of tokens in language model inputs.
                This helps reduce API costs and improve processing efficiency while maintaining information quality.
                Common strategies include summarization, deduplication, and context window optimization.
                """,
                "expected_quality": 0.75
            }
        ]
    
    def evaluate_strategy(self, compressor: TokenCompressor) -> List[EvaluationResult]:
        """评估单个压缩策略"""
        results = []
        
        for test_case in self.test_cases:
            start_time = time.time()
            
            # 执行压缩
            compression_result = compressor.compress(test_case["text"])
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 计算质量分数（简化版）
            quality_score = self._calculate_quality_score(
                compression_result, test_case["expected_quality"]
            )
            
            result = EvaluationResult(
                strategy_name=compressor.name,
                compression_ratio=compression_result.compression_ratio,
                execution_time=execution_time,
                quality_score=quality_score,
                original_tokens=compression_result.original_tokens,
                compressed_tokens=compression_result.compressed_tokens
            )
            
            results.append(result)
        
        return results
    
    def _calculate_quality_score(self, result, expected_quality: float) -> float:
        """计算压缩质量分数"""
        # 基于压缩率和信息保留度的简单评分
        compression_penalty = (1 - result.compression_ratio) * 0.3
        quality_score = expected_quality - compression_penalty
        
        # 确保分数在合理范围内
        return max(0.1, min(1.0, quality_score))
    
    def compare_strategies(self, compressors: List[TokenCompressor]) -> Dict[str, Any]:
        """比较多种压缩策略"""
        comparison_results = {}
        
        for compressor in compressors:
            print(f"正在评估策略: {compressor.name}")
            results = self.evaluate_strategy(compressor)
            comparison_results[compressor.name] = results
        
        return comparison_results
    
    def generate_report(self, comparison_results: Dict[str, List[EvaluationResult]]) -> str:
        """生成评估报告"""
        report = {
            "summary": {},
            "detailed_results": {},
            "recommendations": []
        }
        
        # 计算每种策略的平均指标
        for strategy_name, results in comparison_results.items():
            avg_compression = sum(r.compression_ratio for r in results) / len(results)
            avg_quality = sum(r.quality_score for r in results) / len(results)
            avg_time = sum(r.execution_time for r in results) / len(results)
            
            report["summary"][strategy_name] = {
                "average_compression_ratio": avg_compression,
                "average_quality_score": avg_quality,
                "average_execution_time": avg_time
            }
            
            # 详细结果
            report["detailed_results"][strategy_name] = [
                {
                    "test_case": self.test_cases[i]["name"],
                    "compression_ratio": r.compression_ratio,
                    "quality_score": r.quality_score,
                    "execution_time": r.execution_time,
                    "tokens_saved": r.original_tokens - r.compressed_tokens
                }
                for i, r in enumerate(results)
            ]
        
        # 生成推荐
        self._generate_recommendations(report)
        
        return json.dumps(report, indent=2, ensure_ascii=False)
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> None:
        """生成使用建议"""
        strategies = list(report["summary"].keys())
        
        # 找出最佳压缩率策略
        best_compression = max(
            strategies, 
            key=lambda s: report["summary"][s]["average_compression_ratio"]
        )
        
        # 找出最佳质量策略
        best_quality = max(
            strategies, 
            key=lambda s: report["summary"][s]["average_quality_score"]
        )
        
        # 找出最快策略
        fastest = min(
            strategies, 
            key=lambda s: report["summary"][s]["average_execution_time"]
        )
        
        report["recommendations"] = [
            f"最佳压缩率策略: {best_compression}",
            f"最佳质量策略: {best_quality}", 
            f"最快执行策略: {fastest}",
            "建议根据具体场景选择合适的压缩策略：",
            "- 摘要压缩：适合长文档，需要保留核心信息",
            "- 去重压缩：适合重复内容多的文本",
            "- 上下文最小化：适合对话和历史记录管理"
        ]


def main():
    """主评估函数"""
    print("=== Token 压缩策略评估 ===")
    print()
    
    # 创建评估器
    evaluator = CompressionEvaluator()
    
    # 创建压缩器实例
    compressors = [
        SummaryCompressor(max_summary_length=100),
        DeduplicationCompressor(),
        ContextMinimizer(max_context_length=50)
    ]
    
    # 执行比较评估
    print("开始评估不同压缩策略...")
    comparison_results = evaluator.compare_strategies(compressors)
    
    # 生成报告
    print("\n生成评估报告...")
    report = evaluator.generate_report(comparison_results)
    
    # 输出报告
    print("\n" + "=" * 60)
    print("评估报告:")
    print("=" * 60)
    print(report)
    
    # 保存报告到文件
    with open("compression_evaluation_report.json", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\n报告已保存到: compression_evaluation_report.json")


if __name__ == "__main__":
    main()