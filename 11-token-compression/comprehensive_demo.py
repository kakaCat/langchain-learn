#!/usr/bin/env python3
"""
Token 压缩综合演示
展示所有压缩策略、评估和监控功能
"""

import time
import json
from typing import List, Dict, Any
from token_compression_demo import (
    TokenCompressor, SummaryCompressor, 
    DeduplicationCompressor, ContextMinimizer,
    CompressionResult
)
from compression_evaluator import CompressionEvaluator
from performance_monitor import PerformanceMonitor, PerformanceMetrics
from datetime import datetime


class ComprehensiveDemo:
    """综合演示类"""
    
    def __init__(self):
        self.compressors = [
            SummaryCompressor(max_summary_length=100),
            DeduplicationCompressor(),
            ContextMinimizer(max_context_length=50)
        ]
        self.evaluator = CompressionEvaluator()
        self.monitor = PerformanceMonitor()
    
    def demo_basic_compression(self) -> None:
        """演示基础压缩功能"""
        print("\n" + "=" * 60)
        print("🔧 基础压缩策略演示")
        print("=" * 60)
        
        test_text = """
        Natural language processing (NLP) is a subfield of linguistics, computer science, 
        and artificial intelligence concerned with the interactions between computers and human language.
        It focuses on how to program computers to process and analyze large amounts of natural language data.
        The goal is a computer capable of understanding the contents of documents, including the contextual nuances.
        """
        
        print(f"原始文本 (约 {len(test_text)} 字符):")
        print(test_text[:200] + "..." if len(test_text) > 200 else test_text)
        print()
        
        for compressor in self.compressors:
            print(f"\n📋 使用 {compressor.name}:")
            
            start_time = time.time()
            result = compressor.compress(test_text)
            end_time = time.time()
            
            print(f"   压缩结果: {result.compressed_text}")
            print(f"   压缩率: {result.compression_ratio:.2%}")
            print(f"   Token 节省: {result.original_tokens} → {result.compressed_tokens} (节省 {result.original_tokens - result.compressed_tokens})")
            print(f"   执行时间: {end_time - start_time:.4f}s")
            print(f"   成本节省: ${result.cost_savings:.6f}")
    
    def demo_evaluation(self) -> None:
        """演示评估功能"""
        print("\n" + "=" * 60)
        print("📊 压缩策略评估演示")
        print("=" * 60)
        
        print("正在评估不同压缩策略...")
        comparison_results = self.evaluator.compare_strategies(self.compressors)
        
        report = self.evaluator.generate_report(comparison_results)
        report_data = json.loads(report)
        
        # 显示评估摘要
        print("\n📈 评估摘要:")
        for strategy_name, stats in report_data["summary"].items():
            print(f"\n  {strategy_name}:")
            print(f"    平均压缩率: {stats['average_compression_ratio']:.2%}")
            print(f"    平均质量分数: {stats['average_quality_score']:.2f}")
            print(f"    平均执行时间: {stats['average_execution_time']:.4f}s")
        
        # 显示推荐
        print("\n💡 使用建议:")
        for recommendation in report_data["recommendations"]:
            print(f"  - {recommendation}")
        
        # 保存详细报告
        with open("compression_evaluation_demo.json", "w", encoding="utf-8") as f:
            f.write(report)
        print("\n详细报告已保存到: compression_evaluation_demo.json")
    
    def demo_performance_monitoring(self) -> None:
        """演示性能监控功能"""
        print("\n" + "=" * 60)
        print("📈 性能监控演示")
        print("=" * 60)
        
        # 模拟性能数据记录
        test_texts = [
            "Hello world! This is a short test message.",
            "Natural language processing is important for AI applications.",
            "Token compression helps reduce API costs and improve efficiency."
        ]
        
        print("记录性能数据...")
        for i, text in enumerate(test_texts):
            for compressor in self.compressors:
                start_time = time.time()
                result = compressor.compress(text)
                end_time = time.time()
                
                # 记录性能指标
                metrics = PerformanceMetrics(
                    timestamp=datetime.now(),
                    strategy_name=compressor.name,
                    compression_ratio=result.compression_ratio,
                    execution_time=end_time - start_time,
                    tokens_saved=result.original_tokens - result.compressed_tokens,
                    quality_score=0.8 - (i * 0.05)  # 模拟质量变化
                )
                
                self.monitor.record_metrics(compressor.name, metrics)
            
            print(f"  已完成 {i + 1}/{len(test_texts)} 轮数据记录")
            time.sleep(0.5)
        
        # 生成监控报告
        print("\n生成性能监控报告...")
        report = self.monitor.get_comparison_report()
        
        print("\n📊 性能统计:")
        for strategy_name, stats in report["strategies"].items():
            print(f"\n  {strategy_name}:")
            print(f"    运行次数: {stats['total_runs']}")
            print(f"    平均压缩率: {stats['avg_compression_ratio']:.2%}")
            print(f"    平均执行时间: {stats['avg_execution_time']:.4f}s")
            print(f"    累计节省 Token: {stats['total_tokens_saved']}")
        
        # 导出历史数据
        self.monitor.export_history("performance_monitoring_demo.json")
        print("\n监控数据已导出到: performance_monitoring_demo.json")
    
    def demo_advanced_features(self) -> None:
        """演示高级功能"""
        print("\n" + "=" * 60)
        print("🚀 高级功能演示")
        print("=" * 60)
        
        # 演示批量处理
        print("\n📦 批量压缩演示:")
        texts = [
            "First document with important information.",
            "Second document containing similar content to first.",
            "Third document with unique data points."
        ]
        
        for compressor in self.compressors:
            print(f"\n  使用 {compressor.name} 批量处理:")
            total_saved = 0
            
            for i, text in enumerate(texts):
                result = compressor.compress(text)
                saved = result.original_tokens - result.compressed_tokens
                total_saved += saved
                print(f"    文档 {i + 1}: 节省 {saved} tokens")
            
            print(f"    总计节省: {total_saved} tokens")
        
        # 演示质量与成本的权衡
        print("\n⚖️ 质量与成本权衡:")
        long_text = "This is a very long text that needs compression. " * 10
        
        # 不同压缩强度的比较
        compressors_with_settings = [
            SummaryCompressor(max_summary_length=50),
            SummaryCompressor(max_summary_length=100),
            SummaryCompressor(max_summary_length=200)
        ]
        
        for compressor in compressors_with_settings:
            result = compressor.compress(long_text)
            print(f"\n  {compressor.name} (max_length={compressor.max_summary_length}):")
            print(f"    压缩率: {result.compression_ratio:.2%}")
            print(f"    质量估计: {result.quality_score:.2f}")
            print(f"    成本节省: ${result.cost_savings:.6f}")
    
    def run_complete_demo(self) -> None:
        """运行完整演示"""
        print("🚀 Token 压缩综合演示开始")
        print("=" * 60)
        
        # 执行各个演示环节
        self.demo_basic_compression()
        self.demo_evaluation()
        self.demo_performance_monitoring()
        self.demo_advanced_features()
        
        print("\n" + "=" * 60)
        print("✅ 综合演示完成")
        print("=" * 60)
        print("\n生成的文件:")
        print("  - compression_evaluation_demo.json (评估报告)")
        print("  - performance_monitoring_demo.json (监控数据)")
        print("\n下一步建议:")
        print("  1. 查看生成的报告文件了解详细数据")
        print("  2. 根据实际需求调整压缩策略参数")
        print("  3. 在生产环境中集成性能监控")


def main():
    """主函数"""
    demo = ComprehensiveDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()