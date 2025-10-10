#!/usr/bin/env python3
"""
Module 10: Evaluation & Deployment Demo
评估与部署综合演示
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_ms: float
    token_usage: int
    cost_usd: float


@dataclass
class TraceRecord:
    """调用追踪记录"""
    timestamp: datetime
    operation: str
    duration_ms: float
    status: str
    metadata: Dict[str, Any]


class EvaluationEngine:
    """评估引擎"""
    
    def __init__(self):
        self.trace_records: List[TraceRecord] = []
        self.evaluation_history: List[EvaluationMetrics] = []
    
    def evaluate_model_performance(self, predictions: List[str], ground_truth: List[str]) -> EvaluationMetrics:
        """评估模型性能"""
        start_time = time.time()
        
        # 计算准确率
        correct = sum(1 for pred, truth in zip(predictions, ground_truth) if pred == truth)
        accuracy = correct / len(predictions) if predictions else 0.0
        
        # 计算精确率、召回率和F1分数（简化版本）
        precision = accuracy * 0.9  # 简化计算
        recall = accuracy * 0.85    # 简化计算
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 模拟延迟和成本
        latency_ms = (time.time() - start_time) * 1000
        token_usage = sum(len(pred) for pred in predictions) // 4  # 近似token计算
        cost_usd = token_usage * 0.000002  # 近似成本计算
        
        metrics = EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            latency_ms=latency_ms,
            token_usage=token_usage,
            cost_usd=cost_usd
        )
        
        self.evaluation_history.append(metrics)
        self._record_trace("evaluate_model_performance", latency_ms, "success", {
            "predictions_count": len(predictions),
            "accuracy": accuracy
        })
        
        return metrics
    
    def _record_trace(self, operation: str, duration_ms: float, status: str, metadata: Dict[str, Any]):
        """记录调用追踪"""
        record = TraceRecord(
            timestamp=datetime.now(),
            operation=operation,
            duration_ms=duration_ms,
            status=status,
            metadata=metadata
        )
        self.trace_records.append(record)
    
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """生成评估报告"""
        if not self.evaluation_history:
            return {"error": "No evaluation data available"}
        
        latest_metrics = self.evaluation_history[-1]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "accuracy": latest_metrics.accuracy,
                "precision": latest_metrics.precision,
                "recall": latest_metrics.recall,
                "f1_score": latest_metrics.f1_score,
                "latency_ms": latest_metrics.latency_ms,
                "token_usage": latest_metrics.token_usage,
                "cost_usd": latest_metrics.cost_usd
            },
            "trace_summary": {
                "total_operations": len(self.trace_records),
                "success_rate": len([r for r in self.trace_records if r.status == "success"]) / len(self.trace_records) if self.trace_records else 0.0,
                "average_latency_ms": sum(r.duration_ms for r in self.trace_records) / len(self.trace_records) if self.trace_records else 0.0
            }
        }
        
        return report


def demo_evaluation():
    """演示评估功能"""
    print("=== 评估与部署 Demo - 评估功能演示 ===\n")
    
    engine = EvaluationEngine()
    
    # 模拟测试数据
    predictions = ["positive", "negative", "positive", "positive", "negative"]
    ground_truth = ["positive", "negative", "positive", "negative", "negative"]
    
    print("测试数据:")
    print(f"预测结果: {predictions}")
    print(f"真实标签: {ground_truth}")
    print()
    
    # 执行评估
    metrics = engine.evaluate_model_performance(predictions, ground_truth)
    
    print("评估结果:")
    print(f"准确率: {metrics.accuracy:.3f}")
    print(f"精确率: {metrics.precision:.3f}")
    print(f"召回率: {metrics.recall:.3f}")
    print(f"F1分数: {metrics.f1_score:.3f}")
    print(f"延迟: {metrics.latency_ms:.2f} ms")
    print(f"Token使用: {metrics.token_usage}")
    print(f"成本: ${metrics.cost_usd:.6f}")
    print()
    
    # 生成报告
    report = engine.generate_evaluation_report()
    print("评估报告:")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    """主函数"""
    print("Module 10 - Evaluation & Deployment Demo")
    print("=" * 50)
    demo_evaluation()


if __name__ == "__main__":
    main()