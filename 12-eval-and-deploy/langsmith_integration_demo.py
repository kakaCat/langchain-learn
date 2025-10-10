#!/usr/bin/env python3
"""
LangSmith 集成与高级评估监控演示
深度集成 LangSmith 的评估、追踪和监控功能
"""

import os
import time
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import statistics


class EvaluationType(Enum):
    """评估类型枚举"""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    COST = "cost"
    QUALITY = "quality"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"


@dataclass
class LangSmithConfig:
    """LangSmith 配置"""
    api_key: str
    project_name: str
    environment: str = "development"
    tracing_enabled: bool = True
    evaluation_enabled: bool = True
    monitoring_enabled: bool = True


@dataclass
class AdvancedEvaluationMetrics:
    """高级评估指标"""
    # 基础指标
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_ms: float
    token_usage: int
    cost_usd: float
    
    # 高级指标
    faithfulness_score: float
    answer_relevance_score: float
    context_precision: float
    context_recall: float
    
    # 质量指标
    coherence_score: float
    fluency_score: float
    consistency_score: float
    
    # 业务指标
    user_satisfaction: float
    business_value: float


@dataclass
class TraceSpan:
    """追踪跨度"""
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: datetime
    end_time: datetime
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


class LangSmithIntegration:
    """LangSmith 集成管理器"""
    
    def __init__(self, config: LangSmithConfig):
        self.config = config
        self.trace_spans: Dict[str, TraceSpan] = {}
        self.evaluation_results: List[AdvancedEvaluationMetrics] = []
        self.performance_data: Dict[str, List[float]] = {}
        
    def start_trace_span(self, operation: str, parent_span_id: Optional[str] = None, 
                        tags: Dict[str, str] = None) -> str:
        """开始追踪跨度"""
        span_id = f"span_{len(self.trace_spans) + 1}"
        span = TraceSpan(
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=datetime.now(),
            end_time=datetime.now(),
            status="started",
            tags=tags or {}
        )
        self.trace_spans[span_id] = span
        return span_id
    
    def end_trace_span(self, span_id: str, status: str = "success", 
                      metadata: Dict[str, Any] = None):
        """结束追踪跨度"""
        if span_id in self.trace_spans:
            span = self.trace_spans[span_id]
            span.end_time = datetime.now()
            span.status = status
            span.metadata = metadata or {}
    
    def record_evaluation(self, metrics: AdvancedEvaluationMetrics):
        """记录评估结果"""
        self.evaluation_results.append(metrics)
        
        # 记录性能数据
        self.performance_data.setdefault("latency", []).append(metrics.latency_ms)
        self.performance_data.setdefault("accuracy", []).append(metrics.accuracy)
        self.performance_data.setdefault("cost", []).append(metrics.cost_usd)
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合报告"""
        if not self.evaluation_results:
            return {"error": "No evaluation data available"}
        
        latest_metrics = self.evaluation_results[-1]
        
        # 计算统计信息
        latency_data = self.performance_data.get("latency", [])
        accuracy_data = self.performance_data.get("accuracy", [])
        cost_data = self.performance_data.get("cost", [])
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "project": self.config.project_name,
            "environment": self.config.environment,
            
            "summary_metrics": {
                "total_evaluations": len(self.evaluation_results),
                "average_latency_ms": statistics.mean(latency_data) if latency_data else 0,
                "average_accuracy": statistics.mean(accuracy_data) if accuracy_data else 0,
                "average_cost_usd": statistics.mean(cost_data) if cost_data else 0,
                "p95_latency_ms": statistics.quantiles(latency_data, n=20)[18] if len(latency_data) >= 20 else 0,
            },
            
            "latest_evaluation": {
                "accuracy": latest_metrics.accuracy,
                "precision": latest_metrics.precision,
                "recall": latest_metrics.recall,
                "f1_score": latest_metrics.f1_score,
                "faithfulness_score": latest_metrics.faithfulness_score,
                "answer_relevance_score": latest_metrics.answer_relevance_score,
                "coherence_score": latest_metrics.coherence_score,
                "user_satisfaction": latest_metrics.user_satisfaction,
            },
            
            "trace_summary": {
                "total_spans": len(self.trace_spans),
                "success_rate": len([s for s in self.trace_spans.values() if s.status == "success"]) / len(self.trace_spans) if self.trace_spans else 0,
                "average_span_duration_ms": statistics.mean([(s.end_time - s.start_time).total_seconds() * 1000 for s in self.trace_spans.values()]) if self.trace_spans else 0,
            },
            
            "performance_trends": {
                "latency_trend": "stable" if len(latency_data) < 2 else (
                    "improving" if latency_data[-1] < statistics.mean(latency_data[:-1]) else "degrading"
                ),
                "accuracy_trend": "stable" if len(accuracy_data) < 2 else (
                    "improving" if accuracy_data[-1] > statistics.mean(accuracy_data[:-1]) else "degrading"
                ),
            }
        }
        
        return report


class AutomatedEvaluator:
    """自动化评估器"""
    
    def __init__(self, langsmith_integration: LangSmithIntegration):
        self.langsmith = langsmith_integration
    
    def evaluate_response_quality(self, question: str, response: str, 
                                context: List[str] = None) -> Dict[str, float]:
        """评估响应质量"""
        span_id = self.langsmith.start_trace_span("evaluate_response_quality", 
                                                 tags={"evaluation_type": "quality"})
        
        try:
            # 模拟质量评估计算
            faithfulness = self._calculate_faithfulness(question, response, context)
            relevance = self._calculate_relevance(question, response)
            coherence = self._calculate_coherence(response)
            fluency = self._calculate_fluency(response)
            
            quality_scores = {
                "faithfulness": faithfulness,
                "relevance": relevance,
                "coherence": coherence,
                "fluency": fluency,
                "overall_quality": (faithfulness + relevance + coherence + fluency) / 4
            }
            
            self.langsmith.end_trace_span(span_id, "success", {
                "question": question[:100],  # 截断长问题
                "quality_scores": quality_scores
            })
            
            return quality_scores
            
        except Exception as e:
            self.langsmith.end_trace_span(span_id, "error", {"error": str(e)})
            raise
    
    def _calculate_faithfulness(self, question: str, response: str, context: List[str]) -> float:
        """计算忠实度（基于上下文的准确性）"""
        # 简化实现 - 实际应使用更复杂的NLP方法
        if not context:
            return 0.8  # 默认值
        
        # 检查响应是否与上下文一致
        context_text = " ".join(context)
        keywords_in_context = set(context_text.lower().split()[:10])
        keywords_in_response = set(response.lower().split()[:10])
        
        overlap = len(keywords_in_context.intersection(keywords_in_response))
        return min(overlap / 10, 1.0)
    
    def _calculate_relevance(self, question: str, response: str) -> float:
        """计算相关性"""
        # 简化实现
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        
        overlap = len(question_words.intersection(response_words))
        return min(overlap / max(len(question_words), 1), 1.0)
    
    def _calculate_coherence(self, response: str) -> float:
        """计算连贯性"""
        # 简化实现 - 基于句子结构和长度
        sentences = response.split('.')
        if len(sentences) <= 1:
            return 0.9
        
        # 简单的连贯性评分
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        return min(avg_sentence_length / 20, 1.0)
    
    def _calculate_fluency(self, response: str) -> float:
        """计算流畅度"""
        # 简化实现 - 基于语法正确性假设
        words = response.split()
        if len(words) < 3:
            return 0.8
        
        # 简单的流畅度评分
        return min(len(words) / 50, 1.0)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, langsmith_integration: LangSmithIntegration):
        self.langsmith = langsmith_integration
        self.alerts: List[Dict[str, Any]] = []
    
    def check_performance_sla(self, metrics: AdvancedEvaluationMetrics) -> List[str]:
        """检查性能SLA"""
        violations = []
        
        # 延迟SLA检查
        if metrics.latency_ms > 1000:  # 1秒延迟阈值
            violations.append("High latency")
        
        # 准确性SLA检查
        if metrics.accuracy < 0.8:  # 80%准确性阈值
            violations.append("Low accuracy")
        
        # 成本SLA检查
        if metrics.cost_usd > 0.01:  # 成本阈值
            violations.append("High cost")
        
        # 质量SLA检查
        if metrics.coherence_score < 0.7:
            violations.append("Low coherence")
        
        if violations:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "violations": violations,
                "metrics": {
                    "latency": metrics.latency_ms,
                    "accuracy": metrics.accuracy,
                    "cost": metrics.cost_usd,
                    "coherence": metrics.coherence_score
                }
            }
            self.alerts.append(alert)
        
        return violations
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """获取警报摘要"""
        return {
            "total_alerts": len(self.alerts),
            "recent_alerts": self.alerts[-5:] if self.alerts else [],
            "alert_trend": "increasing" if len(self.alerts) > 5 and len(self.alerts[-5:]) > 2 else "stable"
        }


def demo_langsmith_integration():
    """演示 LangSmith 集成功能"""
    print("=== LangSmith 集成与高级评估监控演示 ===\n")
    
    # 配置 LangSmith
    config = LangSmithConfig(
        api_key="demo_api_key",
        project_name="advanced-evaluation-demo",
        environment="development"
    )
    
    langsmith = LangSmithIntegration(config)
    evaluator = AutomatedEvaluator(langsmith)
    monitor = PerformanceMonitor(langsmith)
    
    # 模拟评估数据
    questions = [
        "什么是人工智能？",
        "机器学习有哪些主要类型？",
        "深度学习与传统机器学习有什么区别？"
    ]
    
    responses = [
        "人工智能是模拟人类智能的计算机系统。",
        "机器学习包括监督学习、无监督学习和强化学习。",
        "深度学习使用神经网络，传统机器学习使用特征工程。"
    ]
    
    contexts = [
        ["人工智能涉及计算机科学、心理学、哲学等领域。"],
        ["监督学习需要标注数据，无监督学习发现数据模式。"],
        ["深度学习可以自动学习特征，传统机器学习需要手动设计特征。"]
    ]
    
    print("执行自动化评估...")
    for i, (question, response, context) in enumerate(zip(questions, responses, contexts)):
        print(f"\n评估 {i+1}: {question}")
        
        # 评估质量
        quality_scores = evaluator.evaluate_response_quality(question, response, context)
        print(f"  质量评分: {quality_scores}")
        
        # 创建高级评估指标
        metrics = AdvancedEvaluationMetrics(
            accuracy=0.85 + i * 0.05,
            precision=0.82 + i * 0.03,
            recall=0.88 + i * 0.02,
            f1_score=0.85 + i * 0.04,
            latency_ms=200 + i * 50,
            token_usage=150 + i * 20,
            cost_usd=0.0003 + i * 0.0001,
            faithfulness_score=quality_scores["faithfulness"],
            answer_relevance_score=quality_scores["relevance"],
            context_precision=0.9,
            context_recall=0.85,
            coherence_score=quality_scores["coherence"],
            fluency_score=quality_scores["fluency"],
            consistency_score=0.9,
            user_satisfaction=0.8 + i * 0.05,
            business_value=0.7 + i * 0.1
        )
        
        # 记录评估结果
        langsmith.record_evaluation(metrics)
        
        # 检查性能SLA
        violations = monitor.check_performance_sla(metrics)
        if violations:
            print(f"  ⚠️  SLA违规: {violations}")
    
    # 生成综合报告
    print("\n" + "="*50)
    print("生成综合评估报告...")
    report = langsmith.generate_comprehensive_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # 显示警报摘要
    print("\n" + "="*50)
    print("性能监控警报摘要:")
    alerts_summary = monitor.get_alerts_summary()
    print(json.dumps(alerts_summary, indent=2, ensure_ascii=False))


def main():
    """主函数"""
    print("LangSmith 集成与高级评估监控演示")
    print("=" * 60)
    demo_langsmith_integration()


if __name__ == "__main__":
    main()