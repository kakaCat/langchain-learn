#!/usr/bin/env python3
"""
Token 压缩性能监控器
实时监控压缩策略的性能指标
"""

import time
import threading
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import deque


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    strategy_name: str
    compression_ratio: float
    execution_time: float
    tokens_saved: int
    quality_score: float


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, max_history: int = 1000):
        self.metrics_history: Dict[str, deque] = {}
        self.max_history = max_history
        self.lock = threading.Lock()
        self.alerts: List[str] = []
    
    def record_metrics(self, strategy_name: str, metrics: PerformanceMetrics) -> None:
        """记录性能指标"""
        with self.lock:
            if strategy_name not in self.metrics_history:
                self.metrics_history[strategy_name] = deque(maxlen=self.max_history)
            
            self.metrics_history[strategy_name].append(metrics)
            
            # 检查性能异常
            self._check_performance_anomalies(strategy_name, metrics)
    
    def _check_performance_anomalies(self, strategy_name: str, metrics: PerformanceMetrics) -> None:
        """检查性能异常"""
        history = self.metrics_history[strategy_name]
        
        if len(history) < 5:  # 需要有足够的历史数据
            return
        
        # 计算最近5次的平均值
        recent_metrics = list(history)[-5:]
        avg_time = sum(m.execution_time for m in recent_metrics) / 5
        avg_ratio = sum(m.compression_ratio for m in recent_metrics) / 5
        
        # 检测执行时间异常
        if metrics.execution_time > avg_time * 2:
            alert = f"警告: {strategy_name} 执行时间异常增加 ({metrics.execution_time:.4f}s > 平均 {avg_time:.4f}s)"
            self.alerts.append(alert)
        
        # 检测压缩率异常下降
        if metrics.compression_ratio < avg_ratio * 0.7:
            alert = f"警告: {strategy_name} 压缩率异常下降 ({metrics.compression_ratio:.2%} < 平均 {avg_ratio:.2%})"
            self.alerts.append(alert)
    
    def get_strategy_stats(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """获取策略统计信息"""
        with self.lock:
            if strategy_name not in self.metrics_history:
                return None
            
            metrics_list = list(self.metrics_history[strategy_name])
            
            if not metrics_list:
                return None
            
            return {
                "total_runs": len(metrics_list),
                "avg_compression_ratio": sum(m.compression_ratio for m in metrics_list) / len(metrics_list),
                "avg_execution_time": sum(m.execution_time for m in metrics_list) / len(metrics_list),
                "total_tokens_saved": sum(m.tokens_saved for m in metrics_list),
                "avg_quality_score": sum(m.quality_score for m in metrics_list) / len(metrics_list),
                "last_updated": metrics_list[-1].timestamp.isoformat()
            }
    
    def get_comparison_report(self) -> Dict[str, Any]:
        """获取策略比较报告"""
        with self.lock:
            report = {
                "timestamp": datetime.now().isoformat(),
                "strategies": {},
                "alerts": self.alerts.copy(),
                "recommendations": []
            }
            
            for strategy_name in self.metrics_history:
                stats = self.get_strategy_stats(strategy_name)
                if stats:
                    report["strategies"][strategy_name] = stats
            
            # 生成推荐
            self._generate_recommendations(report)
            
            return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> None:
        """生成性能优化建议"""
        strategies = list(report["strategies"].keys())
        
        if len(strategies) < 2:
            return
        
        # 找出最佳策略
        best_compression = max(
            strategies,
            key=lambda s: report["strategies"][s]["avg_compression_ratio"]
        )
        
        best_quality = max(
            strategies,
            key=lambda s: report["strategies"][s]["avg_quality_score"]
        )
        
        fastest = min(
            strategies,
            key=lambda s: report["strategies"][s]["avg_execution_time"]
        )
        
        report["recommendations"] = [
            f"推荐使用 {best_compression} 以获得最佳压缩率",
            f"推荐使用 {best_quality} 以获得最佳质量",
            f"推荐使用 {fastest} 以获得最快执行速度",
            f"累计节省 Token 数量: {sum(report['strategies'][s]['total_tokens_saved'] for s in strategies)}"
        ]
    
    def clear_alerts(self) -> None:
        """清除警报"""
        with self.lock:
            self.alerts.clear()
    
    def export_history(self, filepath: str) -> None:
        """导出历史数据"""
        with self.lock:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "metrics_history": {}
            }
            
            for strategy_name, metrics_deque in self.metrics_history.items():
                export_data["metrics_history"][strategy_name] = [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "compression_ratio": m.compression_ratio,
                        "execution_time": m.execution_time,
                        "tokens_saved": m.tokens_saved,
                        "quality_score": m.quality_score
                    }
                    for m in metrics_deque
                ]
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)


class RealTimeMonitor:
    """实时监控器"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.running = False
        self.thread: Optional[threading.Thread] = None
    
    def start_monitoring(self, interval: int = 60) -> None:
        """开始实时监控"""
        if self.running:
            print("监控器已在运行中")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, args=(interval,))
        self.thread.daemon = True
        self.thread.start()
        
        print(f"实时监控已启动，每 {interval} 秒生成一次报告")
    
    def stop_monitoring(self) -> None:
        """停止实时监控"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("实时监控已停止")
    
    def _monitoring_loop(self, interval: int) -> None:
        """监控循环"""
        while self.running:
            time.sleep(interval)
            
            # 生成报告
            report = self.monitor.get_comparison_report()
            
            # 输出摘要信息
            self._print_report_summary(report)
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
    
    def _print_report_summary(self, report: Dict[str, Any]) -> None:
        """打印报告摘要"""
        print("\n" + "=" * 60)
        print(f"性能监控报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        for strategy_name, stats in report["strategies"].items():
            print(f"\n策略: {strategy_name}")
            print(f"  运行次数: {stats['total_runs']}")
            print(f"  平均压缩率: {stats['avg_compression_ratio']:.2%}")
            print(f"  平均执行时间: {stats['avg_execution_time']:.4f}s")
            print(f"  平均质量分数: {stats['avg_quality_score']:.2f}")
            print(f"  累计节省 Token: {stats['total_tokens_saved']}")
        
        if report["alerts"]:
            print(f"\n⚠️  警报 ({len(report['alerts'])} 条):")
            for alert in report["alerts"][-5:]:  # 只显示最近5条警报
                print(f"  - {alert}")
        
        if report["recommendations"]:
            print(f"\n💡 推荐:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")


def demo_monitoring():
    """演示监控功能"""
    print("=== Token 压缩性能监控演示 ===")
    
    # 创建监控器
    monitor = PerformanceMonitor()
    realtime_monitor = RealTimeMonitor(monitor)
    
    # 模拟一些性能数据
    strategies = ["摘要压缩", "去重压缩", "上下文最小化"]
    
    print("\n模拟性能数据记录...")
    for i in range(10):
        for strategy in strategies:
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                strategy_name=strategy,
                compression_ratio=0.7 + (i * 0.02),  # 模拟逐渐改善的压缩率
                execution_time=0.1 + (i * 0.01),     # 模拟逐渐增加的执行时间
                tokens_saved=100 + (i * 20),         # 模拟逐渐增加的节省量
                quality_score=0.8 - (i * 0.01)       # 模拟逐渐下降的质量
            )
            monitor.record_metrics(strategy, metrics)
        time.sleep(0.5)
    
    # 生成报告
    print("\n生成性能报告...")
    report = monitor.get_comparison_report()
    
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # 导出历史数据
    monitor.export_history("performance_history.json")
    print("\n历史数据已导出到: performance_history.json")


if __name__ == "__main__":
    demo_monitoring()