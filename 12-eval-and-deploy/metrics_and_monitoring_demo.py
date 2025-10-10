#!/usr/bin/env python3
"""
性能监控与指标收集演示
"""

import time
import psutil
import threading
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
from contextlib import contextmanager


@dataclass
class MetricPoint:
    """指标数据点"""
    timestamp: datetime
    value: float
    labels: Dict[str, str]


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.lock = threading.Lock()
        self.system_metrics_enabled = False
        self.collection_thread: Optional[threading.Thread] = None
        self.stop_collection = False
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录指标"""
        if labels is None:
            labels = {}
        
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels
        )
        
        with self.lock:
            self.metrics[name].append(point)
    
    def record_counter(self, name: str, increment: int = 1, labels: Dict[str, str] = None):
        """记录计数器"""
        if labels is None:
            labels = {}
        
        # 获取当前计数
        current_count = self.get_metric_sum(name, labels) or 0
        self.record_metric(name, current_count + increment, labels)
    
    def record_timer(self, name: str, duration_seconds: float, labels: Dict[str, str] = None):
        """记录计时器"""
        if labels is None:
            labels = {}
        
        # 记录持续时间
        self.record_metric(f"{name}_duration", duration_seconds, labels)
        # 记录调用次数
        self.record_counter(f"{name}_calls", 1, labels)
    
    @contextmanager
    def timer(self, name: str, labels: Dict[str, str] = None):
        """计时器上下文管理器"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timer(name, duration, labels)
    
    def get_metric_sum(self, name: str, labels: Dict[str, str] = None, 
                      time_window: Optional[timedelta] = None) -> Optional[float]:
        """获取指标总和"""
        with self.lock:
            points = self.metrics.get(name, [])
            
            if time_window:
                cutoff_time = datetime.now() - time_window
                points = [p for p in points if p.timestamp >= cutoff_time]
            
            if labels:
                points = [p for p in points if p.labels == labels]
            
            if not points:
                return None
            
            return sum(p.value for p in points)
    
    def get_metric_avg(self, name: str, labels: Dict[str, str] = None,
                      time_window: Optional[timedelta] = None) -> Optional[float]:
        """获取指标平均值"""
        with self.lock:
            points = self.metrics.get(name, [])
            
            if time_window:
                cutoff_time = datetime.now() - time_window
                points = [p for p in points if p.timestamp >= cutoff_time]
            
            if labels:
                points = [p for p in points if p.labels == labels]
            
            if not points:
                return None
            
            return sum(p.value for p in points) / len(points)
    
    def collect_system_metrics(self):
        """收集系统指标"""
        while not self.stop_collection:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_metric("system_cpu_percent", cpu_percent, {"type": "usage"})
                
                # 内存使用
                memory = psutil.virtual_memory()
                self.record_metric("system_memory_percent", memory.percent, {"type": "usage"})
                self.record_metric("system_memory_used_mb", memory.used / 1024 / 1024, {"type": "absolute"})
                
                # 磁盘IO
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.record_metric("system_disk_read_mb", disk_io.read_bytes / 1024 / 1024, {"type": "io"})
                    self.record_metric("system_disk_write_mb", disk_io.write_bytes / 1024 / 1024, {"type": "io"})
                
                # 网络IO
                net_io = psutil.net_io_counters()
                if net_io:
                    self.record_metric("system_network_sent_mb", net_io.bytes_sent / 1024 / 1024, {"type": "network"})
                    self.record_metric("system_network_recv_mb", net_io.bytes_recv / 1024 / 1024, {"type": "network"})
                
            except Exception as e:
                print(f"收集系统指标时出错: {e}")
            
            time.sleep(5)  # 每5秒收集一次
    
    def start_system_metrics_collection(self):
        """开始系统指标收集"""
        if self.system_metrics_enabled:
            return
        
        self.system_metrics_enabled = True
        self.stop_collection = False
        self.collection_thread = threading.Thread(target=self.collect_system_metrics)
        self.collection_thread.daemon = True
        self.collection_thread.start()
    
    def stop_system_metrics_collection(self):
        """停止系统指标收集"""
        self.stop_collection = True
        if self.collection_thread:
            self.collection_thread.join(timeout=10)
        self.system_metrics_enabled = False
    
    def get_metrics_summary(self, time_window: timedelta = timedelta(minutes=5)) -> Dict[str, Any]:
        """获取指标摘要"""
        summary = {}
        
        with self.lock:
            for metric_name, points in self.metrics.items():
                recent_points = [p for p in points 
                               if p.timestamp >= datetime.now() - time_window]
                
                if not recent_points:
                    continue
                
                values = [p.value for p in recent_points]
                
                summary[metric_name] = {
                    "count": len(recent_points),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "latest": recent_points[-1].value,
                    "latest_timestamp": recent_points[-1].timestamp.isoformat()
                }
        
        return summary


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_thresholds = {
            "system_cpu_percent": 80.0,
            "system_memory_percent": 85.0,
            "api_latency_ms": 1000.0,
            "error_rate": 0.05
        }
        self.alerts: List[Dict[str, Any]] = []
    
    def monitor_api_call(self, api_name: str):
        """监控API调用装饰器"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.metrics_collector.timer(f"api_{api_name}", {"api": api_name}):
                    try:
                        result = func(*args, **kwargs)
                        self.metrics_collector.record_counter(f"api_{api_name}_success", 1, {"api": api_name})
                        return result
                    except Exception as e:
                        self.metrics_collector.record_counter(f"api_{api_name}_errors", 1, {"api": api_name})
                        raise
            return wrapper
        return decorator
    
    def check_alerts(self):
        """检查警报条件"""
        current_alerts = []
        
        # 检查CPU使用率
        cpu_usage = self.metrics_collector.get_metric_avg("system_cpu_percent", 
                                                         time_window=timedelta(minutes=1))
        if cpu_usage and cpu_usage > self.alert_thresholds["system_cpu_percent"]:
            current_alerts.append({
                "metric": "system_cpu_percent",
                "value": cpu_usage,
                "threshold": self.alert_thresholds["system_cpu_percent"],
                "message": f"CPU使用率过高: {cpu_usage:.1f}%"
            })
        
        # 检查内存使用率
        memory_usage = self.metrics_collector.get_metric_avg("system_memory_percent",
                                                            time_window=timedelta(minutes=1))
        if memory_usage and memory_usage > self.alert_thresholds["system_memory_percent"]:
            current_alerts.append({
                "metric": "system_memory_percent",
                "value": memory_usage,
                "threshold": self.alert_thresholds["system_memory_percent"],
                "message": f"内存使用率过高: {memory_usage:.1f}%"
            })
        
        # 检查API错误率
        for metric_name in self.metrics_collector.metrics.keys():
            if metric_name.endswith("_errors"):
                api_name = metric_name.replace("_errors", "")
                success_metric = f"{api_name}_success"
                
                error_count = self.metrics_collector.get_metric_sum(metric_name, 
                                                                   time_window=timedelta(minutes=5))
                success_count = self.metrics_collector.get_metric_sum(success_metric,
                                                                     time_window=timedelta(minutes=5))
                
                if error_count and success_count and (error_count + success_count) > 0:
                    error_rate = error_count / (error_count + success_count)
                    if error_rate > self.alert_thresholds["error_rate"]:
                        current_alerts.append({
                            "metric": f"{api_name}_error_rate",
                            "value": error_rate,
                            "threshold": self.alert_thresholds["error_rate"],
                            "message": f"{api_name} 错误率过高: {error_rate:.2%}"
                        })
        
        self.alerts.extend(current_alerts)
        return current_alerts
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        metrics_summary = self.metrics_collector.get_metrics_summary()
        current_alerts = self.check_alerts()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics_summary": metrics_summary,
            "current_alerts": current_alerts,
            "total_alerts": len(self.alerts),
            "system_status": "healthy" if not current_alerts else "degraded"
        }


def demo_metrics_and_monitoring():
    """演示性能监控与指标收集"""
    print("=== 性能监控与指标收集演示 ===\n")
    
    # 初始化性能监控器
    monitor = PerformanceMonitor()
    
    # 开始收集系统指标
    monitor.metrics_collector.start_system_metrics_collection()
    
    # 模拟API调用
    @monitor.monitor_api_call("predict")
    def mock_predict_api():
        """模拟预测API"""
        time.sleep(0.1)  # 模拟处理时间
        # 模拟90%的成功率
        import random
        if random.random() < 0.9:
            return {"prediction": "success"}
        else:
            raise Exception("Prediction failed")
    
    @monitor.monitor_api_call("search")
    def mock_search_api():
        """模拟搜索API"""
        time.sleep(0.05)  # 模拟处理时间
        return {"results": ["item1", "item2", "item3"]}
    
    # 执行模拟API调用
    print("执行模拟API调用...")
    for i in range(20):
        try:
            if i % 3 == 0:
                mock_search_api()
            else:
                mock_predict_api()
        except Exception:
            pass  # 预期会有一些失败
    
    # 等待系统指标收集
    print("等待系统指标收集...")
    time.sleep(10)
    
    # 获取性能报告
    performance_report = monitor.get_performance_report()
    
    print("\n性能报告:")
    import json
    print(json.dumps(performance_report, indent=2, ensure_ascii=False))
    
    # 停止系统指标收集
    monitor.metrics_collector.stop_system_metrics_collection()
    
    print(f"\n总警报数量: {len(monitor.alerts)}")
    for alert in monitor.alerts:
        print(f"⚠️  {alert['message']}")


if __name__ == "__main__":
    demo_metrics_and_monitoring()