#!/usr/bin/env python3
"""
调用追踪与结构化日志演示
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import contextmanager


@dataclass
class TraceSpan:
    """追踪跨度"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "pending"
    tags: Dict[str, Any] = None
    logs: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.logs is None:
            self.logs = []


class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str = "eval_deploy"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str, extra: Dict[str, Any] = None):
        """记录信息级别日志"""
        log_data = {"message": message}
        if extra:
            log_data.update(extra)
        self.logger.info(json.dumps(log_data, ensure_ascii=False, default=str))
    
    def error(self, message: str, extra: Dict[str, Any] = None):
        """记录错误级别日志"""
        log_data = {"message": message, "level": "ERROR"}
        if extra:
            log_data.update(extra)
        self.logger.error(json.dumps(log_data, ensure_ascii=False, default=str))
    
    def warn(self, message: str, extra: Dict[str, Any] = None):
        """记录警告级别日志"""
        log_data = {"message": message, "level": "WARN"}
        if extra:
            log_data.update(extra)
        self.logger.warning(json.dumps(log_data, ensure_ascii=False, default=str))


class TraceManager:
    """追踪管理器"""
    
    def __init__(self):
        self.spans: List[TraceSpan] = []
        self.logger = StructuredLogger("tracing")
    
    def start_span(self, operation: str, parent_span: Optional[TraceSpan] = None, 
                   trace_id: Optional[str] = None) -> TraceSpan:
        """开始新的追踪跨度"""
        import uuid
        
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        parent_span_id = parent_span.span_id if parent_span else None
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=datetime.now()
        )
        
        self.spans.append(span)
        
        self.logger.info(f"Started span: {operation}", {
            "trace_id": trace_id,
            "span_id": span_id,
            "operation": operation
        })
        
        return span
    
    def end_span(self, span: TraceSpan, status: str = "success", 
                 error_message: Optional[str] = None):
        """结束追踪跨度"""
        span.end_time = datetime.now()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status
        
        if error_message:
            span.logs.append({
                "timestamp": datetime.now(),
                "level": "ERROR",
                "message": error_message
            })
        
        self.logger.info(f"Ended span: {span.operation}", {
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "duration_ms": span.duration_ms,
            "status": status
        })
    
    @contextmanager
    def span(self, operation: str, parent_span: Optional[TraceSpan] = None):
        """追踪跨度上下文管理器"""
        span = self.start_span(operation, parent_span)
        try:
            yield span
            self.end_span(span, "success")
        except Exception as e:
            self.end_span(span, "error", str(e))
            raise
    
    def get_trace_report(self, trace_id: str) -> Dict[str, Any]:
        """获取追踪报告"""
        trace_spans = [span for span in self.spans if span.trace_id == trace_id]
        
        if not trace_spans:
            return {"error": "Trace not found"}
        
        # 构建追踪树
        span_map = {span.span_id: span for span in trace_spans}
        root_spans = [span for span in trace_spans if span.parent_span_id is None]
        
        def build_span_tree(span: TraceSpan) -> Dict[str, Any]:
            children = [build_span_tree(child) for child in trace_spans 
                       if child.parent_span_id == span.span_id]
            
            return {
                "operation": span.operation,
                "span_id": span.span_id,
                "start_time": span.start_time.isoformat(),
                "end_time": span.end_time.isoformat() if span.end_time else None,
                "duration_ms": span.duration_ms,
                "status": span.status,
                "tags": span.tags,
                "children": children
            }
        
        trace_tree = [build_span_tree(root) for root in root_spans]
        
        return {
            "trace_id": trace_id,
            "total_spans": len(trace_spans),
            "success_rate": len([s for s in trace_spans if s.status == "success"]) / len(trace_spans),
            "total_duration_ms": sum(s.duration_ms or 0 for s in trace_spans),
            "trace_tree": trace_tree
        }


def demo_tracing_and_logging():
    """演示追踪与日志功能"""
    print("=== 调用追踪与结构化日志演示 ===\n")
    
    # 初始化追踪管理器和日志记录器
    trace_manager = TraceManager()
    logger = StructuredLogger("demo")
    
    # 记录开始信息
    logger.info("开始追踪演示", {"demo_type": "tracing_and_logging"})
    
    # 创建根跨度
    with trace_manager.span("demo_workflow") as root_span:
        
        # 第一步：数据准备
        with trace_manager.span("data_preparation", root_span) as data_span:
            logger.info("准备测试数据", {"step": "data_preparation"})
            time.sleep(0.1)  # 模拟处理时间
            data_span.tags["data_size"] = 100
            data_span.tags["data_type"] = "test"
        
        # 第二步：模型评估
        with trace_manager.span("model_evaluation", root_span) as eval_span:
            logger.info("执行模型评估", {"step": "model_evaluation"})
            time.sleep(0.2)  # 模拟评估时间
            
            # 模拟评估结果
            eval_span.tags["accuracy"] = 0.85
            eval_span.tags["latency_ms"] = 150
            
            # 记录评估详情
            logger.info("评估完成", {
                "accuracy": 0.85,
                "latency_ms": 150,
                "model": "test_model_v1"
            })
        
        # 第三步：报告生成
        with trace_manager.span("report_generation", root_span) as report_span:
            logger.info("生成评估报告", {"step": "report_generation"})
            time.sleep(0.05)  # 模拟报告生成时间
            
            # 模拟报告数据
            report_data = {
                "summary": "评估完成",
                "metrics": {"accuracy": 0.85, "f1_score": 0.82},
                "timestamp": datetime.now().isoformat()
            }
            
            report_span.tags["report_size"] = len(json.dumps(report_data))
            logger.info("报告生成完成", {"report_size": len(json.dumps(report_data))})
    
    # 获取追踪报告
    trace_report = trace_manager.get_trace_report(root_span.trace_id)
    
    print("\n追踪报告:")
    print(json.dumps(trace_report, indent=2, ensure_ascii=False))
    
    # 记录结束信息
    logger.info("追踪演示完成", {
        "total_spans": len(trace_manager.spans),
        "trace_id": root_span.trace_id
    })


if __name__ == "__main__":
    demo_tracing_and_logging()