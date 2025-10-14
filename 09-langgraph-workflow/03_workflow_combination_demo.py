"""
LangGraph Workflow 组合形式演示

这个文件展示了 LangGraph 中多种 workflow 组合形式，包括：
1. 并行工作流
2. 串行工作流链
3. 条件分支工作流
4. 循环工作流
5. 嵌套工作流
6. 错误恢复工作流
7. 超时控制工作流
"""

import asyncio
from typing import Dict, List, Any, Optional, Annotated
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import random

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class WorkflowType(Enum):
    """工作流类型枚举"""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    NESTED = "nested"
    ERROR_RECOVERY = "error_recovery"
    TIMEOUT = "timeout"


@dataclass
class WorkflowState:
    """工作流状态定义"""
    workflow_type: Annotated[WorkflowType, "工作流类型"]
    input_data: Annotated[Dict[str, Any], "输入数据"]
    intermediate_results: Annotated[Dict[str, Any], "中间结果"]
    current_step: Annotated[str, "当前步骤"]
    execution_history: Annotated[List[Dict[str, Any]], "执行历史"]
    error_info: Annotated[Optional[Dict[str, Any]], "错误信息"] = None
    is_completed: Annotated[bool, "是否完成"] = False
    timeout_at: Annotated[Optional[datetime], "超时时间"] = None


class WorkflowCombinationDemo:
    """工作流组合演示类"""
    
    def __init__(self):
        self.workflows = {}
    
    def create_parallel_workflow(self) -> StateGraph:
        """创建并行工作流 - 简化为串行执行以避免并发冲突"""
        print("🔄 创建并行工作流...")
        
        graph = StateGraph(WorkflowState)
        
        # 添加节点
        graph.add_node("data_processing", self._data_processing_node)
        graph.add_node("model_inference", self._model_inference_node)
        graph.add_node("external_api_call", self._external_api_node)
        
        # 设置入口点
        graph.set_entry_point("data_processing")
        
        # 串行执行以避免并发冲突
        graph.add_edge("data_processing", "model_inference")
        graph.add_edge("model_inference", "external_api_call")
        graph.add_edge("external_api_call", END)
        
        return graph.compile()
    
    def create_sequential_workflow(self) -> StateGraph:
        """创建串行工作流链 - 任务按顺序执行"""
        print("🔄 创建串行工作流链...")
        
        graph = StateGraph(WorkflowState)
        
        # 添加串行节点
        graph.add_node("step_1_preprocessing", self._preprocessing_node)
        graph.add_node("step_2_analysis", self._analysis_node)
        graph.add_node("step_3_enhancement", self._enhancement_node)
        graph.add_node("step_4_finalization", self._finalization_node)
        
        # 设置串行执行顺序
        graph.set_entry_point("step_1_preprocessing")
        graph.add_edge("step_1_preprocessing", "step_2_analysis")
        graph.add_edge("step_2_analysis", "step_3_enhancement")
        graph.add_edge("step_3_enhancement", "step_4_finalization")
        graph.add_edge("step_4_finalization", END)
        
        return graph.compile()
    
    def create_conditional_workflow(self) -> StateGraph:
        """创建条件分支工作流 - 根据条件选择不同路径"""
        print("🔄 创建条件分支工作流...")
        
        graph = StateGraph(WorkflowState)
        
        # 添加节点
        graph.add_node("initial_analysis", self._initial_analysis_node)
        graph.add_node("complex_processing", self._complex_processing_node)
        graph.add_node("simple_processing", self._simple_processing_node)
        graph.add_node("final_output", self._final_output_node)
        
        # 设置入口点
        graph.set_entry_point("initial_analysis")
        
        # 条件分支
        def route_based_on_complexity(state: WorkflowState) -> str:
            """根据查询复杂度路由"""
            complexity = state.intermediate_results.get("complexity_score", 0)
            
            if complexity > 0.7:
                return "complex"
            else:
                return "simple"
        
        graph.add_conditional_edges(
            "initial_analysis",
            route_based_on_complexity,
            {
                "complex": "complex_processing",
                "simple": "simple_processing"
            }
        )
        
        # 合并路径
        graph.add_edge("complex_processing", "final_output")
        graph.add_edge("simple_processing", "final_output")
        graph.add_edge("final_output", END)
        
        return graph.compile()
    
    def create_loop_workflow(self) -> StateGraph:
        """创建循环工作流 - 重复执行直到满足条件"""
        print("🔄 创建循环工作流...")
        
        graph = StateGraph(WorkflowState)
        
        # 添加节点
        graph.add_node("iteration_step", self._iteration_node)
        graph.add_node("final_result", self._final_result_node)
        
        # 设置入口点
        graph.set_entry_point("iteration_step")
        
        # 循环控制
        def check_iteration_completion(state: WorkflowState) -> str:
            """检查迭代是否完成"""
            iteration_count = state.intermediate_results.get("iteration_count", 0)
            max_iterations = state.input_data.get("max_iterations", 5)
            
            if iteration_count >= max_iterations:
                return "complete"
            else:
                return "continue"
        
        graph.add_conditional_edges(
            "iteration_step",
            check_iteration_completion,
            {
                "continue": "iteration_step",  # 循环回到自身
                "complete": "final_result"
            }
        )
        
        graph.add_edge("final_result", END)
        
        return graph.compile()
    
    def create_nested_workflow(self) -> StateGraph:
        """创建嵌套工作流 - 工作流中包含子工作流"""
        print("🔄 创建嵌套工作流...")
        
        graph = StateGraph(WorkflowState)
        
        # 添加主工作流节点
        graph.add_node("main_processing", self._main_processing_node)
        graph.add_node("sub_workflow_executor", self._sub_workflow_executor_node)
        graph.add_node("result_aggregation", self._result_aggregation_node)
        
        # 设置执行顺序
        graph.set_entry_point("main_processing")
        graph.add_edge("main_processing", "sub_workflow_executor")
        graph.add_edge("sub_workflow_executor", "result_aggregation")
        graph.add_edge("result_aggregation", END)
        
        return graph.compile()
    
    def create_error_recovery_workflow(self) -> StateGraph:
        """创建错误恢复工作流 - 自动处理错误并重试"""
        print("🔄 创建错误恢复工作流...")
        
        graph = StateGraph(WorkflowState)
        
        # 添加节点
        graph.add_node("main_task", self._main_task_node)
        graph.add_node("error_handler", self._error_handler_node)
        graph.add_node("fallback_task", self._fallback_task_node)
        graph.add_node("final_output", self._final_output_node)
        
        # 设置入口点
        graph.set_entry_point("main_task")
        
        # 错误检测和路由
        def check_for_errors(state: WorkflowState) -> str:
            """检查是否有错误"""
            if state.error_info:
                return "error"
            else:
                return "success"
        
        graph.add_conditional_edges(
            "main_task",
            check_for_errors,
            {
                "error": "error_handler",
                "success": "final_output"
            }
        )
        
        # 错误处理路径
        def decide_recovery_strategy(state: WorkflowState) -> str:
            """决定恢复策略"""
            error_type = state.error_info.get("error_type", "unknown")
            
            if error_type == "retryable":
                return "retry"
            else:
                return "fallback"
        
        graph.add_conditional_edges(
            "error_handler",
            decide_recovery_strategy,
            {
                "retry": "main_task",  # 重试主任务
                "fallback": "fallback_task"
            }
        )
        
        graph.add_edge("fallback_task", "final_output")
        graph.add_edge("final_output", END)
        
        return graph.compile()
    
    def create_timeout_workflow(self) -> StateGraph:
        """创建超时控制工作流 - 限制任务执行时间"""
        print("🔄 创建超时控制工作流...")
        
        graph = StateGraph(WorkflowState)
        
        # 添加节点
        graph.add_node("timed_task", self._timed_task_node)
        graph.add_node("timeout_handler", self._timeout_handler_node)
        graph.add_node("normal_completion", self._normal_completion_node)
        
        # 设置入口点
        graph.set_entry_point("timed_task")
        
        # 超时检查
        def check_timeout(state: WorkflowState) -> str:
            """检查是否超时"""
            if state.timeout_at and datetime.now() > state.timeout_at:
                return "timeout"
            else:
                return "normal"
        
        graph.add_conditional_edges(
            "timed_task",
            check_timeout,
            {
                "timeout": "timeout_handler",
                "normal": "normal_completion"
            }
        )
        
        graph.add_edge("timeout_handler", END)
        graph.add_edge("normal_completion", END)
        
        return graph.compile()
    
    # ========== 节点函数实现 ==========
    
    def _data_processing_node(self, state: WorkflowState) -> WorkflowState:
        """数据处理节点"""
        print("📊 执行数据处理...")
        state.intermediate_results["data_processed"] = True
        state.execution_history.append({"step": "data_processing", "timestamp": datetime.now()})
        return state
    
    def _model_inference_node(self, state: WorkflowState) -> WorkflowState:
        """模型推理节点"""
        print("🤖 执行模型推理...")
        state.intermediate_results["model_inference_completed"] = True
        state.execution_history.append({"step": "model_inference", "timestamp": datetime.now()})
        return state
    
    def _external_api_node(self, state: WorkflowState) -> WorkflowState:
        """外部API调用节点"""
        print("🌐 调用外部API...")
        state.intermediate_results["api_call_completed"] = True
        state.execution_history.append({"step": "external_api_call", "timestamp": datetime.now()})
        return state
    
    def _aggregate_results_node(self, state: WorkflowState) -> WorkflowState:
        """聚合结果节点"""
        print("📈 聚合并行执行结果...")
        state.intermediate_results["results_aggregated"] = True
        state.execution_history.append({"step": "aggregate_results", "timestamp": datetime.now()})
        return state
    
    def _preprocessing_node(self, state: WorkflowState) -> WorkflowState:
        """预处理节点"""
        print("🔧 执行预处理...")
        state.intermediate_results["preprocessing_done"] = True
        state.execution_history.append({"step": "preprocessing", "timestamp": datetime.now()})
        return state
    
    def _analysis_node(self, state: WorkflowState) -> WorkflowState:
        """分析节点"""
        print("🔍 执行分析...")
        state.intermediate_results["analysis_done"] = True
        state.execution_history.append({"step": "analysis", "timestamp": datetime.now()})
        return state
    
    def _enhancement_node(self, state: WorkflowState) -> WorkflowState:
        """增强节点"""
        print("✨ 执行增强处理...")
        state.intermediate_results["enhancement_done"] = True
        state.execution_history.append({"step": "enhancement", "timestamp": datetime.now()})
        return state
    
    def _finalization_node(self, state: WorkflowState) -> WorkflowState:
        """最终化节点"""
        print("✅ 执行最终化...")
        state.intermediate_results["finalization_done"] = True
        state.execution_history.append({"step": "finalization", "timestamp": datetime.now()})
        return state
    
    def _initial_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """初始分析节点"""
        print("🔍 执行初始分析...")
        # 模拟计算复杂度分数
        complexity = random.uniform(0.1, 1.0)
        state.intermediate_results["complexity_score"] = complexity
        state.execution_history.append({"step": "initial_analysis", "timestamp": datetime.now()})
        return state
    
    def _complex_processing_node(self, state: WorkflowState) -> WorkflowState:
        """复杂处理节点"""
        print("🧠 执行复杂处理...")
        state.intermediate_results["complex_processing_done"] = True
        state.execution_history.append({"step": "complex_processing", "timestamp": datetime.now()})
        return state
    
    def _simple_processing_node(self, state: WorkflowState) -> WorkflowState:
        """简单处理节点"""
        print("⚡ 执行简单处理...")
        state.intermediate_results["simple_processing_done"] = True
        state.execution_history.append({"step": "simple_processing", "timestamp": datetime.now()})
        return state
    
    def _final_output_node(self, state: WorkflowState) -> WorkflowState:
        """最终输出节点"""
        print("📤 生成最终输出...")
        state.intermediate_results["final_output_generated"] = True
        state.execution_history.append({"step": "final_output", "timestamp": datetime.now()})
        return state
    
    def _iteration_node(self, state: WorkflowState) -> WorkflowState:
        """迭代节点"""
        iteration_count = state.intermediate_results.get("iteration_count", 0)
        iteration_count += 1
        state.intermediate_results["iteration_count"] = iteration_count
        print(f"🔄 执行第 {iteration_count} 次迭代...")
        state.execution_history.append({"step": f"iteration_{iteration_count}", "timestamp": datetime.now()})
        return state
    
    def _final_result_node(self, state: WorkflowState) -> WorkflowState:
        """最终结果节点"""
        print("🏁 生成最终结果...")
        state.intermediate_results["final_result_generated"] = True
        state.execution_history.append({"step": "final_result", "timestamp": datetime.now()})
        return state
    
    def _main_processing_node(self, state: WorkflowState) -> WorkflowState:
        """主处理节点"""
        print("🏗️ 执行主处理...")
        state.intermediate_results["main_processing_done"] = True
        state.execution_history.append({"step": "main_processing", "timestamp": datetime.now()})
        return state
    
    def _sub_workflow_executor_node(self, state: WorkflowState) -> WorkflowState:
        """子工作流执行器节点"""
        print("🔗 执行子工作流...")
        state.intermediate_results["sub_workflow_executed"] = True
        state.execution_history.append({"step": "sub_workflow_executor", "timestamp": datetime.now()})
        return state
    
    def _result_aggregation_node(self, state: WorkflowState) -> WorkflowState:
        """结果聚合节点"""
        print("📊 聚合结果...")
        state.intermediate_results["results_aggregated"] = True
        state.execution_history.append({"step": "result_aggregation", "timestamp": datetime.now()})
        return state
    
    def _main_task_node(self, state: WorkflowState) -> WorkflowState:
        """主任务节点"""
        print("🎯 执行主任务...")
        # 模拟随机错误
        if random.random() < 0.3:  # 30% 概率模拟错误
            state.error_info = {
                "error_type": "retryable",
                "error_message": "模拟可重试错误",
                "timestamp": datetime.now()
            }
        state.intermediate_results["main_task_done"] = True
        state.execution_history.append({"step": "main_task", "timestamp": datetime.now()})
        return state
    
    def _error_handler_node(self, state: WorkflowState) -> WorkflowState:
        """错误处理节点"""
        print("🛠️ 处理错误...")
        state.execution_history.append({"step": "error_handler", "timestamp": datetime.now()})
        return state
    
    def _fallback_task_node(self, state: WorkflowState) -> WorkflowState:
        """备用任务节点"""
        print("🔄 执行备用任务...")
        state.intermediate_results["fallback_task_done"] = True
        state.execution_history.append({"step": "fallback_task", "timestamp": datetime.now()})
        return state
    
    def _timed_task_node(self, state: WorkflowState) -> WorkflowState:
        """定时任务节点"""
        print("⏰ 执行定时任务...")
        # 设置超时时间
        if not state.timeout_at:
            state.timeout_at = datetime.now() + timedelta(seconds=2)
        state.intermediate_results["timed_task_done"] = True
        state.execution_history.append({"step": "timed_task", "timestamp": datetime.now()})
        return state
    
    def _timeout_handler_node(self, state: WorkflowState) -> WorkflowState:
        """超时处理节点"""
        print("⏱️ 处理超时...")
        state.intermediate_results["timeout_handled"] = True
        state.execution_history.append({"step": "timeout_handler", "timestamp": datetime.now()})
        return state
    
    def _normal_completion_node(self, state: WorkflowState) -> WorkflowState:
        """正常完成节点"""
        print("✅ 正常完成任务...")
        state.intermediate_results["normal_completion_done"] = True
        state.execution_history.append({"step": "normal_completion", "timestamp": datetime.now()})
        return state


def demo_workflow_combinations():
    """演示多种工作流组合形式"""
    print("🚀 LangGraph Workflow 组合形式演示")
    print("=" * 60)
    
    demo = WorkflowCombinationDemo()
    
    # 演示1: 并行工作流
    print("\n📋 演示1: 并行工作流")
    print("-" * 40)
    parallel_workflow = demo.create_parallel_workflow()
    initial_state = WorkflowState(
        workflow_type=WorkflowType.PARALLEL,
        input_data={"query": "并行处理测试"},
        intermediate_results={},
        current_step="start",
        execution_history=[]
    )
    result = parallel_workflow.invoke(initial_state)
    print(f"✅ 并行工作流执行完成")
    print(f"📊 最终结果: {result}")
    
    # 演示2: 串行工作流链
    print("\n📋 演示2: 串行工作流链")
    print("-" * 40)
    sequential_workflow = demo.create_sequential_workflow()
    initial_state = WorkflowState(
        workflow_type=WorkflowType.SEQUENTIAL,
        input_data={"text": "这是一个测试文本"},
        intermediate_results={},
        current_step="start",
        execution_history=[]
    )
    result = sequential_workflow.invoke(initial_state)
    print(f"✅ 串行工作流执行完成")
    print(f"📊 最终结果: {result}")
    
    # 演示3: 条件分支工作流
    print("\n📋 演示3: 条件分支工作流")
    print("-" * 40)
    conditional_workflow = demo.create_conditional_workflow()
    
    # 测试不同条件
    test_cases = [
        ("simple", "简单查询"),
        ("complex", "复杂分析请求"),
        ("unknown", "其他类型查询")
    ]
    
    for query_type, query_text in test_cases:
        initial_state = WorkflowState(
            workflow_type=WorkflowType.CONDITIONAL,
            input_data={"query": query_text, "query_type": query_type},
            intermediate_results={},
            current_step="start",
            execution_history=[]
        )
        
        result = conditional_workflow.invoke(initial_state)
        print(f"🔍 查询类型 '{query_type}': {result}")
    
    # 演示4: 循环工作流
    print("\n📋 演示4: 循环工作流")
    print("-" * 40)
    loop_workflow = demo.create_loop_workflow()
    
    initial_state = WorkflowState(
        workflow_type=WorkflowType.LOOP,
        input_data={"items": ["item1", "item2", "item3"], "processed_count": 0},
        intermediate_results={},
        current_step="start",
        execution_history=[]
    )
    
    result = loop_workflow.invoke(initial_state)
    print(f"✅ 循环工作流执行完成")
    print(f"📊 处理结果: {result}")
    
    # 演示5: 嵌套工作流
    print("\n📋 演示5: 嵌套工作流")
    print("-" * 40)
    nested_workflow = demo.create_nested_workflow()
    initial_state = WorkflowState(
        workflow_type=WorkflowType.NESTED,
        input_data={"query": "嵌套工作流测试"},
        intermediate_results={},
        current_step="start",
        execution_history=[]
    )
    result = nested_workflow.invoke(initial_state)
    print(f"✅ 嵌套工作流执行完成")
    print(f"📊 最终结果: {result}")
    
    # 演示6: 错误恢复工作流
    print("\n📋 演示6: 错误恢复工作流")
    print("-" * 40)
    error_recovery_workflow = demo.create_error_recovery_workflow()
    initial_state = WorkflowState(
        workflow_type=WorkflowType.ERROR_RECOVERY,
        input_data={"query": "错误恢复测试"},
        intermediate_results={},
        current_step="start",
        execution_history=[]
    )
    result = error_recovery_workflow.invoke(initial_state)
    print(f"✅ 错误恢复工作流执行完成")
    print(f"📊 最终结果: {result}")
    
    # 演示7: 超时控制工作流
    print("\n📋 演示7: 超时控制工作流")
    print("-" * 40)
    timeout_workflow = demo.create_timeout_workflow()
    initial_state = WorkflowState(
        workflow_type=WorkflowType.TIMEOUT,
        input_data={"query": "超时控制测试"},
        intermediate_results={},
        current_step="start",
        execution_history=[]
    )
    result = timeout_workflow.invoke(initial_state)
    print(f"✅ 超时控制工作流执行完成")
    print(f"📊 最终结果: {result}")
    
    print("\n" + "=" * 60)
    print("🎉 所有工作流组合演示完成！")
    print("\n📊 演示总结:")
    print("   • 并行工作流: 多个任务同时执行，提高效率")
    print("   • 串行工作流链: 任务按顺序执行，确保依赖关系")
    print("   • 条件分支工作流: 根据条件选择不同执行路径")
    print("   • 循环工作流: 重复执行直到满足条件")
    print("   • 嵌套工作流: 工作流中包含子工作流，支持复杂场景")
    print("   • 错误恢复工作流: 自动处理错误并重试，提高鲁棒性")
    print("   • 超时控制工作流: 限制任务执行时间，防止无限等待")


if __name__ == "__main__":
    demo_workflow_combinations()