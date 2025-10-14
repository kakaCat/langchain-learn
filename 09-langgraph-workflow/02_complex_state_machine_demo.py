"""
复杂状态机设计演示
展示 LangGraph 中的复杂状态管理和工作流编排
"""

from typing import Dict, List, Any, Optional, TypedDict
from enum import Enum
from datetime import datetime
import json


class WorkflowState(TypedDict):
    """工作流状态定义"""
    current_step: str
    input_data: Dict[str, Any]
    intermediate_results: Dict[str, Any]
    error_info: Optional[Dict[str, Any]]
    retry_count: int
    max_retries: int
    execution_history: List[Dict[str, Any]]
    final_output: Optional[Dict[str, Any]]


class WorkflowStep(Enum):
    """工作流步骤枚举"""
    INITIALIZE = "initialize"
    DATA_PROCESSING = "data_processing"
    MODEL_INFERENCE = "model_inference"
    RESULT_VALIDATION = "result_validation"
    ERROR_HANDLING = "error_handling"
    FINALIZE = "finalize"


class WorkflowStatus(Enum):
    """工作流状态枚举"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class ComplexStateMachine:
    """复杂状态机管理器"""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.workflows: Dict[str, WorkflowState] = {}
    
    def initialize_workflow(self, workflow_id: str, input_data: Dict[str, Any]) -> WorkflowState:
        """初始化工作流"""
        state: WorkflowState = {
            "current_step": WorkflowStep.INITIALIZE.value,
            "input_data": input_data,
            "intermediate_results": {},
            "error_info": None,
            "retry_count": 0,
            "max_retries": self.max_retries,
            "execution_history": [],
            "final_output": None
        }
        
        self.workflows[workflow_id] = state
        self._log_execution(workflow_id, "初始化工作流", state)
        
        return state
    
    def execute_step(self, workflow_id: str, step: WorkflowStep) -> WorkflowState:
        """执行工作流步骤"""
        if workflow_id not in self.workflows:
            raise ValueError(f"工作流 {workflow_id} 不存在")
        
        state = self.workflows[workflow_id]
        state["current_step"] = step.value
        
        try:
            if step == WorkflowStep.INITIALIZE:
                state = self._initialize_step(workflow_id, state)
            elif step == WorkflowStep.DATA_PROCESSING:
                state = self._data_processing_step(workflow_id, state)
            elif step == WorkflowStep.MODEL_INFERENCE:
                state = self._model_inference_step(workflow_id, state)
            elif step == WorkflowStep.RESULT_VALIDATION:
                state = self._result_validation_step(workflow_id, state)
            elif step == WorkflowStep.ERROR_HANDLING:
                state = self._error_handling_step(workflow_id, state)
            elif step == WorkflowStep.FINALIZE:
                state = self._finalize_step(workflow_id, state)
            
            # 重置错误信息
            state["error_info"] = None
            
        except Exception as e:
            state["error_info"] = {
                "step": step.value,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            state["retry_count"] += 1
            
            if state["retry_count"] <= state["max_retries"]:
                state["current_step"] = WorkflowStep.ERROR_HANDLING.value
                self._log_execution(workflow_id, f"步骤 {step.value} 失败，准备重试", state)
            else:
                state["current_step"] = WorkflowStep.FINALIZE.value
                self._log_execution(workflow_id, f"步骤 {step.value} 失败，达到最大重试次数", state)
        
        self.workflows[workflow_id] = state
        return state
    
    def _initialize_step(self, workflow_id: str, state: WorkflowState) -> WorkflowState:
        """初始化步骤"""
        print(f"🔄 [{workflow_id}] 初始化工作流...")
        
        # 验证输入数据
        input_data = state["input_data"]
        required_fields = ["query", "model_type", "max_tokens"]
        
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"缺少必要字段: {field}")
        
        # 设置初始中间结果
        state["intermediate_results"]["start_time"] = datetime.now().isoformat()
        state["intermediate_results"]["query_type"] = self._classify_query(input_data["query"])
        
        self._log_execution(workflow_id, "初始化完成", state)
        return state
    
    def _data_processing_step(self, workflow_id: str, state: WorkflowState) -> WorkflowState:
        """数据处理步骤"""
        print(f"🔄 [{workflow_id}] 处理输入数据...")
        
        input_data = state["input_data"]
        query = input_data["query"]
        
        # 数据预处理
        processed_data = {
            "cleaned_query": query.strip().lower(),
            "query_length": len(query),
            "word_count": len(query.split()),
            "has_special_chars": any(c in query for c in "!@#$%^&*()"),
            "processing_time": datetime.now().isoformat()
        }
        
        state["intermediate_results"]["processed_data"] = processed_data
        
        # 模拟数据验证
        if len(query) < 3:
            raise ValueError("查询太短，请提供更详细的问题")
        
        self._log_execution(workflow_id, "数据处理完成", state)
        return state
    
    def _model_inference_step(self, workflow_id: str, state: WorkflowState) -> WorkflowState:
        """模型推理步骤"""
        print(f"🔄 [{workflow_id}] 执行模型推理...")
        
        input_data = state["input_data"]
        processed_data = state["intermediate_results"]["processed_data"]
        
        # 模拟模型推理
        model_type = input_data["model_type"]
        query = processed_data["cleaned_query"]
        
        # 根据查询类型生成不同响应
        query_type = state["intermediate_results"]["query_type"]
        
        if query_type == "factual":
            response = f"关于 '{query}' 的事实信息：这是一个基于事实的查询，需要准确的信息检索。"
        elif query_type == "creative":
            response = f"关于 '{query}' 的创意回答：这是一个需要创造力的查询，可以生成富有想象力的内容。"
        else:
            response = f"关于 '{query}' 的回答：这是一个常规查询，提供标准化的响应。"
        
        inference_result = {
            "model_type": model_type,
            "response": response,
            "confidence_score": 0.85,
            "inference_time": datetime.now().isoformat(),
            "tokens_used": len(response.split()),
            "query_type": query_type
        }
        
        state["intermediate_results"]["inference_result"] = inference_result
        
        self._log_execution(workflow_id, "模型推理完成", state)
        return state
    
    def _result_validation_step(self, workflow_id: str, state: WorkflowState) -> WorkflowState:
        """结果验证步骤"""
        print(f"🔄 [{workflow_id}] 验证推理结果...")
        
        inference_result = state["intermediate_results"]["inference_result"]
        
        # 验证结果质量
        validation_checks = {
            "response_not_empty": len(inference_result["response"].strip()) > 0,
            "confidence_threshold": inference_result["confidence_score"] > 0.5,
            "reasonable_length": 10 <= inference_result["tokens_used"] <= 500,
            "no_sensitive_content": not any(word in inference_result["response"].lower() 
                                          for word in ["密码", "密钥", "token"])
        }
        
        failed_checks = [check for check, passed in validation_checks.items() if not passed]
        
        if failed_checks:
            raise ValueError(f"验证失败: {failed_checks}")
        
        validation_result = {
            "passed_checks": len([check for check in validation_checks.values() if check]),
            "total_checks": len(validation_checks),
            "validation_time": datetime.now().isoformat(),
            "quality_score": sum([0.25 for check in validation_checks.values() if check])
        }
        
        state["intermediate_results"]["validation_result"] = validation_result
        
        self._log_execution(workflow_id, "结果验证完成", state)
        return state
    
    def _error_handling_step(self, workflow_id: str, state: WorkflowState) -> WorkflowState:
        """错误处理步骤"""
        print(f"🔄 [{workflow_id}] 处理错误...")
        
        error_info = state["error_info"]
        
        if error_info:
            print(f"   错误类型: {error_info['error_type']}")
            print(f"   错误信息: {error_info['error_message']}")
            
            # 根据错误类型决定下一步
            if "验证" in error_info["error_message"]:
                # 验证错误，返回数据预处理步骤
                state["current_step"] = WorkflowStep.DATA_PROCESSING.value
            elif "初始化" in error_info["error_message"]:
                # 初始化错误，需要重新开始
                state["current_step"] = WorkflowStep.INITIALIZE.value
            else:
                # 其他错误，尝试模型推理步骤
                state["current_step"] = WorkflowStep.MODEL_INFERENCE.value
        
        self._log_execution(workflow_id, "错误处理完成", state)
        return state
    
    def _finalize_step(self, workflow_id: str, state: WorkflowState) -> WorkflowState:
        """完成步骤"""
        print(f"🔄 [{workflow_id}] 完成工作流...")
        
        if state["error_info"]:
            # 工作流失败
            state["final_output"] = {
                "status": WorkflowStatus.FAILED.value,
                "error": state["error_info"],
                "execution_time": datetime.now().isoformat()
            }
        else:
            # 工作流成功
            inference_result = state["intermediate_results"]["inference_result"]
            validation_result = state["intermediate_results"]["validation_result"]
            
            state["final_output"] = {
                "status": WorkflowStatus.COMPLETED.value,
                "response": inference_result["response"],
                "confidence": inference_result["confidence_score"],
                "quality_score": validation_result["quality_score"],
                "execution_time": datetime.now().isoformat(),
                "total_steps": len(state["execution_history"])
            }
        
        self._log_execution(workflow_id, "工作流完成", state)
        return state
    
    def _classify_query(self, query: str) -> str:
        """分类查询类型"""
        factual_keywords = ["什么", "如何", "为什么", "何时", "哪里"]
        creative_keywords = ["想象", "创作", "故事", "诗歌", "创意"]
        
        if any(keyword in query for keyword in factual_keywords):
            return "factual"
        elif any(keyword in query for keyword in creative_keywords):
            return "creative"
        else:
            return "general"
    
    def _log_execution(self, workflow_id: str, action: str, state: WorkflowState):
        """记录执行历史"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "step": state["current_step"],
            "retry_count": state["retry_count"]
        }
        
        state["execution_history"].append(log_entry)
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """获取工作流状态"""
        if workflow_id not in self.workflows:
            return {"error": "工作流不存在"}
        
        state = self.workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "current_step": state["current_step"],
            "retry_count": state["retry_count"],
            "max_retries": state["max_retries"],
            "has_error": state["error_info"] is not None,
            "is_completed": state["final_output"] is not None,
            "execution_steps": len(state["execution_history"]),
            "final_output": state["final_output"]
        }


def demo_complex_state_machine():
    """演示复杂状态机"""
    print("🚀 启动复杂状态机演示")
    print("=" * 50)
    
    # 创建状态机实例
    state_machine = ComplexStateMachine(max_retries=2)
    
    # 测试用例1: 正常流程
    print("\n📋 测试用例1: 正常工作流")
    workflow_id = "test_normal_001"
    
    input_data = {
        "query": "什么是人工智能？",
        "model_type": "gpt-4",
        "max_tokens": 1000
    }
    
    # 初始化工作流
    state_machine.initialize_workflow(workflow_id, input_data)
    
    # 执行完整流程
    steps = [
        WorkflowStep.DATA_PROCESSING,
        WorkflowStep.MODEL_INFERENCE,
        WorkflowStep.RESULT_VALIDATION,
        WorkflowStep.FINALIZE
    ]
    
    for step in steps:
        state = state_machine.execute_step(workflow_id, step)
        print(f"   执行步骤: {step.value}")
        print(f"   当前状态: {state['current_step']}")
    
    # 查看最终结果
    status = state_machine.get_workflow_status(workflow_id)
    print(f"\n✅ 工作流完成:")
    print(f"   状态: {status['final_output']['status']}")
    print(f"   响应: {status['final_output']['response']}")
    print(f"   置信度: {status['final_output']['confidence']}")
    print(f"   质量分数: {status['final_output']['quality_score']}")
    
    # 测试用例2: 带错误的重试流程
    print("\n📋 测试用例2: 带错误的重试流程")
    workflow_id2 = "test_error_002"
    
    input_data2 = {
        "query": "AI",  # 太短的查询会触发验证错误
        "model_type": "gpt-4",
        "max_tokens": 1000
    }
    
    state_machine.initialize_workflow(workflow_id2, input_data2)
    
    # 执行流程（会触发错误和重试）
    steps2 = [
        WorkflowStep.DATA_PROCESSING,  # 这里会触发验证错误
        WorkflowStep.ERROR_HANDLING,   # 错误处理
        WorkflowStep.DATA_PROCESSING,  # 重试数据处理
        WorkflowStep.MODEL_INFERENCE,
        WorkflowStep.RESULT_VALIDATION,
        WorkflowStep.FINALIZE
    ]
    
    for step in steps2:
        state = state_machine.execute_step(workflow_id2, step)
        print(f"   执行步骤: {step.value}")
        print(f"   当前状态: {state['current_step']}")
        if state['error_info']:
            print(f"   错误信息: {state['error_info']['error_message']}")
    
    # 查看最终结果
    status2 = state_machine.get_workflow_status(workflow_id2)
    print(f"\n🔄 工作流完成（带重试）:")
    print(f"   状态: {status2['final_output']['status']}")
    print(f"   重试次数: {status2['retry_count']}")
    print(f"   执行步骤数: {status2['execution_steps']}")
    
    # 测试用例3: 分布式工作流编排演示
    print("\n📋 测试用例3: 分布式工作流编排")
    workflow_ids = ["distributed_001", "distributed_002", "distributed_003"]
    
    queries = [
        "机器学习的基本原理是什么？",
        "创作一首关于春天的诗歌",
        "深度学习在医疗领域的应用"
    ]
    
    print("   启动分布式工作流执行...")
    
    # 并行初始化多个工作流
    for i, (wf_id, query) in enumerate(zip(workflow_ids, queries)):
        input_data = {
            "query": query,
            "model_type": "gpt-4",
            "max_tokens": 500
        }
        state_machine.initialize_workflow(wf_id, input_data)
        print(f"   工作流 {wf_id} 已初始化")
    
    # 并行执行数据处理步骤
    print("\n   并行执行数据处理步骤...")
    for wf_id in workflow_ids:
        state_machine.execute_step(wf_id, WorkflowStep.DATA_PROCESSING)
        print(f"   工作流 {wf_id} 数据处理完成")
    
    # 并行执行模型推理步骤
    print("\n   并行执行模型推理步骤...")
    for wf_id in workflow_ids:
        state_machine.execute_step(wf_id, WorkflowStep.MODEL_INFERENCE)
        print(f"   工作流 {wf_id} 模型推理完成")
    
    # 完成所有工作流
    print("\n   完成所有工作流...")
    for wf_id in workflow_ids:
        state_machine.execute_step(wf_id, WorkflowStep.RESULT_VALIDATION)
        state_machine.execute_step(wf_id, WorkflowStep.FINALIZE)
        
        status = state_machine.get_workflow_status(wf_id)
        print(f"   工作流 {wf_id}: {status['final_output']['status']}")
    
    print("\n" + "=" * 50)
    print("🎉 复杂状态机演示完成！")
    print("\n📊 演示总结:")
    print("   • 正常流程执行: 展示了完整的工作流执行")
    print("   • 错误处理与重试: 演示了错误检测和自动重试机制")
    print("   • 分布式编排: 展示了并行工作流执行能力")
    print("   • 状态管理: 实现了复杂的状态跟踪和转换")


if __name__ == "__main__":
    demo_complex_state_machine()