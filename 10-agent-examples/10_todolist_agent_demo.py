#!/usr/bin/env python3
"""
ToDoList Agent Demo

实现一个能够分解复杂任务、管理任务状态并执行工具的智能Agent。
基于LangGraph构建，支持任务分解、状态跟踪和工具调用。
"""

import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# 从当前模块目录加载 .env
def load_environment():
    """加载环境变量"""
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

# 获取配置的语言模型
def get_llm() -> ChatOpenAI:
    """创建并配置语言模型实例"""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    base_url = os.getenv("OPENAI_BASE_URL")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "512"))

    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 120,
        "max_retries": 3,
        "request_timeout": 120,
        "verbose": True,
        "base_url": base_url
    }

    return ChatOpenAI(**kwargs)

class ToDoListState(BaseModel):
    """ToDoList Agent状态"""
    original_task: str = Field(description="原始任务")
    subtasks: List[str] = Field(default_factory=list, description="分解后的子任务列表")
    current_subtask: Optional[str] = Field(default=None, description="当前正在处理的子任务")
    completed_tasks: List[str] = Field(default_factory=list, description="已完成的任务列表")
    pending_tasks: List[str] = Field(default_factory=list, description="待处理的任务列表")
    results: Dict[str, str] = Field(default_factory=dict, description="任务执行结果")
    final_answer: Optional[str] = Field(default=None, description="最终答案")


@tool
def search_web(query: str) -> str:
    """搜索网络获取信息"""
    # 模拟网络搜索
    search_results = {
        "Python教程": "Python是一种高级编程语言，具有简洁易读的语法。",
        "机器学习": "机器学习是人工智能的一个分支，让计算机从数据中学习。",
        "深度学习": "深度学习是机器学习的一个子集，使用神经网络进行学习。",
        "数据分析": "数据分析是检查、清理、转换和建模数据的过程。"
    }
    return search_results.get(query, f"未找到关于'{query}'的信息")


@tool
def calculate_math(expression: str) -> str:
    """计算数学表达式"""
    try:
        # 简单的数学表达式计算
        if "+" in expression:
            parts = expression.split("+")
            result = sum(float(p.strip()) for p in parts)
        elif "-" in expression:
            parts = expression.split("-")
            result = float(parts[0].strip()) - sum(float(p.strip()) for p in parts[1:])
        elif "*" in expression:
            parts = expression.split("*")
            result = 1
            for p in parts:
                result *= float(p.strip())
        elif "/" in expression:
            parts = expression.split("/")
            result = float(parts[0].strip())
            for p in parts[1:]:
                result /= float(p.strip())
        else:
            return f"无法计算表达式: {expression}"
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def write_file(content: str, filename: str) -> str:
    """写入文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"文件 '{filename}' 写入成功"
    except Exception as e:
        return f"文件写入失败: {str(e)}"


@tool
def read_file(filename: str) -> str:
    """读取文件"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"文件内容: {content}"
    except Exception as e:
        return f"文件读取失败: {str(e)}"



def task_decomposition_node(state: ToDoListState) -> ToDoListState:
    """任务分解节点"""
    llm = get_llm()
    
    prompt = f"""
    请将以下复杂任务分解为具体的子任务：
    
    原始任务: {state.original_task}
    
    请提供3-5个具体的子任务，每个子任务应该：
    1. 明确具体
    2. 可执行
    3. 有明确的完成标准
    
    请按编号列出子任务：
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    subtasks = [task.strip() for task in response.content.split('\n') if task.strip() and task.strip()[0].isdigit()]
    
    state.subtasks = subtasks
    state.pending_tasks = subtasks.copy()
    
    print(f"任务分解完成，共{len(subtasks)}个子任务:")
    for i, task in enumerate(subtasks, 1):
        print(f"  {i}. {task}")
    
    return state


def select_next_task_node(state: ToDoListState) -> ToDoListState:
    """选择下一个任务节点"""
    if state.pending_tasks:
        state.current_subtask = state.pending_tasks.pop(0)
        print(f"选择下一个任务: {state.current_subtask}")
    else:
        state.current_subtask = None
        print("所有任务已完成")
    
    return state


def execute_task_node(state: ToDoListState) -> ToDoListState:
    """执行任务节点"""
    if not state.current_subtask:
        return state
    
    llm = get_llm()
    
    prompt = f"""
    请执行以下任务：{state.current_subtask}
    
    可用的工具：
    - search_web: 搜索网络信息
    - calculate_math: 计算数学表达式
    - write_file: 写入文件
    - read_file: 读取文件
    
    请选择合适的工具并执行任务：
    """
    
    # 当前不使用 LLM 响应做工具选择，仅保留结构
    _ = llm.invoke([HumanMessage(content=prompt)])
    
    # 模拟工具执行（避免回调问题，直接调用底层函数）
    normalized = state.current_subtask.strip().lower()
    if ("搜索" in state.current_subtask) or ("search" in normalized):
        result = search_web.func("Python教程")
    elif ("计算" in state.current_subtask) or ("calculate" in normalized):
        result = calculate_math.func("10 + 20 + 30")
    elif ("读取" in state.current_subtask) or ("read" in normalized):
        result = read_file.func("learning_notes.txt")
    elif ("创建" in state.current_subtask) or ("写入" in state.current_subtask) or ("file" in normalized) or ("写" in state.current_subtask):
        result = write_file.func("这是学习笔记内容", "learning_notes.txt")
    else:
        result = f"任务 '{state.current_subtask}' 执行完成"
    
    state.results[state.current_subtask] = result
    state.completed_tasks.append(state.current_subtask)
    
    print(f"任务执行结果: {result}")
    
    return state


def final_answer_node(state: ToDoListState) -> ToDoListState:
    """最终答案节点"""
    llm = get_llm()
    
    prompt = f"""
    基于以下任务执行结果，生成最终答案：
    
    原始任务: {state.original_task}
    
    执行结果:
    """
    
    for task, result in state.results.items():
        prompt += f"\n- {task}: {result}"
    
    prompt += "\n\n请总结所有执行结果，生成完整的最终答案："
    
    response = llm.invoke([HumanMessage(content=prompt)])
    state.final_answer = response.content
    
    print(f"最终答案: {state.final_answer}")
    
    return state


def create_todolist_workflow():
    """创建ToDoList工作流"""
    workflow = StateGraph(ToDoListState)
    
    # 添加节点
    workflow.add_node("task_decomposition", task_decomposition_node)
    workflow.add_node("select_next_task", select_next_task_node)
    workflow.add_node("execute_task", execute_task_node)
    workflow.add_node("final_answer", final_answer_node)
    
    # 设置入口点
    workflow.set_entry_point("task_decomposition")
    
    # 添加边
    workflow.add_edge("task_decomposition", "select_next_task")
    workflow.add_conditional_edges(
        "select_next_task",
        lambda state: "execute_task" if state.current_subtask else "final_answer"
    )
    workflow.add_edge("execute_task", "select_next_task")
    workflow.add_edge("final_answer", END)
    
    return workflow.compile()


def run_todolist_example():
    load_environment()
    """运行ToDoList示例"""
    print("=== ToDoList Agent Demo ===")
    
    # 测试用例
    test_cases = [
        "学习Python编程并创建学习笔记",
        "进行数学计算和数据分析",
        "研究机器学习和深度学习"
    ]
    
    for i, task in enumerate(test_cases, 1):
        print(f"\n--- 测试用例 {i}: {task} ---")
        
        # 初始化状态
        initial_state = ToDoListState(original_task=task)
        
        # 创建工作流
        workflow = create_todolist_workflow()
        
        # 执行工作流
        try:
            final_state = workflow.invoke(initial_state)
            print(f"\n✅ 任务完成!")
            print(f"最终答案: {final_state['final_answer']}")
        except Exception as e:
            print(f"❌ 执行出错: {e}")


if __name__ == "__main__":
    run_todolist_example()