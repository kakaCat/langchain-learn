"""
STORM (Synthesis of Thought with Observation and Reflection for Multi-step reasoning) 模式示例

STORM 模式特点：
- 结合思考、观察和反思的多步推理
- 适用于复杂问题的分解和解决
- 通过反思步骤优化推理过程
- 支持多轮迭代推理
"""

import os
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from dataclasses import dataclass
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

class StormState(BaseModel):
    """STORM 模式的状态定义"""
    question: str = Field(description="用户问题")
    thoughts: List[str] = Field(default_factory=list, description="思考步骤")
    observations: List[str] = Field(default_factory=list, description="观察结果")
    reflections: List[str] = Field(default_factory=list, description="反思内容")
    final_answer: Optional[str] = Field(default=None, description="最终答案")





def thought_node(state: StormState) -> StormState:
    """思考节点 - 分析问题并生成思考步骤"""
    agent = get_llm()
    
    if not agent:
        # 模拟模式
        if len(state.thoughts) == 0:
            state.thoughts.append("分析问题结构，识别关键要素")
        elif len(state.thoughts) == 1:
            state.thoughts.append("分解问题为可管理的子问题")
        else:
            state.thoughts.append("整合信息，形成推理路径")
        return state
    
    # 实际 LLM 调用
    prompt = f"""
    问题: {state.question}
    
    当前思考步骤: {state.thoughts}
    当前观察: {state.observations}
    当前反思: {state.reflections}
    
    请生成下一步的思考内容，帮助解决这个问题。
    """
    
    response = agent.invoke([HumanMessage(content=prompt)])
    state.thoughts.append(response.content)
    return state


def observation_node(state: StormState) -> StormState:
    """观察节点 - 基于思考生成观察结果"""
    agent = get_llm()
    
    if not agent:
        # 模拟模式
        current_thought = state.thoughts[-1] if state.thoughts else "初始思考"
        state.observations.append(f"基于思考 '{current_thought}' 的观察结果")
        return state
    
    prompt = f"""
    问题: {state.question}
    最新思考: {state.thoughts[-1] if state.thoughts else '无'}
    
    请基于这个思考，生成具体的观察结果。
    """
    
    response = agent.invoke([HumanMessage(content=prompt)])
    state.observations.append(response.content)
    return state


def reflection_node(state: StormState) -> StormState:
    """反思节点 - 评估当前进展并优化推理"""
    agent = get_llm()
    
    if not agent:
        # 模拟模式
        state.reflections.append("评估当前推理进展，识别可能的改进方向")
        return state
    
    prompt = f"""
    问题: {state.question}
    所有思考: {state.thoughts}
    所有观察: {state.observations}
    
    请反思当前的推理过程，提出改进建议或识别潜在问题。
    """
    
    response = agent.invoke([HumanMessage(content=prompt)])
    state.reflections.append(response.content)
    return state


def decision_node(state: StormState) -> str:
    """决策节点 - 决定是否继续推理或结束"""
    if len(state.thoughts) >= 3:
        return "end"
    return "continue"


def final_answer_node(state: StormState) -> StormState:
    """最终答案节点 - 生成最终答案"""
    agent = get_llm()
    
    if not agent:
        # 模拟模式
        state.final_answer = f"基于 {len(state.thoughts)} 步思考、{len(state.observations)} 次观察和 {len(state.reflections)} 次反思，得出最终答案"
        return state
    
    prompt = f"""
    问题: {state.question}
    思考过程: {state.thoughts}
    观察结果: {state.observations}
    反思内容: {state.reflections}
    
    请基于以上信息，给出最终答案。
    """
    
    response = agent.invoke([HumanMessage(content=prompt)])
    state.final_answer = response.content
    return state


def create_storm_workflow():
    """创建 STORM 工作流"""
    workflow = StateGraph(StormState)
    
    # 添加节点
    workflow.add_node("thought", thought_node)
    workflow.add_node("observation", observation_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("final_answer", final_answer_node)
    
    # 设置入口点
    workflow.set_entry_point("thought")
    
    # 添加边
    workflow.add_edge("thought", "observation")
    workflow.add_edge("observation", "reflection")
    workflow.add_conditional_edges(
        "reflection",
        decision_node,
        {
            "continue": "thought",
            "end": "final_answer"
        }
    )
    workflow.add_edge("final_answer", END)
    
    return workflow.compile()


def run_storm_example():
    """运行 STORM 示例"""
    load_environment()
    
    # 测试用例
    test_cases = [
        "如何制定一个有效的学习计划来掌握一门新技能？",
        "分析气候变化对全球经济的影响",
        "设计一个可持续发展的城市交通系统"
    ]
    
    for i, question in enumerate(test_cases, 1):
        print(f"测试用例 {i}:")
        print(f"问题: {question}")
        
        # 创建状态
        initial_state = StormState(question=question)
        
        # 运行工作流
        workflow = create_storm_workflow()
        result = workflow.invoke(initial_state)
        
        print(f"思考步骤: {len(result['thoughts'])}")
        print(f"观察结果: {len(result['observations'])}")
        print(f"反思内容: {len(result['reflections'])}")
        print(f"最终答案: {result['final_answer']}")
        print("-" * 50)


if __name__ == "__main__":
    run_storm_example()