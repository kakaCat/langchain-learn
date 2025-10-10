"""
Reason without Observation 模式示例

这种模式的特点：
- 只有推理步骤，没有观察步骤
- 直接基于推理进行决策
- 适用于不需要外部工具调用的场景
- 简化了传统的推理-行动-观察循环
"""

import os
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
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


# 定义状态类型
class ReasonState(Dict[str, Any]):
    """推理状态"""
    messages: List[Any] = add_messages
    reasoning: str = ""
    final_answer: str = ""



def reason_node(state: ReasonState) -> ReasonState:
    """
    推理节点 - 执行推理步骤
    
    在这个模式中，我们只进行推理，不调用外部工具进行观察
    """
    llm = get_llm()
    
    # 构建系统提示词
    system_prompt = """你是一个推理专家。请仔细分析问题，进行逻辑推理，
    然后给出最终答案。不需要调用任何外部工具。
    
    推理步骤：
    1. 理解问题
    2. 分析关键信息
    3. 进行逻辑推理
    4. 得出结论
    
    请确保你的推理过程清晰，最终答案准确。"""
    
    # 获取用户消息
    user_message = state["messages"][-1].content if state["messages"] else ""
    
    # 构建推理提示词
    reasoning_prompt = f"""
问题：{user_message}

请进行推理：
1. 首先理解问题的核心
2. 分析问题的关键要素
3. 进行逻辑推理
4. 得出最终结论

推理过程："""
    
    # 执行推理
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=reasoning_prompt)
    ]
    
    reasoning_response = llm.invoke(messages)
    reasoning_text = reasoning_response.content
    
    # 提取最终答案
    final_answer_prompt = f"""
基于以下推理过程，请给出最终的简洁答案：

推理过程：
{reasoning_text}

最终答案："""
    
    answer_response = llm.invoke([
        SystemMessage(content="请基于推理过程给出简洁的最终答案"),
        HumanMessage(content=final_answer_prompt)
    ])
    
    # 更新状态
    state["reasoning"] = reasoning_text
    state["final_answer"] = answer_response.content
    state["messages"].append(HumanMessage(content=f"推理过程：{reasoning_text}"))
    state["messages"].append(HumanMessage(content=f"最终答案：{answer_response.content}"))
    
    return state

def create_reason_workflow():
    """创建推理工作流"""
    
    # 创建工作流
    workflow = StateGraph(ReasonState)
    
    # 添加节点
    workflow.add_node("reason", reason_node)
    
    # 设置入口点
    workflow.set_entry_point("reason")
    
    # 设置结束点
    workflow.add_edge("reason", END)
    
    return workflow.compile()

def run_reason_example():
    """运行推理示例"""
    
    # 设置环境
    load_environment()    
    # 创建工作流
    workflow = create_reason_workflow()
    
    # 测试用例
    test_cases = [
        "如果我有3个苹果，给了朋友1个，然后又买了2个，现在我有几个苹果？",
        "一个篮球队有5名首发球员，如果每场比赛有3名替补球员，整个球队有多少名球员？",
        "一本书有200页，我第一天读了1/4，第二天读了剩下的1/3，我还剩多少页没读？"
    ]
    
    print("=== Reason without Observation 模式演示 ===\n")
    
    for i, question in enumerate(test_cases, 1):
        print(f"测试用例 {i}:")
        print(f"问题: {question}")
        
        # 初始化状态
        initial_state = ReasonState(
            messages=[HumanMessage(content=question)],
            reasoning="",
            final_answer=""
        )
        
        # 执行工作流
        result = workflow.invoke(initial_state)
        
        print(f"\n推理过程:")
        print(result["reasoning"])
        print(f"\n最终答案: {result['final_answer']}")
        print("-" * 50)

if __name__ == "__main__":
    run_reason_example()