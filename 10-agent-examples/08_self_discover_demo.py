#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_self_discover_demo.py — LangChain v1 create_agent 版本（Self-Discover 自我发现）

统一工具与 CLI。该示例强调：识别缺失信息 → 提出澄清 → 分解任务 → 执行 → 简短反思。
"""

import os
from typing import Optional
from datetime import datetime

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import Optional
from typing_extensions import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import END, START, StateGraph

def load_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

def get_llm() -> ChatOpenAI:

    """创建并配置语言模型实例"""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
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
        "verbose": False,
        "base_url": base_url
    }

    return ChatOpenAI(**kwargs)


# Inline Self-Discover prompts (Hub 不可用时的占位模板)
select_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a self-discovery selector. Read the task and available reasoning modules, then select the most helpful ones.",
    ),
    (
        "human",
        "Task: {task_description}\nAvailable reasoning modules:\n{reasoning_modules}\n\nReturn a concise list of selected module numbers or names and a short rationale.",
    ),
])

adapt_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a self-discovery adapter. Adapt the selected modules for this specific task and context.",
    ),
    (
        "human",
        "Task: {task_description}\nSelected modules:\n{selected_modules}\nAvailable reasoning modules:\n{reasoning_modules}\n\nReturn an adapted list with brief instructions per module.",
    ),
])

structured_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a structurer. Produce a clear step-by-step reasoning structure based on the adapted modules.",
    ),
    (
        "human",
        "Task: {task_description}\nAdapted modules:\n{adapted_modules}\n\nReturn a numbered plan (1., 2., 3., ...) describing the reasoning workflow.",
    ),
])

reasoning_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a reasoner. Follow the provided reasoning structure to answer the task thoroughly and concisely.",
    ),
    (
        "human",
        "Task: {task_description}\nReasoning structure:\n{reasoning_structure}\n\nReturn the final answer in 3-6 sentences.",
    ),
])


def select(inputs):
    select_chain = select_prompt | get_llm() | StrOutputParser()
    return {"selected_modules": select_chain.invoke(inputs)}


def adapt(inputs):
    adapt_chain = adapt_prompt | get_llm() | StrOutputParser()
    return {"adapted_modules": adapt_chain.invoke(inputs)}


def structure(inputs):
    structure_chain = structured_prompt | get_llm() | StrOutputParser()
    return {"reasoning_structure": structure_chain.invoke(inputs)}


def reason(inputs):
    reasoning_chain = reasoning_prompt | get_llm() | StrOutputParser()
    return {"answer": reasoning_chain.invoke(inputs)}

class SelfDiscoverState(TypedDict):
    reasoning_modules: str
    task_description: str
    selected_modules: Optional[str]
    adapted_modules: Optional[str]
    reasoning_structure: Optional[str]
    answer: Optional[str]

def create_workflow():

    workflow = StateGraph(SelfDiscoverState)
    workflow.add_node("select", select)
    workflow.add_node("adapt", adapt)
    workflow.add_node("structure", structure)
    workflow.add_node("reason", reason)
    workflow.add_edge(START, "select")
    workflow.add_edge("select", "adapt")
    workflow.add_edge("adapt", "structure")
    workflow.add_edge("structure", "reason")
    workflow.add_edge("reason", END)

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile()
    app.get_graph().draw_mermaid_png(output_file_path='blog/flow_01.png')
    return app

def main() -> None:
    load_environment()
    """运行工作流并输出最终结果"""
    app = create_workflow()

    reasoning_modules = [
    "1. How could I devise an experiment to help solve that problem?",
    "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    # "3. How could I measure progress on this problem?",
    "4. How can I simplify the problem so that it is easier to solve?",
    "5. What are the key assumptions underlying this problem?",
    "6. What are the potential risks and drawbacks of each solution?",
    "7. What are the alternative perspectives or viewpoints on this problem?",
    "8. What are the long-term implications of this problem and its solutions?",
    "9. How can I break down this problem into smaller, more manageable parts?",
    "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
    "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
    # "12. Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
    "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
    "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
    # "15. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
    "16. What is the core issue or problem that needs to be addressed?",
    "17. What are the underlying causes or factors contributing to the problem?",
    "18. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
    "19. What are the potential obstacles or challenges that might arise in solving this problem?",
    "20. Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
    "21. Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
    "22. What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
    "23. How can progress or success in solving the problem be measured or evaluated?",
    "24. What indicators or metrics can be used?",
    "25. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
    "26. Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
    "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
    "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
    "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
    "30. Is the problem a design challenge that requires creative solutions and innovation?",
    "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
    "32. Is the problem time-sensitive or urgent, requiring immediate attention and action?",
    "33. What kinds of solution typically are produced for this kind of problem specification?",
    "34. Given the problem specification and the current best solution, have a guess about other possible solutions."
    "35. Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?"
    "36. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?"
    "37. Ignoring the current best solution, create an entirely new solution to the problem."
    # "38. Let’s think step by step."
    "39. Let’s make a step by step plan and implement it with good notation and explanation.",
]


    task_example = "Lisa has 10 apples. She gives 3 apples to her friend and then buys 5 more apples from the store. How many apples does Lisa have now?"

    task_example = """This SVG path element <path d="M 55.57,80.69 L 57.38,65.80 M 57.38,65.80 L 48.90,57.46 M 48.90,57.46 L
    45.58,47.78 M 45.58,47.78 L 53.25,36.07 L 66.29,48.90 L 78.69,61.09 L 55.57,80.69"/> draws a:
    (A) circle (B) heptagon (C) hexagon (D) kite (E) line (F) octagon (G) pentagon(H) rectangle (I) sector (J) triangle"""

    reasoning_modules_str = "\n".join(reasoning_modules)

    for s in app.stream(
        {"task_description": task_example, "reasoning_modules": reasoning_modules_str}
    ):
        print(s)


if __name__ == "__main__":
    main()