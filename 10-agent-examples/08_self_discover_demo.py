#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_self_discover_demo.py — LangChain v1 create_agent 版本（Self-Discover 自我发现）

统一工具与 CLI。该示例强调：识别缺失信息 → 提出澄清 → 分解任务 → 执行 → 简短反思。
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


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
# 你是一名自我探索选择器。请阅读任务及可用推理模块，然后选出最有帮助的模块。
# 任务：{task_description}n可用推理模块：可用推理模块：{reasoning_modules}返回一个简洁的选中模块编号或名称列表和一个简短的理由
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

#你是一名自我探索适配器。根据任务描述和选中的推理模块，适配这些模块以适应特定任务和上下文。
#任务：{task_description}\n选中的推理模块：\n{selected_modules}\n可用推理模块：\n{reasoning_modules}\n\n返回一个适配后的列表，每个模块包含简要说明。
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

#你是一名结构化器。根据适配后的推理模块，生成一个清晰的推理结构。
#任务：{task_description}\n适配后的推理模块：\n{adapted_modules}\n\n返回一个编号计划（1.，2.，3.，...），描述推理工作流程。
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

#你是一名推理器。根据提供的推理结构，详细而简洁地回答任务。
#任务：{task_description}\n推理结构：\n{reasoning_structure}\n\n返回最终答案在3-6句话内。
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

class SelfDiscoverState(TypedDict):
    reasoning_modules: str  # 推理模块列表
    task_description: str  # 任务描述
    selected_modules: Optional[str]  # 选中的推理模块
    adapted_modules: Optional[str]  # 适配后的推理模块
    reasoning_structure: Optional[str]  # 推理结构
    answer: Optional[str]  # 最终答案

def select(state: SelfDiscoverState):
    print("[select] 开始选择推理模块...")
    select_chain = select_prompt | get_llm() | StrOutputParser()
    state["selected_modules"] = select_chain.invoke(state)
    print(f"[select] 已选择推理模块: {state['selected_modules']}")
    return state

def adapt(state: SelfDiscoverState):
    print("[adapt] 开始适配推理模块...")
    adapt_chain = adapt_prompt | get_llm() | StrOutputParser()
    state["adapted_modules"] = adapt_chain.invoke(state)
    print(f"[adapt] 已适配推理模块: {state['adapted_modules']}")
    return state

def structure(state: SelfDiscoverState):
    print("[structure] 开始构建推理结构...")
    structure_chain = structured_prompt | get_llm() | StrOutputParser()
    state["reasoning_structure"] = structure_chain.invoke(state)
    print(f"[structure] 已构建推理结构: {state['reasoning_structure']}")
    return state

def reason(state: SelfDiscoverState):
    print("[reason] 开始执行推理...")
    reasoning_chain = reasoning_prompt | get_llm() | StrOutputParser()
    state["answer"] = reasoning_chain.invoke(state)
    print(f"[reason] 推理完成，答案: {state['answer']}")
    return state



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
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "blog" / "self-discover.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    app.get_graph().draw_mermaid_png(output_file_path=str(output_path))
    return app


# 我该如何设计实验来帮助解决这个问题？
# 列出解决该问题的思路清单，将其逐一应用到问题中，观察是否能取得进展。
# （原文注释：已标注 #，暂不翻译）
# 我该如何简化这个问题，使其更容易解决？
# 这个问题背后的核心假设是什么？
# 每种解决方案的潜在风险和弊端是什么？
# 关于这个问题，有哪些不同的视角或观点？
# 这个问题及其解决方案的长期影响是什么？
# 我该如何将这个问题拆解为更小、更易处理的部分？
# 批判性思维：该思维方式要求从不同角度分析问题、质疑假设，并评估现有证据或信息。其核心在于逻辑推理、基于证据的决策，以及识别思考过程中潜在的偏见或漏洞。
# 尝试运用创造性思维，提出跳出常规的思路来解决问题。探索非传统解决方案，突破传统界限思考，激发想象力与原创性。
# （原文注释：已标注 #，暂不翻译）
# 运用系统思维：将问题视为更大系统的一部分，理解各要素间的相互联系。核心是识别影响问题的根本原因、反馈循环及相互依赖关系，并制定针对整个系统的整体性解决方案。
# 运用风险分析：评估不同解决方案或方法相关的潜在风险、不确定性与权衡取舍。重点是分析成功或失败的潜在后果及可能性，基于对风险与收益的平衡分析做出明智决策。
# （原文注释：已标注 #，暂不翻译）
# 需要解决的核心议题或问题是什么？
# 导致该问题的根本原因或影响因素是什么？
# 此前是否尝试过任何潜在的解决方案或策略？若有，结果如何？从中获得了哪些经验教训？
# 解决这个问题可能会遇到哪些潜在障碍或挑战？
# 是否有任何相关数据或信息能为理解该问题提供启发？若有，可获取哪些数据源？如何分析这些数据？
# 该问题是否直接影响某些利益相关者或个人？他们的视角和需求是什么？
# 有效解决该问题需要哪些资源（资金、人力、技术等）？
# 如何衡量解决问题的进展或成功与否？
# 可使用哪些指标或衡量标准？
# 该问题是需要特定专业知识或技能的技术性 / 实操性问题，还是更偏向概念性 / 理论性的问题？
# 该问题是否受物理条件限制，例如资源有限、基础设施不足或空间受限？
# 该问题是否与人类行为相关，例如涉及社会、文化或心理层面的议题？
# 该问题是否涉及决策或规划，需要在不确定环境下或存在目标冲突时做出选择？
# 该问题是否属于分析类问题，需要运用数据分析、建模或优化技术？
# 该问题是否属于设计类挑战，需要创造性解决方案与创新思维？
# 该问题是否需要解决系统性或结构性问题，而非仅处理个别案例？
# 该问题是否具有时间敏感性或紧迫性，需要立即关注和采取行动？
# 针对这类问题描述（problem specification），通常会产生哪些类型的解决方案？
# 结合问题描述与当前最佳解决方案，推测可能存在的其他解决方案。
# 假设当前的最佳解决方案完全错误，还有哪些思考该问题描述的方式？
# 基于你对这类问题描述的了解，修改当前最佳解决方案的最佳方式是什么？
# 暂不考虑当前最佳解决方案，为该问题设计一个全新的解决方案。
# （原文注释：已标注 #，暂不翻译）
# 制定一个逐步实施计划，并用清晰的表述和解释将其落地。
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


    task_example = "莉萨有 10 个苹果。她给了朋友 3 个苹果，之后又从商店买了 5 个苹果。莉萨现在有多少个苹果？"

    # task_example = """This SVG path element <path d="M 55.57,80.69 L 57.38,65.80 M 57.38,65.80 L 48.90,57.46 M 48.90,57.46 L
    # 45.58,47.78 M 45.58,47.78 L 53.25,36.07 L 66.29,48.90 L 78.69,61.09 L 55.57,80.69"/> draws a:
    # (A) circle (B) heptagon (C) hexagon (D) kite (E) line (F) octagon (G) pentagon(H) rectangle (I) sector (J) triangle"""

    reasoning_modules_str = "\n".join(reasoning_modules)
    initial_state = {"task_description": task_example, "reasoning_modules": reasoning_modules_str}
    final_state = app.invoke(initial_state)
    print("最终结果:", final_state.get("answer"))
    # for s in app.stream(
    #     initial_state
    # ):
    #     print("最终结果:", s.get("answer"))

if __name__ == "__main__":
    main()
