#!/usr/bin/env python3
"""
语义识别与澄清系统 (Semantic Recognition and Clarification System)
智能问答系统 - 通过LLM语义分析理解用户意图并主动澄清

功能特点：
1. LLM驱动的意图识别：准确识别用户输入的意图类型
2. LLM驱动的实体提取：智能提取关键实体信息
3. 语义分析：评估语义完整度和可执行性
4. 智能澄清：根据语义缺失生成针对性问题
5. 多轮对话：支持多轮澄清直到需求明确

应用场景：
- 智能客服：理解客户意图并提供精准服务
- 对话系统：构建上下文理解能力
- 需求分析：澄清产品需求和技术细节
- 任务助手：理解并执行用户指令

运行方式：
- 演示模式（自动化）：python 05_clarification_system.py --demo
- 交互模式（真实交互）：python 05_clarification_system.py

环境变量：
- USE_OLLAMA=true: 使用本地Ollama模型
- OPENAI_API_KEY: OpenAI API密钥（默认使用OpenAI）

学习要点：
1. 如何使用LLM进行意图识别和实体提取
2. 如何评估语义完整度
3. 如何使用LLM根据语义缺失生成澄清问题
4. 如何使用 interrupt() 实现 HITL
5. 如何整合多轮对话的语义信息
"""

import sys
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Set
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 初始化 LLM
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"

if USE_OLLAMA:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(
        model="qwen2.5:latest",
        temperature=0,
    )
    print("✓ 使用本地 Ollama qwen2.5")
else:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
    )
    print("✓ 使用 OpenAI GPT-3.5-turbo")


class IntentType(Enum):
    """意图类型"""
    QUERY = "查询"  # 查询信息
    OPERATION = "操作"  # 执行操作
    COMPLAINT = "投诉"  # 问题投诉
    REQUIREMENT = "需求"  # 功能需求
    HELP = "求助"  # 寻求帮助
    UNKNOWN = "未知"  # 无法识别


@dataclass
class SemanticInfo:
    """语义信息"""
    intent: IntentType = IntentType.UNKNOWN
    entities: Dict[str, str] = field(default_factory=dict)
    missing_entities: Set[str] = field(default_factory=set)
    confidence: float = 0.0
    completeness: float = 0.0
    is_executable: bool = False

    def to_dict(self) -> Dict:
        """转换为字典格式便于展示"""
        return {
            "intent": self.intent.value,
            "entities": self.entities,
            "missing_entities": list(self.missing_entities),
            "confidence": f"{self.confidence:.2f}",
            "completeness": f"{self.completeness:.2f}",
            "is_executable": self.is_executable
        }


@dataclass
class ClarificationState:
    """澄清流程状态"""
    user_input: str = ""
    semantic_info: SemanticInfo = field(default_factory=SemanticInfo)
    ambiguity_level: int = 0
    clarifications: List[str] = field(default_factory=list)
    all_questions: List[str] = field(default_factory=list)
    clarification_answers: List[str] = field(default_factory=list)
    current_question: str = ""
    clarified_understanding: str = ""
    is_clear: bool = False
    round_count: int = 0
    max_rounds: int = 3
    conversation_history: List[Dict[str, str]] = field(default_factory=list)


def recognize_intent(text: str) -> IntentType:
    """使用 LLM 识别用户意图"""

    system_prompt = """你是一个智能意图识别助手。请分析用户的输入文本，识别其意图类型。

意图类型定义：
1. QUERY (查询): 用户想要查询信息、查看数据、了解状态
   - 示例："订单状态是什么？"、"显示用户列表"、"查看报表"

2. OPERATION (操作): 用户想要执行具体操作（增删改查）
   - 示例："添加新用户"、"删除这条记录"、"导出数据"

3. COMPLAINT (投诉): 用户反馈问题、报告故障、投诉错误
   - 示例："登录功能坏了"、"系统报错"、"这个有问题"

4. REQUIREMENT (需求): 用户提出新需求、功能建议
   - 示例："能不能增加导出功能"、"希望支持批量操作"

5. HELP (求助): 用户寻求帮助、学习使用方法
   - 示例："怎么使用这个功能"、"教我如何操作"

6. UNKNOWN (未知): 无法明确识别意图

请仅返回以下之一：QUERY, OPERATION, COMPLAINT, REQUIREMENT, HELP, UNKNOWN"""

    user_prompt = f"用户输入：{text}\n\n请识别意图类型（仅返回类型名称，不要其他内容）："

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    response = llm.invoke(messages)
    intent_str = response.content.strip().upper()

    # 解析返回的意图
    intent_mapping = {
        "QUERY": IntentType.QUERY,
        "OPERATION": IntentType.OPERATION,
        "COMPLAINT": IntentType.COMPLAINT,
        "REQUIREMENT": IntentType.REQUIREMENT,
        "HELP": IntentType.HELP,
        "UNKNOWN": IntentType.UNKNOWN
    }

    return intent_mapping.get(intent_str, IntentType.UNKNOWN)


def extract_entities(text: str, intent: IntentType) -> Dict[str, str]:
    """使用 LLM 提取实体信息"""

    system_prompt = """你是一个智能实体提取助手。请从用户输入中提取关键实体信息。

需要提取的实体类型：
1. 对象 - 用户提到的功能、模块、系统名称（如：登录功能、用户管理、报表系统）
2. 时间 - 时间相关信息（如：今天、昨天、早上、现在）
3. 平台 - 平台或设备类型（如：移动端、PC端、Web端、APP）
4. 范围 - 影响范围（如：所有用户、部分用户、特定用户）
5. 问题现象 - 具体的问题描述（仅对投诉类）
6. 操作细节 - 具体的操作说明（仅对操作类）
7. 需求细节 - 具体的需求描述（仅对需求类）
8. 目标 - 用户想达成的目标（仅对求助类）

请以 JSON 格式返回提取的实体，格式如下：
{"对象": "...", "时间": "...", "平台": "..."}

如果某个实体不存在，则不要包含该字段。
仅返回 JSON，不要其他内容。"""

    user_prompt = f"用户输入：{text}\n意图类型：{intent.value}\n\n请提取实体（仅返回JSON）："

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    response = llm.invoke(messages)
    result_text = response.content.strip()

    # 尝试解析 JSON
    # 移除可能的 markdown 代码块标记
    if result_text.startswith("```"):
        result_text = result_text.split("```")[1]
        if result_text.startswith("json"):
            result_text = result_text[4:]
        result_text = result_text.strip()

    entities = json.loads(result_text)
    return entities


def determine_missing_entities(intent: IntentType, current_entities: Dict[str, str]) -> Set[str]:
    """根据意图确定缺失的实体"""
    # 定义不同意图需要的实体
    required_entities = {
        IntentType.QUERY: {"对象"},
        IntentType.OPERATION: {"对象", "操作细节"},
        IntentType.COMPLAINT: {"对象", "问题现象"},
        IntentType.REQUIREMENT: {"对象", "需求细节"},
        IntentType.HELP: {"对象", "目标"}
    }

    required = required_entities.get(intent, set())
    current = set(current_entities.keys())

    return required - current


def calculate_completeness(entities: Dict[str, str], missing: Set[str]) -> float:
    """计算语义完整度"""
    if not missing:
        return 1.0

    total_required = len(entities) + len(missing)
    if total_required == 0:
        return 0.0

    return len(entities) / total_required


def analyze_semantic(state: ClarificationState) -> ClarificationState:
    """语义分析 - 识别意图、提取实体、评估完整度"""
    print(f"\n🔍 语义分析: '{state.user_input}'")

    # 1. 识别意图
    intent = recognize_intent(state.user_input)
    print(f"   意图识别: {intent.value}")

    # 2. 提取实体
    entities = extract_entities(state.user_input, intent)
    print(f"   实体提取: {entities if entities else '无'}")

    # 3. 确定缺失实体
    missing_entities = determine_missing_entities(intent, entities)
    if missing_entities:
        print(f"   缺失实体: {missing_entities}")

    # 4. 计算完整度
    completeness = calculate_completeness(entities, missing_entities)

    # 5. 计算置信度（基于输入长度和实体数量）
    confidence = min(1.0, (len(state.user_input) / 20 + len(entities) * 0.2))

    # 6. 判断是否可执行
    is_executable = completeness >= 0.7 and confidence >= 0.6

    # 7. 计算模糊度（用于兼容原有逻辑）
    ambiguity_score = int((1 - completeness) * 10)

    # 更新语义信息
    state.semantic_info = SemanticInfo(
        intent=intent,
        entities=entities,
        missing_entities=missing_entities,
        confidence=confidence,
        completeness=completeness,
        is_executable=is_executable
    )
    state.ambiguity_level = ambiguity_score

    print(f"   完整度: {completeness:.2f} | 置信度: {confidence:.2f} | 可执行: {is_executable}")
    print(f"   模糊度评分: {ambiguity_score}/10")

    return state


def route_by_clarity(state: ClarificationState) -> str:
    """根据模糊度决定是否需要澄清"""
    if state.ambiguity_level > 3:
        print(f"⚠️  需求不够明确，需要澄清")
        return "generate_questions"
    else:
        print(f"✅ 需求明确，无需澄清")
        state.is_clear = True
        state.clarified_understanding = f"理解：{state.user_input}"
        return "finalize"


def generate_questions_with_llm(
    user_input: str,
    intent: IntentType,
    entities: Dict[str, str],
    missing_entities: Set[str],
    clarification_history: List[str]
) -> List[str]:
    """使用 LLM 生成澄清问题"""

    system_prompt = """你是一个智能澄清问题生成助手。根据用户输入和当前的语义分析结果，生成针对性的澄清问题。

生成原则：
1. 问题应该针对缺失的关键信息
2. 问题应该清晰、具体、易于回答
3. 避免生成过于宽泛的问题
4. 考虑意图类型，生成相关的问题
5. 每次生成1-3个问题

请以 JSON 数组格式返回问题列表，例如：
["问题1", "问题2", "问题3"]

仅返回 JSON 数组，不要其他内容。"""

    user_prompt = f"""用户输入：{user_input}
意图类型：{intent.value}
已识别实体：{entities if entities else "无"}
缺失实体：{list(missing_entities) if missing_entities else "无"}
澄清历史：{clarification_history if clarification_history else "无"}

请生成澄清问题（仅返回JSON数组）："""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    response = llm.invoke(messages)
    result_text = response.content.strip()

    # 移除可能的 markdown 代码块标记
    if result_text.startswith("```"):
        result_text = result_text.split("```")[1]
        if result_text.startswith("json"):
            result_text = result_text[4:]
        result_text = result_text.strip()

    questions = json.loads(result_text)

    if isinstance(questions, list) and questions:
        return questions[:3]  # 最多返回3个问题
    else:
        return ["能否提供更多详细信息？"]


def generate_clarification_questions(state: ClarificationState) -> ClarificationState:
    """基于语义缺失生成澄清问题（使用 LLM）"""
    print(f"\n💭 生成澄清问题（第 {state.round_count + 1} 轮）...")

    # 添加轮次安全检查，防止超过最大轮次
    if state.round_count >= state.max_rounds:
        print(f"⚠️  已达到最大澄清轮次 ({state.max_rounds})，不再生成新问题")
        state.is_clear = True
        return state

    semantic = state.semantic_info

    # 准备澄清历史
    clarification_history = []
    for i, (q, a) in enumerate(zip(state.all_questions, state.clarification_answers)):
        clarification_history.append(f"Q{i+1}: {q}\nA{i+1}: {a}")

    # 使用 LLM 生成澄清问题
    questions = generate_questions_with_llm(
        user_input=state.user_input,
        intent=semantic.intent,
        entities=semantic.entities,
        missing_entities=semantic.missing_entities,
        clarification_history=clarification_history
    )

    state.clarifications = questions
    state.all_questions.extend(questions)  # 保存到历史记录
    state.round_count += 1

    print(f"📝 已生成 {len(questions)} 个澄清问题")
    for i, q in enumerate(questions, 1):
        print(f"   {i}. {q}")

    return state


def ask_clarification(state: ClarificationState) -> ClarificationState:
    """询问澄清问题 - HITL 暂停点"""
    # 选择第一个未回答的问题（使用all_questions作为总问题数）
    question_index = len(state.clarification_answers)

    if question_index < len(state.all_questions):
        state.current_question = state.all_questions[question_index]
        print(f"\n❓ 澄清问题 {question_index + 1}: {state.current_question}")
        print(f"⏸️  等待用户回答...")

        # 暂停等待人工输入
        interrupt("等待用户回答澄清问题")

    return state


def update_understanding(state: ClarificationState) -> ClarificationState:
    """根据澄清答案更新理解并重新分析语义"""
    print(f"\n🔄 更新理解...")

    # 1. 整合所有信息
    understanding_parts = [f"原始需求：{state.user_input}"]

    # 显示所有已回答的问题和答案
    for i, answer in enumerate(state.clarification_answers):
        # 从历史问题列表中获取对应的问题
        if i < len(state.all_questions):
            question = state.all_questions[i]
        else:
            question = "（问题记录丢失）"
        understanding_parts.append(f"澄清{i+1}：{question} -> {answer}")

    # 2. 重新分析语义 - 整合原始输入和所有澄清答案
    if state.clarification_answers:
        # 将最后一个澄清答案合并到语义分析中
        last_answer = state.clarification_answers[-1]

        # 重新提取实体（从最新的答案中）
        new_entities = extract_entities(last_answer, state.semantic_info.intent)

        # 合并实体信息
        state.semantic_info.entities.update(new_entities)

        # 重新计算缺失实体
        state.semantic_info.missing_entities = determine_missing_entities(
            state.semantic_info.intent,
            state.semantic_info.entities
        )

        # 重新计算完整度
        state.semantic_info.completeness = calculate_completeness(
            state.semantic_info.entities,
            state.semantic_info.missing_entities
        )

        # 重新判断是否可执行
        state.semantic_info.is_executable = (
            state.semantic_info.completeness >= 0.7 and
            state.semantic_info.confidence >= 0.6
        )

        print(f"   更新后的语义信息：")
        print(f"   - 意图: {state.semantic_info.intent.value}")
        print(f"   - 实体: {state.semantic_info.entities}")
        print(f"   - 完整度: {state.semantic_info.completeness:.2f}")
        print(f"   - 可执行: {state.semantic_info.is_executable}")

    state.clarified_understanding = "\n".join(understanding_parts)

    print(f"\n✅ 更新后的理解：")
    print(state.clarified_understanding)

    # 判断是否需要继续澄清
    all_answered = len(state.clarification_answers) >= len(state.all_questions)
    within_max_rounds = state.round_count < state.max_rounds

    # 如果所有问题都回答了，检查回答质量和语义完整度
    if all_answered:
        # 检查语义完整度
        if state.semantic_info.is_executable:
            state.is_clear = True
            print(f"✅ 需求已澄清完成！语义完整度达标。")
        elif state.semantic_info.completeness >= 0.5:
            # 检查是否有实质性内容
            has_substantial_answer = any(
                len(answer) >= 10 and not any(
                    vague in answer for vague in ["就是", "那个", "这个", "不知道", "随便", "跳过"]
                )
                for answer in state.clarification_answers
            )

            if has_substantial_answer:
                if not within_max_rounds:
                    state.is_clear = True
                    print(f"⚠️  已达最大澄清轮次，使用当前理解（完整度: {state.semantic_info.completeness:.2f}）")
                else:
                    print(f"⚠️  语义完整度尚可，但仍可继续澄清（完整度: {state.semantic_info.completeness:.2f}）")
            else:
                state.is_clear = True
                print(f"⚠️  用户回答模糊，使用当前理解")
        else:
            if not within_max_rounds:
                state.is_clear = True
                print(f"⚠️  已达最大澄清轮次，使用当前理解")
            else:
                print(f"⚠️  语义完整度较低，需要继续澄清（完整度: {state.semantic_info.completeness:.2f}）")
    elif not within_max_rounds:
        state.is_clear = True
        print(f"⚠️  已达最大澄清轮次，使用当前理解")

    return state


def route_after_clarification(state: ClarificationState) -> str:
    """澄清后的路由决策"""
    # 优先检查是否已经明确（避免在 is_clear 为 True 时继续循环）
    if state.is_clear:
        return "finalize"

    # 检查是否还有未回答的问题（基于all_questions）
    if len(state.clarification_answers) < len(state.all_questions):
        return "ask_more"

    # 检查是否还能继续澄清
    if state.round_count < state.max_rounds:
        # 继续生成新的澄清问题
        return "generate_more"
    else:
        # 达到最大轮次，强制结束
        state.is_clear = True
        return "finalize"


def finalize_understanding(state: ClarificationState) -> ClarificationState:
    """最终确定理解"""
    print(f"\n" + "="*60)
    print(f"✅ 最终理解确认")
    print(f"="*60)
    print(state.clarified_understanding)
    print(f"\n【语义分析结果】")
    print(f"意图类型：{state.semantic_info.intent.value}")
    print(f"识别实体：{state.semantic_info.entities}")
    print(f"语义完整度：{state.semantic_info.completeness:.2f}")
    print(f"识别置信度：{state.semantic_info.confidence:.2f}")
    print(f"是否可执行：{'是' if state.semantic_info.is_executable else '否'}")
    print(f"\n【对话统计】")
    print(f"澄清轮次：{state.round_count}")
    print(f"回答数量：{len(state.clarification_answers)}")
    print(f"="*60)

    return state


def create_clarification_workflow():
    """创建语义识别与澄清系统工作流"""
    workflow = StateGraph(ClarificationState)

    # 添加节点
    workflow.add_node("analyze", analyze_semantic)
    workflow.add_node("generate_questions", generate_clarification_questions)
    workflow.add_node("ask", ask_clarification)
    workflow.add_node("update", update_understanding)
    workflow.add_node("finalize", finalize_understanding)

    # 设置入口
    workflow.set_entry_point("analyze")

    # 分析后根据模糊度路由
    workflow.add_conditional_edges(
        "analyze",
        route_by_clarity,
        {
            "generate_questions": "generate_questions",
            "finalize": "finalize"
        }
    )

    # 生成问题后开始询问
    workflow.add_edge("generate_questions", "ask")

    # 询问后更新理解
    workflow.add_edge("ask", "update")

    # 更新后决定下一步
    workflow.add_conditional_edges(
        "update",
        route_after_clarification,
        {
            "ask_more": "ask",
            "generate_more": "generate_questions",
            "finalize": "finalize"
        }
    )

    # 完成后结束
    workflow.add_edge("finalize", END)

    return workflow


def run_demo_mode():
    """演示模式 - 自动化演示"""
    print("="*60)
    print("🎯 语义识别与澄清系统演示模式")
    print("="*60)
    print("功能：LLM驱动的意图识别 + 实体提取 + 智能澄清")
    print("="*60)

    # 创建工作流
    workflow = create_clarification_workflow()
    app = workflow.compile(checkpointer=MemorySaver())

    # === 场景 1: 投诉类 - 模糊的技术问题 ===
    print("\n" + "🎬 场景 1: 投诉类 - 模糊的技术问题")
    print("-"*60)
    print("用户输入: '这个东西有问题'")
    print("预期：识别为投诉意图，缺少关键实体，需要多轮澄清")

    state1 = ClarificationState(
        user_input="这个东西有问题"
    )
    config1 = {"configurable": {"thread_id": "demo_1"}}

    # 启动工作流
    print("\n>>> 启动澄清流程...")
    app.invoke(state1, config1)

    # 模拟回答第一个澄清问题
    current_state = app.get_state(config1)
    print(f"\n👤 用户回答: 登录功能无法使用，点击登录按钮后没有反应")
    app.update_state(
        config1,
        {"clarification_answers": ["登录功能无法使用，点击登录按钮后没有反应"]}
    )

    # 继续执行
    app.invoke(None, config1)

    # 回答第二个问题
    current_state = app.get_state(config1)
    print(f"\n👤 用户回答: 今天早上开始的，影响所有用户，我尝试过清除缓存但没有效果")
    app.update_state(
        config1,
        {"clarification_answers": current_state.values["clarification_answers"] + ["今天早上开始的，影响所有用户，我尝试过清除缓存但没有效果"]}
    )
    app.invoke(None, config1)

    # 回答第三个问题（如果有）
    current_state = app.get_state(config1)
    all_questions_count = len(current_state.values.get("all_questions", []))
    answers_count = len(current_state.values.get("clarification_answers", []))

    if answers_count < all_questions_count:
        print(f"\n👤 用户回答: 主要是移动端网页，PC端正常")
        app.update_state(
            config1,
            {"clarification_answers": current_state.values["clarification_answers"] + ["主要是移动端网页，PC端正常"]}
        )
        app.invoke(None, config1)

    # === 场景 2: 需求类 - 相对明确的需求 ===
    print("\n\n" + "🎬 场景 2: 需求类 - 相对明确的需求")
    print("-"*60)
    print("用户输入: '我需要在用户管理页面添加批量导出用户数据为 Excel 的功能'")
    print("预期：识别为需求意图，实体较完整，可能无需澄清或只需少量澄清")

    state2 = ClarificationState(
        user_input="我需要在用户管理页面添加批量导出用户数据为 Excel 的功能"
    )
    config2 = {"configurable": {"thread_id": "demo_2"}}

    print("\n>>> 启动澄清流程...")
    app.invoke(state2, config2)

    # === 场景 3: 需求类 - 极度模糊的需求，需要多轮澄清 ===
    print("\n\n" + "🎬 场景 3: 需求类 - 极度模糊的需求（多轮澄清）")
    print("-"*60)
    print("用户输入: '能不能弄一下'")
    print("预期：意图不明，实体缺失，需要多轮澄清才能理解")

    state3 = ClarificationState(
        user_input="能不能弄一下"
    )
    config3 = {"configurable": {"thread_id": "demo_3"}}

    print("\n>>> 启动澄清流程...")
    app.invoke(state3, config3)

    # 第一轮回答
    print(f"\n👤 用户回答: 就是那个功能")
    app.update_state(
        config3,
        {"clarification_answers": ["就是那个功能"]}
    )
    app.invoke(None, config3)

    # 继续处理后续的澄清轮次
    max_attempts = 5  # 防止无限循环
    attempt = 0
    while attempt < max_attempts:
        current_state = app.get_state(config3)

        # 检查是否已完成
        if current_state.values.get("is_clear", False):
            break

        # 检查是否有未回答的问题（使用all_questions）
        answers_count = len(current_state.values.get("clarification_answers", []))
        questions_count = len(current_state.values.get("all_questions", []))

        if answers_count < questions_count:
            # 有新问题需要回答
            round_num = current_state.values.get("round_count", 0)

            # 根据轮次提供不同的回答
            if round_num == 2:
                answer = "报表导出功能，现在导出很慢，能不能优化一下"
            elif round_num == 3:
                answer = "最好能后台异步导出，然后发邮件通知用户下载"
            else:
                answer = "额外的详细信息"

            print(f"\n👤 用户回答: {answer}")
            current_answers = current_state.values.get("clarification_answers", [])
            app.update_state(
                config3,
                {"clarification_answers": current_answers + [answer]}
            )
            app.invoke(None, config3)
        else:
            # 没有新问题，退出循环
            break

        attempt += 1

    print("\n" + "="*60)
    print("✨ 所有演示场景完成！")
    print("="*60)


def run_interactive_mode():
    """交互模式 - 真实人机交互"""
    print("="*60)
    print("🚀 语义识别与澄清系统 - 交互模式")
    print("="*60)
    print("功能：LLM驱动的智能识别意图、提取实体、主动澄清模糊需求")
    print("\n💡 提示：如果无法输入，请使用演示模式: python 05_clarification_system.py --demo\n")

    # 创建工作流
    workflow = create_clarification_workflow()
    app = workflow.compile(checkpointer=MemorySaver())

    try:
        # 获取用户输入
        user_input = input("📝 请输入您的需求（可以模糊一些，系统会帮您澄清）: ").strip()

        if not user_input:
            user_input = "这个有问题"
            print(f"使用默认输入: '{user_input}'")

        state = ClarificationState(user_input=user_input)
        config = {"configurable": {"thread_id": "interactive"}}

        # 启动工作流
        print("\n>>> 启动澄清流程...")
        app.invoke(state, config)

        # 循环处理澄清问题
        while True:
            current_state = app.get_state(config)

            # 检查是否已完成
            if current_state.values.get("is_clear", False):
                break

            # 检查是否在等待回答
            current_question = current_state.values.get("current_question", "")
            answers = current_state.values.get("clarification_answers", [])
            questions = current_state.values.get("all_questions", [])

            if current_question and len(answers) < len(questions):
                # 获取用户回答
                answer = input(f"\n👤 您的回答: ").strip()

                if not answer:
                    answer = "（跳过此问题）"

                # 更新状态
                new_answers = answers + [answer]
                app.update_state(config, {"clarification_answers": new_answers})

                # 继续执行
                app.invoke(None, config)
            else:
                # 可能需要生成新问题
                app.invoke(None, config)

        # 显示最终结果
        final_state = app.get_state(config)
        print("\n" + "="*60)
        print("✅ 澄清完成！")
        print("="*60)
        print(f"\n最终理解：\n{final_state.values['clarified_understanding']}")
        print(f"\n澄清轮次：{final_state.values['round_count']}")
        print("\n✨ 流程完成！")
        print("="*60)

    except (EOFError, KeyboardInterrupt):
        print("\n\n❌ 检测到无法接收输入或用户中断")
        print("💡 请尝试以下方式之一：")
        print("   1. 在真实终端中运行: python 05_clarification_system.py")
        print("   2. 使用演示模式: python 05_clarification_system.py --demo")
        sys.exit(1)


def main():
    """主函数 - 根据参数选择模式"""
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo_mode()
    else:
        run_interactive_mode()


if __name__ == "__main__":
    main()
