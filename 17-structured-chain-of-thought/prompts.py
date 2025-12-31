"""
Centralized Prompt Templates for DeepSeek-R1 Agent V2

This module contains all prompt templates used in the agent system,
following the DeepSeek-R1 style with <think> and <answer> tags.
"""

from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# Gate Prompt - Task Complexity Classification
# ============================================================================

GATE_PROMPT = ChatPromptTemplate.from_template(
    """你是一个任务复杂度分类器。请判断用户输入的任务类型。

分类标准：
- SIMPLE: 问候语、简单事实查询、基础算术（如 2+2）
- MEDIUM: 数学应用题、逻辑推理、需要多步计算的问题
- COMPLEX: 多步骤设计任务、复杂推理、需要综合分析的问题

用户输入：{input}

请仅输出以下之一：SIMPLE、MEDIUM、COMPLEX
不要包含任何解释或其他文字。"""
)


# ============================================================================
# Direct Prompt - Simple Questions
# ============================================================================

DIRECT_PROMPT = ChatPromptTemplate.from_template(
    """你是一个乐于助人的助手。请直接、简洁地回答用户的问题。

问题：{input}

回答："""
)


# ============================================================================
# Single Think Prompt - Recommended for Most Cases
# ============================================================================

SINGLE_THINK_PROMPT = ChatPromptTemplate.from_template(
    """你是 DeepSeek-R1 风格的推理助手。请使用 <think> 和 <answer> 标签来组织你的回答。

**格式要求：**
<think>
[在这里进行详细的推理过程]
1. 理解问题：明确题目要求和关键信息
2. 分析条件：列出已知条件和未知量
3. 选择方法：确定解题思路和方法
4. 逐步推理：如果是数学题，请列出每一步计算；如果是逻辑题，请展示推理链条
5. 验证答案：检查答案的合理性和正确性
</think>

<answer>
[在这里给出简洁明确的最终答案]
</answer>

**重要规则：**
1. 必须严格基于题目给出的信息，不要假设或添加原题中不存在的条件
2. 数学计算必须准确，复杂计算可以分步展示
3. 如果遇到模糊信息，按最直接、最合理的理解来解答
4. 保持推理过程的连贯性和逻辑性
5. 最终答案要明确、具体

问题：{input}"""
)


# ============================================================================
# Tool Enhanced Prompt - For ReAct Agent with Tools
# ============================================================================

TOOL_ENHANCED_PROMPT = ChatPromptTemplate.from_template(
    """你是一个可以使用工具的推理助手。你可以使用以下工具来辅助推理和计算。

可用工具：
{tools}

工具名称列表：{tool_names}

**使用格式（ReAct 模式）：**
Thought: [你的思考过程，解释你打算做什么]
Action: [工具名称，从上面的列表中选择]
Action Input: [传递给工具的输入]
Observation: [工具返回的结果]
... (重复 Thought/Action/Action Input/Observation 直到得到最终答案)
Thought: 我现在知道最终答案了
Final Answer: [最终答案]

**重要提示：**
- 对于数学计算，优先使用 calculator 工具来确保准确性
- 仔细检查工具的输入格式，确保正确使用
- 如果工具返回错误，尝试调整输入或换一个思路
- 最终答案要基于工具的计算结果

问题：{input}

{agent_scratchpad}"""
)


# ============================================================================
# Multi-Reflect Prompts - For Complex Scenarios
# ============================================================================

MULTI_REFLECT_VERIFY_PROMPT = ChatPromptTemplate.from_template(
    """你是一个批判性思维专家。请仔细验证以下推理过程，找出可能的逻辑漏洞或计算错误。

原问题：
{input}

推理过程：
{previous_think}

请执行以下检查：
1. 是否正确理解了题目要求？
2. 是否引入了题目中不存在的假设或条件？
3. 计算步骤是否准确？
4. 逻辑推理是否连贯？
5. 最终答案是否合理？

请指出具体的问题（如果有），或确认推理无误。"""
)


MULTI_REFLECT_REFINE_PROMPT = ChatPromptTemplate.from_template(
    """基于验证结果，请给出最终的正确答案。

原问题：
{input}

验证结果：
{verification}

请使用 <answer> 标签给出最终答案：
<answer>
[修正后的最终答案]
</answer>"""
)


MULTI_REFLECT_PROMPTS = {
    "verify": MULTI_REFLECT_VERIFY_PROMPT,
    "refine": MULTI_REFLECT_REFINE_PROMPT
}


# ============================================================================
# 幻觉检测提示词
# ============================================================================

HALLUCINATION_CHECK_PROMPT = ChatPromptTemplate.from_template(
    """你是一个严格的逻辑验证专家。请检查以下推理过程是否引入了原问题中不存在的信息或假设。

原问题：
{original_question}

推理过程：
{reasoning}

请仔细检查：
1. 推理中是否使用了原问题中明确提到的所有信息？
2. 推理中是否引入了原问题中没有的新假设、条件或概念？
3. 如果引入了新信息，请逐条列出

**输出格式：**
如果没有问题，仅输出"无问题"。
如果有问题，列出：
- 问题1: [具体描述引入的不存在的信息]
- 问题2: [...]
"""
)


# ============================================================================
# V2.5: Structured 4-Stage Prompts (用户要求的结构化4阶段模式)
# ============================================================================

STAGE1_PROBLEM_DEF_V2_5 = ChatPromptTemplate.from_template(
    """你是 DeepSeek-R1 风格的问题分析助手。

**用户问题**:
{input}

**任务**: 分析问题的关键信息

使用 <think> 和 <answer> 标签输出：

<think>
1. 这是什么类型的问题？（数学/逻辑/常识/...）
2. 题目给出了哪些已知信息？（逐条列出）
3. 需要求解什么？（明确目标）
4. 是否涉及计算？需要什么工具？
</think>

<answer>
[简洁总结关键信息，格式化输出]
- 问题类型: ...
- 已知: ...
- 求解: ...
- 工具需求: ...
</answer>

**规则**: 严格基于题目，不添加假设
""")


STAGE2_BLOOM_V2_5 = ChatPromptTemplate.from_template(
    """你是路径探索助手。基于问题分析，探索解决路径。

**原始问题**:
{original_question}

**阶段 1 分析**:
{stage1_output}

**完整历史**:
{chat_history}

**可用工具**:
- calculator: 精确数学计算

---

**任务**: 探索 2-3 种可能的解题路径

<think>
路径 1: [描述方法和步骤]
  - 步骤1: ...
  - 步骤2: ...
  - 预期结果: ...

路径 2: [如果有其他合理方法]
  ...

路径 3: [如果问题复杂]
  ...

推荐路径: [选择最直接/可靠的方法]
理由: ...
</think>

<answer>
推荐使用路径 X，因为...
[如果涉及计算，请使用 calculator 工具验证关键步骤]
</answer>

**规则**:
- 路径必须基于阶段1的已知信息
- 计算使用工具验证
- 不假设题目未提及的信息
""")


STAGE3_VALIDATION_V2_5 = ChatPromptTemplate.from_template(
    """你是一个验证助手，负责快速检查推理的准确性。

**原始问题**:
{original_question}

**阶段 2 输出（路径探索）**:
{stage2_output}

---

**验证任务（2分钟内完成）**:

1. **信息准确性**: 推理是否基于原始问题？是否引入了不存在的信息？

2. **逻辑一致性**: 推理步骤是否连贯？结论是否合理？

**输出格式**:
<think>
[简短验证，只检查关键问题]
</think>

<answer>
状态: [通过/需要修正]
问题: [如有问题简述；无问题输出"无问题"]
</answer>

**重要规则**:
- ✅ 只进行一次性检查，不要重复验证
- ✅ 发现问题只报告，不尝试修正
- ❌ 不要创造新假设或过度批判
- ⏱️ 2分钟内必须完成
""")



STAGE4_FINAL_V2_5 = ChatPromptTemplate.from_template(
    """你是最终决策助手。基于完整推理历史，输出最终答案。

**原始问题**:
{original_question}

**推理历史**:
{chat_history}

**阶段 3 验证结果**:
{stage3_validation}

---

**任务**: 综合所有信息，给出最终答案

如果阶段 3 验证通过:
- 直接输出答案
- 如果是逻辑题且验证中明确了唯一解，使用该唯一解

如果阶段 3 发现问题:
- 根据建议修正
- 重新计算（如需要）
- 输出修正后的答案

🔍 **特别注意逻辑题**:
- 如果阶段3给出了"唯一解"字段，必须使用该唯一解作为最终答案
- 不要再引入其他可能性或犹豫
- 答案必须清晰明确（如: A是Knight, B是Knave, C是Knave）

**输出格式**:
<answer>
[简洁明确的最终答案，包含必要的单位和说明]
对于逻辑题: 直接给出每个角色的最终判定
对于数学题: 给出数值和单位
</answer>

**规则**:
- 答案必须直接回答原始问题
- 数值答案包含单位
- 如果有修正，简要说明修正原因
- 对于逻辑题，输出格式必须清晰（列出每个角色的身份）
- ⚠️ 不要输出"可能是"、"也许"等模糊表述，必须给出确定答案
""")


if __name__ == "__main__":
    # 测试提示词模板
    print("=== 测试 GATE_PROMPT ===")
    test_input = "Janet's ducks lay 16 eggs per day"
    formatted = GATE_PROMPT.format(input=test_input)
    print(formatted[:200] + "...\n")

    print("=== 测试 SINGLE_THINK_PROMPT ===")
    formatted = SINGLE_THINK_PROMPT.format(input=test_input)
    print(formatted[:300] + "...\n")

    print("=== 测试 STAGE1_PROBLEM_DEF_V2_5 ===")
    formatted = STAGE1_PROBLEM_DEF_V2_5.format(input=test_input)
    print(formatted[:300] + "...\n")

    print("所有提示词模板加载成功！")
