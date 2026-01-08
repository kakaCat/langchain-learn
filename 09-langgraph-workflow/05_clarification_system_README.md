# 语义识别与澄清系统 (Semantic Recognition and Clarification System)

## 概述

这是一个基于 LangGraph 的智能问答系统，通过**语义分析**理解用户意图并**主动澄清**模糊需求。相比原始的简单澄清系统，这个版本增加了完整的语义理解能力。

## 核心功能

### 1. 意图识别 (Intent Recognition)
自动识别用户输入的意图类型：
- **查询 (QUERY)**: 查询信息、查看数据
- **操作 (OPERATION)**: 执行具体操作（添加、删除、修改等）
- **投诉 (COMPLAINT)**: 问题反馈、故障报告
- **需求 (REQUIREMENT)**: 功能需求、新增需求
- **求助 (HELP)**: 寻求帮助、学习指导
- **未知 (UNKNOWN)**: 无法识别的意图

### 2. 实体提取 (Entity Extraction)
从用户输入中提取关键实体信息：
- **对象实体**: 登录功能、报表系统、用户管理等
- **时间实体**: 今天、昨天、早上、现在等
- **平台实体**: 移动端、PC端、Web端等
- **范围实体**: 所有用户、部分用户等

### 3. 语义分析 (Semantic Analysis)
评估语义质量：
- **完整度 (Completeness)**: 0.0-1.0，基于必需实体的覆盖度
- **置信度 (Confidence)**: 0.0-1.0，基于输入长度和实体数量
- **可执行性 (Is Executable)**: 判断需求是否足够明确可执行
- **缺失实体 (Missing Entities)**: 列出缺少的关键信息

### 4. 智能澄清 (Intelligent Clarification)
根据语义缺失生成针对性澄清问题：
- 基于意图类型生成特定问题
- 识别缺失的关键实体
- 支持多轮澄清直到需求明确
- 动态更新语义理解

### 5. 多轮对话 (Multi-turn Dialogue)
- 保持对话上下文
- 每次澄清后重新分析语义
- 合并新旧实体信息
- 智能判断是否需要继续澄清

## 技术架构

### 状态管理
```python
class ClarificationState:
    user_input: str                          # 用户原始输入
    semantic_info: SemanticInfo              # 语义信息（意图、实体、完整度等）
    clarifications: List[str]                # 澄清问题列表
    clarification_answers: List[str]         # 用户答案列表
    is_clear: bool                           # 是否已明确
    round_count: int                         # 澄清轮次
```

### 工作流节点
1. **analyze_semantic**: 语义分析（意图识别 + 实体提取 + 完整度评估）
2. **generate_clarification_questions**: 基于语义缺失生成澄清问题
3. **ask_clarification**: 询问问题（HITL暂停点）
4. **update_understanding**: 整合答案并重新分析语义
5. **finalize_understanding**: 输出最终理解结果

### 工作流路由
- `route_by_clarity`: 根据语义完整度决定是否需要澄清
- `route_after_clarification`: 决定继续澄清还是结束

## 运行方式

### 演示模式（推荐）
```bash
python 05_clarification_system.py --demo
```

演示模式包含3个场景：
1. **投诉类** - 模糊的技术问题："这个东西有问题"
2. **操作类** - 相对明确的需求："我需要在用户管理页面添加批量导出功能"
3. **需求类** - 极度模糊的需求："能不能弄一下"

### 交互模式
```bash
python 05_clarification_system.py
```

交互模式支持真实的人机对话，系统会根据您的输入主动澄清。

## 示例输出

### 场景1：投诉类问题
```
🔍 语义分析: '这个东西有问题'
   意图识别: 投诉
   实体提取: 无
   缺失实体: {'对象', '问题现象'}
   完整度: 0.00 | 置信度: 0.35 | 可执行: False
   模糊度评分: 10/10

💭 生成澄清问题（第 1 轮）...
📝 已生成 4 个澄清问题
   1. 能否详细描述问题的具体现象？例如：错误提示、操作失败的具体步骤等。
   2. 请问是哪个功能或模块出现了问题？
   3. 问题是什么时候开始出现的？影响范围有多大？
   4. 能否提供更多背景信息？比如具体的使用场景、涉及的对象等。
```

经过多轮澄清后：
```
【语义分析结果】
意图类型：投诉
识别实体：{'对象': '登录功能', '时间': '今天', '范围': '所有用户', '平台': '移动端'}
语义完整度：0.80
识别置信度：0.75
是否可执行：是
```

## 关键改进点

相比原始的澄清系统，新版本的主要改进：

1. **从模糊度检测 → 语义分析**
   - 原版：简单的关键词匹配计算模糊度
   - 新版：完整的意图识别 + 实体提取 + 语义评估

2. **从通用问题 → 针对性问题**
   - 原版：基于规则生成固定问题
   - 新版：根据意图类型和缺失实体动态生成

3. **从静态理解 → 动态更新**
   - 原版：简单拼接问答内容
   - 新版：每轮澄清后重新分析语义，合并新实体

4. **从简单判断 → 智能决策**
   - 原版：基于回答长度判断
   - 新版：基于语义完整度和可执行性判断

## 应用场景

### 1. 智能客服系统
- 理解客户的模糊问题
- 主动澄清问题细节
- 提供精准的服务

### 2. 对话机器人
- 构建上下文理解能力
- 处理不完整的用户输入
- 引导用户表达需求

### 3. 需求分析工具
- 澄清产品需求细节
- 识别需求的完整度
- 辅助需求文档编写

### 4. 任务管理助手
- 理解模糊的任务描述
- 提取任务的关键信息
- 确保任务可执行

## 扩展建议

### 1. 集成 LLM
将规则匹配替换为 LLM 调用，提升识别准确度：
```python
def recognize_intent_with_llm(text: str) -> IntentType:
    """使用 LLM 识别意图"""
    prompt = f"识别以下文本的意图类型：{text}"
    response = llm.invoke(prompt)
    # 解析响应并返回意图类型
```

### 2. 使用 NER 模型
使用专业的命名实体识别模型：
```python
import spacy
nlp = spacy.load("zh_core_web_sm")

def extract_entities_with_ner(text: str) -> Dict[str, str]:
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entities[ent.label_] = ent.text
    return entities
```

### 3. 对话历史管理
维护完整的对话历史，支持上下文引用：
```python
state.conversation_history.append({
    "role": "user",
    "content": user_input,
    "semantic_info": semantic_info.to_dict()
})
```

### 4. 意图优先级
处理复合意图（同时包含多个意图类型）：
```python
def recognize_multi_intent(text: str) -> List[IntentType]:
    """识别可能包含的多个意图"""
    # 返回意图列表及优先级
```

## 学习要点

通过本示例，您可以学习：

1. ✅ 如何设计语义识别系统（意图 + 实体）
2. ✅ 如何评估语义完整度和可执行性
3. ✅ 如何基于语义缺失生成澄清问题
4. ✅ 如何实现多轮对话的语义累积
5. ✅ 如何使用 LangGraph 的 interrupt() 实现 HITL
6. ✅ 如何设计智能路由决策逻辑

## 文件说明

- `05_clarification_system.py`: 完整的语义识别与澄清系统实现
- `05_clarification_system_README.md`: 本文档

## 依赖项

```bash
pip install langgraph
```

## 许可证

本代码仅用于学习和演示目的。
