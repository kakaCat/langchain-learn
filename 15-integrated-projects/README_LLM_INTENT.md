# LLM 意图识别智能客服系统

## 概述

这是一个升级版的智能客服系统，使用 **LLM（大语言模型）** 进行意图识别，相比传统的基于规则的方法，具有更强的理解能力和灵活性。

## 主要改进

### 1. LLM 驱动的意图识别

使用 LangChain + OpenAI GPT-4o-mini 实现智能意图识别：

- ✅ **自然语言理解**: 能够理解复杂、口语化的表达
- ✅ **上下文感知**: 准确识别一句话中包含的多个意图
- ✅ **智能实体提取**: 自动提取订单号、产品名、金额等关键信息
- ✅ **情感分析**: 精准识别客户情绪（积极/中性/消极）
- ✅ **紧急程度评估**: 智能判断问题的紧急程度
- ✅ **推理能力**: 提供分析依据和建议行动

### 2. 降级保护机制

系统具有完善的降级保护：

- 🔄 **自动回退**: LLM 调用失败时自动切换到规则方法
- ⚙️ **灵活配置**: 可通过参数选择使用 LLM 或规则方法
- 🛡️ **稳定可靠**: 确保系统在任何情况下都能正常工作

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│          IntegratedCustomerService                      │
│  (集成智能客服系统)                                      │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼────────┐    ┌────────▼────────┐
│ Workflow       │    │ Monitor         │
│ (工作流管理)    │    │ (性能监控)       │
└───────┬────────┘    └─────────────────┘
        │
┌───────▼────────────────────────────────────┐
│      IntentRecognizer                      │
│      (意图识别器)                           │
├────────────────────────────────────────────┤
│  🤖 LLM Mode                               │
│  • ChatOpenAI (GPT-4o-mini)               │
│  • Structured Output (Pydantic)           │
│  • JSON Parser                            │
│  • Prompt Engineering                     │
├────────────────────────────────────────────┤
│  📋 Rule-based Mode (Fallback)            │
│  • Regex Pattern Matching                │
│  • Keyword Analysis                       │
│  • Entity Extraction                      │
└────────────────────────────────────────────┘
```

## 安装依赖

```bash
# 安装必要的包
pip install langchain langchain-openai langchain-core

# 或者从 requirements.txt 安装
pip install -r requirements.txt
```

## 环境配置

```bash
# 设置 OpenAI API Key
export OPENAI_API_KEY="your-openai-api-key-here"

# 可选：设置 LangSmith (用于追踪和监控)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-langsmith-api-key"
export LANGCHAIN_PROJECT="customer-service-intent"
```

## 使用方法

### 基础使用

```python
from smart_customer_service import IntegratedCustomerService

# 创建客服系统（默认使用 LLM）
service = IntegratedCustomerService(use_llm_intent=True)

# 处理客户消息
result = service.process_customer_message(
    customer_id="customer_001",
    message_content="我的订单 ORD123456 三天了还没发货，很着急！"
)

# 查看结果
print(f"识别意图: {result['current_intent']}")
print(f"置信度: {result['confidence']}")
print(f"提取实体: {result['collected_info']}")
print(f"系统响应: {result['system_response']}")
```

### 使用规则方法（备用）

```python
# 创建使用规则的客服系统
service = IntegratedCustomerService(use_llm_intent=False)

# 其余使用方式相同
```

### 运行演示

```bash
# 运行基础演示
python 01_smart_customer_service.py

# 运行对比演示（LLM vs 规则）
python demo_llm_intent_recognition.py
```

## LLM vs 规则方法对比

| 特性 | LLM 方法 | 规则方法 |
|------|---------|---------|
| **准确性** | ⭐⭐⭐⭐⭐ 高 | ⭐⭐⭐ 中等 |
| **灵活性** | ⭐⭐⭐⭐⭐ 很强 | ⭐⭐ 有限 |
| **上下文理解** | ⭐⭐⭐⭐⭐ 优秀 | ⭐⭐ 较弱 |
| **实体提取** | ⭐⭐⭐⭐⭐ 智能 | ⭐⭐⭐ 固定模式 |
| **响应速度** | ⭐⭐⭐ 较快 (API调用) | ⭐⭐⭐⭐⭐ 很快 |
| **成本** | 💰 有 API 成本 | 💰 无额外成本 |
| **离线能力** | ❌ 需要网络 | ✅ 可离线 |
| **可维护性** | ⭐⭐⭐⭐⭐ 易维护 | ⭐⭐ 需手动更新规则 |

## 测试场景示例

### 场景 1: 复杂意图识别

**客户消息**: "我昨天买的手机一直开不了机，太生气了！马上给我退款！"

**LLM 分析**:
- 意图: `complaint` (投诉) + `refund` (退款)
- 情感: `negative` (消极)
- 紧急程度: `high` (高)
- 实体: 产品类型 = 手机
- 建议行动: 安抚情绪 → 技术诊断 → 处理退款

**规则分析**:
- 意图: `refund` (仅识别退款关键词)
- 情感: `negative`
- 紧急程度: `low`
- 实体: 无
- 建议行动: 处理退款

### 场景 2: 口语化表达

**客户消息**: "app老是闪退，根本没法用，这怎么整？"

**LLM 分析**:
- 意图: `support` (技术支持)
- 情感: `negative`
- 紧急程度: `medium`
- 问题描述: app 闪退
- 建议行动: 收集设备信息 → 技术支持

**规则分析**:
- 意图: `general` (无法匹配规则)
- 情感: `neutral`
- 紧急程度: `low`

## 代码结构

```
15-integrated-projects/
├── 01_smart_customer_service.py        # 主系统实现
├── demo_llm_intent_recognition.py      # LLM vs 规则对比演示
├── README_LLM_INTENT.md                # 本文档
└── requirements.txt                    # 依赖包列表
```

## 核心类说明

### IntentRecognizer

意图识别器，支持 LLM 和规则两种模式：

```python
class IntentRecognizer:
    def __init__(self, use_llm: bool = True):
        """初始化意图识别器"""

    def analyze_intent(self, message: CustomerMessage) -> IntentAnalysis:
        """分析客户意图"""

    def _analyze_intent_with_llm(self, message: CustomerMessage) -> IntentAnalysis:
        """使用 LLM 分析"""

    def _analyze_intent_with_rules(self, message: CustomerMessage) -> IntentAnalysis:
        """使用规则分析（备用）"""
```

### IntentAnalysisOutput (Pydantic 模型)

LLM 输出的结构化模型：

```python
class IntentAnalysisOutput(BaseModel):
    intent: str                    # 意图类型
    confidence: float              # 置信度 (0-1)
    entities: Dict[str, Any]       # 提取的实体
    sentiment: str                 # 情感 (positive/neutral/negative)
    urgency_level: str             # 紧急程度 (low/medium/high)
    reasoning: str                 # 推理过程
    suggested_actions: List[str]   # 建议行动
```

## Prompt 工程

系统使用精心设计的 Prompt 来指导 LLM 进行意图分析：

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的客服意图分析助手。

    请分析客户消息，提取：
    1. 意图分类 (inquiry/complaint/support/order/payment/refund/general)
    2. 置信度 (0-1)
    3. 实体提取 (订单号、产品名、金额等)
    4. 情感分析 (positive/neutral/negative)
    5. 紧急程度 (low/medium/high)
    6. 推理过程
    7. 建议行动

    {format_instructions}"""),
    ("human", "客户消息：{customer_message}")
])
```

## 性能优化建议

1. **模型选择**: 使用 `gpt-4o-mini` 平衡性能和成本
2. **温度设置**: `temperature=0.1` 保证输出稳定性
3. **缓存策略**: 对相似问题可以实现缓存机制
4. **批量处理**: 高并发场景可以考虑批量调用
5. **降级策略**: LLM 失败时自动回退到规则方法

## 监控和调试

系统集成了 LangSmith 监控功能：

```python
# 查看监控面板
dashboard = service.get_monitoring_dashboard()

print(f"平均置信度: {dashboard['summary']['average_confidence']}")
print(f"意图分布: {dashboard['intent_distribution']}")
print(f"系统健康: {dashboard['system_health']}")
```

## 常见问题

### Q1: LLM 调用失败怎么办？

系统会自动回退到规则方法，确保服务不中断。

### Q2: 如何降低 API 成本？

- 使用 `gpt-4o-mini` 而非 `gpt-4`
- 实现缓存机制
- 对简单问题使用规则方法

### Q3: 如何提高识别准确率？

- 优化 Prompt 模板
- 提供更多上下文信息
- 使用少样本学习（Few-shot learning）

### Q4: 能否使用本地 LLM？

可以，替换 `ChatOpenAI` 为本地模型接口即可：

```python
from langchain_community.llms import Ollama

self.llm = Ollama(model="llama2")
```

## 后续优化方向

- [ ] 支持多轮对话的上下文记忆
- [ ] 实现意图识别的 A/B 测试
- [ ] 添加用户反馈学习机制
- [ ] 集成更多 LLM 提供商（Azure、Claude等）
- [ ] 实现意图识别的在线评估

## 许可证

MIT License

## 作者

LangChain 学习项目
