#!/usr/bin/env python3
"""
智能客服系统集成项目
综合应用 LangChain、LangGraph 和 LangSmith 技术栈
"""

import os
import json
import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from datetime import datetime, timedelta
import re

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Pydantic imports - 兼容不同版本
try:
    from langchain_core.pydantic_v1 import BaseModel, Field
except ImportError:
    try:
        from pydantic.v1 import BaseModel, Field
    except ImportError:
        from pydantic import BaseModel, Field


class CustomerIntent(Enum):
    """客户意图枚举"""
    INQUIRY = "inquiry"  # 咨询
    COMPLAINT = "complaint"  # 投诉
    SUPPORT = "support"  # 技术支持
    ORDER = "order"  # 订单相关
    PAYMENT = "payment"  # 支付问题
    REFUND = "refund"  # 退款
    GENERAL = "general"  # 一般问题


class ConversationState(Enum):
    """对话状态枚举"""
    INITIAL = "initial"  # 初始状态
    INTENT_RECOGNITION = "intent_recognition"  # 意图识别
    INFORMATION_GATHERING = "information_gathering"  # 信息收集
    PROBLEM_SOLVING = "problem_solving"  # 问题解决
    ESCALATION = "escalation"  # 升级处理
    RESOLUTION = "resolution"  # 问题解决
    FOLLOW_UP = "follow_up"  # 后续跟进


@dataclass
class CustomerMessage:
    """客户消息"""
    message_id: str
    customer_id: str
    content: str
    timestamp: datetime
    channel: str  # web, mobile, wechat, etc.
    sentiment: float = 0.0
    urgency: float = 0.0


@dataclass
class IntentAnalysis:
    """意图分析结果"""
    intent: CustomerIntent
    confidence: float
    entities: Dict[str, Any]
    sentiment: str
    urgency_level: str
    suggested_actions: List[str]


# Pydantic 模型用于 LLM 输出结构化
class IntentAnalysisOutput(BaseModel):
    """意图分析输出模型"""
    intent: str = Field(description="客户意图：inquiry(咨询), complaint(投诉), support(技术支持), order(订单), payment(支付), refund(退款), general(一般问题)")
    confidence: float = Field(description="置信度，范围0-1", ge=0.0, le=1.0)
    entities: Dict[str, Any] = Field(description="提取的实体信息，如订单号、产品名称、金额等")
    sentiment: str = Field(description="情感分析：positive(积极), neutral(中性), negative(消极)")
    urgency_level: str = Field(description="紧急程度：low(低), medium(中), high(高)")
    reasoning: str = Field(description="分析推理过程")
    suggested_actions: List[str] = Field(description="建议的处理行动")


@dataclass
class ConversationContext:
    """对话上下文"""
    conversation_id: str
    customer_id: str
    current_state: ConversationState
    history: List[Dict[str, Any]]
    collected_info: Dict[str, Any]
    intent_analysis: Optional[IntentAnalysis] = None
    escalation_reason: Optional[str] = None
    resolution_summary: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class SystemResponse:
    """系统响应"""
    response_id: str
    content: str
    actions: List[str]
    next_state: ConversationState
    confidence: float
    reasoning: str
    requires_human: bool = False


class IntentRecognizer:
    """意图识别器 (LangChain 应用) - 使用 LLM 进行意图识别"""

    def __init__(self, use_llm: bool = True):
        """
        初始化意图识别器

        Args:
            use_llm: 是否使用 LLM 进行意图识别，默认为 True。设置为 False 则使用基于规则的方法
        """
        self.use_llm = use_llm

        # 始终初始化规则相关的属性，作为备用方案
        self.intent_patterns = {
            CustomerIntent.INQUIRY: [
                r"请问.*", r"咨询.*", r"了解.*", r"想问.*"
            ],
            CustomerIntent.COMPLAINT: [
                r"投诉.*", r"不满意.*", r"问题.*", r"糟糕.*", r"差评.*"
            ],
            CustomerIntent.SUPPORT: [
                r"帮助.*", r"支持.*", r"故障.*", r"无法.*", r"错误.*"
            ],
            CustomerIntent.ORDER: [
                r"订单.*", r"下单.*", r"购买.*", r"发货.*", r"物流.*"
            ],
            CustomerIntent.PAYMENT: [
                r"支付.*", r"付款.*", r"费用.*", r"价格.*"
            ],
            CustomerIntent.REFUND: [
                r"退款.*", r"退货.*", r"取消.*", r"撤销.*"
            ]
        }

        self.sentiment_keywords = {
            "positive": ["满意", "不错", "很好", "感谢", "谢谢"],
            "negative": ["不满意", "糟糕", "差劲", "生气", "投诉"],
            "urgent": ["紧急", "立刻", "马上", "尽快", "着急"]
        }

        if self.use_llm:
            try:
                # 初始化 LLM
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.1,  # 低温度保证输出稳定
                    api_key=os.getenv("OPENAI_API_KEY")
                )

                # 创建输出解析器
                self.parser = JsonOutputParser(pydantic_object=IntentAnalysisOutput)

                # 创建提示模板
                self.prompt = ChatPromptTemplate.from_messages([
                    ("system", """你是一个专业的客服意图分析助手。你的任图是分析客户消息，识别其意图、情感和紧急程度。

请仔细分析客户消息，提取以下信息：

1. **意图分类**（必须从以下选项中选择）：
   - inquiry: 一般咨询、询问信息
   - complaint: 投诉、抱怨、不满
   - support: 技术支持、故障报修、使用帮助
   - order: 订单查询、物流追踪、发货问题
   - payment: 支付问题、付款失败、价格咨询
   - refund: 退款、退货、取消订单
   - general: 其他一般性问题

2. **置信度**：你对意图判断的确定程度（0-1之间的浮点数）

3. **实体提取**：从消息中提取关键信息，如：
   - order_number: 订单号
   - product_name: 产品名称
   - amount: 金额（数字）
   - date: 日期
   - 其他相关信息

4. **情感分析**（必须从以下选项中选择）：
   - positive: 积极、满意、友好
   - neutral: 中性、平和
   - negative: 消极、不满、愤怒

5. **紧急程度**（必须从以下选项中选择）：
   - low: 不紧急，可以稍后处理
   - medium: 需要及时关注
   - high: 紧急，需要立即处理

6. **推理过程**：简要说明你的分析依据

7. **建议行动**：基于分析结果，建议采取的处理措施

{format_instructions}"""),
                    ("human", "客户消息：{customer_message}")
                ])

                # 创建处理链
                self.chain = self.prompt | self.llm | self.parser

            except Exception as e:
                print(f"LLM 初始化失败: {e}，将使用规则方法")
                self.use_llm = False

    def analyze_intent(self, message: CustomerMessage) -> IntentAnalysis:
        """分析客户意图"""
        if self.use_llm:
            return self._analyze_intent_with_llm(message)
        else:
            return self._analyze_intent_with_rules(message)

    def _analyze_intent_with_llm(self, message: CustomerMessage) -> IntentAnalysis:
        """使用 LLM 分析意图"""
        try:
            # 调用 LLM 链
            result = self.chain.invoke({
                "customer_message": message.content,
                "format_instructions": self.parser.get_format_instructions()
            })

            # 转换为 IntentAnalysis 对象
            intent_str = result.get("intent", "general")
            try:
                intent = CustomerIntent(intent_str)
            except ValueError:
                intent = CustomerIntent.GENERAL

            return IntentAnalysis(
                intent=intent,
                confidence=result.get("confidence", 0.5),
                entities=result.get("entities", {}),
                sentiment=result.get("sentiment", "neutral"),
                urgency_level=result.get("urgency_level", "low"),
                suggested_actions=result.get("suggested_actions", [])
            )

        except Exception as e:
            print(f"LLM 意图识别失败: {e}，使用规则回退")
            # 如果 LLM 失败，回退到基于规则的方法
            return self._analyze_intent_with_rules(message)

    def _analyze_intent_with_rules(self, message: CustomerMessage) -> IntentAnalysis:
        """使用规则分析意图（备用方案）"""
        content = message.content.lower()

        # 意图识别
        intent, confidence = self._detect_intent(content)

        # 情感分析
        sentiment = self._analyze_sentiment(content)

        # 紧急程度分析
        urgency_level = self._analyze_urgency(content)

        # 实体提取
        entities = self._extract_entities(content)

        # 建议行动
        suggested_actions = self._suggest_actions(intent, sentiment, urgency_level)

        return IntentAnalysis(
            intent=intent,
            confidence=confidence,
            entities=entities,
            sentiment=sentiment,
            urgency_level=urgency_level,
            suggested_actions=suggested_actions
        )

    def _detect_intent(self, content: str) -> tuple[CustomerIntent, float]:
        """检测意图"""
        max_matches = 0
        detected_intent = CustomerIntent.GENERAL

        for intent, patterns in self.intent_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, content))
            if matches > max_matches:
                max_matches = matches
                detected_intent = intent

        confidence = min(max_matches / 2, 1.0)  # 简化置信度计算
        return detected_intent, confidence

    def _analyze_sentiment(self, content: str) -> str:
        """分析情感"""
        positive_count = sum(1 for word in self.sentiment_keywords["positive"] if word in content)
        negative_count = sum(1 for word in self.sentiment_keywords["negative"] if word in content)

        if negative_count > positive_count:
            return "negative"
        elif positive_count > negative_count:
            return "positive"
        else:
            return "neutral"

    def _analyze_urgency(self, content: str) -> str:
        """分析紧急程度"""
        urgent_count = sum(1 for word in self.sentiment_keywords["urgent"] if word in content)

        if urgent_count >= 2:
            return "high"
        elif urgent_count == 1:
            return "medium"
        else:
            return "low"

    def _extract_entities(self, content: str) -> Dict[str, Any]:
        """提取实体"""
        entities = {}

        # 提取订单号
        order_pattern = r"订单[号]?[：:]?\s*(\w+)"
        order_match = re.search(order_pattern, content)
        if order_match:
            entities["order_number"] = order_match.group(1)

        # 提取产品名称
        product_pattern = r"产品[：:]?\s*(\w+)"
        product_match = re.search(product_pattern, content)
        if product_match:
            entities["product_name"] = product_match.group(1)

        # 提取金额
        amount_pattern = r"(\d+(?:\.\d+)?)元"
        amount_match = re.search(amount_pattern, content)
        if amount_match:
            entities["amount"] = float(amount_match.group(1))

        return entities

    def _suggest_actions(self, intent: CustomerIntent, sentiment: str, urgency: str) -> List[str]:
        """建议行动"""
        actions = []

        if intent == CustomerIntent.COMPLAINT:
            actions.append("安抚客户情绪")
            actions.append("收集问题详细信息")
            if urgency == "high":
                actions.append("优先处理")

        elif intent == CustomerIntent.SUPPORT:
            actions.append("提供技术支持")
            actions.append("收集故障信息")

        elif intent == CustomerIntent.ORDER:
            actions.append("查询订单状态")
            actions.append("提供物流信息")

        elif intent == CustomerIntent.PAYMENT:
            actions.append("检查支付状态")
            actions.append("提供支付解决方案")

        elif intent == CustomerIntent.REFUND:
            actions.append("检查退款政策")
            actions.append("处理退款申请")

        # 基于情感的额外行动
        if sentiment == "negative":
            actions.append("加强客户关怀")

        return actions


class CustomerServiceWorkflow:
    """客服工作流管理器 (LangGraph 应用)"""

    def __init__(self, use_llm_intent: bool = True):
        """
        初始化客服工作流

        Args:
            use_llm_intent: 是否使用 LLM 进行意图识别
        """
        self.intent_recognizer = IntentRecognizer(use_llm=use_llm_intent)
        self.conversations: Dict[str, ConversationContext] = {}
    
    def start_conversation(self, customer_id: str, initial_message: CustomerMessage) -> ConversationContext:
        """开始对话"""
        conversation_id = str(uuid.uuid4())
        
        context = ConversationContext(
            conversation_id=conversation_id,
            customer_id=customer_id,
            current_state=ConversationState.INITIAL,
            history=[],
            collected_info={}
        )
        
        self.conversations[conversation_id] = context
        
        # 处理初始消息
        return self.process_message(conversation_id, initial_message)
    
    def process_message(self, conversation_id: str, message: CustomerMessage) -> ConversationContext:
        """处理客户消息"""
        if conversation_id not in self.conversations:
            raise ValueError(f"对话 {conversation_id} 不存在")
        
        context = self.conversations[conversation_id]
        context.last_activity = datetime.now()
        
        # 记录消息历史
        context.history.append({
            "type": "customer",
            "content": message.content,
            "timestamp": message.timestamp,
            "sentiment": message.sentiment,
            "urgency": message.urgency
        })
        
        # 根据当前状态处理消息
        if context.current_state == ConversationState.INITIAL:
            return self._handle_initial_state(context, message)
        elif context.current_state == ConversationState.INTENT_RECOGNITION:
            return self._handle_intent_recognition(context, message)
        elif context.current_state == ConversationState.INFORMATION_GATHERING:
            return self._handle_information_gathering(context, message)
        elif context.current_state == ConversationState.PROBLEM_SOLVING:
            return self._handle_problem_solving(context, message)
        elif context.current_state == ConversationState.ESCALATION:
            return self._handle_escalation(context, message)
        else:
            return self._handle_other_states(context, message)
    
    def _handle_initial_state(self, context: ConversationContext, message: CustomerMessage) -> ConversationContext:
        """处理初始状态"""
        # 分析意图
        intent_analysis = self.intent_recognizer.analyze_intent(message)
        context.intent_analysis = intent_analysis
        
        # 根据意图转移到下一个状态
        if intent_analysis.intent in [CustomerIntent.COMPLAINT, CustomerIntent.SUPPORT]:
            context.current_state = ConversationState.INFORMATION_GATHERING
        else:
            context.current_state = ConversationState.PROBLEM_SOLVING
        
        return context
    
    def _handle_intent_recognition(self, context: ConversationContext, message: CustomerMessage) -> ConversationContext:
        """处理意图识别状态"""
        # 重新分析意图（如果需要）
        if not context.intent_analysis:
            context.intent_analysis = self.intent_recognizer.analyze_intent(message)
        
        context.current_state = ConversationState.INFORMATION_GATHERING
        return context
    
    def _handle_information_gathering(self, context: ConversationContext, message: CustomerMessage) -> ConversationContext:
        """处理信息收集状态"""
        # 提取和存储信息
        entities = self.intent_recognizer._extract_entities(message.content)
        context.collected_info.update(entities)
        
        # 检查是否收集到足够信息
        required_info = self._get_required_info(context.intent_analysis.intent)
        if all(info in context.collected_info for info in required_info):
            context.current_state = ConversationState.PROBLEM_SOLVING
        
        return context
    
    def _handle_problem_solving(self, context: ConversationContext, message: CustomerMessage) -> ConversationContext:
        """处理问题解决状态"""
        # 基于收集的信息和意图解决问题
        solution = self._generate_solution(context)
        
        if solution.get("requires_escalation", False):
            context.current_state = ConversationState.ESCALATION
            context.escalation_reason = solution["escalation_reason"]
        else:
            context.current_state = ConversationState.RESOLUTION
            context.resolution_summary = solution["resolution"]
        
        return context
    
    def _handle_escalation(self, context: ConversationContext, message: CustomerMessage) -> ConversationContext:
        """处理升级状态"""
        # 等待人工处理或特殊处理
        context.current_state = ConversationState.RESOLUTION
        return context
    
    def _handle_other_states(self, context: ConversationContext, message: CustomerMessage) -> ConversationContext:
        """处理其他状态"""
        # 默认处理
        return context
    
    def _get_required_info(self, intent: CustomerIntent) -> List[str]:
        """获取所需信息"""
        info_requirements = {
            CustomerIntent.ORDER: ["order_number"],
            CustomerIntent.PAYMENT: ["order_number", "amount"],
            CustomerIntent.REFUND: ["order_number", "product_name"],
            CustomerIntent.SUPPORT: ["product_name"],
            CustomerIntent.COMPLAINT: ["order_number", "product_name"]
        }
        
        return info_requirements.get(intent, [])
    
    def _generate_solution(self, context: ConversationContext) -> Dict[str, Any]:
        """生成解决方案"""
        intent = context.intent_analysis.intent
        collected_info = context.collected_info
        
        solutions = {
            CustomerIntent.ORDER: {
                "resolution": f"已查询订单 {collected_info.get('order_number', '未知')} 状态：处理中",
                "requires_escalation": False
            },
            CustomerIntent.PAYMENT: {
                "resolution": f"支付问题已处理，订单 {collected_info.get('order_number', '未知')}",
                "requires_escalation": False
            },
            CustomerIntent.REFUND: {
                "resolution": f"退款申请已提交，订单 {collected_info.get('order_number', '未知')}",
                "requires_escalation": False
            },
            CustomerIntent.SUPPORT: {
                "resolution": f"技术支持问题已记录，产品 {collected_info.get('product_name', '未知')}",
                "requires_escalation": collected_info.get('product_name') == "复杂产品"
            },
            CustomerIntent.COMPLAINT: {
                "resolution": f"投诉问题已升级处理",
                "requires_escalation": True,
                "escalation_reason": "客户投诉需要人工处理"
            }
        }
        
        return solutions.get(intent, {
            "resolution": "问题已处理",
            "requires_escalation": False
        })
    
    def generate_response(self, context: ConversationContext) -> SystemResponse:
        """生成系统响应"""
        response_id = str(uuid.uuid4())
        
        # 基于当前状态和意图生成响应
        if context.current_state == ConversationState.INITIAL:
            content = "您好！欢迎使用智能客服系统，请问有什么可以帮助您的？"
            actions = ["意图识别", "信息收集"]
            next_state = ConversationState.INTENT_RECOGNITION
            
        elif context.current_state == ConversationState.INTENT_RECOGNITION:
            intent_name = context.intent_analysis.intent.value if context.intent_analysis else "未知"
            content = f"已识别您的意图为：{intent_name}，正在为您处理..."
            actions = ["信息收集", "问题分析"]
            next_state = ConversationState.INFORMATION_GATHERING
            
        elif context.current_state == ConversationState.INFORMATION_GATHERING:
            required_info = self._get_required_info(context.intent_analysis.intent)
            missing_info = [info for info in required_info if info not in context.collected_info]
            
            if missing_info:
                info_prompts = {
                    "order_number": "请提供您的订单号",
                    "product_name": "请提供产品名称",
                    "amount": "请提供相关金额"
                }
                prompt = ", ".join(info_prompts.get(info, info) for info in missing_info)
                content = f"为了更好帮助您，{prompt}。"
                actions = ["继续收集信息"]
                next_state = ConversationState.INFORMATION_GATHERING
            else:
                content = "信息收集完成，正在分析您的问题..."
                actions = ["问题分析", "生成解决方案"]
                next_state = ConversationState.PROBLEM_SOLVING
                
        elif context.current_state == ConversationState.PROBLEM_SOLVING:
            solution = self._generate_solution(context)
            content = solution.get("resolution", "正在处理您的问题...")
            
            if solution.get("requires_escalation", False):
                actions = ["升级处理", "通知人工客服"]
                next_state = ConversationState.ESCALATION
            else:
                actions = ["确认解决方案", "结束对话"]
                next_state = ConversationState.RESOLUTION
                
        elif context.current_state == ConversationState.ESCALATION:
            content = f"您的问题已升级处理，原因：{context.escalation_reason}。请稍等，人工客服将很快联系您。"
            actions = ["等待人工处理", "记录处理进度"]
            next_state = ConversationState.RESOLUTION
            
        elif context.current_state == ConversationState.RESOLUTION:
            content = f"问题处理完成：{context.resolution_summary}。感谢您的咨询！"
            actions = ["满意度调查", "结束对话"]
            next_state = ConversationState.FOLLOW_UP
            
        else:
            content = "正在处理您的请求..."
            actions = ["继续处理"]
            next_state = context.current_state
        
        return SystemResponse(
            response_id=response_id,
            content=content,
            actions=actions,
            next_state=next_state,
            confidence=context.intent_analysis.confidence if context.intent_analysis else 0.5,
            reasoning=f"基于当前状态 {context.current_state.value} 和意图 {context.intent_analysis.intent.value if context.intent_analysis else '未知'} 生成响应",
            requires_human=context.current_state == ConversationState.ESCALATION
        )


class LangSmithMonitor:
    """LangSmith 监控器 (LangSmith 应用)"""
    
    def __init__(self):
        self.metrics_history = []
        self.traces = []
        self.alerts = []
    
    def record_conversation_metrics(self, context: ConversationContext, response: SystemResponse):
        """记录对话指标"""
        metrics = {
            "conversation_id": context.conversation_id,
            "customer_id": context.customer_id,
            "timestamp": datetime.now(),
            "intent": context.intent_analysis.intent.value if context.intent_analysis else "unknown",
            "confidence": context.intent_analysis.confidence if context.intent_analysis else 0.0,
            "sentiment": context.intent_analysis.sentiment if context.intent_analysis else "neutral",
            "urgency": context.intent_analysis.urgency_level if context.intent_analysis else "low",
            "current_state": context.current_state.value,
            "response_confidence": response.confidence,
            "requires_human": response.requires_human,
            "processing_time": (datetime.now() - context.last_activity).total_seconds()
        }
        
        self.metrics_history.append(metrics)
        
        # 检查是否需要警报
        self._check_alerts(metrics)
    
    def record_trace(self, conversation_id: str, action: str, details: Dict[str, Any]):
        """记录调用追踪"""
        trace = {
            "trace_id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "timestamp": datetime.now(),
            "action": action,
            "details": details
        }
        
        self.traces.append(trace)
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """检查警报条件"""
        # 低置信度警报
        if metrics["confidence"] < 0.3:
            self.alerts.append({
                "alert_id": str(uuid.uuid4()),
                "type": "low_confidence",
                "conversation_id": metrics["conversation_id"],
                "message": f"意图识别置信度过低: {metrics['confidence']}",
                "timestamp": datetime.now(),
                "severity": "warning"
            })
        
        # 高紧急程度警报
        if metrics["urgency"] == "high":
            self.alerts.append({
                "alert_id": str(uuid.uuid4()),
                "type": "high_urgency",
                "conversation_id": metrics["conversation_id"],
                "message": "客户问题紧急程度高",
                "timestamp": datetime.now(),
                "severity": "high"
            })
        
        # 负面情感警报
        if metrics["sentiment"] == "negative":
            self.alerts.append({
                "alert_id": str(uuid.uuid4()),
                "type": "negative_sentiment",
                "conversation_id": metrics["conversation_id"],
                "message": "客户情感负面，需要特别关注",
                "timestamp": datetime.now(),
                "severity": "medium"
            })
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """生成监控报告"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-10:]  # 最近10条记录
        
        # 计算平均指标
        avg_confidence = sum(m["confidence"] for m in recent_metrics) / len(recent_metrics)
        avg_processing_time = sum(m["processing_time"] for m in recent_metrics) / len(recent_metrics)
        
        # 意图分布
        intent_distribution = {}
        for m in recent_metrics:
            intent = m["intent"]
            intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
        
        # 状态分布
        state_distribution = {}
        for m in recent_metrics:
            state = m["current_state"]
            state_distribution[state] = state_distribution.get(state, 0) + 1
        
        return {
            "report_id": str(uuid.uuid4()),
            "timestamp": datetime.now(),
            "summary": {
                "total_conversations": len(set(m["conversation_id"] for m in self.metrics_history)),
                "recent_metrics_count": len(recent_metrics),
                "average_confidence": round(avg_confidence, 3),
                "average_processing_time": round(avg_processing_time, 3),
                "active_alerts": len([a for a in self.alerts if (datetime.now() - a["timestamp"]).total_seconds() < 3600])
            },
            "intent_distribution": intent_distribution,
            "state_distribution": state_distribution,
            "recent_alerts": self.alerts[-5:],  # 最近5个警报
            "system_health": "healthy" if avg_confidence > 0.5 else "degraded"
        }


class IntegratedCustomerService:
    """集成智能客服系统"""

    def __init__(self, use_llm_intent: bool = True):
        """
        初始化集成客服系统

        Args:
            use_llm_intent: 是否使用 LLM 进行意图识别
        """
        self.workflow = CustomerServiceWorkflow(use_llm_intent=use_llm_intent)
        self.monitor = LangSmithMonitor()
    
    def process_customer_message(self, customer_id: str, message_content: str, channel: str = "web") -> Dict[str, Any]:
        """处理客户消息"""
        # 创建客户消息
        message = CustomerMessage(
            message_id=str(uuid.uuid4()),
            customer_id=customer_id,
            content=message_content,
            timestamp=datetime.now(),
            channel=channel
        )
        
        # 记录追踪
        self.monitor.record_trace(
            "system",
            "process_message_start",
            {"customer_id": customer_id, "message_content": message_content}
        )
        
        # 处理对话
        if customer_id not in [ctx.customer_id for ctx in self.workflow.conversations.values()]:
            # 新对话
            context = self.workflow.start_conversation(customer_id, message)
        else:
            # 现有对话
            conversation_id = next(
                ctx.conversation_id for ctx in self.workflow.conversations.values() 
                if ctx.customer_id == customer_id
            )
            context = self.workflow.process_message(conversation_id, message)
        
        # 生成响应
        response = self.workflow.generate_response(context)
        
        # 记录指标
        self.monitor.record_conversation_metrics(context, response)
        
        # 记录追踪
        self.monitor.record_trace(
            context.conversation_id,
            "process_message_complete",
            {
                "response_content": response.content,
                "next_state": response.next_state.value,
                "confidence": response.confidence
            }
        )
        
        return {
            "conversation_id": context.conversation_id,
            "customer_id": customer_id,
            "system_response": response.content,
            "suggested_actions": response.actions,
            "next_state": response.next_state.value,
            "confidence": response.confidence,
            "requires_human": response.requires_human,
            "current_intent": context.intent_analysis.intent.value if context.intent_analysis else "unknown",
            "collected_info": context.collected_info
        }
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """获取监控仪表板数据"""
        return self.monitor.generate_monitoring_report()


def demo_integrated_customer_service():
    """演示集成智能客服系统"""
    print("=" * 60)
    print("智能客服系统集成演示 (使用 LLM 意图识别)")
    print("=" * 60)

    # 创建集成系统（使用 LLM 进行意图识别）
    service = IntegratedCustomerService(use_llm_intent=True)
    
    # 模拟客户对话
    test_cases = [
        ("customer_001", "我想咨询一下我的订单状态"),
        ("customer_001", "订单号是 ORD123456"),
        ("customer_002", "我要投诉产品质量问题"),
        ("customer_002", "产品名称是 智能音箱"),
        ("customer_003", "支付遇到问题，无法完成付款"),
        ("customer_004", "技术支持：设备无法连接网络")
    ]
    
    print("\n开始模拟客户对话...")
    for customer_id, message in test_cases:
        print(f"\n客户 {customer_id}: {message}")
        
        result = service.process_customer_message(customer_id, message)
        
        print(f"系统响应: {result['system_response']}")
        print(f"建议行动: {', '.join(result['suggested_actions'])}")
        print(f"当前意图: {result['current_intent']}")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"需要人工: {'是' if result['requires_human'] else '否'}")
        
        # 短暂暂停以模拟处理时间
        import time
        time.sleep(1)
    
    # 显示监控报告
    print("\n" + "=" * 60)
    print("系统监控报告")
    print("=" * 60)
    
    dashboard = service.get_monitoring_dashboard()
    
    if dashboard and dashboard.get("status") != "no_data":
        summary = dashboard.get("summary", {})
        print(f"总对话数: {summary.get('total_conversations', 0)}")
        print(f"平均置信度: {summary.get('average_confidence', 0):.2f}")
        print(f"平均处理时间: {summary.get('average_processing_time', 0):.2f}秒")
        print(f"活跃警报: {summary.get('active_alerts', 0)}")
        
        print(f"\n意图分布: {dashboard.get('intent_distribution', {})}")
        print(f"状态分布: {dashboard.get('state_distribution', {})}")
        print(f"系统健康状态: {dashboard.get('system_health', 'unknown')}")
        
        if dashboard.get("recent_alerts"):
            print("\n最近警报:")
            for alert in dashboard["recent_alerts"]:
                print(f"  - {alert.get('type', 'unknown')}: {alert.get('message', 'unknown')} (严重性: {alert.get('severity', 'low')})")
    else:
        print("暂无监控数据")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    demo_integrated_customer_service()