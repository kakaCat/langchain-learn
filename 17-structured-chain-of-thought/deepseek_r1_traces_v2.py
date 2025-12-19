"""
DeepSeek-R1 Agent V2 - Improved Chain-of-Thought Reasoning

This is a complete rewrite of the structured thinking chain agent,
designed to mimic real LLM thinking patterns (like DeepSeek-R1, OpenAI o1)
using <think> and <answer> tags.

Key Improvements over V1:
1. Natural single-pass thinking instead of forced 4-stage pipeline
2. <think> and <answer> tags for structured reasoning
3. Tool integration (calculator) for mathematical validation
4. Loop detection to prevent infinite repetition
5. Hallucination detection to avoid introducing non-existent information
6. Larger model (32b) for better reasoning capability
"""

import os
import signal
import functools
from typing import Literal, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


# 超时异常类
class StageTimeoutError(Exception):
    """Stage execution timeout exception"""
    pass


# 超时装饰器
def with_timeout(seconds=120):
    """
    为 Stage 方法添加超时保护

    Args:
        seconds: 超时时间（秒）,默认 120 秒（2 分钟）
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise StageTimeoutError(f"{func.__name__} exceeded {seconds} seconds")

            # 设置超时信号
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # 取消超时
                return result
            except StageTimeoutError:
                signal.alarm(0)
                # 超时时返回简化的默认响应
                stage_name = func.__name__.replace("_stage", "Stage ").replace("_", " ")
                print(f"\n⏱️ {stage_name} 超时 ({seconds}s)，使用简化响应")
                return "<answer>继续处理（超时简化）</answer>"
            finally:
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper
    return decorator

# Memory 支持（兼容新旧版本）
try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    # 完全独立的 Memory 实现，不依赖任何 langchain 基类
    class ConversationBufferMemory:
        def __init__(self, return_messages=True, memory_key="chat_history"):
            self.messages = []
            self.return_messages = return_messages
            self.memory_key = memory_key

        def load_memory_variables(self, inputs):
            return {self.memory_key: self.messages}

        def save_context(self, inputs, outputs):
            self.messages.append({"role": "user", "content": str(inputs)})
            self.messages.append({"role": "assistant", "content": str(outputs)})

        def clear(self):
            self.messages = []

# 工具支持（可选）
try:
    from langchain.agents import AgentExecutor, create_react_agent
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    AgentExecutor = None
    create_react_agent = None

from prompts import (
    GATE_PROMPT,
    DIRECT_PROMPT,
    SINGLE_THINK_PROMPT,
    TOOL_ENHANCED_PROMPT,
    MULTI_REFLECT_PROMPTS,
    # V2.5: Structured 4-Stage Prompts
    STAGE1_PROBLEM_DEF_V2_5,
    STAGE2_BLOOM_V2_5,
    STAGE3_VALIDATION_V2_5,
    STAGE4_FINAL_V2_5
)
from tools import ToolRegistry
from validators import LoopBreaker, HallucinationDetector
from parsers import ThinkTagParser

# 加载环境变量
load_dotenv()


def get_llm(model: Optional[str] = None, temperature: float = 0.2):
    """
    创建并配置 LLM 实例 (支持 DeepSeek API 或 Ollama)

    Args:
        model: 模型名称，默认从环境变量读取
        temperature: 温度参数，越低越确定性

    Returns:
        Chat LLM 实例
    """
    use_backend = os.getenv("USE_BACKEND", "ollama")

    if use_backend == "deepseek_api":
        # 使用 DeepSeek 官方 API
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        model_name = model or os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")

        if not api_key:
            raise ValueError("使用 DeepSeek API 需要设置 DEEPSEEK_API_KEY 环境变量")

        print(f"正在使用 DeepSeek API: {model_name} (URL: {base_url})")

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature
        )
    else:
        # 使用本地 Ollama
        model_name = model or os.getenv("OLLAMA_MODEL", "deepseek-r1:32b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        print(f"正在使用 Ollama 本地模型: {model_name} (URL: {base_url})")

        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature
        )


class DeepSeekR1AgentV2:
    """
    混合式思维链 Agent (V2)

    核心改进：
    1. 使用 deepseek-r1:32b 大模型
    2. 模仿真实 LLM 的 <think> 标签模式
    3. 集成工具调用（计算器/Python）
    4. 循环和幻觉检测
    5. 三种推理模式自适应切换
    """

    MODES = {
        "direct": "直接回答（简单问题）",
        "single_think": "单次深度思考（快速推理）",
        "structured_4stage": "结构化4阶段（复杂推理，V2.5新增）",
        "multi_reflect": "多轮反思（极复杂场景）"
    }

    def __init__(
        self,
        model: str = "deepseek-r1:32b",
        enable_tools: bool = True,
        enable_loop_detection: bool = True,
        enable_hallucination_detection: bool = False  # 默认关闭以提高速度
    ):
        """
        初始化 Agent

        Args:
            model: 模型名称
            enable_tools: 是否启用工具
            enable_loop_detection: 是否启用循环检测
            enable_hallucination_detection: 是否启用幻觉检测
        """
        self.llm = get_llm(model)
        self.parser = ThinkTagParser()
        self.enable_tools = enable_tools
        self.enable_loop_detection = enable_loop_detection
        self.enable_hallucination_detection = enable_hallucination_detection

        # 初始化组件
        if enable_loop_detection:
            self.loop_breaker = LoopBreaker(similarity_threshold=0.85)
        if enable_hallucination_detection:
            self.hallucination_detector = HallucinationDetector(self.llm)

        # 获取工具
        if enable_tools:
            self.tools = ToolRegistry.get_basic_tools()
        else:
            self.tools = []

        # V2.5: 初始化 Memory for structured_4stage mode
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )

        # 构建推理链
        self._build_chains()

    def _build_chains(self):
        """构建推理链"""
        # 门控链
        self.gate_chain = GATE_PROMPT | self.llm | StrOutputParser()

        # 直接回答链
        self.direct_chain = DIRECT_PROMPT | self.llm | StrOutputParser()

        # 单次思考链（推荐）
        self.single_think_chain = SINGLE_THINK_PROMPT | self.llm | StrOutputParser()

        # 工具增强链
        if self.enable_tools and self.tools and AGENTS_AVAILABLE:
            try:
                from langchain_core.prompts import PromptTemplate

                # 创建简单的 ReAct prompt
                react_prompt = PromptTemplate.from_template(TOOL_ENHANCED_PROMPT.template)

                self.tool_agent = create_react_agent(
                    llm=self.llm,
                    tools=self.tools,
                    prompt=react_prompt
                )
                self.tool_executor = AgentExecutor(
                    agent=self.tool_agent,
                    tools=self.tools,
                    verbose=True,
                    max_iterations=5,
                    handle_parsing_errors=True
                )
            except Exception as e:
                print(f"工具链初始化失败: {e}")
                self.tool_executor = None
        else:
            self.tool_executor = None

    def classify_complexity(self, user_input: str) -> Literal["direct", "single_think", "structured_4stage", "multi_reflect"]:
        """
        分类任务复杂度（V2.5 更新：支持 structured_4stage）

        Args:
            user_input: 用户输入

        Returns:
            推理模式：direct, single_think, structured_4stage, 或 multi_reflect
        """
        # 简单问题的关键词
        simple_keywords = ["你好", "hello", "hi", "谢谢", "thanks", "再见", "bye"]
        if any(kw in user_input.lower() for kw in simple_keywords):
            return "direct"

        # V2.5: 复杂问题指标 → structured_4stage
        complexity_indicators = [
            len(user_input) > 100,  # 问题描述较长
            user_input.count('，') > 3 or user_input.count(',') > 3,  # 多个子句
            any(kw in user_input.lower() for kw in ["首先", "然后", "接着", "最后", "first", "then", "next", "finally"]),  # 多步骤
            any(kw in user_input.lower() for kw in ["设计", "规划", "分析", "比较", "design", "plan", "analyze", "compare"])  # 复杂任务
        ]

        if sum(complexity_indicators) >= 2:
            return "structured_4stage"

        # 数学/逻辑问题 → single_think（更快）
        math_keywords = ["多少", "how many", "计算", "solve", "算", "几", "total"]
        if any(kw in user_input.lower() for kw in math_keywords):
            return "single_think"

        # 默认使用 single_think（最稳定）
        return "single_think"

    def run(self, user_input: str, mode: Optional[str] = None, verbose: bool = True) -> str:
        """
        执行推理

        Args:
            user_input: 用户输入
            mode: 强制指定推理模式（可选）
            verbose: 是否打印详细信息

        Returns:
            最终答案
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"输入: {user_input}")
            # 兼容 ChatOllama (使用 .model) 和 ChatOpenAI (使用 .model_name)
            model_name = getattr(self.llm, 'model', None) or getattr(self.llm, 'model_name', 'unknown')
            print(f"模型: {model_name}")

        # 1. 分类或使用指定模式
        if mode is None:
            mode = self.classify_complexity(user_input)

        if verbose:
            print(f"推理模式: {self.MODES.get(mode, mode)}")
            print(f"{'='*60}\n")

        # 2. 执行对应模式
        try:
            if mode == "direct":
                output = self._direct_answer(user_input)
            elif mode == "single_think":
                output = self._single_think(user_input, verbose=verbose)
            elif mode == "structured_4stage":
                # V2.5: 新增的结构化4阶段模式
                output = self._structured_4stage(user_input, verbose=verbose)
            elif mode == "multi_reflect":
                output = self._multi_reflect(user_input, verbose=verbose)
            else:
                # 默认使用 single_think
                output = self._single_think(user_input, verbose=verbose)

            # 3. 解析输出
            parsed = self.parser.parse(output)

            if verbose:
                print(f"\n{'='*60}")
                if parsed['think']:
                    print(f"思考过程:\n{parsed['think'][:500]}...")
                print(f"\n最终答案:\n{parsed['answer']}")
                print(f"{'='*60}\n")

            return parsed['answer']

        except Exception as e:
            error_msg = f"执行失败: {str(e)}"
            print(f"\n错误: {error_msg}")
            return error_msg

    def _direct_answer(self, user_input: str) -> str:
        """直接回答模式（简单问题）"""
        return self.direct_chain.invoke({"input": user_input})

    def _single_think(self, user_input: str, verbose: bool = True) -> str:
        """单次深度思考模式（推荐）"""
        # 先尝试纯 LLM
        output = self.single_think_chain.invoke({"input": user_input})

        # 检查循环
        if self.enable_loop_detection:
            is_loop, suggestion = self.loop_breaker.check_and_break(output)
            if is_loop:
                if verbose:
                    print(f"⚠️ {suggestion}")

                # 如果检测到循环且工具可用，使用工具重新尝试
                if self.tool_executor:
                    if verbose:
                        print("正在使用工具重新计算...")
                    try:
                        result = self.tool_executor.invoke({"input": user_input})
                        output = result.get("output", output)
                    except Exception as e:
                        if verbose:
                            print(f"工具执行失败: {e}")

        # 检查幻觉
        if self.enable_hallucination_detection:
            parsed = self.parser.parse(output)
            if parsed['think']:
                validation = self.hallucination_detector.validate(user_input, parsed['think'])
                if not validation['is_valid'] and verbose:
                    print(f"⚠️ 检测到可能的幻觉:")
                    for issue in validation['issues']:
                        print(f"  - {issue}")

        return output

    def _multi_reflect(self, user_input: str, verbose: bool = True) -> str:
        """多轮反思模式（复杂场景）"""
        # Think → Verify → Refine → Answer
        if verbose:
            print("第1步：初始思考...")
        think_output = self.single_think_chain.invoke({"input": user_input})

        # Verify step
        if verbose:
            print("第2步：验证推理...")
        verify_prompt = MULTI_REFLECT_PROMPTS["verify"]
        verification = (verify_prompt | self.llm | StrOutputParser()).invoke({
            "input": user_input,
            "previous_think": think_output
        })

        # Refine step
        if verbose:
            print("第3步：修正答案...")
        refine_prompt = MULTI_REFLECT_PROMPTS["refine"]
        final_output = (refine_prompt | self.llm | StrOutputParser()).invoke({
            "input": user_input,
            "verification": verification
        })

        return final_output

    # ========================================================================
    # V2.5: Structured 4-Stage Reasoning Mode (用户要求的结构化4阶段)
    # ========================================================================

    def _structured_4stage(self, user_input: str, verbose: bool = True) -> str:
        """
        结构化 4 阶段推理模式 (V2.5)

        核心改进：
        1. 使用 Memory 管理各阶段上下文
        2. Stage 3 改为"验证"而非"魔鬼代言人"
        3. 工具在 Stage 2 可用
        4. 检测器在 Stage 3 集成

        Args:
            user_input: 用户输入
            verbose: 是否打印详细信息

        Returns:
            最终答案字符串
        """
        if verbose:
            print("\n🔄 使用结构化 4 阶段推理模式 (V2.5)")
            print("="*60)

        # 初始化 memory
        self.memory.clear()
        self.memory.save_context(
            {"input": user_input},
            {"output": "开始 4 阶段推理"}
        )

        # Stage 1: Problem Definition
        if verbose:
            print("\n📋 阶段 1/4: 问题定义")
            print("-"*60)
        stage1_output = self._stage1_problem_definition(user_input, verbose=verbose)
        self.memory.save_context(
            {"input": f"[Stage 1: Problem Definition]\n{user_input}"},
            {"output": stage1_output}
        )

        # Stage 2: Bloom (Path Exploration with Tools)
        if verbose:
            print("\n🌸 阶段 2/4: 路径探索（带工具支持）")
            print("-"*60)
        stage2_output = self._stage2_bloom_with_tools(user_input, stage1_output, verbose=verbose)
        self.memory.save_context(
            {"input": "[Stage 2: Bloom - Path Exploration]"},
            {"output": stage2_output}
        )

        # Stage 3: Validation (NOT Devil's Advocate!)
        if verbose:
            print("\n✅ 阶段 3/4: 验证（非魔鬼代言人）")
            print("-"*60)
        stage3_output = self._stage3_validation_not_devil(user_input, stage1_output, stage2_output, verbose=verbose)
        self.memory.save_context(
            {"input": "[Stage 3: Validation]"},
            {"output": stage3_output}
        )

        # Stage 4: Final Decision
        if verbose:
            print("\n🎯 阶段 4/4: 最终决策")
            print("-"*60)
        final_output = self._stage4_final_decision(user_input, stage3_output, verbose=verbose)

        if verbose:
            print("\n" + "="*60)
            print("✨ 4 阶段推理完成")
            print("="*60)

        return final_output

    @with_timeout(seconds=120)  # Stage 1 最多 2 分钟
    def _stage1_problem_definition(self, user_input: str, verbose: bool = True) -> str:
        """
        阶段 1: 问题定义

        分析问题的关键信息，提取已知条件和目标
        """
        chain = STAGE1_PROBLEM_DEF_V2_5 | self.llm | StrOutputParser()
        output = chain.invoke({"input": user_input})

        if verbose:
            parsed = self.parser.parse(output)
            if parsed['think']:
                print(f"思考: {parsed['think'][:200]}...")
            print(f"定义: {parsed['answer'][:300]}...")

        return output

    @with_timeout(seconds=120)  # Stage 2 最多 2 分钟
    def _stage2_bloom_with_tools(self, user_input: str, stage1_output: str, verbose: bool = True) -> str:
        """
        阶段 2: 路径探索（带工具支持）

        基于问题定义，探索 2-3 种解决路径，可使用 calculator 工具
        """
        # 获取历史
        history = self.memory.load_memory_variables({})
        chat_history_str = self._format_chat_history(history.get("chat_history", []))

        # 构造提示词
        prompt_text = STAGE2_BLOOM_V2_5.format(
            original_question=user_input,
            stage1_output=stage1_output,
            chat_history=chat_history_str
        )

        # 如果工具可用，尝试使用工具
        if self.tool_executor:
            try:
                result = self.tool_executor.invoke({"input": prompt_text})
                output = result.get("output", "")
            except Exception as e:
                if verbose:
                    print(f"⚠️ 工具执行失败，使用纯 LLM: {e}")
                chain = STAGE2_BLOOM_V2_5 | self.llm | StrOutputParser()
                output = chain.invoke({
                    "original_question": user_input,
                    "stage1_output": stage1_output,
                    "chat_history": chat_history_str
                })
        else:
            chain = STAGE2_BLOOM_V2_5 | self.llm | StrOutputParser()
            output = chain.invoke({
                "original_question": user_input,
                "stage1_output": stage1_output,
                "chat_history": chat_history_str
            })

        if verbose:
            parsed = self.parser.parse(output)
            if parsed['think']:
                print(f"路径探索: {parsed['think'][:200]}...")
            print(f"推荐: {parsed['answer'][:300]}...")

        return output

    @with_timeout(seconds=120)  # Stage 3 最多 2 分钟（关键阶段，容易超时）
    def _stage3_validation_not_devil(self, user_input: str, stage1_output: str, stage2_output: str, verbose: bool = True) -> str:
        """
        阶段 3: 验证（非魔鬼代言人）

        验证推理准确性，使用循环和幻觉检测器
        关键：仅验证，不创造新假设
        """
        # 获取历史
        history = self.memory.load_memory_variables({})
        chat_history_str = self._format_chat_history(history.get("chat_history", []))

        chain = STAGE3_VALIDATION_V2_5 | self.llm | StrOutputParser()
        output = chain.invoke({
            "original_question": user_input,
            "stage1_output": stage1_output,
            "stage2_output": stage2_output,
            "chat_history": chat_history_str
        })

        # 循环检测
        if self.enable_loop_detection:
            is_loop, msg = self.loop_breaker.check_and_break(output)
            if is_loop and verbose:
                print(f"⚠️ {msg}")

        # 幻觉检测
        if self.enable_hallucination_detection:
            parsed = self.parser.parse(output)
            if parsed['think']:
                validation = self.hallucination_detector.validate(user_input, parsed['think'])
                if not validation['is_valid'] and verbose:
                    print(f"⚠️ 检测到可能的幻觉:")
                    for issue in validation['issues']:
                        print(f"  - {issue}")

        if verbose:
            parsed = self.parser.parse(output)
            if parsed['think']:
                print(f"验证: {parsed['think'][:200]}...")
            print(f"结果: {parsed['answer'][:300]}...")

        return output

    @with_timeout(seconds=120)  # Stage 4 最多 2 分钟
    def _stage4_final_decision(self, user_input: str, stage3_validation: str, verbose: bool = True) -> str:
        """
        阶段 4: 最终决策

        基于完整推理历史和验证结果，输出最终答案
        """
        # 获取历史
        history = self.memory.load_memory_variables({})
        chat_history_str = self._format_chat_history(history.get("chat_history", []))

        chain = STAGE4_FINAL_V2_5 | self.llm | StrOutputParser()
        output = chain.invoke({
            "original_question": user_input,
            "chat_history": chat_history_str,
            "stage3_validation": stage3_validation
        })

        if verbose:
            parsed = self.parser.parse(output)
            print(f"最终答案: {parsed.get('answer', output)[:500]}...")

        return output

    def _format_chat_history(self, messages) -> str:
        """
        格式化聊天历史为字符串

        Args:
            messages: 消息列表

        Returns:
            格式化的历史字符串
        """
        if not messages:
            return "无历史记录"

        formatted = []
        for i, msg in enumerate(messages):
            if hasattr(msg, 'content'):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get('content', msg.get('output', str(msg)))
            else:
                content = str(msg)

            formatted.append(f"[{i+1}] {content[:200]}...")

        return "\n".join(formatted)


# ============================================================================
# 便捷函数
# ============================================================================

def quick_run(question: str, model: str = "deepseek-r1:32b", verbose: bool = True) -> str:
    """
    快速运行接口

    Args:
        question: 问题
        model: 模型名称
        verbose: 是否打印详细信息

    Returns:
        答案
    """
    agent = DeepSeekR1AgentV2(
        model=model,
        enable_tools=True,
        enable_loop_detection=True,
        enable_hallucination_detection=False  # 默认关闭以提高速度
    )
    return agent.run(question, verbose=verbose)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DeepSeek-R1 Agent V2 - 测试运行")
    print("="*60)

    # 创建 Agent
    agent = DeepSeekR1AgentV2(
        model=os.getenv("OLLAMA_MODEL", "deepseek-r1:32b"),
        enable_tools=True,
        enable_loop_detection=True,
        enable_hallucination_detection=False  # 可以设置为 True 以启用幻觉检测
    )

    # 测试案例 1: 简单数学题
    print("\n测试案例 1: 简单数学")
    question1 = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    answer1 = agent.run(question1)
    print(f"\n最终答案: {answer1}")

    # 测试案例 2: 问候语（应该使用 direct 模式）
    print("\n\n测试案例 2: 问候")
    answer2 = agent.run("你好", mode="direct")
    print(f"\n最终答案: {answer2}")

    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
