import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()

def get_llm() -> ChatOllama:
    """创建并配置本地 Ollama LLM 实例"""
    # 默认使用 deepseek-r1 或 llama3，用户可以通过环境变量覆盖
    model = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
    
    print(f"正在使用本地模型: {model} (URL: {base_url})")
    
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature
    )

class DeepSeekR1Traces:
    """
    结构化思维链 (DeepSeek-R1 Traces) 实现
    
    此类实现了一个针对高复杂度任务的 4 阶段推理框架：
    1. 问题定义阶段 (Problem Definition)：重新表述并明确核心任务。
    2. 绽放周期 (Bloom Cycle)：问题分解与假设生成。
    3. 重构周期 (Rumination Cycle)：自我验证与修正。
    4. 最终决策阶段 (Final Decision)：输出结论与置信度。
    """
    
    def __init__(self):
        self.llm = get_llm()
        self._build_chains()
        
    def _build_chains(self):
        # 阶段 0：任务复杂度门控 (Gate)
        self.gate_prompt = ChatPromptTemplate.from_template(
            """
            你是一个任务复杂度分类器。
            请判断用户输入的任务是 "SIMPLE" (简单) 还是 "COMPLEX" (复杂)。
            
            示例：
            输入: "你好" -> SIMPLE
            输入: "2+2等于几？" -> SIMPLE
            输入: "What is 2+2?" -> SIMPLE
            输入: "法国的首都是哪里？" -> SIMPLE
            输入: "如果我有3个苹果，吃了一个，还剩几个？" -> SIMPLE
            输入: "Which is heavier: a pound of bricks or a pound of feathers?" -> SIMPLE
            
            输入: "解决鸡兔同笼问题：头20，脚50" -> COMPLEX
            输入: "Solve the chicken and rabbit problem: 20 heads, 50 legs" -> COMPLEX
            输入: "Three people (A, B, C) are either Knights or Knaves..." -> COMPLEX
            输入: "Design a rate limiter algorithm..." -> COMPLEX
            输入: "写一个Python脚本来抓取网页" -> COMPLEX
            输入: "解释量子纠缠" -> COMPLEX
            输入: "How many letters 'r' are in the word 'Strawberry'?" -> COMPLEX
            输入: "When I was 6, my sister was half my age. Now I am 70. How old is my sister?" -> COMPLEX
            输入: "There are 3 killers in a room..." -> COMPLEX
            
            用户输入：{input}
            
            请仅输出 "SIMPLE" 或 "COMPLEX"，不要包含其他内容。不要输出任何解释。
            """
        )
        self.gate_chain = self.gate_prompt | self.llm | StrOutputParser()
        
        # 简单模式链 (Baseline)
        self.baseline_prompt = ChatPromptTemplate.from_template(
            """
            你是一个乐于助人的助手。请直接回答用户的问题。
            
            问题：{input}
            """
        )
        self.baseline_chain = self.baseline_prompt | self.llm | StrOutputParser()

        # 阶段 1：问题定义
        # 目标：重述核心任务并校准注意力。
        self.problem_definition_prompt = ChatPromptTemplate.from_template(
            """
            你正处于结构化推理过程的 **问题定义阶段 (Problem Definition Phase)**。
            
            你的目标是在尝试解决问题之前，澄清并重新定义用户的请求，以确保完全理解。
            
            用户请求：{input}
            
            指令：
            1. 识别任务的核心目标。
            2. 列出任何隐含或明确的约束条件。
            3. 用精确、技术性的语言重新表述问题。
            4. 突出显示需要关注的关键实体或概念。
            
            输出格式：
            - 核心目标 (Core Objective)：...
            - 约束条件 (Constraints)：...
            - 问题重述 (Rephrased Problem)：...
            - 关键概念 (Key Concepts)：...
            """
        )
        self.problem_definition_chain = self.problem_definition_prompt | self.llm | StrOutputParser()

        # 阶段 2：绽放周期
        # 目标：初步分解与假设生成。
        self.bloom_cycle_prompt = ChatPromptTemplate.from_template(
            """
            你正处于结构化推理过程的 **绽放周期阶段 (Bloom Cycle Phase)**。
            
            基于已定义的问题，生成初步假设并将问题分解。
            
            问题定义：
            {problem_definition}
            
            指令：
            1. 将问题分解为更小、可管理的子问题。
            2. 构思至少 3 种不同的解决途径或假设。
            3. 对于每种途径，简要列出其优缺点。
            4. 选择最有希望的途径作为初步方案。
            
            输出格式：
            - 子问题 (Sub-problems)：...
            - 假设/途径 (Hypotheses/Approaches)：
              1. ... (优/缺)
              2. ... (优/缺)
              3. ... (优/缺)
            - 选定途径 (Selected Approach)：...
            """
        )
        self.bloom_cycle_chain = self.bloom_cycle_prompt | self.llm | StrOutputParser()

        # 阶段 3：重构周期
        # 目标：自我验证与修正。
        self.rumination_cycle_prompt = ChatPromptTemplate.from_template(
            """
            你正处于结构化推理过程的 **重构周期阶段 (Rumination Cycle Phase)**。
            
            你的目标是批判性地评估绽放周期中选定的途径，识别潜在缺陷，并完善计划。
            把这看作是一个“魔鬼代言人” (Devil's Advocate) 阶段。
            
            问题定义：
            {problem_definition}
            
            绽放周期输出 (初步计划)：
            {bloom_output}
            
            指令：
            1. 识别选定途径中的潜在逻辑漏洞、边缘情况或错误。
            2. 验证所做的任何假设。
            3. 如果发现错误，提出修正方案或替代路径。
            4. 基于此分析完善计划。
            
            输出格式：
            - 批判/验证 (Critique/Verification)：...
            - 识别出的缺陷/风险 (Identified Flaws/Risks)：...
            - 修正/完善 (Corrections/Refinements)：...
            - 最终计划 (Finalized Plan)：...
            """
        )
        self.rumination_cycle_chain = self.rumination_cycle_prompt | self.llm | StrOutputParser()

        # 阶段 4：最终决策
        # 目标：输出结论与置信度。
        self.final_decision_prompt = ChatPromptTemplate.from_template(
            """
            你正处于结构化推理过程的 **最终决策阶段 (Final Decision Phase)**。
            
            执行最终计划，并针对用户的原始请求提供最终答案。
            
            原始请求：{input}
            完善后的计划 (来自重构周期)：
            {rumination_output}
            
            指令：
            1. 基于完善后的计划综合得出最终答案。
            2. 确保答案直接回应了原始请求。
            3. 提供一个置信度评分 (0-10) 并简要说明理由。
            
            输出格式：
            - 最终答案 (Final Answer)：...
            - 置信度评分 (Confidence Score)：.../10
            - 理由 (Justification)：...
            """
        )
        self.final_decision_chain = self.final_decision_prompt | self.llm | StrOutputParser()

    def run(self, user_input: str):
        print(f"\n{'='*20} 开始 DeepSeek-R1 Traces 推理 {'='*20}")
        print(f"原始输入：{user_input}\n")
        print(f"使用模型：{self.llm.model}")

        try:
            # 步骤 0：门控判断
            print(f"{'-'*10} 阶段 0：门控 (Gate) {'-'*10}")
            # gate_decision = self.gate_chain.invoke({"input": user_input}).strip().upper()
            gate_decision = "COMPLEX" # 强制使用复杂模式以测试思维链效果
            print(f"任务复杂度判定: {gate_decision} (Forced for testing)\n")
            
            # if "SIMPLE" in gate_decision:
            if False: # 强制禁用简单模式
                 print(f"{'-'*10} 执行简单模式 (Baseline) {'-'*10}")
                 final_out = self.baseline_chain.invoke({"input": user_input})
                 print(final_out)
                 print(f"\n{'='*20} 推理完成 {'='*20}\n")
                 return final_out

            # 步骤 1：问题定义
            print(f"{'-'*10} 阶段 1：问题定义 (Problem Definition) {'-'*10}")
            problem_def = self.problem_definition_chain.invoke({"input": user_input})
            print(problem_def)
            print("\n")

            # 步骤 2：绽放周期
            print(f"{'-'*10} 阶段 2：绽放周期 (Bloom Cycle) {'-'*10}")
            bloom_out = self.bloom_cycle_chain.invoke({"problem_definition": problem_def})
            print(bloom_out)
            print("\n")

            # 步骤 3：重构周期
            print(f"{'-'*10} 阶段 3：重构周期 (Rumination Cycle) {'-'*10}")
            rumination_out = self.rumination_cycle_chain.invoke({
                "problem_definition": problem_def,
                "bloom_output": bloom_out
            })
            print(rumination_out)
            print("\n")

            # 步骤 4：最终决策
            print(f"{'-'*10} 阶段 4：最终决策 (Final Decision) {'-'*10}")
            final_out = self.final_decision_chain.invoke({
                "input": user_input,
                "rumination_output": rumination_out
            })
            print(final_out)
            print(f"\n{'='*20} 推理完成 {'='*20}\n")
            
            return final_out
            
        except Exception as e:
            print(f"\n执行过程中出错：{e}")
            print("请检查您的 Ollama 服务是否已启动，以及 OLLAMA_MODEL 是否正确。")
            return str(e)

if __name__ == "__main__":
    # 示例用法
    try:
        tracer = DeepSeekR1Traces()
        
        # 示例任务：一个中等复杂度的逻辑/数学问题或代码设计任务
        # complex_task = "芝加哥有多少位钢琴调音师？请使用费米估算并逐步推导。"
        complex_task = "设计一个在高并发分布式环境下的 API 限流系统。"
        
        tracer.run(complex_task)
    except Exception as e:
        print(f"初始化错误：{e}")
