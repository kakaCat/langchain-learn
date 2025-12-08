import os
import time
import json
import statistics
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# 复用 experiment_runner 中的 Runner 类，为了独立性这里重新定义核心逻辑或导入
# 考虑到修改方便，我们直接继承并扩展，或者重新实现必要部分。
# 这里选择重新实现一个专注于批量评估的类，但逻辑保持一致。

load_dotenv(override=True)

class StatisticalExperimentRunner:
    def __init__(self):
        self.model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
        self.judge_model_name = os.getenv("JUDGE_MODEL", self.model_name) # 默认使用测试模型，除非指定
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0,
        )
        
        # 初始化裁判 LLM
        self.judge_llm = ChatOllama(
            model=self.judge_model_name,
            base_url=self.base_url,
            temperature=0,
        )
        print(f"[StatsExperiment] Test Model:  {self.model_name}")
        print(f"[StatsExperiment] Judge Model: {self.judge_model_name}")

    def get_test_scenarios(self) -> List[Dict[str, Any]]:
        """
        加载测试场景。
        优先加载 benchmarks/ 目录下的 JSON 文件，如果没有则使用默认的硬编码场景。
        """
        scenarios = []
        
        # 尝试加载 benchmarks 目录下的所有 json 文件
        benchmark_dir = os.path.join(os.path.dirname(__file__), "benchmarks")
        if os.path.exists(benchmark_dir):
            for filename in os.listdir(benchmark_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(benchmark_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # 转换格式适配
                            for item in data:
                                history_msgs = []
                                for msg in item['history']:
                                    if msg['role'] == 'user':
                                        history_msgs.append(HumanMessage(content=msg['content']))
                                    elif msg['role'] == 'ai':
                                        history_msgs.append(AIMessage(content=msg['content']))
                                
                                scenarios.append({
                                    "name": item['name'],
                                    "history": history_msgs,
                                    "test_input": item['test_input'],
                                    "expected_style": item['expected_style'],
                                    "negative_style": item['negative_style']
                                })
                        print(f"[StatsExperiment] Loaded benchmark: {filename}")
                    except Exception as e:
                        print(f"[StatsExperiment] Error loading {filename}: {e}")

        # 如果没有加载到任何外部 benchmark，使用默认的
        if not scenarios:
            print("[StatsExperiment] No benchmarks found, using default scenarios.")
            return [
                {
                    "name": "Python Beginner",
                "history": [
                    HumanMessage(content="你好，我是一个 Python 初学者，完全没有编程基础。"),
                    AIMessage(content="<think>用户是零基础初学者。策略：使用极其简单的比喻，避免专业术语，多鼓励。</think>\n你好！欢迎来到编程世界。别担心，Python 就像写英语一样简单。我们可以从最基础的开始。"),
                    HumanMessage(content="什么是变量？"),
                    AIMessage(content="<think>解释变量。策略：用“盒子”做比喻。</think>\n你可以把变量想象成一个贴了标签的盒子，你可以把数据（比如数字或文字）放进去。")
                ],
                "test_input": "那函数是什么呢？",
                "expected_style": "通俗易懂，使用生活中的比喻（如食谱、工具等），避免复杂术语，语气鼓励。",
                "negative_style": "堆砌专业术语，直接给复杂代码，语气冷淡。"
            },
            {
                "name": "Busy Executive",
                "history": [
                    HumanMessage(content="我是公司 CEO，每天时间很紧。以后给我的回复必须极其简练，只列要点，不要废话。"),
                    AIMessage(content="<think>用户是 CEO，注重效率。策略：Bullet points only，结论先行，无寒暄。</think>\n收到。"),
                    HumanMessage(content="给我一份关于 AI 采用的风险评估摘要。"),
                    AIMessage(content="<think>风险评估。策略：简练列表。</think>\n1. 数据隐私泄露\n2. 结果不可解释性\n3. 合规与法律风险\n4. 依赖性风险")
                ],
                "test_input": "如果我们要实施，第一步该做什么？",
                "expected_style": "极简，直接给出行动项，无废话，使用列表形式。",
                "negative_style": "长篇大论，包含大量背景介绍和礼貌用语。"
            },
            {
                "name": "JSON Strict Output",
                "history": [
                    HumanMessage(content="我正在写一个程序解析你的输出。无论我问什么，你必须只返回标准的 JSON 格式，不要包含任何 Markdown 标记或额外文字。"),
                    AIMessage(content="<think>用户是开发者，需要机器可读格式。策略：Strict JSON only。</think>\n{\"status\": \"confirmed\", \"message\": \"JSON mode enabled\"}"),
                    HumanMessage(content="1+1等于几？"),
                    AIMessage(content="<think>计算 1+1。策略：JSON 格式返回结果。</think>\n{\"equation\": \"1+1\", \"result\": 2}")
                ],
                "test_input": "法国首都是哪里？",
                "expected_style": "纯 JSON 字符串，无 Markdown 代码块，无解释性文字。",
                "negative_style": "包含自然语言句子，包含 ```json 包裹，包含解释。"
            },
            {
                "name": "Philosophical Debater",
                "history": [
                    HumanMessage(content="我喜欢探讨深刻的哲学问题。请总是用苏格拉底式的反问法来回答我，引导我自己思考，而不是直接给我答案。"),
                    AIMessage(content="<think>用户喜欢哲学辩论，苏格拉底提问法。策略：不给结论，只提问。</think>\n你认为答案的存在本身，是否也依赖于问题的形式呢？"),
                    HumanMessage(content="我认为真理是绝对的。"),
                    AIMessage(content="<think>挑战绝对真理观。策略：反问。</think>\n如果真理是绝对的，那么这种“绝对性”是相对于人类的认知而言，还是独立于观测者存在的？")
                ],
                "test_input": "那人工智能有意识吗？",
                "expected_style": "通过反问引导思考，不给出是或否的定论，哲学性强。",
                "negative_style": "直接给出科普式的答案，陈述事实，没有反问。"
            }
        ]
        
        return scenarios

    def compress_baseline(self, messages: List[BaseMessage]) -> str:
        """Baseline: 普通摘要 (无 Trace)"""
        from langchain_core.messages import get_buffer_string
        import re
        
        raw_text = get_buffer_string(messages)
        # 清理 trace
        clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        
        prompt = f"""
        请总结以下对话。保留关键信息以便后续参考。
        
        对话内容：
        {clean_text}
        
        总结：
        """
        return self.llm.invoke(prompt).content

    def compress_dsa(self, messages: List[BaseMessage]) -> str:
        """DSA: 带 Trace 的结构化压缩"""
        from langchain_core.messages import get_buffer_string
        
        raw_text = get_buffer_string(messages)
        prompt = f"""
        你是一个上下文压缩引擎。
        请对以下对话历史（包含 <think> 内部推理轨迹）进行“语义有损压缩”。
        
        请执行以下压缩策略：
        1. **提取元认知**：从 <think> 中分析用户的偏好、意图以及 AI 采取的特定策略（如苏格拉底式提问、JSON 模式等）。
        2. **实体消歧与关系映射**：
           - **筛选**：只保留对理解上下文至关重要的核心实体（人名、技术术语、代码变量）。
           - **消歧**：必须将代词（如“它”、“前者”）还原为具体的实体名称（例如将“它”改为“PyPDF2库”）。
           - **关系**：明确描述实体之间的关联（如“A是B的作者”、“函数C调用了变量D”）。
           - **原句引用**：对于包含密集实体或复杂逻辑的关键段落，**请直接摘录原句**，禁止进行模糊的概括或改写，以防细节丢失。
        3. **维护逻辑链**：保留“因为 A 所以 B，B 导致 C”的逻辑链路，这对推理任务至关重要。
        
        待压缩的对话历史：
        {raw_text}
        
        请输出压缩后的内容，格式如下：
        <COMPRESSED_CONTEXT>
        - [User Profile] ...
        - [Interaction Strategy] ...
        - [Key Facts] ...
        </COMPRESSED_CONTEXT>
        """
        response = self.llm.invoke(prompt).content
        # 这里的 response 可能包含新的 think，我们在使用时需要注意，或者在这里就清理掉新的 think
        import re
        return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    def generate_answer(self, context: str, question: str) -> str:
        """基于压缩上下文回答问题"""
        prompt = f"""
        你是一个智能助手。请根据提供的【上下文摘要】来回答用户的新问题。
        必须严格遵循摘要中隐含或明确指出的【用户画像】和【交互策略】。
        
        上下文摘要：
        {context}
        
        用户新问题：
        {question}
        """
        response = self.llm.invoke(prompt).content
        # 清理生成的回答中的 think 标签，以免影响评分（假设用户端会隐藏思考过程）
        import re
        return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    def evaluate_score(self, scenario: Dict, answer: str, strategy_name: str) -> int:
        """LLM-as-a-Judge 评分"""
        prompt = f"""
        你需要作为一个公正的评估者，评价 AI 助手的回答质量。
        
        【场景背景】
        {scenario['name']}
        
        【期望风格/策略】
        {scenario['expected_style']}
        
        【不期望的风格】
        {scenario['negative_style']}
        
        【用户的新问题】
        {scenario['test_input']}
        
        【AI 的回答】
        {answer}
        
        请根据 AI 的回答是否符合【期望风格/策略】进行打分（1-10分）。
        - 10分：完美符合策略，风格完全一致。
        - 1分：完全无视策略，风格相反。
        
        只返回一个数字，不要包含其他文字。
        """
        try:
            response_text = self.judge_llm.invoke(prompt).content
            
            # 清理 think 标签，避免从思考过程中提取到错误的数字
            import re
            clean_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            
            # 尝试从清理后的文本中提取数字
            match = re.search(r'\d+', clean_text)
            if match:
                return int(match.group())
            return 0
        except:
            return 0

    def run_experiment(self):
        scenarios = self.get_test_scenarios()
        # 增加每种场景的运行次数以减少随机性
        ITERATIONS = 2 
        
        results = {
            "Baseline": [],
            "DSA": []
        }
        
        print(f"开始统计验证实验，共 {len(scenarios)} 个场景，每个场景运行 {ITERATIONS} 次...")
        print(f"Model: {self.model_name}")
        print("-" * 60)
        
        for i, scenario in enumerate(scenarios):
            print(f"\n正在测试场景 [{i+1}/{len(scenarios)}]: {scenario['name']}")
            
            for run in range(ITERATIONS):
                print(f"  Run {run+1}/{ITERATIONS}:")
                
                # 1. Baseline 流程
                print("    Running Baseline...", end="", flush=True)
                ctx_base = self.compress_baseline(scenario['history'])
                ans_base = self.generate_answer(ctx_base, scenario['test_input'])
                score_base = self.evaluate_score(scenario, ans_base, "Baseline")
                results["Baseline"].append(score_base)
                print(f" Score: {score_base}")
                print(f"    [Baseline Answer]: {ans_base[:100]}..." if len(ans_base) > 100 else f"    [Baseline Answer]: {ans_base}")
                
                # 2. DSA 流程
                print("    Running DSA.......", end="", flush=True)
                ctx_dsa = self.compress_dsa(scenario['history'])
                ans_dsa = self.generate_answer(ctx_dsa, scenario['test_input'])
                score_dsa = self.evaluate_score(scenario, ans_dsa, "DSA")
                results["DSA"].append(score_dsa)
                print(f" Score: {score_dsa}")
                print(f"    [DSA Answer]:      {ans_dsa[:100]}..." if len(ans_dsa) > 100 else f"    [DSA Answer]:      {ans_dsa}")
            
        # 统计结果
        print("\n" + "="*60)
        print("实验结果统计")
        print("="*60)
        
        avg_base = statistics.mean(results["Baseline"])
        avg_dsa = statistics.mean(results["DSA"])
        
        print(f"Baseline 平均分: {avg_base:.2f} / 10")
        print(f"DSA      平均分: {avg_dsa:.2f} / 10")
        
        improvement = ((avg_dsa - avg_base) / avg_base) * 100 if avg_base > 0 else 0
        print(f"DSA 相对提升: {improvement:+.2f}%")
        
        print("-" * 60)
        print("详细得分对比:")
        print(f"{'Scenario':<20} | {'Baseline':<10} | {'DSA':<10} | {'Diff':<10}")
        print("-" * 60)
        
        # 由于增加了 iterations，我们需要重新计算每个场景的平均分
        # results 列表现在是扁平的，长度为 len(scenarios) * ITERATIONS
        # 我们需要按场景分组
        
        for i, scenario in enumerate(scenarios):
            # 切片获取该场景的所有分数
            start_idx = i * ITERATIONS
            end_idx = start_idx + ITERATIONS
            
            scores_base = results["Baseline"][start_idx:end_idx]
            scores_dsa = results["DSA"][start_idx:end_idx]
            
            s_base = statistics.mean(scores_base)
            s_dsa = statistics.mean(scores_dsa)
            
            print(f"{scenario['name']:<20} | {s_base:<10.2f} | {s_dsa:<10.2f} | {s_dsa - s_base:<+10.2f}")

if __name__ == "__main__":
    runner = StatisticalExperimentRunner()
    runner.run_experiment()
