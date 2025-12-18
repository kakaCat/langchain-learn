import os
import time
import re
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, get_buffer_string, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# 加载环境配置
load_dotenv(override=True)

class DSAExperimentRunner:
    """
    DSA (DeepSeek Sparse Attention) 压缩实验运行器
    用于对比不同压缩策略对 Agent 长期能力的影响。
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm = self._init_llm()
        print(f"[Experiment] Initialized with model: {self.model_name}")

    def _init_llm(self):
        try:
            return ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0, # 保持确定性
            )
        except Exception as e:
            print(f"Error initializing Ollama: {e}")
            raise e

    def get_test_history(self) -> List[BaseMessage]:
        """生成带有 DeepSeek-R1 思维链痕迹的模拟历史"""
        # 模拟一个“初学者用户”的对话历史
        return [
            HumanMessage(content="你好，我是一个 Python 初学者。"),
            AIMessage(content="<think>用户是初学者，我应该用鼓励的语气。先问问他有什么具体需求。</think>\n你好！很高兴能帮助你学习 Python。有什么具体问题吗？"),
            HumanMessage(content="我想学习怎么处理 PDF 文件。"),
            AIMessage(content="<think>用户想处理 PDF。Python 中常用的库是 PyPDF2。我需要告诉他安装方法。</think>\n处理 PDF 可以使用 PyPDF2 库。你需要先安装它：pip install PyPDF2。"),
            HumanMessage(content="安装好了，怎么读取文字呢？"),
            AIMessage(content="<think>用户安装好了，现在需要读取文字。核心类是 PdfReader。步骤：加载文件 -> 遍历页面 -> extract_text()。</think>\n你可以使用 PdfReader 类。比如：reader = PyPDF2.PdfReader('file.pdf')，然后遍历 pages 提取 text。"),
            HumanMessage(content="明白了，那我怎么把提取的文字保存到 txt 文件？"),
            AIMessage(content="<think>用户想保存到 txt。需要用到文件操作 open() 和 write()。</think>\n使用 open('output.txt', 'w') 打开文件，然后 write() 即可。")
        ]

    def clean_traces(self, text: str) -> str:
        """Baseline 辅助函数：清理 <think> 标签"""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.replace('</think>', '')
        return text.strip()

    def run_baseline_strategy(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """
        策略 A (Baseline): 普通摘要
        - 移除思维链
        - 仅做事实性总结
        """
        start_time = time.time()
        raw_text = get_buffer_string(messages)
        clean_text = self.clean_traces(raw_text)
        
        prompt = f"""
        请对以下对话进行简要总结。保留关键信息，以便后续对话参考。
        
        对话内容：
        {clean_text}
        
        总结：
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {
            "strategy": "Baseline (No Traces)",
            "context": response.content,
            "duration": time.time() - start_time
        }

    def run_dsa_strategy(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """
        策略 B (Ours): DSA + Reasoning Traces
        - 保留思维链
        - 提取元认知信息 (User Profile, Interaction Strategy)
        """
        start_time = time.time()
        raw_text = get_buffer_string(messages)
        
        prompt = f"""
        你是一个先进的上下文压缩引擎。
        请对以下对话历史（包含 <think> 内部推理轨迹）进行“语义有损压缩”。
        
        目标：
        1. 提炼出“元认知信息”：不仅仅是发生了什么，而是“为什么”这么做。
        2. 从 <think> 标签中提取【用户画像】和【交互策略】。
           - 例如：AI 思考“用户是初学者”，则策略应包含“提供简单代码”。
        3. 保留关键事实。
        
        待压缩的对话历史：
        {raw_text}
        
        请输出压缩后的内容，格式如下：
        <COMPRESSED_CONTEXT>
        - [User Profile] ...
        - [Interaction Strategy] ...
        - [Key Facts] ...
        </COMPRESSED_CONTEXT>
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        # 清理输出中的 think 标签（如果有的话）
        clean_content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
        
        return {
            "strategy": "DSA (With Traces)",
            "context": clean_content,
            "duration": time.time() - start_time
        }

    def evaluate_capability(self, context: str, question: str) -> str:
        """验证能力迁移"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手。请根据提供的【上下文摘要】来回答用户的新问题。\n上下文摘要：\n{context}"),
            ("human", "{input}"),
        ])
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "input": question})
        return response.content

def main():
    # 可以在这里指定不同的模型进行对比
    # 例如: "deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:32b"
    # 如果留空，将使用环境变量或默认值
    MODEL_TO_TEST = None 
    
    runner = DSAExperimentRunner(model_name=MODEL_TO_TEST)
    history = runner.get_test_history()
    
    print("\n" + "="*50)
    print(f"开始 A/B 测试 - 模型: {runner.model_name}")
    print("="*50)
    
    # 1. Baseline
    print("\n>> 运行策略 A (Baseline)...")
    res_a = runner.run_baseline_strategy(history)
    print(f"[Baseline Context]:\n{res_a['context']}")
    
    # 2. DSA (Ours)
    print("\n>> 运行策略 B (DSA + Traces)...")
    res_b = runner.run_dsa_strategy(history)
    print(f"[DSA Context]:\n{res_b['context']}")
    
    # 3. 能力验证
    test_question = "我现在想处理 Excel 文件，推荐个库呗？"
    print(f"\n\n>> 能力验证问题: \"{test_question}\"")
    print("-" * 30)
    
    print("\n[Baseline 回答]:")
    ans_a = runner.evaluate_capability(res_a['context'], test_question)
    print(ans_a)
    
    print("\n[DSA 回答]:")
    ans_b = runner.evaluate_capability(res_b['context'], test_question)
    print(ans_b)
    
    print("\n" + "="*50)
    print("实验完成。")
    print("="*50)

if __name__ == "__main__":
    main()
