import os
import re
import tiktoken
from typing import Dict, List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, get_buffer_string, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug

# 1. 加载环境配置
load_dotenv(override=True)

def get_llm():
    """创建并配置语言模型实例"""
    try:
        ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
        # print(f"Using Local Model: {ollama_model}")
        return ChatOllama(
            model=ollama_model,
            base_url="http://localhost:11434",
            temperature=0,
        )
    except Exception as e:
        print(f"Ollama Error: {e}")
        return ChatOpenAI()

def get_raw_history_with_traces() -> List[BaseMessage]:
    """生成带有 DeepSeek-R1 思维链痕迹的模拟历史"""
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

def clean_traces(text: str) -> str:
    """清理 <think> 标签，模拟普通模型的视角"""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('</think>', '')
    return text.strip()

# --- 策略 A：普通摘要压缩 (Baseline) ---
def compression_strategy_baseline(llm, messages: List[BaseMessage]) -> str:
    """
    传统压缩：移除思维链，仅对对话内容进行摘要。
    这是大多数 Memory 系统的默认做法。
    """
    raw_text = get_buffer_string(messages)
    clean_text = clean_traces(raw_text) # 移除 traces，模拟普通摘要看不到思考
    
    prompt = f"""
    请对以下对话进行简要总结。保留关键信息，以便后续对话参考。
    
    对话内容：
    {clean_text}
    
    总结：
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# --- 策略 B：DSA + 思维链提取 (Ours - Hybrid with Anchors) ---
def compression_strategy_dsa_hybrid(llm, messages: List[BaseMessage]) -> str:
    """
    我们的方法（混合策略）：
    1. 提取元认知 (User Profile, Strategy)
    2. 实体消歧与关系映射
    3. 知识锚点 (Knowledge Anchors) - 显式提取索引键
    """
    raw_text = get_buffer_string(messages) # 保留 traces
    
    prompt = f"""
    你是一个高级上下文压缩专家。
    你的任务是对对话历史进行高保真压缩，特别要保留**推理逻辑**和**关键细节**。
    
    输入包含用户的对话和 AI 的思维链（<think>标签）。
    
    请执行以下压缩策略：
    1. **提取元认知**：从 <think> 中分析用户的偏好、意图以及 AI 采取的特定策略。
    2. **实体消歧与关系映射**：
       - **筛选**：只保留对理解上下文至关重要的核心实体（人名、技术术语、代码变量）。
       - **消歧**：必须将代词（如“它”、“前者”）还原为具体的实体名称（例如将“它”改为“PyPDF2库”）。
       - **关系**：明确描述实体之间的关联（如“A是B的作者”、“函数C调用了变量D”）。
    3. **知识锚点 (Knowledge Anchors)**：
       - 将所有可能作为未来检索键的关键词（实体名、专有名词、数字常量）提取到单独的列表中。
    
    待压缩的对话历史：
    {raw_text}
    
    请严格按照以下 XML 格式输出压缩结果：
    <COMPRESSED_CONTEXT>
    [User Profile]
    (用户是谁？有什么偏好？)
    
    [Interaction Strategy]
    (AI 应该保持什么样的语气、格式或策略？)
    
    [Critical Facts & Logic]
    - (关键实体与事实 1)
    - (关键实体与事实 2)
    - (逻辑推导链)
    
    [Knowledge Anchors]
    (关键词1, 关键词2, 关键词3...)
    </COMPRESSED_CONTEXT>
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    # 提取 <think> 以外的内容（如果压缩模型本身也输出了 think）
    content = response.content
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return content

# --- 评估函数 ---
def evaluate_performance(llm, compressed_context: str, strategy_name: str):
    print(f"\n{'='*20} 策略测试: {strategy_name} {'='*20}")
    print(f"压缩后上下文:\n{compressed_context}\n")
    
    # 新问题：测试 Agent 是否能迁移“初学者”策略
    new_question = "我现在想处理 Excel 文件，推荐个库呗？"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能助手。请根据提供的【上下文摘要】来回答用户的新问题。\n上下文摘要：\n{context}"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm
    
    print(f"User: {new_question}")
    print("AI Response:")
    response = chain.invoke({"context": compressed_context, "input": new_question})
    print(response.content)
    print("-" * 60)

def main():
    # set_debug(True)
    llm = get_llm()
    history = get_raw_history_with_traces()
    
    print(f"测试模型: {os.getenv('OLLAMA_MODEL', 'deepseek-r1:1.5b')}")
    print("正在运行 A/B 测试对比...\n")
    
    # 1. 运行基线策略
    print(">> 运行策略 A: 普通摘要 (Baseline)...")
    baseline_context = compression_strategy_baseline(llm, history)
    
    # 2. 运行我们的策略
    print(">> 运行策略 B: DSA + Hybrid Anchors (Ours)...")
    dsa_context = compression_strategy_dsa_hybrid(llm, history)
    
    # 3. 评估对比
    evaluate_performance(llm, baseline_context, "普通摘要 (Baseline)")
    evaluate_performance(llm, dsa_context, "DSA + Hybrid Anchors (Ours)")

if __name__ == "__main__":
    main()
