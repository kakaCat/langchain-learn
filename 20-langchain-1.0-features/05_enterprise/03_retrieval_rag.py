"""
LangChain 1.0 - 检索增强生成 (RAG) - 官方文档完整实现

严格按照 https://docs.langchain.com/oss/python/langchain/rag 实现

核心功能:
1. 基础 RAG 流程
2. 文档加载和分割
3. 向量存储和检索
4. 检索链 (Retrieval Chain)
5. 多文档检索
6. 高级检索策略
7. RAG 评估和优化

参考文档:
- https://python.langchain.com/docs/tutorials/rag/
- https://python.langchain.com/docs/how_to/#qa-with-rag
"""

import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()


# ========== 1. 基础 RAG 示例 ==========
def demo_basic_rag():
    """最简单的 RAG 示例"""
    print("=" * 70)
    print("1. 基础 RAG 示例")
    print("=" * 70)

    # 1. 准备文档
    documents = [
        Document(
            page_content="LangChain 是一个用于开发由语言模型驱动的应用程序的框架。",
            metadata={"source": "docs", "page": 1}
        ),
        Document(
            page_content="LangChain 1.0 引入了 create_agent API，替代了旧的 initialize_agent。",
            metadata={"source": "docs", "page": 2}
        ),
        Document(
            page_content="Middleware 系统是 LangChain 1.0 的核心特性，包括 PII 保护和自动摘要。",
            metadata={"source": "docs", "page": 3}
        ),
    ]

    # 2. 创建向量存储
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 3. 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 4. 创建 LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    # 5. 创建提示模板
    prompt = ChatPromptTemplate.from_template("""
    根据以下上下文回答问题:

    上下文: {context}

    问题: {question}

    回答:
    """)

    # 6. 创建 RAG 链
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. 查询
    question = "LangChain 1.0 有什么新特性?"
    print(f"\n问题: {question}")

    answer = rag_chain.invoke(question)
    print(f"回答: {answer}")


# ========== 2. 文档加载和分割 ==========
def demo_document_loading():
    """文档加载和分割"""
    print("\n\n" + "=" * 70)
    print("2. 文档加载和分割")
    print("=" * 70)

    # 创建长文档
    long_text = """
    LangChain 是一个强大的框架。它支持多种语言模型。

    主要特性包括:
    1. Agents - 智能代理系统
    2. Chains - 组件链接
    3. Memory - 对话记忆
    4. Tools - 工具集成
    5. Callbacks - 回调系统

    LangChain 1.0 的新特性:
    - create_agent API
    - Middleware 系统
    - Checkpointer 持久化
    - 流式输出增强

    使用场景:
    - 聊天机器人
    - 问答系统
    - 文档分析
    - 代码助手
    """

    # 创建文档
    doc = Document(page_content=long_text, metadata={"source": "guide"})

    # 文档分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  # 每块字符数
        chunk_overlap=20,  # 重叠字符数
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents([doc])

    print(f"\n原始文档长度: {len(long_text)} 字符")
    print(f"分割后块数: {len(chunks)}")
    print("\n前3个块:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n块 {i}:")
        print(f"  内容: {chunk.page_content[:50]}...")
        print(f"  长度: {len(chunk.page_content)}")
        print(f"  元数据: {chunk.metadata}")


# ========== 3. 向量存储和检索 ==========
def demo_vector_store():
    """向量存储和检索"""
    print("\n\n" + "=" * 70)
    print("3. 向量存储和检索")
    print("=" * 70)

    # 准备文档
    documents = [
        Document(page_content="Python 是一种高级编程语言"),
        Document(page_content="JavaScript 用于Web开发"),
        Document(page_content="Java 是面向对象的语言"),
        Document(page_content="Go 语言适合并发编程"),
        Document(page_content="Rust 注重内存安全"),
    ]

    # 创建向量存储
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 1. 相似度搜索
    print("\n1️⃣ 相似度搜索:")
    query = "哪种语言适合网页开发?"
    similar_docs = vectorstore.similarity_search(query, k=2)

    for i, doc in enumerate(similar_docs, 1):
        print(f"  {i}. {doc.page_content}")

    # 2. 带分数的搜索
    print("\n2️⃣ 带分数的搜索:")
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)

    for doc, score in docs_with_scores:
        print(f"  {doc.page_content} (分数: {score:.3f})")

    # 3. MMR 搜索 (最大边际相关性)
    print("\n3️⃣ MMR 搜索 (多样性):")
    mmr_docs = vectorstore.max_marginal_relevance_search(query, k=3)

    for i, doc in enumerate(mmr_docs, 1):
        print(f"  {i}. {doc.page_content}")


# ========== 4. 检索链 ==========
def demo_retrieval_chain():
    """使用检索链"""
    print("\n\n" + "=" * 70)
    print("4. 检索链 (Retrieval Chain)")
    print("=" * 70)

    # 准备知识库
    documents = [
        Document(
            page_content="FastAPI 是一个现代、快速的 Web 框架",
            metadata={"category": "backend"}
        ),
        Document(
            page_content="Vue.js 是一个渐进式 JavaScript 框架",
            metadata={"category": "frontend"}
        ),
        Document(
            page_content="PostgreSQL 是一个强大的关系型数据库",
            metadata={"category": "database"}
        ),
        Document(
            page_content="Docker 用于容器化部署应用",
            metadata={"category": "devops"}
        ),
    ]

    # 创建向量存储
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 创建 LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    # 创建提示模板
    system_prompt = """
    你是一个技术专家助手。根据提供的上下文回答用户问题。

    上下文:
    {context}

    如果上下文中没有相关信息，请明确说明。
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # 创建文档链
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 创建检索链
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 查询
    questions = [
        "推荐一个后端框架",
        "什么数据库比较好?",
        "如何部署应用?"
    ]

    for question in questions:
        print(f"\n问题: {question}")

        result = retrieval_chain.invoke({"input": question})

        print(f"回答: {result['answer']}")

        # 显示使用的文档
        print(f"引用文档:")
        for doc in result['context']:
            print(f"  - {doc.page_content} (类别: {doc.metadata['category']})")


# ========== 5. 多文档检索 ==========
def demo_multi_document_retrieval():
    """多文档源检索"""
    print("\n\n" + "=" * 70)
    print("5. 多文档源检索")
    print("=" * 70)

    # 不同来源的文档
    python_docs = [
        Document(page_content="Python 支持列表推导式", metadata={"source": "python_guide"}),
        Document(page_content="Python 有丰富的标准库", metadata={"source": "python_guide"}),
    ]

    js_docs = [
        Document(page_content="JavaScript 支持异步编程", metadata={"source": "js_guide"}),
        Document(page_content="JavaScript 有 Promise 和 async/await", metadata={"source": "js_guide"}),
    ]

    # 合并文档
    all_docs = python_docs + js_docs

    # 创建向量存储
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(all_docs, embeddings)

    # 创建检索器 - 按来源过滤
    def create_filtered_retriever(source: str):
        """创建过滤特定来源的检索器"""
        return vectorstore.as_retriever(
            search_kwargs={
                "k": 2,
                "filter": {"source": source}
            }
        )

    # 测试不同来源
    print("\n只搜索 Python 文档:")
    python_retriever = create_filtered_retriever("python_guide")
    python_results = python_retriever.invoke("有什么特性?")
    for doc in python_results:
        print(f"  - {doc.page_content}")

    print("\n只搜索 JavaScript 文档:")
    js_retriever = create_filtered_retriever("js_guide")
    js_results = js_retriever.invoke("有什么特性?")
    for doc in js_results:
        print(f"  - {doc.page_content}")


# ========== 6. 高级检索策略 ==========
def demo_advanced_retrieval():
    """高级检索策略"""
    print("\n\n" + "=" * 70)
    print("6. 高级检索策略")
    print("=" * 70)

    documents = [
        Document(page_content=f"文档 {i}: 这是关于主题{i%3}的内容")
        for i in range(10)
    ]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 1. 相似度阈值
    print("\n1️⃣ 相似度阈值检索:")
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.8,
            "k": 5
        }
    )
    results = retriever.invoke("主题1")
    print(f"  找到 {len(results)} 个高相关文档")

    # 2. MMR (最大边际相关性)
    print("\n2️⃣ MMR 检索 (平衡相关性和多样性):")
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,  # 先获取10个，再选3个
            "lambda_mult": 0.5  # 0=最大多样性, 1=最大相关性
        }
    )
    results = retriever.invoke("主题内容")
    for doc in results:
        print(f"  - {doc.page_content}")

    # 3. 自定义检索器
    print("\n3️⃣ 自定义检索逻辑:")

    class CustomRetriever:
        def __init__(self, vectorstore, min_docs=2, max_docs=5):
            self.vectorstore = vectorstore
            self.min_docs = min_docs
            self.max_docs = max_docs

        def retrieve(self, query: str) -> List[Document]:
            # 先获取候选文档
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, k=self.max_docs
            )

            # 自定义过滤逻辑
            filtered = []
            for doc, score in docs_with_scores:
                if score < 1.0:  # 分数阈值
                    filtered.append(doc)

            # 确保最少数量
            if len(filtered) < self.min_docs:
                filtered = [doc for doc, _ in docs_with_scores[:self.min_docs]]

            return filtered[:self.max_docs]

    custom_retriever = CustomRetriever(vectorstore)
    results = custom_retriever.retrieve("查询内容")
    print(f"  自定义检索返回 {len(results)} 个文档")


# ========== 7. RAG 评估 ==========
def demo_rag_evaluation():
    """RAG 系统评估"""
    print("\n\n" + "=" * 70)
    print("7. RAG 评估")
    print("=" * 70)

    # 准备测试数据
    documents = [
        Document(page_content="Python 由 Guido van Rossum 创建于 1991 年"),
        Document(page_content="Python 强调代码可读性"),
        Document(page_content="Python 支持多种编程范式"),
    ]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 测试问题和预期答案
    test_cases = [
        {
            "question": "谁创建了 Python?",
            "expected_answer": "Guido van Rossum",
            "expected_docs": ["Python 由 Guido van Rossum 创建"]
        },
        {
            "question": "Python 有什么特点?",
            "expected_answer": "可读性",
            "expected_docs": ["Python 强调代码可读性"]
        }
    ]

    print("\n评估 RAG 系统:")

    for i, test in enumerate(test_cases, 1):
        print(f"\n测试 {i}:")
        print(f"  问题: {test['question']}")

        # 检索文档
        retrieved_docs = retriever.invoke(test['question'])

        # 评估检索质量
        retrieved_content = [doc.page_content for doc in retrieved_docs]
        print(f"  检索到 {len(retrieved_docs)} 个文档")

        # 检查是否包含预期文档
        found = False
        for expected in test['expected_docs']:
            if any(expected in content for content in retrieved_content):
                found = True
                break

        if found:
            print(f"  ✅ 检索正确")
        else:
            print(f"  ❌ 检索错误")

        # 显示检索到的文档
        print(f"  检索结果:")
        for doc in retrieved_docs:
            print(f"    - {doc.page_content}")


# ========== 8. 最佳实践 ==========
def best_practices():
    """RAG 最佳实践"""
    print("\n\n" + "=" * 70)
    print("8. 最佳实践")
    print("=" * 70)

    print("""
🎯 RAG 系统架构:

┌─────────────────┐
│   用户问题      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  文档检索       │ ← 向量数据库
│  (Retrieval)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  上下文增强     │ ← 检索到的文档
│  (Augmentation) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM 生成       │
│  (Generation)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   最终答案      │
└─────────────────┘

💡 关键设计决策:

1. **文档分割策略**
   ✅ 根据内容类型选择:
      - 技术文档: 按段落/标题
      - 代码: 按函数/类
      - 对话: 按轮次
   ✅ chunk_size: 500-1000 字符
   ✅ chunk_overlap: 10-20%

2. **Embedding 模型选择**
   - text-embedding-3-small (推荐, 性价比高)
   - text-embedding-3-large (更高精度)
   - 自定义模型 (特定领域)

3. **检索策略**
   ✅ 相似度搜索 (默认)
   ✅ MMR (需要多样性)
   ✅ 混合搜索 (关键词 + 向量)
   ✅ 重排序 (提高精度)

4. **检索参数调优**
   - k: 检索文档数 (2-5 为宜)
   - score_threshold: 相似度阈值 (0.7-0.9)
   - fetch_k: MMR 候选数 (k * 2-3)

5. **提示工程**
   ✅ 明确角色定位
   ✅ 强调使用上下文
   ✅ 处理无答案情况
   ✅ 引用来源

⚠️ 常见问题:

Q1: 检索不到相关文档怎么办?
A: 1) 调整 chunk_size
   2) 降低 score_threshold
   3) 增加 k 值
   4) 检查 embedding 质量

Q2: 答案不准确?
A: 1) 改进提示模板
   2) 使用更强的 LLM
   3) 重排序检索结果
   4) 增加上下文长度

Q3: 如何处理长文档?
A: 1) Map-Reduce 策略
   2) Refine 策略 (逐步优化)
   3) 分层检索 (先章节后段落)

Q4: 如何评估 RAG 系统?
A: 1) 检索准确率 (Retrieval Accuracy)
   2) 答案相关性 (Answer Relevance)
   3) 忠实度 (Faithfulness)
   4) 端到端评估

📊 性能优化:

1. **缓存策略**
   - 缓存 embedding 结果
   - 缓存频繁查询
   - 使用持久化向量存储

2. **并行处理**
   - 批量 embedding
   - 异步检索
   - 并发 LLM 调用

3. **成本优化**
   - 使用更小的 embedding 模型
   - 减少检索文档数
   - 缓存减少重复调用

🔗 参考资源:
- RAG Tutorial: https://python.langchain.com/docs/tutorials/rag/
- Vector Stores: https://python.langchain.com/docs/integrations/vectorstores/
- Retrievers: https://python.langchain.com/docs/how_to/#retrievers
- RAG Evaluation: https://python.langchain.com/docs/guides/evaluation/
    """)


def main():
    """运行所有示例"""
    demo_basic_rag()
    demo_document_loading()
    demo_vector_store()
    demo_retrieval_chain()
    demo_multi_document_retrieval()
    demo_advanced_retrieval()
    demo_rag_evaluation()
    best_practices()

    print("\n\n" + "=" * 70)
    print("✅ RAG 所有示例演示完成")
    print("=" * 70)
    print("\n核心要点:")
    print("  ✓ 文档分割 - chunk_size + overlap")
    print("  ✓ 向量存储 - FAISS/Chroma")
    print("  ✓ 检索策略 - 相似度/MMR/混合")
    print("  ✓ 检索链 - create_retrieval_chain")
    print("  ✓ 评估优化 - 准确率 + 相关性")


if __name__ == "__main__":
    main()
