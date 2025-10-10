#!/usr/bin/env python3
"""
高级 RAG 检索优化 Demo
实现多向量检索、混合检索、重排序、查询扩展等优化技术
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict


class RetrievalStrategy(Enum):
    """检索策略枚举"""
    DENSE = "dense"  # 稠密检索
    SPARSE = "sparse"  # 稀疏检索
    HYBRID = "hybrid"  # 混合检索
    MULTI_VECTOR = "multi_vector"  # 多向量检索


@dataclass
class Document:
    """文档数据类"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    """检索结果数据类"""
    document: Document
    score: float
    retrieval_strategy: RetrievalStrategy
    rerank_score: Optional[float] = None


class QueryExpander:
    """查询扩展器"""
    
    def expand_query(self, query: str) -> List[str]:
        """扩展查询词"""
        # 在实际应用中，这里会使用同义词扩展、相关词挖掘等技术
        base_terms = query.lower().split()
        
        # 简单的同义词扩展
        synonym_map = {
            "ai": ["artificial intelligence", "machine learning"],
            "learn": ["study", "understand", "master"],
            "system": ["platform", "framework", "architecture"]
        }
        
        expanded_queries = [query]
        
        for term in base_terms:
            if term in synonym_map:
                for synonym in synonym_map[term]:
                    expanded_query = query.replace(term, synonym)
                    expanded_queries.append(expanded_query)
        
        print(f"🔍 查询扩展: {query} -> {expanded_queries}")
        return expanded_queries


class Reranker:
    """重排序器"""
    
    def rerank(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """对检索结果进行重排序"""
        # 在实际应用中，这里会使用更复杂的重排序模型
        # 这里使用简单的基于内容和查询相关性的重排序
        
        query_terms = set(query.lower().split())
        
        for result in results:
            # 计算查询词在文档中的出现频率
            content_terms = set(result.document.content.lower().split())
            overlap = len(query_terms.intersection(content_terms))
            
            # 计算重排序分数（基础分数 + 重叠度奖励）
            rerank_score = result.score + (overlap * 0.1)
            result.rerank_score = min(rerank_score, 1.0)  # 归一化
        
        # 按重排序分数排序
        results.sort(key=lambda x: x.rerank_score or x.score, reverse=True)
        
        print(f"🔄 重排序完成，前3个结果分数: {[r.rerank_score for r in results[:3]]}")
        return results


class AdvancedRetrievalSystem:
    """高级检索系统"""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.query_expander = QueryExpander()
        self.reranker = Reranker()
        
    def add_documents(self, documents: List[Document]):
        """添加文档到检索系统"""
        self.documents.extend(documents)
        print(f"📚 添加了 {len(documents)} 个文档到检索系统")
    
    def dense_retrieval(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """稠密检索（模拟实现）"""
        # 在实际应用中，这里会使用向量数据库进行相似度检索
        results = []
        
        for doc in self.documents[:top_k]:
            # 模拟相似度计算
            similarity = np.random.uniform(0.6, 0.95)
            results.append(RetrievalResult(
                document=doc,
                score=similarity,
                retrieval_strategy=RetrievalStrategy.DENSE
            ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def sparse_retrieval(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """稀疏检索（模拟实现）"""
        # 在实际应用中，这里会使用 BM25、TF-IDF 等传统检索方法
        results = []
        
        for doc in self.documents[:top_k]:
            # 模拟基于关键词的检索
            query_terms = set(query.lower().split())
            content_terms = set(doc.content.lower().split())
            overlap = len(query_terms.intersection(content_terms))
            
            score = overlap / len(query_terms) if query_terms else 0
            results.append(RetrievalResult(
                document=doc,
                score=min(score, 1.0),
                retrieval_strategy=RetrievalStrategy.SPARSE
            ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def hybrid_retrieval(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """混合检索"""
        dense_results = self.dense_retrieval(query, top_k * 2)
        sparse_results = self.sparse_retrieval(query, top_k * 2)
        
        # 合并结果并去重
        all_results = {}
        for result in dense_results + sparse_results:
            doc_id = result.document.id
            if doc_id not in all_results:
                all_results[doc_id] = result
            else:
                # 取最高分
                if result.score > all_results[doc_id].score:
                    all_results[doc_id] = result
        
        results = list(all_results.values())
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 标记为混合检索
        for result in results:
            result.retrieval_strategy = RetrievalStrategy.HYBRID
        
        return results[:top_k]
    
    def multi_vector_retrieval(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """多向量检索（模拟实现）"""
        # 在实际应用中，这里会使用多个不同粒度的向量表示
        expanded_queries = self.query_expander.expand_query(query)
        
        all_results = []
        for expanded_query in expanded_queries:
            # 对每个扩展查询进行检索
            results = self.dense_retrieval(expanded_query, top_k)
            all_results.extend(results)
        
        # 合并和去重
        merged_results = {}
        for result in all_results:
            doc_id = result.document.id
            if doc_id not in merged_results:
                merged_results[doc_id] = result
            else:
                # 取最高分
                if result.score > merged_results[doc_id].score:
                    merged_results[doc_id] = result
        
        results = list(merged_results.values())
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 标记为多向量检索
        for result in results:
            result.retrieval_strategy = RetrievalStrategy.MULTI_VECTOR
        
        return results[:top_k]
    
    def retrieve(self, query: str, strategy: RetrievalStrategy, top_k: int = 5, 
                 use_reranking: bool = True) -> List[RetrievalResult]:
        """执行检索"""
        print(f"🔍 执行 {strategy.value} 检索: '{query}'")
        
        if strategy == RetrievalStrategy.DENSE:
            results = self.dense_retrieval(query, top_k)
        elif strategy == RetrievalStrategy.SPARSE:
            results = self.sparse_retrieval(query, top_k)
        elif strategy == RetrievalStrategy.HYBRID:
            results = self.hybrid_retrieval(query, top_k)
        elif strategy == RetrievalStrategy.MULTI_VECTOR:
            results = self.multi_vector_retrieval(query, top_k)
        else:
            results = []
        
        # 应用重排序
        if use_reranking and results:
            results = self.reranker.rerank(results, query)
        
        return results


def create_sample_documents() -> List[Document]:
    """创建示例文档"""
    documents = [
        Document(
            id="doc1",
            content="LangChain 是一个用于开发由语言模型驱动的应用程序的框架",
            metadata={"source": "官方文档", "type": "framework"}
        ),
        Document(
            id="doc2", 
            content="RAG 技术结合了检索和生成，能够提供更准确的信息",
            metadata={"source": "技术文章", "type": "technique"}
        ),
        Document(
            id="doc3",
            content="多模态 AI 可以处理图像、文本、音频等多种类型的数据",
            metadata={"source": "研究论文", "type": "research"}
        ),
        Document(
            id="doc4",
            content="向量检索是 modern 信息检索的核心技术之一",
            metadata={"source": "教科书", "type": "textbook"}
        ),
        Document(
            id="doc5",
            content="查询扩展和重排序可以显著提升检索系统的性能",
            metadata={"source": "最佳实践", "type": "practice"}
        )
    ]
    return documents


def demo_retrieval_optimization():
    """演示检索优化功能"""
    print("🚀 开始高级 RAG 检索优化演示")
    print("=" * 50)
    
    # 创建检索系统
    retrieval_system = AdvancedRetrievalSystem()
    
    # 添加示例文档
    documents = create_sample_documents()
    retrieval_system.add_documents(documents)
    
    # 测试查询
    test_query = "AI 学习系统"
    
    print(f"\n📝 测试查询: '{test_query}'")
    print("-" * 30)
    
    # 测试不同检索策略
    strategies = [
        RetrievalStrategy.DENSE,
        RetrievalStrategy.SPARSE, 
        RetrievalStrategy.HYBRID,
        RetrievalStrategy.MULTI_VECTOR
    ]
    
    for strategy in strategies:
        print(f"\n🎯 使用 {strategy.value} 策略:")
        results = retrieval_system.retrieve(test_query, strategy, top_k=3)
        
        for i, result in enumerate(results, 1):
            score = result.rerank_score if result.rerank_score else result.score
            print(f"   {i}. [{score:.3f}] {result.document.content[:50]}...")
    
    print("\n✅ 检索优化演示完成")


if __name__ == "__main__":
    demo_retrieval_optimization()