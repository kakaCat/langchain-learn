#!/usr/bin/env python3
"""
高级 RAG 技术 Demo
实现检索增强生成的高级优化技术
"""

import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import re
from datetime import datetime, timedelta


class DocumentType(Enum):
    """文档类型枚举"""
    TEXT = "text"
    CODE = "code"
    MARKDOWN = "markdown"
    PDF = "pdf"
    HTML = "html"


class RetrievalStrategy(Enum):
    """检索策略枚举"""
    SEMANTIC = "semantic"  # 语义检索
    KEYWORD = "keyword"    # 关键词检索
    HYBRID = "hybrid"      # 混合检索
    MULTI_HOP = "multi_hop" # 多跳检索


class ChunkingStrategy(Enum):
    """分块策略枚举"""
    FIXED_SIZE = "fixed_size"      # 固定大小分块
    SENTENCE = "sentence"          # 按句子分块
    PARAGRAPH = "paragraph"        # 按段落分块
    SEMANTIC = "semantic"          # 语义分块
    OVERLAP = "overlap"            # 重叠分块


class EmbeddingModel:
    """嵌入模型模拟类"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.model_name = model_name
        self.dimension = 1536
    
    def embed(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        # 模拟嵌入生成
        np.random.seed(hash(text) % 2**32)
        return list(np.random.randn(self.dimension))
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class DocumentChunk:
    """文档分块"""
    
    def __init__(self, content: str, doc_id: str, chunk_id: str, 
                 chunk_index: int, metadata: Dict[str, Any] = None):
        self.content = content
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.chunk_index = chunk_index
        self.metadata = metadata or {}
        self.embedding = None
        self.score = 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "score": self.score
        }


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self):
        self.embedding_model = EmbeddingModel()
    
    def chunk_document(self, content: str, doc_id: str, 
                      strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
                      chunk_size: int = 500, overlap: int = 50) -> List[DocumentChunk]:
        """文档分块处理"""
        chunks = []
        
        if strategy == ChunkingStrategy.FIXED_SIZE:
            # 固定大小分块
            for i in range(0, len(content), chunk_size - overlap):
                chunk_content = content[i:i + chunk_size]
                if chunk_content.strip():
                    chunk_id = f"{doc_id}_chunk_{i//chunk_size}"
                    chunk = DocumentChunk(
                        content=chunk_content,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        chunk_index=i//chunk_size,
                        metadata={
                            "strategy": strategy.value,
                            "chunk_size": chunk_size,
                            "overlap": overlap
                        }
                    )
                    chunks.append(chunk)
        
        elif strategy == ChunkingStrategy.SENTENCE:
            # 按句子分块
            sentences = re.split(r'[.!?。！？]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            for i, sentence in enumerate(sentences):
                chunk_id = f"{doc_id}_sentence_{i}"
                chunk = DocumentChunk(
                    content=sentence,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    chunk_index=i,
                    metadata={
                        "strategy": strategy.value,
                        "sentence_index": i
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """为分块生成嵌入向量"""
        for chunk in chunks:
            chunk.embedding = self.embedding_model.embed(chunk.content)
        return chunks


class VectorStore:
    """向量存储"""
    
    def __init__(self):
        self.chunks: Dict[str, DocumentChunk] = {}
        self.embeddings: List[List[float]] = []
        self.chunk_ids: List[str] = []
        self.access_stats = defaultdict(int)
        self.last_access = {}
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """添加分块到向量存储"""
        for chunk in chunks:
            if chunk.chunk_id not in self.chunks:
                self.chunks[chunk.chunk_id] = chunk
                if chunk.embedding:
                    self.embeddings.append(chunk.embedding)
                    self.chunk_ids.append(chunk.chunk_id)
    
    def semantic_search(self, query: str, top_k: int = 5, 
                       strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC) -> List[DocumentChunk]:
        """语义搜索"""
        embedding_model = EmbeddingModel()
        query_embedding = embedding_model.embed(query)
        
        # 计算相似度
        similarities = []
        for i, chunk_embedding in enumerate(self.embeddings):
            similarity = embedding_model.similarity(query_embedding, chunk_embedding)
            similarities.append((similarity, self.chunk_ids[i]))
        
        # 排序并返回top_k
        similarities.sort(reverse=True)
        
        results = []
        for similarity, chunk_id in similarities[:top_k]:
            chunk = self.chunks[chunk_id]
            chunk.score = similarity
            
            # 更新访问统计
            self.access_stats[chunk_id] += 1
            self.last_access[chunk_id] = datetime.now()
            
            results.append(chunk)
        
        return results
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """关键词搜索"""
        query_words = set(query.lower().split())
        
        scores = []
        for chunk_id, chunk in self.chunks.items():
            content_words = set(chunk.content.lower().split())
            
            # 计算关键词匹配度
            intersection = query_words.intersection(content_words)
            score = len(intersection) / len(query_words) if query_words else 0
            
            scores.append((score, chunk_id))
        
        # 排序并返回top_k
        scores.sort(reverse=True)
        
        results = []
        for score, chunk_id in scores[:top_k]:
            chunk = self.chunks[chunk_id]
            chunk.score = score
            results.append(chunk)
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5, 
                     semantic_weight: float = 0.7) -> List[DocumentChunk]:
        """混合搜索"""
        semantic_results = self.semantic_search(query, top_k * 2)
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # 合并结果并重新排序
        all_results = {}
        
        for chunk in semantic_results:
            all_results[chunk.chunk_id] = {
                "chunk": chunk,
                "semantic_score": chunk.score,
                "keyword_score": 0.0
            }
        
        for chunk in keyword_results:
            if chunk.chunk_id in all_results:
                all_results[chunk.chunk_id]["keyword_score"] = chunk.score
            else:
                all_results[chunk.chunk_id] = {
                    "chunk": chunk,
                    "semantic_score": 0.0,
                    "keyword_score": chunk.score
                }
        
        # 计算综合分数
        scored_results = []
        for data in all_results.values():
            combined_score = (data["semantic_score"] * semantic_weight + 
                            data["keyword_score"] * (1 - semantic_weight))
            data["chunk"].score = combined_score
            scored_results.append((combined_score, data["chunk"]))
        
        # 排序并返回top_k
        scored_results.sort(reverse=True)
        return [chunk for _, chunk in scored_results[:top_k]]


class Reranker:
    """重排序器"""
    
    @staticmethod
    def diversity_rerank(chunks: List[DocumentChunk], top_k: int = 5) -> List[DocumentChunk]:
        """多样性重排序"""
        if not chunks:
            return []
        
        # 简单多样性重排序：选择不同文档的分块
        selected = []
        doc_seen = set()
        
        for chunk in chunks:
            if chunk.doc_id not in doc_seen:
                selected.append(chunk)
                doc_seen.add(chunk.doc_id)
            
            if len(selected) >= top_k:
                break
        
        # 如果多样性不足，补充高分分块
        if len(selected) < top_k:
            for chunk in chunks:
                if chunk not in selected:
                    selected.append(chunk)
                if len(selected) >= top_k:
                    break
        
        return selected
    
    @staticmethod
    def recency_rerank(chunks: List[DocumentChunk], vector_store: VectorStore, 
                      top_k: int = 5) -> List[DocumentChunk]:
        """时效性重排序"""
        # 根据最后访问时间排序
        chunks_with_time = []
        for chunk in chunks:
            last_access = vector_store.last_access.get(chunk.chunk_id, datetime.min)
            chunks_with_time.append((last_access, chunk))
        
        # 按时间倒序排列
        chunks_with_time.sort(reverse=True)
        return [chunk for _, chunk in chunks_with_time[:top_k]]


class QueryExpander:
    """查询扩展器"""
    
    @staticmethod
    def synonym_expansion(query: str) -> List[str]:
        """同义词扩展"""
        synonyms = {
            "AI": ["人工智能", "机器学习", "深度学习"],
            "开发": ["编程", "编写", "创建"],
            "框架": ["库", "工具包", "平台"],
            "学习": ["掌握", "了解", "研究"],
            "技术": ["方法", "技巧", "手段"]
        }
        
        expanded_queries = [query]
        words = query.split()
        
        for i, word in enumerate(words):
            if word in synonyms:
                for synonym in synonyms[word]:
                    new_words = words.copy()
                    new_words[i] = synonym
                    expanded_queries.append(' '.join(new_words))
        
        return expanded_queries
    
    @staticmethod
    def contextual_expansion(query: str, context_chunks: List[DocumentChunk]) -> str:
        """上下文扩展"""
        if not context_chunks:
            return query
        
        # 从相关分块中提取关键词
        all_content = ' '.join(chunk.content for chunk in context_chunks[:3])
        words = re.findall(r'\b\w+\b', all_content.lower())
        
        # 统计词频
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 2:  # 过滤短词
                word_freq[word] += 1
        
        # 选择高频词
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        expanded_query = query
        
        for word, _ in top_words:
            if word not in query.lower():
                expanded_query += f" {word}"
        
        return expanded_query


class AdvancedRAGSystem:
    """高级 RAG 系统"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.reranker = Reranker()
        self.query_expander = QueryExpander()
        self.query_history = deque(maxlen=100)
    
    def add_document(self, content: str, doc_id: str, 
                    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE):
        """添加文档到系统"""
        print(f"📄 处理文档: {doc_id}")
        
        # 文档分块
        chunks = self.document_processor.chunk_document(
            content, doc_id, chunking_strategy
        )
        print(f"  生成分块: {len(chunks)} 个")
        
        # 生成嵌入
        chunks_with_embeddings = self.document_processor.generate_embeddings(chunks)
        print(f"  生成嵌入: 完成")
        
        # 添加到向量存储
        self.vector_store.add_chunks(chunks_with_embeddings)
        print(f"  存储完成: 总共有 {len(self.vector_store.chunks)} 个分块")
    
    def search(self, query: str, strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
              top_k: int = 5, use_reranking: bool = True, 
              use_query_expansion: bool = True) -> Dict[str, Any]:
        """高级搜索"""
        print(f"🔍 执行搜索: {query}")
        print(f"  策略: {strategy.value}, top_k: {top_k}")
        
        # 记录查询历史
        self.query_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy.value
        })
        
        # 查询扩展
        expanded_query = query
        if use_query_expansion and strategy != RetrievalStrategy.KEYWORD:
            expanded_queries = self.query_expander.synonym_expansion(query)
            expanded_query = expanded_queries[0]  # 使用第一个扩展查询
            print(f"  查询扩展: {expanded_query}")
        
        # 执行搜索
        if strategy == RetrievalStrategy.SEMANTIC:
            results = self.vector_store.semantic_search(expanded_query, top_k * 2)
        elif strategy == RetrievalStrategy.KEYWORD:
            results = self.vector_store.keyword_search(expanded_query, top_k * 2)
        else:  # HYBRID or MULTI_HOP
            results = self.vector_store.hybrid_search(expanded_query, top_k * 2)
        
        print(f"  初步结果: {len(results)} 个分块")
        
        # 重排序
        if use_reranking and results:
            results = self.reranker.diversity_rerank(results, top_k)
            print(f"  重排序后: {len(results)} 个分块")
        
        # 准备返回结果
        result_chunks = results[:top_k]
        
        return {
            "query": query,
            "expanded_query": expanded_query,
            "strategy": strategy.value,
            "results": [chunk.to_dict() for chunk in result_chunks],
            "total_results": len(result_chunks),
            "search_time": datetime.now().isoformat()
        }
    
    def multi_hop_search(self, query: str, max_hops: int = 3, 
                        top_k_per_hop: int = 3) -> Dict[str, Any]:
        """多跳检索"""
        print(f"🔄 执行多跳检索: {query}")
        
        current_query = query
        all_results = []
        hop_results = []
        
        for hop in range(max_hops):
            print(f"  第 {hop + 1} 跳检索...")
            
            # 执行搜索
            results = self.search(
                current_query, 
                RetrievalStrategy.HYBRID, 
                top_k_per_hop, 
                use_reranking=True,
                use_query_expansion=True
            )
            
            hop_results.append({
                "hop": hop + 1,
                "query": current_query,
                "results": results["results"]
            })
            
            # 收集结果
            all_results.extend(results["results"])
            
            # 生成下一跳查询
            if hop < max_hops - 1 and results["results"]:
                # 从结果中提取信息生成新查询
                context_content = ' '.join(
                    result["content"] for result in results["results"][:2]
                )
                current_query = self.query_expander.contextual_expansion(
                    query, 
                    [DocumentChunk(**result) for result in results["results"][:2]]
                )
                print(f"  下一跳查询: {current_query}")
            else:
                break
        
        # 去重和排序
        unique_results = {}
        for result in all_results:
            if result["chunk_id"] not in unique_results:
                unique_results[result["chunk_id"]] = result
            elif result["score"] > unique_results[result["chunk_id"]]["score"]:
                unique_results[result["chunk_id"]] = result
        
        final_results = sorted(
            unique_results.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )[:top_k_per_hop * max_hops]
        
        return {
            "original_query": query,
            "max_hops": max_hops,
            "hop_results": hop_results,
            "final_results": final_results,
            "total_unique_results": len(final_results)
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "total_chunks": len(self.vector_store.chunks),
            "total_documents": len(set(chunk.doc_id for chunk in self.vector_store.chunks.values())),
            "total_queries": len(self.query_history),
            "most_accessed_chunks": sorted(
                self.vector_store.access_stats.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            "recent_queries": list(self.query_history)[-5:]
        }


def demo_advanced_rag():
    """演示高级 RAG 功能"""
    print("🚀 启动高级 RAG 系统演示")
    print("=" * 50)
    
    # 创建系统实例
    rag_system = AdvancedRAGSystem()
    
    # 添加示例文档
    sample_documents = [
        {
            "id": "doc_ai_intro",
            "content": """
人工智能（AI）是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器。
机器学习是AI的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。
深度学习是机器学习的一个子领域，使用神经网络模拟人脑的工作方式。
            """
        },
        {
            "id": "doc_rag_tech", 
            "content": """
检索增强生成（RAG）是一种结合检索系统和生成模型的技术。
RAG系统首先从知识库中检索相关信息，然后使用这些信息来生成更准确和相关的回答。
高级RAG技术包括多跳检索、查询扩展、重排序等优化方法。
            """
        },
        {
            "id": "doc_langchain",
            "content": """
LangChain是一个用于开发由语言模型驱动的应用程序的框架。
它提供了链式调用、代理、记忆等组件，简化了复杂AI应用的开发。
LangChain支持多种语言模型和向量数据库的集成。
            """
        }
    ]
    
    # 处理文档
    for doc in sample_documents:
        rag_system.add_document(doc["content"], doc["id"])
    
    print("\n📊 系统统计信息:")
    stats = rag_system.get_system_stats()
    print(f"   总文档数: {stats['total_documents']}")
    print(f"   总分块数: {stats['total_chunks']}")
    
    print("\n🔍 演示不同检索策略:")
    
    # 语义检索演示
    print("\n1. 语义检索:")
    semantic_results = rag_system.search(
        "什么是人工智能", 
        RetrievalStrategy.SEMANTIC,
        top_k=3
    )
    print(f"   找到 {semantic_results['total_results']} 个相关结果")
    for i, result in enumerate(semantic_results["results"][:2], 1):
        print(f"   {i}. 分数: {result['score']:.3f}")
        print(f"      内容: {result['content'][:100]}...")
    
    # 混合检索演示
    print("\n2. 混合检索:")
    hybrid_results = rag_system.search(
        "RAG技术的工作原理", 
        RetrievalStrategy.HYBRID,
        top_k=3
    )
    print(f"   找到 {hybrid_results['total_results']} 个相关结果")
    for i, result in enumerate(hybrid_results["results"][:2], 1):
        print(f"   {i}. 分数: {result['score']:.3f}")
        print(f"      内容: {result['content'][:100]}...")
    
    # 多跳检索演示
    print("\n3. 多跳检索:")
    multi_hop_results = rag_system.multi_hop_search(
        "AI和RAG的关系",
        max_hops=2,
        top_k_per_hop=2
    )
    print(f"   经过 {len(multi_hop_results['hop_results'])} 跳检索")
    print(f"   找到 {multi_hop_results['total_unique_results']} 个唯一结果")
    
    for hop in multi_hop_results["hop_results"]:
        print(f"   第{hop['hop']}跳查询: {hop['query']}")
    
    print("\n✅ 高级 RAG 演示完成!")


if __name__ == "__main__":
    demo_advanced_rag()