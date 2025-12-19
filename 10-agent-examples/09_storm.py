"""
STORM（Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking）长文写作示例

借鉴论文与实践总结（参考知乎文章）实现：
- 视角发现：发现不同写作视角（基本事实、历史、应用等）
- 多轮提问：每个视角生成多轮高价值问题
- 检索与整合：用简易搜索检索并总结成可信信息
- 大纲合成：根据主题与对话信息生成结构化大纲
- 章节写作：按大纲写作并引用来源

说明：
- 无外部搜索 API 时，使用 Tavily 轻量搜索（如不可用则降级模拟）
- 无 LLM API Key 时，降级为规则式/占位写作，保证可运行
"""

import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# =========================
# 环境与模型
# =========================

def load_environment():
    """加载 .env 环境变量（就近目录）。"""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)


def get_llm() -> Optional[ChatOpenAI]:
    """创建并配置语言模型实例；无 key 时返回 None 以触发降级模式。"""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    base_url = os.getenv("OPENAI_BASE_URL")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))

    if not api_key:
        return None

    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 120,
        "max_retries": 3,
        "request_timeout": 120,
        "base_url": base_url,
    }
    try:
        return ChatOpenAI(**kwargs)
    except Exception:
        return None


# =========================
# 轻量检索（Tavily）
# =========================

def tavily_search(query: str, k: int = 5) -> List[Dict[str, str]]:
    """使用 Tavily 进行轻量检索；不可用时返回占位结果。
    返回项包含 title / link / snippet。
    """
    try:
        # 优先尝试官方包
        from langchain_tavily import TavilySearch
        tavily_tool = TavilySearch(max_results=k)
        results = tavily_tool.invoke({"query": query})
        
        # 转换结果格式以保持兼容性
        formatted_results = []
        for r in results:
            formatted_results.append({
                "title": r.get("title") or "",
                "link": r.get("url") or r.get("link") or "",
                "snippet": r.get("content") or r.get("snippet") or r.get("body") or "",
            })
        return formatted_results
    except Exception:
        try:
            # 降级到社区版
            from langchain_community.tools.tavily_search import TavilySearchResults
            tavily_tool = TavilySearchResults(k=k)
            results = tavily_tool.invoke({"query": query})
            
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "title": r.get("title") or "",
                    "link": r.get("url") or r.get("link") or "",
                    "snippet": r.get("content") or r.get("snippet") or r.get("body") or "",
                })
            return formatted_results
        except Exception:
            # 最终降级占位：避免运行时失败
            return [{
                "title": "占位搜索结果",
                "link": "https://example.com",
                "snippet": f"无法使用实时搜索。这里是关于'{query}'的占位信息。",
            }]


# =========================
# 数据结构
# =========================

class QAItem(BaseModel):
    perspective: str = Field(description="视角名称")
    question: str = Field(description="问题")
    answer: Optional[str] = Field(default=None, description="答案/总结")
    sources: List[str] = Field(default_factory=list, description="信息来源链接")


class StormWriteState(BaseModel):
    """STORM 写作状态"""
    topic: str = Field(description="写作主题")
    perspectives: List[str] = Field(default_factory=list, description="写作视角集合")
    qa: List[QAItem] = Field(default_factory=list, description="多视角问答集合")
    outline: List[str] = Field(default_factory=list, description="文章大纲（章节标题）")
    article: Optional[str] = Field(default=None, description="最终文章内容")
    iteration: int = Field(default=0, description="多轮对话轮次计数")


# =========================
# 节点实现
# =========================

def discover_perspectives_node(state: StormWriteState) -> StormWriteState:
    """视角发现：基于主题提出 3-5 个多维视角。"""
    llm = get_llm()
 
    prompt = f"""
你是一名资深百科写作者。请针对主题《{state.topic}》提出 4-6 个互补的写作视角。
要求：
- 覆盖：基本事实、历史/演进、应用/影响、挑战/争议、未来/趋势 等维度
- 输出：用逗号分隔的短语列表，不要解释
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    raw = resp.content.strip()
    perspectives = [p.strip() for p in raw.replace("，", ",").split(",") if p.strip()]
    # 去重与裁剪
    seen = set()
    clean = []
    for p in perspectives:
        if p not in seen:
            clean.append(p)
            seen.add(p)
        if len(clean) >= 6:
            break
    state.perspectives = clean or ["基本事实", "历史脉络", "关键应用", "挑战与争议", "未来趋势"]
    return state


def questioning_node(state: StormWriteState) -> StormWriteState:
    """多轮提问：每个视角生成 2 条高价值问题。"""
    llm = get_llm()
    if llm is None:
        # 降级直接生成模板问题
        for p in state.perspectives:
            state.qa.append(QAItem(perspective=p, question=f"关于‘{p}’的关键事实是什么？"))
            state.qa.append(QAItem(perspective=p, question=f"在‘{p}’下最重要的案例/证据有哪些？"))
        state.iteration += 1
        return state

    prompt = f"""
主题：{state.topic}
视角：{state.perspectives}
请为每个视角生成两条高价值问题（简短精炼，便于检索），仅输出按视角分组的问题清单：
- 输出格式示例：
基本事实：问题A；问题B
历史脉络：问题C；问题D
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    lines = [ln.strip() for ln in resp.content.splitlines() if ln.strip()]
    for ln in lines:
        if "：" in ln:
            k, v = ln.split("：", 1)
            questions = [q.strip() for q in v.replace("；", ";").split(";") if q.strip()]
            for q in questions[:2]:
                state.qa.append(QAItem(perspective=k.strip(), question=q))
    state.iteration += 1
    return state


def retrieve_node(state: StormWriteState) -> StormWriteState:
    """检索并总结：对尚未回答的问题进行轻量检索并总结为答案。"""
    llm = get_llm()
    # 找到未回答的问题
    pending = [item for item in state.qa if not item.answer]
    for item in pending:
        results = tavily_search(f"{state.topic} {item.question}", k=4)
        sources = [r.get("link", "") for r in results if r.get("link")]
        snippets = [r.get("snippet", "") for r in results if r.get("snippet")]
        item.sources = sources
        if llm is None:
            # 降级总结：拼接摘要片段
            summary = "；".join(snippets) or f"基于公开资料，对问题‘{item.question}’进行占位总结。"
            item.answer = summary[:800]
        else:
            prompt = f"""
你是信息整合助手。请基于以下检索摘要，针对问题生成简洁、可信的回答，并在结尾列出参考链接：
问题：{item.question}
摘要：{snippets}
"""
            resp = llm.invoke([HumanMessage(content=prompt)])
            item.answer = resp.content.strip()
            # 若回答未包含链接，追加来源
            if item.sources and ("http" not in item.answer):
                item.answer += "\n参考：" + " | ".join(item.sources[:3])
    return state


def outline_node(state: StormWriteState) -> StormWriteState:
    """大纲合成：根据主题与多视角问答生成结构化大纲。"""
    llm = get_llm()
    qa_view = [f"[{q.perspective}] {q.question} -> {q.answer[:120] if q.answer else ''}" for q in state.qa]
    if llm is None:
        # 降级固定大纲
        state.outline = [
            "引言：主题背景与重要性",
            "基本概念与核心事实",
            "历史演进与关键节点",
            "代表性应用与影响",
            "挑战、争议与风险",
            "未来发展趋势与展望",
            "结论与参考资料"
        ]
        return state

    prompt = f"""
请基于主题《{state.topic}》以及下列多视角问答，合成 6-8 条结构化大纲条目（仅输出章节标题列表，每行一条，不要解释）。
问答摘要：{qa_view}
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    lines = [ln.strip("- • ").strip() for ln in resp.content.splitlines() if ln.strip()]
    # 去重
    seen = set()
    outline = []
    for ln in lines:
        if ln and ln not in seen:
            outline.append(ln)
            seen.add(ln)
        if len(outline) >= 8:
            break
    state.outline = outline or [
        "引言：主题背景与重要性",
        "基本概念与核心事实",
        "历史演进与关键节点",
        "代表性应用与影响",
        "挑战、争议与风险",
        "未来发展趋势与展望",
        "结论与参考资料"
    ]
    return state


def writing_node(state: StormWriteState) -> StormWriteState:
    """章节写作：按大纲生成带引用的文章。"""
    llm = get_llm()
    qa_context = "\n".join([f"[{q.perspective}] Q: {q.question}\nA: {q.answer}" for q in state.qa])
    outline = state.outline or ["引言", "正文", "结论"]

    if llm is None:
        # 降级写作：模板化拼接
        sections = []
        for sec in outline:
            sections.append(f"## {sec}\n\n基于对主题与公开资料的综合，本文就‘{sec}’进行占位性论述。\n")
        refs = []
        for q in state.qa:
            refs.extend(q.sources)
        refs = list(dict.fromkeys(refs))
        state.article = "\n".join(sections) + ("\n参考资料:\n" + "\n".join(refs[:10]) if refs else "")
        return state

    prompt = f"""
你是一名维基百科风格的写作者。请基于大纲与多视角问答，撰写结构化的长文草稿：
- 文风客观、清晰，避免主观形容
- 每个大纲段落 2-4 段
- 在适当位置引用参考链接（可使用括号形式）

主题：{state.topic}
大纲：{outline}
问答上下文：\n{qa_context}
只输出文章正文，不要额外说明。
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    state.article = resp.content.strip()
    return state


# =========================
# 决策与工作流
# =========================

def decision_after_retrieve(state: StormWriteState) -> str:
    """是否继续新一轮提问检索；限制 2 轮。"""
    if state.iteration >= 2:
        return "outline"
    return "question"


def create_storm_workflow():
    """创建 STORM 写作工作流。"""
    workflow = StateGraph(StormWriteState)

    # 主题方向
    workflow.add_node("discover", discover_perspectives_node)
    # 资料
    workflow.add_node("question", questioning_node)
    # 补充资料
    workflow.add_node("retrieve", retrieve_node)
    # 生成大纲
    workflow.add_node("outline", outline_node)
    # 生成文章
    workflow.add_node("write", writing_node)

    # 流程起点
    workflow.set_entry_point("discover")
    workflow.add_edge("discover", "question")
    workflow.add_edge("question", "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        decision_after_retrieve,
        {
            "question": "question",
            "outline": "outline",
        },
    )
    workflow.add_edge("outline", "write")
    workflow.add_edge("write", END)

    return workflow.compile()


# =========================
# 运行示例
# =========================

def run_storm_example():
    """运行 STORM 写作示例。"""
    load_environment()

    examples = [
        "生成一篇介绍‘生成式 AI 在教育中的应用’的百科式文章",
        "分析‘气候变化对全球经济的影响’并形成结构化长文",
    ]

    graph = create_storm_workflow()

    for i, topic in enumerate(examples, 1):
        print(f"\n===== 示例 {i} =====")
        print(f"主题：{topic}")
        init = StormWriteState(topic=topic)
        result = graph.invoke(init)
        print(f"视角：{result['perspectives']}")
        print(f"大纲：{result['outline']}")
        print("\n正文预览（前 600 字）：")
        body = result.get("article") or "(无正文)"
        print(body[:600])
        print("\n—— 完 ——")


if __name__ == "__main__":
    run_storm_example()