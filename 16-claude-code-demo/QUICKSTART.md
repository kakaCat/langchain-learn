# 快速开始指南

## 5 分钟运行增强版 Claude Code Agent

### 第 1 步：安装依赖（1 分钟）

```bash
cd 10-agent-examples
pip install -r requirements.txt
```

### 第 2 步：配置环境变量（2 分钟）

#### 选项 A：使用 OpenAI（推荐）

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填写 API Key
nano .env  # 或使用你喜欢的编辑器
```

在 `.env` 中设置：

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

#### 选项 B：使用本地 Ollama（免费）

```bash
# 1. 安装 Ollama
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# 2. 下载模型
ollama pull llama3.1:8b

# 3. 启动服务（新终端）
ollama serve

# 4. 配置 .env
cp .env.example .env
```

在 `.env` 中设置：

```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
```

### 第 3 步：验证配置（1 分钟）

```bash
python test_setup.py
```

你应该看到类似这样的输出：

```
✅ langchain               LangChain 核心库
✅ langgraph               LangGraph 工作流引擎
✅ duckduckgo_search       Web 搜索工具
...
✅ LLM 连接: 连接成功！响应: OK

🎉 所有检查通过！你可以运行增强版 Agent 了
```

### 第 4 步：运行演示（1 分钟）

```bash
python 11_claude_code_style_enhanced.py
```

等待几分钟，你将看到完整的研究流程和最终报告！

## 自定义你的任务

编辑 `11_claude_code_style_enhanced.py` 中的 `run_enhanced_demo()` 函数：

```python
tasks = [
    {
        "request": "你的研究问题",  # 修改这里
        "budget": 40000,
        "max_loops": 6,
    },
]
```

示例任务：

- "研究 2025 年最流行的 Python Web 框架及其优缺点"
- "分析 Rust 和 Go 在微服务开发中的性能对比"
- "总结 LangChain vs LlamaIndex 的核心区别和选择建议"
- "调研 Vector 数据库的主流选择（Pinecone、Weaviate、Qdrant）"

## 预期输出

程序会输出：

1. **研究规划**：自动拆解你的问题为 3-5 个具体研究方面
2. **执行过程**：每个 SubAgent 的思考、行动、观察过程
3. **研究记忆**：所有收集的信息和引用来源
4. **目标评估**：任务完成度和缺失方面
5. **最终报告**：Markdown 格式的综合研究报告

## 常见问题

### Q: 安装 duckduckgo-search 失败

```bash
# 尝试指定版本
pip install duckduckgo-search==6.3.5

# 或使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple duckduckgo-search
```

### Q: OpenAI API 调用超时

在 `.env` 中增加超时时间（暂不支持，需修改代码）或减少 `max_loops`。

### Q: Ollama 内存不足

使用更小的模型：

```bash
ollama pull llama3.1:8b  # 而不是 70b
# 或
ollama pull mistral:7b
```

### Q: JSON 解析总是失败

1. 使用更强的模型（GPT-4 或 Llama 3.1 70B）
2. 检查 `OPENAI_MAX_TOKENS` 是否足够（建议 1500+）

## 下一步

- 阅读 [README_enhanced.md](README_enhanced.md) 了解详细架构
- 查看 [COMPARISON.md](COMPARISON.md) 理解增强点
- 尝试添加自定义工具（参考 `ToolRegistry` 类）

## 获取帮助

如果遇到问题：

1. 运行 `python test_setup.py` 诊断
2. 查看 [README_enhanced.md](README_enhanced.md) 的故障排除部分
3. 提交 Issue 到 GitHub（包含 `test_setup.py` 的输出）

祝你使用愉快！🚀
