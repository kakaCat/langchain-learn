# LangChain 全栈 AI 应用开发学习项目

本项目是一个完整的 LangChain 学习体系，从基础的多模态 AI 应用到高级的 Agent 架构、工作流编排、评估部署等全栈开发技能。

## 项目结构

```
01-chatbots-basic/           # 基础聊天机器人与多模态应用
02-prompt-templates/         # 提示词模板与结构化输出
03-semantic-intent/          # 语义路由与意图识别
04-memory/                   # 记忆管理与上下文压缩
05-structured-output/        # 结构化输出验证
06-rag-advanced/             # 高级 RAG 优化技术
07-tool-calling/             # 工具调用与函数集成
08-mcp-server/               # MCP 模型上下文协议
09-langgraph-workflow/       # LangGraph 工作流与状态管理
10-agent-examples/           # Agent 模式与编排策略
11-token-compression/        # Token 压缩与上下文管理
12-eval-and-deploy/          # 评估指标与生产部署
13-multimodal-processing/    # 多模态数据处理
14-advanced-rag/             # 高级 RAG 检索优化
15-integrated-projects/      # 集成实践项目
```

## 学习路线

### 🚀 基础模块 (1-4)
- **01-chatbots-basic**: 基础聊天机器人、多模态应用（图像、音频、视频）
- **02-prompt-templates**: 提示词模板、结构化输出、Few-shot 学习
- **03-semantic-intent**: 语义路由、意图识别、对话管理
- **04-memory**: 记忆管理、上下文压缩、向量存储记忆

### 🔧 核心能力 (6-10)  
- **06-rag-advanced**: 高级 RAG 优化技术、多模态检索
- **07-tools**: 工具调用、函数集成、外部 API 连接
- **08-mcp**: MCP 协议、上下文管理、服务端/客户端实现
- **09-langgraph-workflow**: LangGraph 工作流、状态管理、条件分支
- **10-agent-examples**: Agent 模式、编排策略、反思与推理

### 🎯 高级架构 (11-15)
- **11-token-compression**: Token 压缩、性能监控、上下文优化
- **12-eval-and-deploy**: 评估指标、监控追踪、生产部署
- **13-multimodal-processing**: 多模态数据处理、图像理解
- **14-advanced-rag**: 高级 RAG 检索优化、复杂查询处理
- **15-integrated-projects**: 集成实践项目、智能客服系统

## 核心功能特性

### 多模态 AI 应用
- **图像理解**: 图像描述、问答交互、Base64 编码
- **音频处理**: 语音转文本、文本转语音、内容分析  
- **视频分析**: 关键帧提取、内容描述、摘要生成

### 提示词工程
- **模板系统**: PromptTemplate、ChatPromptTemplate、Few-shot 模板
- **结构化输出**: Pydantic 模型集成、数据验证
- **性能优化**: 模板缓存、批量处理、国际化支持

### 记忆与上下文管理
- **记忆类型**: 对话历史、实体记忆、摘要记忆、向量存储
- **Token 压缩**: 智能压缩策略、性能监控、上下文优化
- **状态管理**: LangGraph 状态机、检查点、会话恢复

### Agent 与工作流
- **Agent 模式**: ReAct、Plan-and-Solve、Reflection、ToDoList
- **工作流编排**: LangGraph 状态图、条件分支、工具集成
- **评估部署**: 性能监控、调用追踪、CI/CD 管道

## 快速开始

### 环境配置

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **配置环境变量**
在项目根目录创建 `.env` 文件：
```env
# Qwen-Omni 配置
QWEN_API_KEY=your_qwen_api_key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

3. **获取 API Key**
- 访问 [阿里云 DashScope](https://dashscope.aliyuncs.com/)
- 注册账号并创建 API Key
- 配置到环境变量中

### 运行示例

**基础多模态应用**:
```bash
python 01-chatbots-basic/03_image_understanding_demo.py
python 01-chatbots-basic/04_audio_processing_demo.py  
python 01-chatbots-basic/05_video_analysis_demo.py
```

**提示词模板**:
```bash
python 02-prompt-templates/01_prompt_templates_demo.py
```

**工作流编排**:
```bash
python 09-langgraph-workflow/01_langgraph_workflow_chatbot.py
```

**Agent 模式**:
```bash
python 10-agent-examples/01_react_demo.py
```

## 学习目标

### 🎓 技能掌握
- **LangChain 核心概念**: 模型、提示词、记忆、工具、Agent
- **多模态 AI 开发**: 图像、音频、视频处理与理解
- **工作流编排**: LangGraph 状态管理、条件分支、工具集成
- **Agent 架构**: ReAct、反思、计划求解等高级模式
- **生产部署**: 评估指标、监控追踪、CI/CD 管道

### 💼 实践能力
- 构建端到端的 AI 应用
- 设计复杂的 Agent 工作流
- 优化性能与成本效益
- 部署到生产环境

### 🚀 进阶方向
- 企业级 AI 应用架构
- 多 Agent 协作系统
- 实时 AI 服务开发
- AI 应用性能调优

## 技术栈特点

### 🎯 模型优势 (Qwen-Omni)
- **多模态支持**: 文本、图像、音频、视频统一处理
- **中文优化**: 在中文场景下表现优异
- **成本效益**: 相比 OpenAI 更具成本优势
- **本地化**: 更好的中文理解和生成能力

### 🛠 实现特点
- **统一架构**: 遵循 LangChain 最佳实践
- **模块化设计**: 每个模块独立可运行
- **生产就绪**: 包含评估、监控、部署全流程
- **代码质量**: 清晰的文档和示例

### 📚 学习价值
- **系统性**: 从基础到高级的完整学习路径
- **实践性**: 每个概念都有可运行的代码示例
- **扩展性**: 易于基于现有代码进行二次开发
- **社区友好**: 标准的项目结构和文档

## 学习建议

### 📖 学习顺序
1. 从 01-chatbots-basic 开始，掌握基础的多模态应用
2. 学习 02-prompt-templates，理解提示词工程
3. 逐步深入记忆管理、工具调用、工作流编排
4. 最后学习 Agent 模式和评估部署

### 🔧 实践建议
- 每个模块都运行对应的示例代码
- 修改参数和输入，观察输出变化
- 基于现有代码进行扩展实验
- 参考各模块的 README 获取详细说明

## 注意事项

1. **API 限制**: 注意 Qwen-Omni 的 API 调用频率和配额限制
2. **文件格式**: 确保输入的文件格式符合要求
3. **网络连接**: 需要稳定的网络连接访问 API
4. **隐私保护**: 避免上传敏感或隐私相关的文件

## 扩展开发

可以基于现有代码扩展以下功能：
- 集成其他大模型 (OpenAI、Claude 等)
- 开发 Web 界面或移动应用
- 添加批处理和异步处理
- 构建企业级 AI 应用

## 许可证

MIT License