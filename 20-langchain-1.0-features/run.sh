#!/bin/bash

# LangChain Agents 快速启动脚本

echo "======================================"
echo "LangChain Agents 学习项目"
echo "======================================"
echo ""

# 检查 Python 版本
echo "🔍 检查 Python 环境..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"
echo ""

# 检查 .env 文件
if [ ! -f .env ]; then
    echo "⚠️  未找到 .env 文件"
    echo "正在从模板创建 .env 文件..."
    cp .env.example .env
    echo "✅ .env 文件已创建"
    echo "⚠️  请编辑 .env 文件，填入你的 OPENAI_API_KEY"
    echo ""
    exit 1
else
    echo "✅ .env 文件存在"
fi

# 检查依赖
echo ""
echo "🔍 检查依赖包..."
if python3 -c "import langchain" 2>/dev/null; then
    echo "✅ langchain 已安装"
else
    echo "⚠️  langchain 未安装"
    echo "正在安装依赖..."
    pip install -r requirements.txt
fi

echo ""
echo "======================================"
echo "选择要运行的示例:"
echo "======================================"
echo "1. 基础 Agent (01_basic_agent.py)"
echo "2. 自定义工具 (02_custom_tools.py)"
echo "3. ReAct Agent (03_react_agent.py)"
echo "4. OpenAI Functions Agent (04_openai_functions_agent.py)"
echo "5. 对话记忆 Agent (05_conversational_agent.py)"
echo "6. LangGraph Agent (06_langgraph_agent.py)"
echo "7. 错误处理 (07_error_handling.py)"
echo "8. 工具组合 (08_tool_composition.py)"
echo "9. 性能优化 (09_optimization_best_practices.py)"
echo "0. 退出"
echo ""

read -p "请输入选项 (1-9): " choice

case $choice in
    1)
        echo "运行: 基础 Agent"
        python3 01_basic_agent.py
        ;;
    2)
        echo "运行: 自定义工具"
        python3 02_custom_tools.py
        ;;
    3)
        echo "运行: ReAct Agent"
        python3 03_react_agent.py
        ;;
    4)
        echo "运行: OpenAI Functions Agent"
        python3 04_openai_functions_agent.py
        ;;
    5)
        echo "运行: 对话记忆 Agent"
        python3 05_conversational_agent.py
        ;;
    6)
        echo "运行: LangGraph Agent"
        python3 06_langgraph_agent.py
        ;;
    7)
        echo "运行: 错误处理"
        python3 07_error_handling.py
        ;;
    8)
        echo "运行: 工具组合"
        python3 08_tool_composition.py
        ;;
    9)
        echo "运行: 性能优化"
        python3 09_optimization_best_practices.py
        ;;
    0)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac
