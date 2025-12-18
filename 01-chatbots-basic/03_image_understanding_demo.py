"""
图像理解示例 - 使用 Qwen-Omni 多模态模型实现图像描述和问答

功能：
- 加载本地图像文件
- 使用 Qwen-Omni 多模态模型进行图像描述
- 支持基于图像的问答对话
"""

import os
import base64
from PIL import Image
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_community.document_loaders import ImageCaptionLoader
from dotenv import load_dotenv


# 从当前模块目录加载 .env
def load_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)


def get_llm() -> ChatOpenAI:

    """创建并配置 Qwen-Omni 语言模型实例"""
    api_key = os.getenv("QWEN_API_KEY")
    model = os.getenv("QWEN_MODEL", "qwen-omni")
    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    temperature = float(os.getenv("QWEN_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("QWEN_MAX_TOKENS", "1024"))
    
    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 120,
        "max_retries": 3,
        "request_timeout": 120,
        "verbose": True,
        "base_url": base_url
    }
    
    return ChatOpenAI(**kwargs)


def describe_image(image_path, model):
    """描述图像内容"""
    try:
        # 加载图像
        image = Image.open(image_path)
        
        # 读取图像并转换为 base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 构建多模态消息
        message = HumanMessage(
            content=[
                {"type": "text", "text": "请描述这张图片的内容。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        )
        
        # 使用模型生成描述
        response = model.invoke([message])
        description = response.content
        
        return description
    except Exception as e:
        return f"图像处理错误: {str(e)}"

def image_qa_demo(image_path, question, model):
    """基于图像的问答演示"""
    try:
        image = Image.open(image_path)
        
        # 读取图像并转换为 base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 构建多模态消息
        message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        )
        
        # 使用模型进行问答
        response = model.invoke([message])
        answer = response.content
        
        return answer
    except Exception as e:
        return f"问答处理错误: {str(e)}"

def main():
    """主函数 - 图像理解演示"""
    print("=== 图像理解演示 ===")
   
    # 加载模型
    print("正在加载 Qwen-Omni 图像理解模型...")
    load_environment()
    model = get_llm()
    print("Qwen-Omni 模型加载完成!")
    
    # 图像路径
    image_path = input("请输入图像文件路径: ").strip()
    
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return
    
    # 图像描述
    print("\n--- 图像描述 ---")
    description = describe_image(image_path, model)
    print(f"图像描述: {description}")
    
    # 图像问答
    print("\n--- 图像问答 ---")
    while True:
        question = input("请输入关于图像的问题 (输入'退出'结束): ").strip()
        if question.lower() in ['退出', 'exit', 'quit']:
            break
        
        if question:
            answer = image_qa_demo(image_path, question, model)
            print(f"回答: {answer}")

if __name__ == "__main__":
    main()