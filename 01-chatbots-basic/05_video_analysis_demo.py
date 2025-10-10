"""
视频分析示例 - 使用 Qwen-Omni 实现视频内容理解和分析

功能：
- 视频关键帧提取
- 视频内容描述
- 视频问答
- 视频摘要生成
"""

import os
import cv2
import tempfile
import base64
from PIL import Image
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

def extract_video_keyframes(video_path, num_frames=5):
    """提取视频关键帧"""
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算采样间隔
        interval = max(1, total_frames // num_frames)
        
        frames = []
        frame_count = 0
        
        while len(frames) < num_frames and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                # 保存帧为临时图像文件
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                    cv2.imwrite(temp_file.name, frame)
                    frames.append(temp_file.name)
            
            frame_count += 1
        
        cap.release()
        return frames
    except Exception as e:
        print(f"关键帧提取错误: {str(e)}")
        return []

def load_video_analysis_model():
    """加载视频分析模型 - 使用 Qwen-Omni"""
    model = ChatOpenAI(
        model="qwen-omni",
        api_key=os.getenv("QWEN_API_KEY"),
        base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        max_tokens=2048,
        temperature=0
    )
    return model

def describe_video_content(frames, model):
    """描述视频内容"""
    try:
        # 构建多模态消息
        content = [{"type": "text", "text": "请根据这些关键帧描述视频的主要内容。"}]
        
        # 添加所有关键帧
        for frame_path in frames:
            # 读取图像并转换为 base64
            with open(frame_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            })
        
        message = HumanMessage(content=content)
        
        # 使用模型生成描述
        response = model.invoke([message])
        description = response.content
        
        return description
    except Exception as e:
        return f"视频描述错误: {str(e)}"

def video_qa_demo(frames, question, model):
    """视频问答演示"""
    try:
        # 构建多模态消息
        content = [{"type": "text", "text": question}]
        
        # 添加所有关键帧
        for frame_path in frames:
            # 读取图像并转换为 base64
            with open(frame_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            })
        
        message = HumanMessage(content=content)
        
        # 使用模型进行问答
        response = model.invoke([message])
        answer = response.content
        
        return answer
    except Exception as e:
        return f"视频问答错误: {str(e)}"

def generate_video_summary(frames, model):
    """生成视频摘要"""
    try:
        # 构建多模态消息
        content = [{"type": "text", "text": "请根据这些关键帧生成视频的详细摘要，包括场景描述、主要活动和整体内容概述。"}]
        
        # 添加所有关键帧
        for frame_path in frames:
            # 读取图像并转换为 base64
            with open(frame_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            })
        
        message = HumanMessage(content=content)
        
        # 使用模型生成摘要
        response = model.invoke([message])
        summary = response.content
        
        return summary
    except Exception as e:
        return f"视频摘要生成错误: {str(e)}"

def video_analysis_demo():
    """视频分析演示"""
    print("=== 视频分析演示 ===")
    
    # 加载模型
    print("正在加载 Qwen-Omni 视频分析模型...")
    model = load_video_analysis_model()
    print("Qwen-Omni 模型加载完成!")
    
    # 视频路径
    video_path = input("请输入视频文件路径: ").strip()
    
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return
    
    # 提取关键帧
    print("正在提取视频关键帧...")
    frames = extract_video_keyframes(video_path, num_frames=5)
    
    if not frames:
        print("关键帧提取失败")
        return
    
    print(f"成功提取 {len(frames)} 个关键帧")
    
    while True:
        print("\n请选择功能:")
        print("1. 视频内容描述")
        print("2. 视频问答")
        print("3. 视频摘要生成")
        print("4. 退出")
        
        choice = input("请输入选择 (1-4): ").strip()
        
        if choice == "1":
            # 视频内容描述
            print("\n--- 视频内容描述 ---")
            description = describe_video_content(frames, model)
            print(f"视频描述:\n{description}")
            
        elif choice == "2":
            # 视频问答
            print("\n--- 视频问答 ---")
            question = input("请输入关于视频的问题: ").strip()
            if question:
                answer = video_qa_demo(frames, question, model)
                print(f"回答: {answer}")
            
        elif choice == "3":
            # 视频摘要生成
            print("\n--- 视频摘要生成 ---")
            summary = generate_video_summary(frames, model)
            print(f"视频摘要:\n{summary}")
            
        elif choice == "4":
            print("退出视频分析演示")
            # 清理临时文件
            for frame_path in frames:
                try:
                    os.unlink(frame_path)
                except:
                    pass
            break
        else:
            print("无效选择，请重新输入")

def main():
    """主函数 - 视频分析演示"""
    video_analysis_demo()

if __name__ == "__main__":
    main()