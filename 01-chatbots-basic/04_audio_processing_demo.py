"""
音频处理示例 - 使用 Qwen-Omni 实现语音识别和语音合成

功能：
- 语音识别（语音转文本）
- 语音合成（文本转语音）
- 音频内容分析
"""

import os
import tempfile
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_community.document_loaders import AudioLoader
from langchain_community.utilities import SpeechToText
from langchain_community.utilities import TextToSpeech

def load_audio_processing_models():
    """加载音频处理模型"""
    # 语音识别模型
    stt_model = SpeechToText()
    
    # 语音合成模型
    tts_model = TextToSpeech()
    
    # 文本分析模型 - 使用 Qwen-Omni
    text_model = ChatOpenAI(
        model="qwen-omni",
        api_key=os.getenv("QWEN_API_KEY"),
        base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        max_tokens=1024,
        temperature=0
    )
    
    return stt_model, tts_model, text_model

def speech_to_text(audio_path, stt_model):
    """语音转文本"""
    try:
        # 使用语音识别模型
        text = stt_model.transcribe(audio_path)
        return text
    except Exception as e:
        return f"语音识别错误: {str(e)}"

def text_to_speech(text, tts_model, output_path=None):
    """文本转语音"""
    try:
        if output_path is None:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                output_path = temp_file.name
        
        # 使用语音合成模型
        tts_model.synthesize(text, output_path)
        return output_path
    except Exception as e:
        return f"语音合成错误: {str(e)}"

def analyze_audio_content(text, text_model):
    """分析音频内容"""
    try:
        message = HumanMessage(content=f"请分析以下音频转录内容:\n\n{text}\n\n请总结主要内容、情感倾向和关键信息。")
        response = text_model.invoke([message])
        analysis = response.content
        return analysis
    except Exception as e:
        return f"内容分析错误: {str(e)}"

def audio_processing_demo():
    """音频处理演示"""
    print("=== 音频处理演示 ===")
    
    # 加载模型
    print("正在加载 Qwen-Omni 音频处理模型...")
    stt_model, tts_model, text_model = load_audio_processing_models()
    print("Qwen-Omni 模型加载完成!")
    
    while True:
        print("\n请选择功能:")
        print("1. 语音转文本")
        print("2. 文本转语音")
        print("3. 音频内容分析")
        print("4. 退出")
        
        choice = input("请输入选择 (1-4): ").strip()
        
        if choice == "1":
            # 语音转文本
            audio_path = input("请输入音频文件路径: ").strip()
            if os.path.exists(audio_path):
                text = speech_to_text(audio_path, stt_model)
                print(f"\n转录文本: {text}")
            else:
                print("音频文件不存在")
                
        elif choice == "2":
            # 文本转语音
            text = input("请输入要转换为语音的文本: ").strip()
            if text:
                output_path = text_to_speech(text, tts_model)
                print(f"\n语音文件已生成: {output_path}")
                print("请使用音频播放器打开文件收听")
                
        elif choice == "3":
            # 音频内容分析
            audio_path = input("请输入音频文件路径: ").strip()
            if os.path.exists(audio_path):
                # 先转录
                text = speech_to_text(audio_path, stt_model)
                print(f"\n转录文本: {text}")
                
                # 再分析
                analysis = analyze_audio_content(text, text_model)
                print(f"\n内容分析:\n{analysis}")
            else:
                print("音频文件不存在")
                
        elif choice == "4":
            print("退出音频处理演示")
            break
        else:
            print("无效选择，请重新输入")

def main():
    """主函数 - 音频处理演示"""
    audio_processing_demo()

if __name__ == "__main__":
    main()