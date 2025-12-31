#!/usr/bin/env python3
"""
测试 DeepSeek API 配置
"""

import os
from dotenv import load_dotenv
from deepseek_r1_traces_v2 import DeepSeekR1AgentV2

load_dotenv(override=True)

def main():
    print("="*60)
    print("测试 DeepSeek API 配置")
    print("="*60)

    # 检查配置
    use_backend = os.getenv("USE_BACKEND", "ollama")
    print(f"\n当前使用的后端: {use_backend}")

    if use_backend == "deepseek_api":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        model = os.getenv("DEEPSEEK_MODEL")

        print(f"API Key: {api_key[:20]}... (已隐藏)")
        print(f"Base URL: {base_url}")
        print(f"Model: {model}")

        # 创建 Agent
        print("\n创建 V2.5 Agent...")
        agent = DeepSeekR1AgentV2(
            model=model,
            enable_tools=True,
            enable_loop_detection=True,
            enable_hallucination_detection=False
        )

        # 简单测试
        print("\n" + "="*60)
        print("测试简单问题")
        print("="*60)
        test_question = "1 + 1 = ?"

        print(f"\n问题: {test_question}")
        print("="*60)

        try:
            answer = agent.run(
                test_question,
                mode="single_think",
                verbose=True
            )

            print("\n" + "="*60)
            print("最终答案:")
            print("="*60)
            print(answer)

            print("\n✅ DeepSeek API 配置成功!")

        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("\n当前使用 Ollama 本地模式,不测试 DeepSeek API")
        print("如需测试 DeepSeek API,请在 .env 中设置: USE_BACKEND=deepseek_api")

if __name__ == "__main__":
    main()
