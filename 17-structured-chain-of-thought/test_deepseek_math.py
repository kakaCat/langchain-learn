#!/usr/bin/env python3
"""
测试 DeepSeek API - 数学问题
"""

import os
from dotenv import load_dotenv
from deepseek_r1_traces_v2 import DeepSeekR1AgentV2

load_dotenv(override=True)

def main():
    print("="*60)
    print("DeepSeek API 数学问题测试")
    print("="*60)

    use_backend = os.getenv("USE_BACKEND", "ollama")

    if use_backend == "deepseek_api":
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
        print(f"\n使用模型: {model_name} (DeepSeek API)")
    else:
        model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:32b")
        print(f"\n使用模型: {model_name} (Ollama)")

    # 创建 Agent
    agent = DeepSeekR1AgentV2(
        model=model_name,
        enable_tools=True,
        enable_loop_detection=True,
        enable_hallucination_detection=False
    )

    # 测试一个简单的数学问题
    question = "A farmer has 15 chickens and 20 ducks. If each chicken lays 2 eggs per day and each duck lays 1 egg per day, how many eggs are laid in total per day?"

    print(f"\n问题: {question}")
    print("="*60)

    try:
        answer = agent.run(
            question,
            mode="structured_4stage",
            verbose=True
        )

        print("\n" + "="*60)
        print("最终答案:")
        print("="*60)
        print(answer)

        # 正确答案应该是 15*2 + 20*1 = 30 + 20 = 50
        if "50" in answer:
            print("\n✅ 答案正确!")
        else:
            print("\n❌ 答案可能不正确")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
