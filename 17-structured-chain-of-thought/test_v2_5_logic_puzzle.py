#!/usr/bin/env python3
"""
V2.5 专项测试 - Knights and Knaves 逻辑谜题
只测试V1失败的关键案例
"""

import os
from dotenv import load_dotenv
from deepseek_r1_traces_v2 import DeepSeekR1AgentV2

load_dotenv(override=True)

def main():
    print("=" * 60)
    print("V2.5 Knights and Knaves 逻辑谜题专项测试")
    print("=" * 60)

    # 创建 V2.5 Agent (根据 USE_BACKEND 自动选择模型)
    use_backend = os.getenv("USE_BACKEND", "ollama")

    if use_backend == "deepseek_api":
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
        print(f"\n使用模型: {model_name} (DeepSeek API)")
    else:
        model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:32b")
        print(f"\n使用模型: {model_name} (Ollama)")

    agent = DeepSeekR1AgentV2(
        model=model_name,
        enable_tools=True,
        enable_loop_detection=True,
        enable_hallucination_detection=False
    )

    # V1 失败的关键案例
    question = """Three people (A, B, C) are either Knights (always tell truth) or Knaves (always lie).
A says: 'B is a knave'.
B says: 'A and C are the same type'.
C says: 'I am a Knight'.
Determine who is who."""

    print(f"\n问题:\n{question}")
    print(f"\n正确答案: A is a Knight, B is a Knave, C is a Knave")
    print(f"V1 错误答案: A is a Knave, B is a Knight, C is a Knave")
    print("\n" + "=" * 60)
    print("开始 V2.5 推理 (structured_4stage 模式)")
    print("=" * 60 + "\n")

    try:
        # 强制使用 structured_4stage 模式
        answer = agent.run(
            question,
            mode="structured_4stage",
            verbose=True  # 显示详细推理过程
        )

        print("\n" + "=" * 60)
        print("V2.5 最终答案:")
        print("=" * 60)
        print(answer)
        print("\n" + "=" * 60)

        # 简单判断是否正确
        if "A is a Knight" in answer or "A=Knight" in answer or "A: Knight" in answer:
            if "B is a Knave" in answer or "B=Knave" in answer or "B: Knave" in answer:
                if "C is a Knave" in answer or "C=Knave" in answer or "C: Knave" in answer:
                    print("✅ V2.5 答案正确！成功修复 V1 的逻辑一致性问题！")
                    return

        print("❌ V2.5 答案可能不正确，需要人工检查")

    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
