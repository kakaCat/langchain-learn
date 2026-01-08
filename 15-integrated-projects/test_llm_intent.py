#!/usr/bin/env python3
"""
快速测试 LLM 意图识别功能
"""

import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_llm_intent():
    """测试 LLM 意图识别"""
    print("=" * 60)
    print("LLM 意图识别快速测试")
    print("=" * 60)

    # 检查 API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  警告: 未检测到 OPENAI_API_KEY 环境变量")
        print("系统将使用规则方法进行意图识别\n")
        use_llm = False
    else:
        print("\n✅ 检测到 OPENAI_API_KEY，将使用 LLM 进行意图识别\n")
        use_llm = True

    try:
        # 导入模块
        print("正在导入模块...")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "smart_customer_service",
            os.path.join(os.path.dirname(__file__), "01_smart_customer_service.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        IntegratedCustomerService = module.IntegratedCustomerService
        print("✅ 模块导入成功\n")

        # 创建服务
        print(f"正在创建客服系统 ({'LLM模式' if use_llm else '规则模式'})...")
        service = IntegratedCustomerService(use_llm_intent=use_llm)
        print("✅ 系统创建成功\n")

        # 测试用例
        test_messages = [
            "我的订单 ORD123 还没发货，很着急！",
            "产品质量太差了，我要投诉！",
            "请问如何使用这个功能？"
        ]

        print("开始测试...\n")
        print("-" * 60)

        for i, message in enumerate(test_messages, 1):
            print(f"\n测试 {i}: {message}")

            try:
                result = service.process_customer_message(
                    customer_id=f"test_{i}",
                    message_content=message
                )

                print(f"  ✓ 意图: {result['current_intent']}")
                print(f"  ✓ 置信度: {result['confidence']:.2f}")
                print(f"  ✓ 实体: {result['collected_info']}")
                print(f"  ✓ 响应: {result['system_response'][:50]}...")

            except Exception as e:
                print(f"  ✗ 错误: {str(e)}")

        print("\n" + "-" * 60)
        print("\n✅ 测试完成！")

        if use_llm:
            print("\n💡 提示: 系统正在使用 LLM 进行意图识别")
            print("   - 更准确的意图理解")
            print("   - 更智能的实体提取")
            print("   - 更全面的情感分析")
        else:
            print("\n💡 提示: 系统正在使用规则方法进行意图识别")
            print("   - 要使用 LLM，请设置 OPENAI_API_KEY 环境变量")
            print("   - export OPENAI_API_KEY='your-key-here'")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_llm_intent()
    sys.exit(0 if success else 1)
