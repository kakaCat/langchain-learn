#!/usr/bin/env python3
"""
演示 LLM 意图识别 vs 规则意图识别的对比
展示 LLM 在复杂场景下的优势
"""

import os
from datetime import datetime
from typing import Dict, List
import uuid

# 从主文件导入
# 注意：需要先运行 01_smart_customer_service.py 或将其作为模块导入
import sys
sys.path.append('.')

try:
    from 01_smart_customer_service import (
        IntegratedCustomerService,
        CustomerMessage
    )
except ImportError:
    # 如果作为脚本运行，尝试直接导入
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "smart_customer_service",
        "01_smart_customer_service.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    IntegratedCustomerService = module.IntegratedCustomerService
    CustomerMessage = module.CustomerMessage


def compare_intent_recognition():
    """对比 LLM 和规则两种意图识别方法"""
    print("=" * 80)
    print("意图识别方法对比演示")
    print("=" * 80)

    # 创建两个客服系统：一个使用 LLM，一个使用规则
    service_llm = IntegratedCustomerService(use_llm_intent=True)
    service_rules = IntegratedCustomerService(use_llm_intent=False)

    # 测试用例：包含一些复杂的、需要上下文理解的场景
    test_cases = [
        {
            "customer_id": "test_001",
            "message": "我昨天买的手机一直开不了机，太生气了！马上给我退款！",
            "description": "复杂场景：既涉及技术支持，又涉及退款，且情绪激动"
        },
        {
            "customer_id": "test_002",
            "message": "你们这是什么破服务？我等了三天订单还没发货，投诉！",
            "description": "负面情绪 + 订单问题 + 投诉意图"
        },
        {
            "customer_id": "test_003",
            "message": "请问我能不能取消刚刚下的订单 ORD789012？",
            "description": "礼貌询问 + 退款/取消意图 + 实体提取"
        },
        {
            "customer_id": "test_004",
            "message": "我这边支付的时候钱扣了但是订单显示未支付，怎么办？",
            "description": "支付问题描述（无明显关键词）"
        },
        {
            "customer_id": "test_005",
            "message": "能帮我查一下为什么我的会员积分没有到账吗？",
            "description": "一般咨询（关于积分系统）"
        },
        {
            "customer_id": "test_006",
            "message": "app老是闪退，根本没法用，这怎么整？",
            "description": "口语化的技术支持请求"
        },
        {
            "customer_id": "test_007",
            "message": "我要立即和你们经理谈谈这个质量问题，这太不像话了！",
            "description": "高紧急度 + 负面情绪 + 投诉升级"
        }
    ]

    results_comparison = []

    for test_case in test_cases:
        customer_id = test_case["customer_id"]
        message = test_case["message"]
        description = test_case["description"]

        print(f"\n{'=' * 80}")
        print(f"测试场景: {description}")
        print(f"客户消息: {message}")
        print(f"\n{'-' * 80}")

        # LLM 方法
        print("🤖 LLM 意图识别结果:")
        result_llm = service_llm.process_customer_message(customer_id + "_llm", message)
        print(f"  意图: {result_llm['current_intent']}")
        print(f"  置信度: {result_llm['confidence']:.2f}")
        print(f"  提取实体: {result_llm['collected_info']}")
        print(f"  建议行动: {', '.join(result_llm['suggested_actions'][:3])}")
        print(f"  需要人工: {'是' if result_llm['requires_human'] else '否'}")

        print(f"\n{'-' * 80}")

        # 规则方法
        print("📋 规则意图识别结果:")
        result_rules = service_rules.process_customer_message(customer_id + "_rules", message)
        print(f"  意图: {result_rules['current_intent']}")
        print(f"  置信度: {result_rules['confidence']:.2f}")
        print(f"  提取实体: {result_rules['collected_info']}")
        print(f"  建议行动: {', '.join(result_rules['suggested_actions'][:3])}")
        print(f"  需要人工: {'是' if result_rules['requires_human'] else '否'}")

        # 记录对比结果
        results_comparison.append({
            "message": message,
            "description": description,
            "llm": {
                "intent": result_llm['current_intent'],
                "confidence": result_llm['confidence'],
                "entities": result_llm['collected_info']
            },
            "rules": {
                "intent": result_rules['current_intent'],
                "confidence": result_rules['confidence'],
                "entities": result_rules['collected_info']
            }
        })

    # 生成对比总结
    print(f"\n{'=' * 80}")
    print("对比总结")
    print(f"{'=' * 80}\n")

    intent_match_count = sum(
        1 for r in results_comparison
        if r['llm']['intent'] == r['rules']['intent']
    )

    print(f"总测试用例数: {len(test_cases)}")
    print(f"意图识别一致: {intent_match_count}/{len(test_cases)}")
    print(f"意图识别差异: {len(test_cases) - intent_match_count}/{len(test_cases)}")

    avg_confidence_llm = sum(r['llm']['confidence'] for r in results_comparison) / len(results_comparison)
    avg_confidence_rules = sum(r['rules']['confidence'] for r in results_comparison) / len(results_comparison)

    print(f"\nLLM 平均置信度: {avg_confidence_llm:.2f}")
    print(f"规则 平均置信度: {avg_confidence_rules:.2f}")

    # 展示差异案例
    print("\n差异案例分析:")
    for i, result in enumerate(results_comparison):
        if result['llm']['intent'] != result['rules']['intent']:
            print(f"\n  案例 {i + 1}: {result['description']}")
            print(f"    消息: {result['message']}")
            print(f"    LLM 识别: {result['llm']['intent']} (置信度: {result['llm']['confidence']:.2f})")
            print(f"    规则识别: {result['rules']['intent']} (置信度: {result['rules']['confidence']:.2f})")

    print(f"\n{'=' * 80}")
    print("LLM 意图识别的优势:")
    print(f"{'=' * 80}")
    print("""
1. 上下文理解能力强
   - 能够理解复杂、口语化的表达
   - 可以处理一句话包含多个意图的情况

2. 更准确的情感和紧急程度分析
   - 能够识别隐含的情绪
   - 基于上下文判断紧急程度

3. 更灵活的实体提取
   - 不依赖固定的正则表达式模式
   - 可以理解同义词和变体

4. 更智能的建议行动
   - 基于全面的意图和情感分析
   - 考虑多维度因素

5. 自适应和可扩展性
   - 无需手动维护规则库
   - 可以通过 prompt 工程快速调整
    """)

    print(f"{'=' * 80}")
    print("演示完成！")
    print(f"{'=' * 80}")


def demo_llm_intent_with_history():
    """演示 LLM 意图识别在多轮对话中的表现"""
    print("\n" + "=" * 80)
    print("多轮对话场景演示")
    print("=" * 80)

    service = IntegratedCustomerService(use_llm_intent=True)

    # 模拟一个完整的客服对话流程
    conversation = [
        ("customer_999", "你好，我的订单有点问题"),
        ("customer_999", "订单号是 ORD888999，已经三天了还没发货"),
        ("customer_999", "这个太慢了，我很着急用，能不能加急？"),
        ("customer_999", "算了，我还是取消退款吧"),
    ]

    print("\n多轮对话流程:")
    for i, (customer_id, message) in enumerate(conversation, 1):
        print(f"\n第 {i} 轮对话:")
        print(f"客户: {message}")

        result = service.process_customer_message(customer_id, message)

        print(f"系统响应: {result['system_response']}")
        print(f"识别意图: {result['current_intent']}")
        print(f"当前状态: {result['next_state']}")
        print(f"收集信息: {result['collected_info']}")
        print(f"置信度: {result['confidence']:.2f}")

    print(f"\n{'=' * 80}")
    print("多轮对话演示完成！")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: 未设置 OPENAI_API_KEY 环境变量")
        print("LLM 意图识别将无法工作，系统会自动回退到规则方法")
        print("\n请设置环境变量:")
        print('export OPENAI_API_KEY="your-api-key-here"')
        print()

    # 运行对比演示
    compare_intent_recognition()

    # 运行多轮对话演示
    demo_llm_intent_with_history()
