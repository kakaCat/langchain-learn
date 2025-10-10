"""
07 - 性能优化 Demo (Performance Optimization)

演示模板缓存、批量处理和内存优化策略。
"""

import time
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
from functools import lru_cache
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def get_llm():
    """获取 LLM 模型实例"""
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE")
    )

# 模板缓存装饰器
@lru_cache(maxsize=100)
def get_cached_template(template_text: str, input_variables: tuple):
    """
    缓存模板实例，避免重复创建
    """
    return PromptTemplate(
        template=template_text,
        input_variables=list(input_variables)
    )

def template_caching_demo():
    """
    模板缓存演示
    """
    print("=== 模板缓存演示 ===")
    
    # 相同的模板文本和变量
    template_text = "请为{product}写一段广告文案，目标客户是{audience}。"
    input_vars = ("product", "audience")
    
    # 第一次调用 - 创建新模板
    start_time = time.time()
    template1 = get_cached_template(template_text, input_vars)
    time1 = time.time() - start_time
    
    # 第二次调用 - 使用缓存模板
    start_time = time.time()
    template2 = get_cached_template(template_text, input_vars)
    time2 = time.time() - start_time
    
    print(f"第一次调用耗时: {time1:.6f} 秒")
    print(f"第二次调用耗时: {time2:.6f} 秒")
    print(f"缓存加速比: {time1/time2:.2f}x")
    
    # 验证模板功能
    formatted_text = template1.format(product="智能手机", audience="年轻人")
    print(f"\n格式化结果: {formatted_text}")

def batch_processing_demo():
    """
    批量处理演示
    """
    print("\n=== 批量处理演示 ===")
    
    # 创建基础模板
    review_template = PromptTemplate(
        template="请分析以下产品评论的情感倾向：{review}",
        input_variables=["review"]
    )
    
    # 批量评论数据
    reviews = [
        "这个产品非常好用，质量很棒！",
        "不太满意，功能比预期的要少。",
        "性价比很高，推荐购买。",
        "物流太慢了，等了很久才收到。",
        "外观设计很漂亮，使用体验不错。"
    ]
    
    # 单次处理
    print("单次处理模式:")
    single_start = time.time()
    single_prompts = []
    for review in reviews:
        prompt = review_template.format(review=review)
        single_prompts.append(prompt)
    single_time = time.time() - single_start
    
    # 批量处理
    print("批量处理模式:")
    batch_start = time.time()
    batch_prompts = review_template.format_prompt(review=reviews)
    batch_time = time.time() - batch_start
    
    print(f"单次处理耗时: {single_time:.6f} 秒")
    print(f"批量处理耗时: {batch_time:.6f} 秒")
    print(f"批量处理加速比: {single_time/batch_time:.2f}x")
    
    print(f"\n生成的提示词数量: {len(single_prompts)}")
    print("第一个提示词示例:", single_prompts[0][:50] + "...")

def memory_optimization_demo():
    """
    内存优化演示
    """
    print("\n=== 内存优化演示 ===")
    
    # 大型模板数据集
    large_templates = []
    
    # 方法1: 直接存储所有模板（内存消耗大）
    print("方法1 - 直接存储所有模板:")
    start_memory = get_memory_usage()
    
    for i in range(1000):
        template = PromptTemplate(
            template=f"这是第{i}个模板，用于处理{{data}}数据。",
            input_variables=["data"]
        )
        large_templates.append(template)
    
    end_memory = get_memory_usage()
    memory_used = end_memory - start_memory
    print(f"内存使用量: {memory_used:.2f} MB")
    
    # 方法2: 使用模板工厂（内存优化）
    print("\n方法2 - 使用模板工厂:")
    start_memory = get_memory_usage()
    
    def template_factory(template_id: int, data: str):
        """按需创建模板"""
        template_text = f"这是第{template_id}个模板，用于处理{data}数据。"
        return PromptTemplate(
            template=template_text,
            input_variables=[]
        )
    
    # 只存储模板ID，需要时再创建
    template_ids = list(range(1000))
    end_memory = get_memory_usage()
    memory_used = end_memory - start_memory
    print(f"内存使用量: {memory_used:.2f} MB")
    
    # 演示按需使用
    sample_template = template_factory(1, "示例")
    print(f"\n按需创建的模板示例: {sample_template.template}")

def get_memory_usage():
    """
    获取当前进程内存使用量（MB）
    """
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def lazy_loading_demo():
    """
    懒加载演示
    """
    print("\n=== 懒加载演示 ===")
    
    class LazyTemplateManager:
        """懒加载模板管理器"""
        
        def __init__(self):
            self._templates = {}
            
        def get_template(self, name: str, template_text: str, input_vars: list):
            """获取模板，如果不存在则创建"""
            if name not in self._templates:
                print(f"创建新模板: {name}")
                self._templates[name] = PromptTemplate(
                    template=template_text,
                    input_variables=input_vars
                )
            else:
                print(f"使用缓存模板: {name}")
            return self._templates[name]
    
    # 使用懒加载管理器
    manager = LazyTemplateManager()
    
    # 第一次获取 - 创建新模板
    template1 = manager.get_template(
        "ad_template",
        "请为{product}写广告文案，面向{audience}。",
        ["product", "audience"]
    )
    
    # 第二次获取 - 使用缓存模板
    template2 = manager.get_template(
        "ad_template", 
        "请为{product}写广告文案，面向{audience}。",
        ["product", "audience"]
    )
    
    # 验证模板功能
    result = template1.format(product="笔记本电脑", audience="学生")
    print(f"\n模板格式化结果: {result}")
    print(f"管理的模板数量: {len(manager._templates)}")

def main():
    """主函数"""
    print("性能优化演示程序")
    print("=" * 50)
    
    # 演示模板缓存
    template_caching_demo()
    
    # 演示批量处理
    batch_processing_demo()
    
    # 演示内存优化
    memory_optimization_demo()
    
    # 演示懒加载
    lazy_loading_demo()
    
    print("\n" + "=" * 50)
    print("性能优化演示完成！")

if __name__ == "__main__":
    main()