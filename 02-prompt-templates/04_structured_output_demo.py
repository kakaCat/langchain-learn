#!/usr/bin/env python3
"""
结构化输出示例：JSON 模式输出、Pydantic 模型约束、函数调用模板
"""

from __future__ import annotations

import os
from typing import List, Optional
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_openai import ChatOpenAI

# 从当前模块目录加载 .env，避免在仓库根运行时找不到配置
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

# 定义 Pydantic 模型用于结构化输出
class UserInfo(BaseModel):
    """用户信息模型"""
    name: str = Field(description="用户姓名")
    age: int = Field(description="用户年龄", ge=0, le=150)
    email: str = Field(description="用户邮箱")
    interests: List[str] = Field(description="用户兴趣列表")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "张三",
                "age": 25,
                "email": "zhangsan@example.com",
                "interests": ["编程", "阅读", "运动"]
            }
        }

def json_output_demo() -> None:
    """JSON 模式输出示例"""
    print("=== JSON 模式输出示例 ===")
    
    # 创建 JSON 输出解析器
    parser = JsonOutputParser()
    
    # 创建提示词模板
    template = """
    请从以下文本中提取用户信息，并以 JSON 格式输出：
    
    文本：{text}
    
    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # 示例文本
    text = """
    我叫李四，今年30岁，邮箱是 lisi@example.com。
    我喜欢打篮球、看电影和听音乐。
    """
    
    # 格式化提示词
    formatted_prompt = prompt.format(text=text)
    print("提示词：")
    print(formatted_prompt)
    print("\n" + "="*50 + "\n")

def pydantic_output_demo() -> None:
    """Pydantic 模型约束输出示例"""
    print("=== Pydantic 模型约束输出示例 ===")
    
    # 创建 Pydantic 输出解析器
    parser = PydanticOutputParser(pydantic_object=UserInfo)
    
    # 创建提示词模板
    template = """
    请从以下文本中提取用户信息，并按照指定格式输出：
    
    文本：{text}
    
    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # 示例文本
    text = """
    用户信息：
    姓名：王五
    年龄：28
    邮箱：wangwu@example.com
    兴趣：旅游、摄影、美食
    """
    
    # 格式化提示词
    formatted_prompt = prompt.format(text=text)
    print("提示词：")
    print(formatted_prompt)
    print("\n" + "="*50 + "\n")

def function_calling_demo() -> None:
    """函数调用模板示例"""
    print("=== 函数调用模板示例 ===")
    
    # 定义函数调用模板
    template = """
    根据用户查询，调用相应的函数：
    
    查询：{query}
    
    可用的函数：
    - get_weather(location: str): 获取天气信息
    - search_products(keyword: str): 搜索产品
    - calculate_price(quantity: int, unit_price: float): 计算价格
    
    请选择最合适的函数并返回函数调用信息。
    """
    
    prompt = PromptTemplate.from_template(template)
    
    # 示例查询
    queries = [
        "今天北京的天气怎么样？",
        "我想买一台笔记本电脑",
        "计算购买5件单价为29.9元的商品总价"
    ]
    
    for query in queries:
        formatted_prompt = prompt.format(query=query)
        print(f"查询：{query}")
        print("提示词：")
        print(formatted_prompt)
        print("-" * 30)

def main() -> None:
    """主函数"""
    print("结构化输出示例演示\n")
    
    # 演示 JSON 模式输出
    json_output_demo()
    
    # 演示 Pydantic 模型约束输出
    pydantic_output_demo()
    
    # 演示函数调用模板
    function_calling_demo()

if __name__ == "__main__":
    main()