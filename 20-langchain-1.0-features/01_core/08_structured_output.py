"""
LangChain 1.0 - 结构化输出 (Structured Output)

严格按照官方文档实现
https://docs.langchain.org.cn/oss/python/langchain/structured-output

核心功能:
1. with_structured_output() - 强制结构化输出
2. Pydantic 模型定义
3. 信息提取
4. 表单填充
5. 数据验证

参考文档:
- https://python.langchain.com/docs/how_to/structured_output/
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, ValidationError

load_dotenv()


# ========== 1. 基础结构化输出 ==========
def demo_basic_structured_output():
    """最简单的结构化输出"""
    print("=" * 70)
    print("1. 基础结构化输出")
    print("=" * 70)

    # 定义输出结构
    class Person(BaseModel):
        name: str = Field(description="人名")
        age: int = Field(description="年龄")

    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(Person)

    # 提取信息
    text = "我叫张三，今年25岁"
    result = structured_llm.invoke(text)

    print(f"\n输入: {text}")
    print(f"输出类型: {type(result)}")
    print(f"姓名: {result.name}")
    print(f"年龄: {result.age}")


# ========== 2. 复杂结构提取 ==========
def demo_complex_extraction():
    """复杂结构的信息提取"""
    print("\n\n" + "=" * 70)
    print("2. 复杂结构提取")
    print("=" * 70)

    class Address(BaseModel):
        city: str = Field(description="城市")
        district: Optional[str] = Field(default=None, description="区/县")

    class UserInfo(BaseModel):
        name: str = Field(description="姓名")
        age: int = Field(ge=0, le=150, description="年龄(0-150)")
        email: str = Field(description="邮箱")
        address: Address = Field(description="地址")
        hobbies: List[str] = Field(default=[], description="爱好列表")

    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(UserInfo)

    text = """
    我叫李四，今年30岁，邮箱是lisi@example.com。
    我住在北京市朝阳区。
    我喜欢游泳、阅读和旅行。
    """

    result = structured_llm.invoke(text)

    print(f"\n提取结果:")
    print(f"  姓名: {result.name}")
    print(f"  年龄: {result.age}")
    print(f"  邮箱: {result.email}")
    print(f"  城市: {result.address.city}")
    print(f"  区县: {result.address.district}")
    print(f"  爱好: {', '.join(result.hobbies)}")


# ========== 3. 多个实体提取 ==========
def demo_multiple_entities():
    """提取多个实体"""
    print("\n\n" + "=" * 70)
    print("3. 多个实体提取")
    print("=" * 70)

    class Person(BaseModel):
        name: str = Field(description="人名")
        title: str = Field(description="职位")

    class Meeting(BaseModel):
        topic: str = Field(description="会议主题")
        participants: List[Person] = Field(description="参会人员")
        date: str = Field(description="会议日期")

    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(Meeting)

    text = """
    明天下午2点召开产品规划会议。
    参会人员包括：
    - CEO 张总
    - CTO 王工
    - 产品经理 李明
    会议日期是2024年12月20日。
    """

    result = structured_llm.invoke(text)

    print(f"\n会议信息:")
    print(f"  主题: {result.topic}")
    print(f"  日期: {result.date}")
    print(f"  参会人员:")
    for p in result.participants:
        print(f"    - {p.title} {p.name}")


# ========== 4. 数据验证 ==========
def demo_validation():
    """数据验证示例"""
    print("\n\n" + "=" * 70)
    print("4. 数据验证")
    print("=" * 70)

    class Product(BaseModel):
        name: str = Field(min_length=1, description="产品名称")
        price: float = Field(gt=0, description="价格(必须大于0)")
        quantity: int = Field(ge=0, description="库存(非负)")
        category: str = Field(description="分类")

    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(Product)

    text = "iPhone 15 Pro售价7999元，库存50台，属于电子产品类别"

    try:
        result = structured_llm.invoke(text)
        print(f"\n✅ 验证通过:")
        print(f"  产品: {result.name}")
        print(f"  价格: ¥{result.price}")
        print(f"  库存: {result.quantity}")
        print(f"  分类: {result.category}")
    except ValidationError as e:
        print(f"\n❌ 验证失败:")
        print(e)


# ========== 5. 分类任务 ==========
def demo_classification():
    """文本分类"""
    print("\n\n" + "=" * 70)
    print("5. 文本分类")
    print("=" * 70)

    class Classification(BaseModel):
        category: str = Field(description="分类: 技术/商业/娱乐/体育")
        confidence: float = Field(ge=0, le=1, description="置信度(0-1)")
        keywords: List[str] = Field(description="关键词")

    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(Classification)

    texts = [
        "OpenAI发布了最新的GPT-4模型，性能大幅提升",
        "苹果公司股价创历史新高，市值突破3万亿美元",
        "中国队在世界杯预选赛中1:0击败对手"
    ]

    print("\n分类结果:")
    for text in texts:
        result = structured_llm.invoke(text)
        print(f"\n  文本: {text[:30]}...")
        print(f"  分类: {result.category}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  关键词: {', '.join(result.keywords)}")


# ========== 6. 表单填充 ==========
def demo_form_filling():
    """自动填充表单"""
    print("\n\n" + "=" * 70)
    print("6. 表单填充")
    print("=" * 70)

    class OrderForm(BaseModel):
        customer_name: str = Field(description="客户姓名")
        phone: str = Field(description="联系电话")
        product: str = Field(description="产品名称")
        quantity: int = Field(gt=0, description="数量")
        delivery_address: str = Field(description="送货地址")
        notes: Optional[str] = Field(default=None, description="备注")

    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(OrderForm)

    text = """
    客户王小明来电订购，电话是13800138000。
    需要iPhone 15 Pro两台，送到上海市浦东新区张江高科技园区。
    请尽快发货。
    """

    result = structured_llm.invoke(text)

    print(f"\n订单表单:")
    print(f"  客户: {result.customer_name}")
    print(f"  电话: {result.phone}")
    print(f"  产品: {result.product}")
    print(f"  数量: {result.quantity}")
    print(f"  地址: {result.delivery_address}")
    print(f"  备注: {result.notes or '无'}")


# ========== 7. 情感分析 ==========
def demo_sentiment_analysis():
    """结构化情感分析"""
    print("\n\n" + "=" * 70)
    print("7. 情感分析")
    print("=" * 70)

    class Sentiment(BaseModel):
        sentiment: str = Field(description="情感倾向: 积极/中性/消极")
        score: float = Field(ge=-1, le=1, description="情感分数(-1到1)")
        aspects: List[str] = Field(description="提及的方面")

    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(Sentiment)

    reviews = [
        "这款手机性能很好，拍照清晰，但是电池续航一般",
        "服务态度很差，等了很久，非常失望",
        "产品质量不错，物流也很快"
    ]

    print("\n情感分析结果:")
    for review in reviews:
        result = structured_llm.invoke(review)
        print(f"\n  评论: {review}")
        print(f"  情感: {result.sentiment}")
        print(f"  分数: {result.score:.2f}")
        print(f"  方面: {', '.join(result.aspects)}")


# ========== 8. 实体关系提取 ==========
def demo_relation_extraction():
    """提取实体关系"""
    print("\n\n" + "=" * 70)
    print("8. 实体关系提取")
    print("=" * 70)

    class Relation(BaseModel):
        subject: str = Field(description="主体")
        predicate: str = Field(description="关系")
        object_: str = Field(description="客体", alias="object")

    class Knowledge(BaseModel):
        relations: List[Relation] = Field(description="关系三元组列表")

    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(Knowledge)

    text = """
    张三是ABC公司的CEO。
    ABC公司位于北京。
    ABC公司成立于2010年。
    """

    result = structured_llm.invoke(text)

    print(f"\n提取的知识:")
    for rel in result.relations:
        print(f"  {rel.subject} --[{rel.predicate}]--> {rel.object_}")


# ========== 9. JSON模式 ==========
def demo_json_mode():
    """JSON模式输出"""
    print("\n\n" + "=" * 70)
    print("9. JSON模式输出")
    print("=" * 70)

    class Recipe(BaseModel):
        name: str = Field(description="菜名")
        ingredients: List[str] = Field(description="食材列表")
        steps: List[str] = Field(description="步骤")
        cooking_time: int = Field(description="烹饪时间(分钟)")

    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(Recipe)

    result = structured_llm.invoke("告诉我西红柿炒鸡蛋的做法")

    print(f"\n菜谱:")
    print(f"  菜名: {result.name}")
    print(f"  食材: {', '.join(result.ingredients)}")
    print(f"  步骤:")
    for i, step in enumerate(result.steps, 1):
        print(f"    {i}. {step}")
    print(f"  烹饪时间: {result.cooking_time}分钟")


# ========== 10. 批量提取 ==========
def demo_batch_extraction():
    """批量结构化提取"""
    print("\n\n" + "=" * 70)
    print("10. 批量提取")
    print("=" * 70)

    class Contact(BaseModel):
        name: str = Field(description="姓名")
        company: str = Field(description="公司")
        email: str = Field(description="邮箱")

    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(Contact)

    texts = [
        "张三，ABC科技公司，邮箱zhangsan@abc.com",
        "李四，XYZ企业，联系方式lisi@xyz.com",
        "王五在DEF公司工作，邮箱是wangwu@def.com"
    ]

    results = structured_llm.batch(texts)

    print(f"\n批量提取 {len(results)} 条联系人:")
    for i, result in enumerate(results, 1):
        print(f"\n  {i}. {result.name}")
        print(f"     公司: {result.company}")
        print(f"     邮箱: {result.email}")


# ========== 11. 最佳实践 ==========
def best_practices():
    """结构化输出最佳实践"""
    print("\n\n" + "=" * 70)
    print("11. 最佳实践")
    print("=" * 70)

    print("""
🎯 何时使用 with_structured_output:

✅ 推荐场景:
1. 信息提取 - 从文本中提取结构化数据
2. 表单填充 - 自动填写表单字段
3. 数据分类 - 文本分类、情感分析
4. 知识图谱 - 提取实体关系
5. 数据验证 - 确保输出符合格式

❌ 不推荐场景:
1. 开放式对话 - 需要灵活回答
2. 创意生成 - 不需要固定结构
3. Agent应用 - 应该用 bind_tools()

💡 vs bind_tools() 的区别:

┌─────────────────────┬─────────────────────────────────────┐
│ with_structured_output │ bind_tools()                      │
├─────────────────────┼─────────────────────────────────────┤
│ 强制返回结构        │ 灵活选择工具(0-N个)                │
│ 用于信息提取        │ 用于Agent应用                       │
│ 返回 Pydantic 对象  │ 返回 AIMessage                      │
│ 单一确定输出        │ 多种可能输出                        │
└─────────────────────┴─────────────────────────────────────┘

📝 设计技巧:

1. **清晰的字段描述**
   ✅ name: str = Field(description="用户的完整姓名")
   ❌ name: str

2. **数据验证**
   ✅ age: int = Field(ge=0, le=150)
   ✅ email: str = Field(pattern=r'^[a-z0-9]+@[a-z]+\.[a-z]{2,3}$')

3. **可选字段**
   ✅ notes: Optional[str] = Field(default=None)

4. **嵌套结构**
   ✅ 使用嵌套的 Pydantic 模型

5. **列表字段**
   ✅ tags: List[str] = Field(description="标签列表")

⚠️ 常见问题:

Q1: 输出格式不稳定怎么办?
A: 1) 使用更强的模型 (gpt-4o)
   2) 在描述中明确要求
   3) 添加示例

Q2: 如何处理提取失败?
A:
   try:
       result = structured_llm.invoke(text)
   except ValidationError as e:
       # 处理验证错误
       pass

Q3: 可以提取可选信息吗?
A: 可以，使用 Optional[Type] = Field(default=None)

Q4: 如何提高准确率?
A: 1) 清晰的字段描述
   2) 使用更好的模型
   3) 添加提示词引导

🔗 参考资源:
- Structured Output: https://python.langchain.com/docs/how_to/structured_output/
- Pydantic: https://docs.pydantic.dev/
- Tool Calling: https://python.langchain.com/docs/how_to/tool_calling/
    """)


def main():
    """运行所有示例"""
    demo_basic_structured_output()
    demo_complex_extraction()
    demo_multiple_entities()
    demo_validation()
    demo_classification()
    demo_form_filling()
    demo_sentiment_analysis()
    demo_relation_extraction()
    demo_json_mode()
    demo_batch_extraction()
    best_practices()

    print("\n\n" + "=" * 70)
    print("✅ Structured Output 所有示例演示完成")
    print("=" * 70)
    print("\n核心要点:")
    print("  ✓ with_structured_output() - 强制结构化")
    print("  ✓ Pydantic 模型 - 数据验证")
    print("  ✓ 信息提取 - 文本→结构")
    print("  ✓ 表单填充 - 自动化")
    print("  ✓ 数据分类 - 结构化分析")


if __name__ == "__main__":
    main()
