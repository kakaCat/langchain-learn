"""
LangChain 1.0 - 工具 (Tools) - 官方文档完整实现

严格按照 https://docs.langchain.com/oss/python/langchain/tools 实现

核心功能:
1. @tool 装饰器 - 最简单的方式
2. Tool 类 - 传统方式
3. StructuredTool - 复杂参数
4. 自定义 BaseTool 子类
5. 工具错误处理
6. 工具返回类型
7. 工具组合使用

参考文档:
- https://docs.langchain.com/oss/python/langchain/tools
- https://python.langchain.com/docs/how_to/custom_tools/
"""

import os
from typing import Optional, Type, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool, Tool, StructuredTool, BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_openai import ChatOpenAI

load_dotenv()


# ========== 1. @tool 装饰器 - 最简单 ==========
def demo_tool_decorator():
    """@tool 装饰器 - 最推荐的方式"""
    print("=" * 70)
    print("1. @tool 装饰器 - 最简单最推荐")
    print("=" * 70)

    # 基础用法
    # 默认方法名为工具名称
    @tool
    def search(query: str) -> str:
        """搜索信息

        Args:
            query: 搜索关键词
        """
        return f"搜索结果: 关于'{query}'的信息..."
    
    print(f"\n工具名称: {search.name}")
    print(f"工具描述: {search.description}")
    print(f"工具参数: {search.args}")

    # 调用工具
    result = search.invoke({"query": "Python"})
    print(f"调用结果: {result}")

    # 带类型注解
    @tool
    def calculate(a: float, b: float, operation: str = "add") -> float:
        """执行数学计算

        Args:
            a: 第一个数字
            b: 第二个数字
            operation: 操作类型 (add/subtract/multiply/divide)

        Returns:
            计算结果
        """
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            return a / b if b != 0 else 0

    result = calculate.invoke({"a": 10, "b": 5, "operation": "multiply"})
    print(f"\n计算结果: {result}")


# ========== 2. 自定义工具名称和参数 ==========
def demo_custom_tool():
    """自定义工具名称和参数schema"""
    print("\n\n" + "=" * 70)
    print("2. 自定义工具名称和参数")
    print("=" * 70)

    # 自定义名称
    @tool("weather-search")
    def get_weather(city: str) -> str:
        """获取天气信息"""
        return f"{city}: 晴天, 25°C"

    print(f"\n自定义名称: {get_weather.name}")

    # 使用 Pydantic 模型定义参数
    class SearchInput(BaseModel):
        query: str = Field(description="搜索关键词")
        max_results: int = Field(default=10, description="最大结果数")
        category: Optional[str] = Field(default=None, description="搜索类别")

    @tool(args_schema=SearchInput)
    def advanced_search(query: str, max_results: int = 10, category: Optional[str] = None) -> str:
        """高级搜索功能"""
        result = f"搜索'{query}'"
        if category:
            result += f" 在类别'{category}'"
        result += f", 返回前{max_results}条结果"
        return result

    print(f"\n高级搜索参数schema: {advanced_search.args_schema.schema()}")
    result = advanced_search.invoke({"query": "AI", "max_results": 5, "category": "技术"})
    print(f"搜索结果: {result}")


# ========== 3. Tool 类 - 传统方式 ==========
def demo_tool_class():
    """使用 Tool 类创建工具"""
    print("\n\n" + "=" * 70)
    print("3. Tool 类 - 传统方式")
    print("=" * 70)

    # 定义函数
    def multiply(a: float, b: float) -> float:
        """将两个数相乘"""
        return a * b

    # 创建 Tool 对象
    multiply_tool = Tool(
        name="multiply",
        description="将两个数字相乘。输入应该是两个数字。",
        func=multiply
    )

    print(f"\n工具名称: {multiply_tool.name}")
    print(f"工具描述: {multiply_tool.description}")

    # 调用工具
    result = multiply_tool.run("5, 3")  # 注意: Tool.run() 接受字符串
    print(f"调用结果: {result}")


# ========== 4. StructuredTool - 复杂参数 ==========
def demo_structured_tool():
    """StructuredTool - 处理复杂参数"""
    print("\n\n" + "=" * 70)
    print("4. StructuredTool - 复杂参数")
    print("=" * 70)

    # 定义参数schema
    class DatabaseQueryInput(BaseModel):
        table: str = Field(description="表名")
        filters: dict = Field(description="过滤条件")
        limit: int = Field(default=10, description="返回记录数")

    def query_database(table: str, filters: dict, limit: int = 10) -> str:
        """查询数据库"""
        return f"从表'{table}'查询, 条件:{filters}, 限制:{limit}条"

    # 创建 StructuredTool
    db_tool = StructuredTool.from_function(
        func=query_database,
        name="database_query",
        description="查询数据库",
        args_schema=DatabaseQueryInput
    )

    result = db_tool.invoke({
        "table": "users",
        "filters": {"age": ">18", "city": "北京"},
        "limit": 5
    })
    print(f"\n查询结果: {result}")


# ========== 5. 继承 BaseTool - 完全控制 ==========
def demo_base_tool():
    """继承 BaseTool 类 - 最大控制"""
    print("\n\n" + "=" * 70)
    print("5. 继承 BaseTool - 完全控制")
    print("=" * 70)

    class CustomSearchTool(BaseTool):
        name: str = "custom_search"
        description: str = "自定义搜索工具，用于搜索特定数据源"

        # 定义参数schema
        class ArgsSchema(BaseModel):
            query: str = Field(description="搜索关键词")
            source: str = Field(default="all", description="数据源")

        args_schema: Type[BaseModel] = ArgsSchema

        def _run(
            self,
            query: str,
            source: str = "all",
            run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
            """同步执行"""
            return f"在'{source}'中搜索'{query}': 找到10条结果"

        async def _arun(
            self,
            query: str,
            source: str = "all",
            run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
            """异步执行"""
            return f"[异步] 在'{source}'中搜索'{query}': 找到10条结果"

    # 使用自定义工具
    search_tool = CustomSearchTool()
    print(f"\n工具名称: {search_tool.name}")
    print(f"工具描述: {search_tool.description}")

    result = search_tool.invoke({"query": "LangChain", "source": "docs"})
    print(f"搜索结果: {result}")


# ========== 6. 错误处理 ==========
def demo_error_handling():
    """工具错误处理"""
    print("\n\n" + "=" * 70)
    print("6. 工具错误处理")
    print("=" * 70)

    @tool
    def divide(a: float, b: float) -> str:
        """安全的除法操作，处理除零错误"""
        try:
            if b == 0:
                return "错误: 除数不能为零"
            result = a / b
            return f"结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

    # 正常情况
    result1 = divide.invoke({"a": 10, "b": 2})
    print(f"\n正常除法: {result1}")

    # 除零情况
    result2 = divide.invoke({"a": 10, "b": 0})
    print(f"除零处理: {result2}")

    # 使用 handle_tool_error 参数
    @tool(handle_tool_error=True)
    def risky_operation(value: str) -> str:
        """可能失败的操作"""
        if value == "error":
            raise ValueError("触发了错误")
        return f"成功处理: {value}"

    result3 = risky_operation.invoke({"value": "normal"})
    print(f"\n成功: {result3}")

    result4 = risky_operation.invoke({"value": "error"})
    print(f"错误处理: {result4}")


# ========== 7. 返回直接结果 ==========
def demo_return_direct():
    """返回直接结果，不经过LLM处理"""
    print("\n\n" + "=" * 70)
    print("7. 返回直接结果 (return_direct)")
    print("=" * 70)

    @tool(return_direct=True)
    def get_current_time() -> str:
        """获取当前时间，直接返回结果"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\nreturn_direct={get_current_time.return_direct}")
    print("这意味着结果会直接返回给用户，不经过LLM处理")


# ========== 8. 工具组合使用 ==========
def demo_tool_combination():
    """多个工具组合使用"""
    print("\n\n" + "=" * 70)
    print("8. 工具组合使用")
    print("=" * 70)

    # 定义多个工具
    @tool
    def fetch_data(source: str) -> str:
        """从数据源获取数据"""
        return f"从{source}获取的数据: [1,2,3,4,5]"

    @tool
    def process_data(data: str) -> str:
        """处理数据"""
        return f"处理后的数据: {data.upper()}"

    @tool
    def save_result(result: str, destination: str) -> str:
        """保存结果"""
        return f"结果已保存到{destination}: {result}"

    # 工具列表
    tools = [fetch_data, process_data, save_result]

    print(f"\n定义了 {len(tools)} 个工具:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    # 创建支持这些工具的模型
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm_with_tools = llm.bind_tools(tools)

    response = llm_with_tools.invoke("请从数据库获取数据，处理后保存到文件")
    print(f"\nLLM 决定调用的工具:")
    for tool_call in response.tool_calls:
        print(f"  - {tool_call['name']}({tool_call['args']})")


# ========== 9. 工具元数据 ==========
def demo_tool_metadata():
    """工具元数据和配置"""
    print("\n\n" + "=" * 70)
    print("9. 工具元数据")
    print("=" * 70)

    @tool
    def advanced_tool(
        required_param: str,
        optional_param: Optional[str] = None,
        flag: bool = False
    ) -> str:
        """展示工具元数据的高级示例

        Args:
            required_param: 必需参数
            optional_param: 可选参数
            flag: 布尔标志
        """
        return f"执行: required={required_param}, optional={optional_param}, flag={flag}"

    print("\n工具元数据:")
    print(f"  名称: {advanced_tool.name}")
    print(f"  描述: {advanced_tool.description}")
    print(f"  参数schema: {advanced_tool.args}")
    print(f"  是否直接返回: {advanced_tool.return_direct}")


# ========== 10. 实际应用示例 ==========
def demo_real_world_example():
    """实际应用：客户服务工具集"""
    print("\n\n" + "=" * 70)
    print("10. 实际应用：客户服务工具集")
    print("=" * 70)

    @tool
    def query_order_status(order_id: str) -> str:
        """查询订单状态"""
        orders = {
            "ORD001": "配送中",
            "ORD002": "已送达",
            "ORD003": "处理中"
        }
        status = orders.get(order_id, "未找到")
        return f"订单 {order_id}: {status}"

    @tool
    def query_customer_info(customer_id: str) -> str:
        """查询客户信息"""
        return f"客户{customer_id}: 张三, VIP会员, 积分1000"

    @tool
    def create_ticket(issue: str, priority: str = "medium") -> str:
        """创建客服工单"""
        ticket_id = "TKT" + str(hash(issue))[:6]
        return f"工单已创建: {ticket_id}, 问题: {issue}, 优先级: {priority}"

    @tool
    def send_sms(phone: str, message: str) -> str:
        """发送短信通知"""
        return f"短信已发送到 {phone}: {message[:20]}..."

    # 工具集合
    customer_service_tools = [
        query_order_status,
        query_customer_info,
        create_ticket,
        send_sms
    ]

    print(f"\n客户服务工具集 ({len(customer_service_tools)} 个工具):")
    for tool in customer_service_tools:
        print(f"  ✓ {tool.name}")

    print("\n示例调用:")
    print(f"  订单查询: {query_order_status.invoke({'order_id': 'ORD001'})}")
    print(f"  客户信息: {query_customer_info.invoke({'customer_id': 'C001'})}")
    print(f"  创建工单: {create_ticket.invoke({'issue': '产品质量问题', 'priority': 'high'})}")


def main():
    """运行所有示例"""
    demo_tool_decorator()
    demo_custom_tool()
    demo_tool_class()
    demo_structured_tool()
    demo_base_tool()
    demo_error_handling()
    demo_return_direct()
    demo_tool_combination()
    demo_tool_metadata()
    demo_real_world_example()

    print("\n\n" + "=" * 70)
    print("✅ Tools 所有示例演示完成")
    print("=" * 70)
    print("\n核心要点:")
    print("  ✓ @tool 装饰器 - 最推荐（90%场景）")
    print("  ✓ 清晰的描述和类型注解")
    print("  ✓ 错误处理 - 返回友好信息")
    print("  ✓ 单一职责原则")
    print("  ✓ 与 bind_tools() 配合使用")


if __name__ == "__main__":
    main()
