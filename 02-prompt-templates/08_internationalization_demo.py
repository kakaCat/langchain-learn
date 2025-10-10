"""
08 - 模板国际化与本地化 Demo (Internationalization)

演示多语言模板支持、区域化格式处理和动态语言切换。
"""

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
import locale

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

class InternationalizationManager:
    """国际化模板管理器"""
    
    def __init__(self):
        self.language_templates = {
            "zh-CN": {
                "greeting": "你好，{name}！欢迎使用我们的服务。",
                "product_description": "这款{product}具有以下特点：{features}。",
                "order_confirmation": "您的订单 #{order_id} 已确认，总金额为 {amount}。",
                "date_format": "%Y年%m月%d日",
                "currency_symbol": "¥"
            },
            "en-US": {
                "greeting": "Hello, {name}! Welcome to our service.",
                "product_description": "This {product} has the following features: {features}.",
                "order_confirmation": "Your order #{order_id} has been confirmed. Total amount: {amount}.",
                "date_format": "%B %d, %Y",
                "currency_symbol": "$"
            },
            "ja-JP": {
                "greeting": "こんにちは、{name}さん！当社のサービスへようこそ。",
                "product_description": "この{product}には以下の特徴があります：{features}。",
                "order_confirmation": "注文 #{order_id} が確認されました。合計金額: {amount}。",
                "date_format": "%Y年%m月%d日",
                "currency_symbol": "¥"
            },
            "fr-FR": {
                "greeting": "Bonjour, {name} ! Bienvenue dans notre service.",
                "product_description": "Ce {product} a les caractéristiques suivantes : {features}.",
                "order_confirmation": "Votre commande #{order_id} a été confirmée. Montant total : {amount}.",
                "date_format": "%d %B %Y",
                "currency_symbol": "€"
            }
        }
    
    def get_template(self, template_key: str, language: str = "zh-CN"):
        """获取指定语言的模板"""
        if language not in self.language_templates:
            language = "zh-CN"  # 默认语言
        
        template_text = self.language_templates[language].get(template_key)
        if not template_text:
            raise ValueError(f"Template key '{template_key}' not found for language '{language}'")
        
        # 提取输入变量
        input_vars = self._extract_variables(template_text)
        
        return PromptTemplate(
            template=template_text,
            input_variables=input_vars
        )
    
    def _extract_variables(self, template_text: str):
        """从模板文本中提取变量名"""
        import re
        variables = re.findall(r'\{(\w+)\}', template_text)
        return list(set(variables))  # 去重
    
    def get_locale_info(self, language: str):
        """获取区域化信息"""
        return self.language_templates.get(language, self.language_templates["zh-CN"])

def multilingual_greeting_demo():
    """
    多语言问候演示
    """
    print("=== 多语言问候演示 ===")
    
    manager = InternationalizationManager()
    
    languages = ["zh-CN", "en-US", "ja-JP", "fr-FR"]
    name = "张三"
    
    for lang in languages:
        template = manager.get_template("greeting", lang)
        greeting = template.format(name=name)
        print(f"{lang}: {greeting}")

def localized_date_format_demo():
    """
    本地化日期格式演示
    """
    print("\n=== 本地化日期格式演示 ===")
    
    manager = InternationalizationManager()
    current_date = datetime.now()
    
    languages = ["zh-CN", "en-US", "ja-JP", "fr-FR"]
    
    for lang in languages:
        locale_info = manager.get_locale_info(lang)
        date_format = locale_info["date_format"]
        formatted_date = current_date.strftime(date_format)
        print(f"{lang}: {formatted_date}")

def currency_localization_demo():
    """
    货币本地化演示
    """
    print("\n=== 货币本地化演示 ===")
    
    manager = InternationalizationManager()
    
    languages = ["zh-CN", "en-US", "ja-JP", "fr-FR"]
    amount = 299.99
    order_id = "ORD123456"
    
    for lang in languages:
        template = manager.get_template("order_confirmation", lang)
        locale_info = manager.get_locale_info(lang)
        currency_symbol = locale_info["currency_symbol"]
        
        # 格式化金额（简单示例）
        formatted_amount = f"{currency_symbol}{amount:.2f}"
        
        confirmation = template.format(
            order_id=order_id,
            amount=formatted_amount
        )
        print(f"{lang}: {confirmation}")

def dynamic_language_switching_demo():
    """
    动态语言切换演示
    """
    print("\n=== 动态语言切换演示 ===")
    
    manager = InternationalizationManager()
    llm = get_llm()
    
    # 用户偏好设置
    user_preferences = {
        "user1": "zh-CN",
        "user2": "en-US", 
        "user3": "ja-JP",
        "user4": "fr-FR"
    }
    
    product = "智能手机"
    features = "高清摄像头、长续航电池、快速充电"
    
    for user_id, preferred_lang in user_preferences.items():
        # 获取对应语言的模板
        template = manager.get_template("product_description", preferred_lang)
        
        # 构建多语言消息
        system_message = SystemMessage(
            content=f"你是一个多语言产品助手，请使用{preferred_lang}语言回答用户问题。"
        )
        
        human_message = HumanMessage(
            content=template.format(product=product, features=features)
        )
        
        print(f"\n用户 {user_id} ({preferred_lang}):")
        print(f"提示词: {human_message.content}")
        
        # 在实际应用中，这里会调用 LLM
        # response = llm.invoke([system_message, human_message])
        # print(f"AI 回复: {response.content}")

def locale_based_formatting_demo():
    """
    基于区域设置的格式化演示
    """
    print("\n=== 基于区域设置的格式化演示 ===")
    
    # 模拟不同区域的数字格式化
    locales = [
        ("zh_CN", "1,234.56"),
        ("de_DE", "1.234,56"), 
        ("fr_FR", "1 234,56"),
        ("en_US", "1,234.56")
    ]
    
    number = 1234.56
    
    print("不同区域的数字格式化:")
    for locale_name, expected_format in locales:
        print(f"{locale_name}: {expected_format}")
    
    # 时间格式化示例
    current_time = datetime.now()
    time_formats = [
        ("zh-CN", "%Y年%m月%d日 %H时%M分%S秒"),
        ("en-US", "%B %d, %Y %I:%M:%S %p"),
        ("ja-JP", "%Y年%m月%d日 %H時%M分%S秒"),
        ("fr-FR", "%d %B %Y %H:%M:%S")
    ]
    
    print("\n不同区域的时间格式化:")
    for locale, time_format in time_formats:
        formatted_time = current_time.strftime(time_format)
        print(f"{locale}: {formatted_time}")

def template_translation_demo():
    """
    模板翻译演示
    """
    print("\n=== 模板翻译演示 ===")
    
    manager = InternationalizationManager()
    
    # 源语言模板
    source_template = manager.get_template("product_description", "zh-CN")
    source_text = source_template.format(
        product="智能手表",
        features="心率监测、运动追踪、消息提醒"
    )
    
    print(f"源文本 (中文): {source_text}")
    
    # 翻译到其他语言
    target_languages = ["en-US", "ja-JP", "fr-FR"]
    
    for target_lang in target_languages:
        target_template = manager.get_template("product_description", target_lang)
        target_text = target_template.format(
            product="smartwatch",  # 产品名称也需要翻译
            features="heart rate monitoring, activity tracking, message notifications"
        )
        print(f"目标文本 ({target_lang}): {target_text}")

def main():
    """主函数"""
    print("模板国际化与本地化演示程序")
    print("=" * 50)
    
    # 多语言问候演示
    multilingual_greeting_demo()
    
    # 本地化日期格式演示
    localized_date_format_demo()
    
    # 货币本地化演示
    currency_localization_demo()
    
    # 动态语言切换演示
    dynamic_language_switching_demo()
    
    # 基于区域设置的格式化演示
    locale_based_formatting_demo()
    
    # 模板翻译演示
    template_translation_demo()
    
    print("\n" + "=" * 50)
    print("国际化演示完成！")

if __name__ == "__main__":
    main()