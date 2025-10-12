"""
08 - 模板国际化与本地化 Demo (Internationalization)

演示广告文案生成的多语言模板支持、区域化格式处理和动态语言切换。
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
    """国际化模板管理器：广告文案生成"""
    
    def __init__(self):
        self.language_templates = {
            "zh-CN": {
                "ad_copy": "请为{product}创作{style}风格的广告文案。",
                "product_intro": "这款{product}具有以下特点：{features}，适合{style}风格的广告宣传。",
                "campaign_slogan": "{product} - 让生活更{style}！",
                "date_format": "%Y年%m月%d日",
                "currency_symbol": "¥"
            },
            "en-US": {
                "ad_copy": "Please create a {style}-style advertisement copy for {product}.",
                "product_intro": "This {product} has the following features: {features}, suitable for {style}-style advertising.",
                "campaign_slogan": "{product} - Make life more {style}!",
                "date_format": "%B %d, %Y",
                "currency_symbol": "$"
            },
            "ja-JP": {
                "ad_copy": "{product}のための{style}スタイルの広告コピーを作成してください。",
                "product_intro": "この{product}には以下の特徴があります：{features}、{style}スタイルの広告に適しています。",
                "campaign_slogan": "{product} - 生活をより{style}に！",
                "date_format": "%Y年%m月%d日",
                "currency_symbol": "¥"
            },
            "fr-FR": {
                "ad_copy": "Veuillez créer un texte publicitaire de style {style} pour {product}.",
                "product_intro": "Ce {product} a les caractéristiques suivantes : {features}, adapté à la publicité de style {style}.",
                "campaign_slogan": "{product} - Rendez la vie plus {style} !",
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

def multilingual_ad_copy_demo():
    """
    多语言广告文案生成演示
    """
    print("=== 多语言广告文案生成演示 ===")
    
    manager = InternationalizationManager()
    
    languages = ["zh-CN", "en-US", "ja-JP", "fr-FR"]
    product = "智能手机"
    style = "科幻"
    
    for lang in languages:
        template = manager.get_template("ad_copy", lang)
        message = template.format(product=product, style=style)
        print(f"{lang}: {message}")

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
    动态语言切换演示：广告文案
    """
    print("\n=== 动态语言切换演示：广告文案 ===")
    
    manager = InternationalizationManager()
    
    # 模拟不同市场的广告需求
    market_campaigns = [
        {"market": "中国市场", "product": "智能手机", "style": "科幻", "language": "zh-CN"},
        {"market": "美国市场", "product": "smartphone", "style": "sci-fi", "language": "en-US"},
        {"market": "日本市场", "product": "スマートフォン", "style": "SF", "language": "ja-JP"},
        {"market": "法国市场", "product": "smartphone", "style": "science-fiction", "language": "fr-FR"}
    ]
    
    for campaign in market_campaigns:
        template = manager.get_template("ad_copy", campaign["language"])
        message = template.format(product=campaign["product"], style=campaign["style"])
        print(f"{campaign['market']} ({campaign['language']}): {message}")

def localized_formatting_demo():
    """
    本地化格式处理演示：广告文案
    """
    print("\n=== 本地化格式处理演示：广告文案 ===")
    
    manager = InternationalizationManager()
    
    languages = ["zh-CN", "en-US", "ja-JP", "fr-FR"]
    product = "智能手机"
    features = "全息投影、量子通信、AI助手"
    style = "科幻"
    
    for lang in languages:
        # 广告文案请求
        ad_template = manager.get_template("ad_copy", lang)
        ad_message = ad_template.format(product=product, style=style)
        
        # 产品介绍
        intro_template = manager.get_template("product_intro", lang)
        intro_message = intro_template.format(product=product, features=features, style=style)
        
        # 广告口号
        slogan_template = manager.get_template("campaign_slogan", lang)
        slogan_message = slogan_template.format(product=product, style=style)
        
        # 日期格式化
        locale_info = manager.get_locale_info(lang)
        date_format = locale_info["date_format"]
        current_date = datetime.now().strftime(date_format)
        
        print(f"{lang}:")
        print(f"  广告文案请求: {ad_message}")
        print(f"  产品介绍: {intro_message}")
        print(f"  广告口号: {slogan_message}")
        print(f"  生成日期: {current_date}")
        print()

def language_detection_demo():
    """
    语言检测与自动切换演示：广告文案
    """
    print("\n=== 语言检测与自动切换演示：广告文案 ===")
    
    manager = InternationalizationManager()
    
    # 模拟不同语言的广告需求
    ad_requests = [
        "请为智能手机创作科幻风格的广告文案",
        "Please create a sci-fi style advertisement copy for smartphone",
        "スマートフォンのためのSFスタイルの広告コピーを作成してください",
        "Veuillez créer un texte publicitaire de style science-fiction pour smartphone"
    ]
    
    # 简单的语言检测
    def detect_language(text):
        if any(char in text for char in "请为"):
            return "zh-CN"
        elif any(char in text for char in "Please"):
            return "en-US"
        elif any(char in text for char in "ください"):
            return "ja-JP"
        elif any(char in text for char in "Veuillez"):
            return "fr-FR"
        else:
            return "en-US"
    
    for request in ad_requests:
        detected_lang = detect_language(request)
        template = manager.get_template("ad_copy", detected_lang)
        message = template.format(product="智能手机", style="科幻")
        print(f"请求: {request}")
        print(f"检测语言: {detected_lang}")
        print(f"标准化请求: {message}")
        print()

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