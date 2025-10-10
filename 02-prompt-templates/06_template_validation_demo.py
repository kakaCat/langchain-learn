#!/usr/bin/env python3
"""
模板变量验证与错误处理示例：变量验证、错误处理、边界情况处理
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.base import BasePromptTemplate

# 从当前模块目录加载 .env，避免在仓库根运行时找不到配置
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

def validate_template_variables(template: str, input_variables: List[str]) -> ValidationResult:
    """验证模板变量"""
    errors = []
    warnings = []
    
    # 提取模板中的所有变量
    pattern = r'\{([^}]+)\}'
    template_vars = re.findall(pattern, template)
    
    # 检查未定义的变量
    undefined_vars = set(template_vars) - set(input_variables)
    if undefined_vars:
        errors.append(f"模板中使用了未定义的变量: {', '.join(undefined_vars)}")
    
    # 检查未使用的输入变量
    unused_vars = set(input_variables) - set(template_vars)
    if unused_vars:
        warnings.append(f"输入变量未在模板中使用: {', '.join(unused_vars)}")
    
    # 检查变量命名规范
    for var in template_vars:
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var):
            errors.append(f"变量名不符合规范: {var}")
    
    # 检查嵌套大括号
    nested_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}'
    if re.search(nested_pattern, template):
        errors.append("模板中存在嵌套的大括号")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def safe_template_format(template: PromptTemplate, **kwargs) -> Tuple[bool, str, List[str]]:
    """安全格式化模板，包含错误处理"""
    errors = []
    
    try:
        # 验证输入参数
        missing_vars = set(template.input_variables) - set(kwargs.keys())
        if missing_vars:
            errors.append(f"缺少必需的变量: {', '.join(missing_vars)}")
            return False, "", errors
        
        # 格式化模板
        result = template.format(**kwargs)
        return True, result, errors
        
    except KeyError as e:
        errors.append(f"变量错误: {str(e)}")
        return False, "", errors
    except Exception as e:
        errors.append(f"格式化错误: {str(e)}")
        return False, "", errors

def basic_validation_demo() -> None:
    """基础验证演示"""
    print("=== 基础验证演示 ===")
    
    # 测试模板
    test_templates = [
        ("你好，{name}！今天感觉怎么样？", ["name"]),
        ("产品：{product_name}，价格：{price}，库存：{stock}", ["product_name", "price"]),
        ("欢迎{user-name}来到{company}", ["user-name", "company"]),
        ("嵌套{test{inner}}变量", ["test", "inner"]),
    ]
    
    for template_str, input_vars in test_templates:
        print(f"\n模板: {template_str}")
        print(f"输入变量: {input_vars}")
        
        result = validate_template_variables(template_str, input_vars)
        
        if result.is_valid:
            print("✅ 验证通过")
        else:
            print("❌ 验证失败")
            for error in result.errors:
                print(f"  错误: {error}")
        
        for warning in result.warnings:
            print(f"⚠️  警告: {warning}")
    
    print("\n" + "="*50 + "\n")

def error_handling_demo() -> None:
    """错误处理演示"""
    print("=== 错误处理演示 ===")
    
    # 创建模板
    template = PromptTemplate.from_template(
        "产品 '{product}' 的价格是 {price} 元，库存 {stock} 件。"
    )
    
    # 测试不同的输入情况
    test_cases = [
        {"product": "手机", "price": "1999", "stock": "50"},  # 正常情况
        {"product": "电脑"},  # 缺少变量
        {"price": "2999", "stock": "100"},  # 缺少必需变量
        {},  # 空输入
        {"product": "平板", "price": "1299", "stock": "30", "extra": "多余"},  # 多余变量
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {case}")
        
        success, result, errors = safe_template_format(template, **case)
        
        if success:
            print(f"✅ 成功: {result}")
        else:
            print("❌ 失败")
            for error in errors:
                print(f"  错误: {error}")
    
    print("\n" + "="*50 + "\n")

def boundary_cases_demo() -> None:
    """边界情况处理演示"""
    print("=== 边界情况处理演示 ===")
    
    # 特殊字符和边界值测试
    boundary_cases = [
        # 空值和None
        ("欢迎{user}！", {"user": ""}, "空字符串"),
        ("产品：{name}", {"name": None}, "None值"),
        
        # 特殊字符
        ("消息：{msg}", {"msg": "Hello\nWorld"}, "换行符"),
        ("路径：{path}", {"path": "C:\\Users\\test"}, "反斜杠"),
        ("代码：{code}", {"code": "print(\"hello\")"}, "引号"),
        
        # 超长内容
        ("摘要：{summary}", {"summary": "A" * 1000}, "超长文本"),
        
        # 数字和布尔值
        ("数量：{count}", {"count": 42}, "整数"),
        ("价格：{price}", {"price": 99.99}, "浮点数"),
        ("状态：{active}", {"active": True}, "布尔值"),
    ]
    
    for template_str, inputs, description in boundary_cases:
        print(f"\n{description}:")
        print(f"模板: {template_str}")
        print(f"输入: {inputs}")
        
        try:
            template = PromptTemplate.from_template(template_str)
            result = template.format(**inputs)
            print(f"✅ 结果: {result[:100]}{'...' if len(result) > 100 else ''}")
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    print("\n" + "="*50 + "\n")

def template_sanitization() -> None:
    """模板清理和规范化"""
    print("=== 模板清理和规范化 ===")
    
    def sanitize_template(template: str) -> str:
        """清理模板字符串"""
        # 移除多余的空格和换行
        template = re.sub(r'\s+', ' ', template)
        template = template.strip()
        
        # 规范化变量格式
        template = re.sub(r'\{\s*([^}]+)\s*\}', r'{\1}', template)
        
        return template
    
    messy_templates = [
        "  你好，  {  name  } ！  今天  {weather}  怎么样？  ",
        "产品：{product}\n\n价格：{price}\n\n",
        "{  user  } 说： \"{message}\" ",
    ]
    
    for messy in messy_templates:
        print(f"\n原始模板: '{messy}'")
        cleaned = sanitize_template(messy)
        print(f"清理后: '{cleaned}'")
        
        # 验证清理后的模板
        vars_in_template = re.findall(r'\{([^}]+)\}', cleaned)
        print(f"提取的变量: {vars_in_template}")
    
    print("\n" + "="*50 + "\n")

def advanced_validation_rules() -> None:
    """高级验证规则"""
    print("=== 高级验证规则 ===")
    
    def validate_with_rules(template: str, rules: Dict[str, Any]) -> ValidationResult:
        """使用规则验证模板"""
        errors = []
        warnings = []
        
        # 提取变量
        variables = re.findall(r'\{([^}]+)\}', template)
        
        # 应用规则
        if "max_variables" in rules and len(variables) > rules["max_variables"]:
            warnings.append(f"变量数量超过限制: {len(variables)} > {rules['max_variables']}")
        
        if "required_variables" in rules:
            missing = set(rules["required_variables"]) - set(variables)
            if missing:
                errors.append(f"缺少必需变量: {', '.join(missing)}")
        
        if "forbidden_patterns" in rules:
            for pattern in rules["forbidden_patterns"]:
                if re.search(pattern, template):
                    errors.append(f"包含禁止的模式: {pattern}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    # 定义验证规则
    validation_rules = {
        "max_variables": 5,
        "required_variables": ["user", "action"],
        "forbidden_patterns": [
            r"密码|password",  # 禁止包含敏感词汇
            r"\$\{.*\}",       # 禁止特定变量格式
        ]
    }
    
    test_templates = [
        "用户 {user} 执行了 {action} 操作",
        "{user} 的密码是 {password}",
        "执行 ${command} 命令",
        "{var1} {var2} {var3} {var4} {var5} {var6}",  # 超过变量限制
    ]
    
    for template in test_templates:
        print(f"\n模板: {template}")
        result = validate_with_rules(template, validation_rules)
        
        if result.is_valid:
            print("✅ 验证通过")
        else:
            print("❌ 验证失败")
            for error in result.errors:
                print(f"  错误: {error}")
        
        for warning in result.warnings:
            print(f"⚠️  警告: {warning}")

def main() -> None:
    """主函数"""
    print("模板变量验证与错误处理示例演示\n")
    
    # 基础验证演示
    basic_validation_demo()
    
    # 错误处理演示
    error_handling_demo()
    
    # 边界情况处理
    boundary_cases_demo()
    
    # 模板清理
    template_sanitization()
    
    # 高级验证规则
    advanced_validation_rules()

if __name__ == "__main__":
    main()