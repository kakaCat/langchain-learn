#!/usr/bin/env python3
"""
结构化输出验证 Demo
实现复杂数据结构的验证、转换和标准化输出
"""

import json
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime
from pydantic import BaseModel, Field, validator, ValidationError


class ContentType(Enum):
    """内容类型枚举"""
    TEXT = "text"
    IMAGE = "image" 
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"


class PriorityLevel(Enum):
    """优先级枚举"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class ValidationRule:
    """验证规则静态工具类"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """验证邮箱格式"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """验证手机号格式"""
        pattern = r'^1[3-9]\d{9}$'
        return bool(re.match(pattern, phone))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """验证URL格式"""
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def validate_content_length(content: str, min_length: int = 1, max_length: int = 1000) -> bool:
        """验证内容长度"""
        return min_length <= len(content) <= max_length


class DataConverter:
    """数据转换器静态工具类"""
    
    @staticmethod
    def to_snake_case(text: str) -> str:
        """转换为蛇形命名"""
        return re.sub(r'(?<!^)(?=[A-Z])', '_', text).lower()
    
    @staticmethod
    def to_camel_case(text: str) -> str:
        """转换为驼峰命名"""
        words = text.split('_')
        return words[0] + ''.join(word.capitalize() for word in words[1:])
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """标准化文本"""
        # 去除多余空格，首字母大写
        return ' '.join(text.strip().split()).capitalize()
    
    @staticmethod
    def format_timestamp(timestamp: Union[str, int, float]) -> str:
        """格式化时间戳"""
        if isinstance(timestamp, str):
            try:
                timestamp = float(timestamp)
            except ValueError:
                return timestamp
        
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, OSError):
            return str(timestamp)


class ContentData(BaseModel):
    """内容数据模型"""
    id: str = Field(..., description="内容ID")
    title: str = Field(..., min_length=1, max_length=200, description="内容标题")
    content: str = Field(..., min_length=1, description="内容主体")
    content_type: ContentType = Field(..., description="内容类型")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    priority: PriorityLevel = Field(default=PriorityLevel.MEDIUM, description="优先级")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="创建时间")
    
    @validator('title')
    def validate_title(cls, v):
        """验证标题"""
        if not v.strip():
            raise ValueError('标题不能为空')
        return DataConverter.normalize_text(v)
    
    @validator('tags')
    def validate_tags(cls, v):
        """验证标签"""
        # 去重并限制标签数量
        unique_tags = list(set(tag.strip() for tag in v if tag.strip()))
        return unique_tags[:10]  # 最多10个标签


class UserData(BaseModel):
    """用户数据模型"""
    user_id: str = Field(..., description="用户ID")
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: str = Field(..., description="邮箱")
    phone: Optional[str] = Field(None, description="手机号")
    profile: Dict[str, Any] = Field(default_factory=dict, description="用户资料")
    
    @validator('email')
    def validate_email(cls, v):
        """验证邮箱格式"""
        if not ValidationRule.validate_email(v):
            raise ValueError('邮箱格式不正确')
        return v.lower()
    
    @validator('phone')
    def validate_phone(cls, v):
        """验证手机号格式"""
        if v and not ValidationRule.validate_phone(v):
            raise ValueError('手机号格式不正确')
        return v


class ContentWrapper:
    """内容包装器"""
    
    @staticmethod
    def wrap_for_api(content_data: ContentData) -> Dict[str, Any]:
        """包装为API响应格式"""
        return {
            "success": True,
            "data": {
                "id": content_data.id,
                "title": content_data.title,
                "content": content_data.content,
                "type": content_data.content_type.value,
                "tags": content_data.tags,
                "priority": content_data.priority.value,
                "metadata": content_data.metadata,
                "created_at": DataConverter.format_timestamp(content_data.created_at)
            },
            "message": "内容处理成功"
        }
    
    @staticmethod
    def wrap_for_cache(content_data: ContentData) -> Dict[str, Any]:
        """包装为缓存格式"""
        return {
            "id": content_data.id,
            "title": content_data.title,
            "content_preview": content_data.content[:100] + "..." if len(content_data.content) > 100 else content_data.content,
            "type": content_data.content_type.value,
            "priority": content_data.priority.value,
            "cache_timestamp": datetime.now().timestamp()
        }


class ContentService:
    """内容服务"""
    
    def __init__(self):
        self.contents: Dict[str, ContentData] = {}
    
    def create_content(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建内容"""
        try:
            # 数据验证和转换
            validated_data = ContentData(**content_data)
            
            # 存储内容
            self.contents[validated_data.id] = validated_data
            
            # 返回包装后的响应
            return ContentWrapper.wrap_for_api(validated_data)
            
        except ValidationError as e:
            return {
                "success": False,
                "error": f"数据验证失败: {str(e)}",
                "details": e.errors()
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"创建内容失败: {str(e)}"
            }
    
    def get_content(self, content_id: str, include_metadata: bool = True) -> Dict[str, Any]:
        """获取内容"""
        if content_id not in self.contents:
            return {
                "success": False,
                "error": f"内容不存在: {content_id}"
            }
        
        content_data = self.contents[content_id]
        
        if include_metadata:
            return ContentWrapper.wrap_for_api(content_data)
        else:
            return ContentWrapper.wrap_for_cache(content_data)
    
    def search_contents(self, query: str, content_type: Optional[ContentType] = None) -> Dict[str, Any]:
        """搜索内容"""
        results = []
        
        for content in self.contents.values():
            # 简单的关键词搜索
            if (query.lower() in content.title.lower() or 
                query.lower() in content.content.lower()):
                
                if content_type is None or content.content_type == content_type:
                    results.append(ContentWrapper.wrap_for_cache(content))
        
        return {
            "success": True,
            "data": {
                "query": query,
                "content_type": content_type.value if content_type else "all",
                "results": results,
                "total_count": len(results)
            }
        }


def demo_structured_validation():
    """演示结构化验证功能"""
    print("🚀 开始结构化输出验证演示")
    print("=" * 50)
    
    # 创建内容服务
    content_service = ContentService()
    
    # 测试用例1: 正常创建内容
    print("\n📝 测试用例1: 正常创建内容")
    print("-" * 30)
    
    valid_content = {
        "id": "content_001",
        "title": "LangChain 学习指南",
        "content": "LangChain 是一个强大的 AI 应用开发框架，支持多种语言模型和工具集成。",
        "content_type": ContentType.TEXT,
        "tags": ["AI", "开发", "框架"],
        "priority": PriorityLevel.HIGH,
        "metadata": {"author": "AI助手", "version": "1.0"}
    }
    
    result = content_service.create_content(valid_content)
    print(f"创建结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    # 测试用例2: 数据验证失败
    print("\n📝 测试用例2: 数据验证失败")
    print("-" * 30)
    
    invalid_content = {
        "id": "",  # 空的ID
        "title": "",  # 空的标题
        "content": "测试内容",
        "content_type": "invalid_type",  # 无效的类型
        "tags": ["tag1", "tag1", "tag1"],  # 重复的标签
    }
    
    result = content_service.create_content(invalid_content)
    print(f"创建结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    # 测试用例3: 获取内容
    print("\n📝 测试用例3: 获取内容")
    print("-" * 30)
    
    result = content_service.get_content("content_001")
    print(f"获取结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    # 测试用例4: 搜索内容
    print("\n📝 测试用例4: 搜索内容")
    print("-" * 30)
    
    result = content_service.search_contents("LangChain")
    print(f"搜索结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    # 测试用例5: 数据转换演示
    print("\n📝 测试用例5: 数据转换演示")
    print("-" * 30)
    
    test_text = "HelloWorldExample"
    snake_case = DataConverter.to_snake_case(test_text)
    camel_case = DataConverter.to_camel_case("hello_world_example")
    
    print(f"原始文本: {test_text}")
    print(f"蛇形命名: {snake_case}")
    print(f"驼峰命名: {camel_case}")
    
    # 测试用例6: 验证规则演示
    print("\n📝 测试用例6: 验证规则演示")
    print("-" * 30)
    
    test_email = "test@example.com"
    test_phone = "13800138000"
    test_url = "https://example.com"
    
    print(f"邮箱验证: {test_email} -> {ValidationRule.validate_email(test_email)}")
    print(f"手机验证: {test_phone} -> {ValidationRule.validate_phone(test_phone)}")
    print(f"URL验证: {test_url} -> {ValidationRule.validate_url(test_url)}")
    
    print("\n✅ 结构化验证演示完成")


if __name__ == "__main__":
    demo_structured_validation()