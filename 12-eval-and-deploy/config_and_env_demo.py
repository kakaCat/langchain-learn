#!/usr/bin/env python3
"""
配置管理与环境适配演示
"""

import os
import json
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from pathlib import Path
import hashlib
import secrets


class Environment(Enum):
    """环境枚举"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ConfigSource(Enum):
    """配置来源枚举"""
    ENV_VARS = "environment_variables"
    FILE = "file"
    SECRETS = "secrets"
    DEFAULT = "default"


@dataclass
class ConfigValue:
    """配置值"""
    value: Any
    source: ConfigSource
    timestamp: str
    sensitive: bool = False


@dataclass
class ValidationRule:
    """验证规则"""
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_string(value: str, rule: ValidationRule) -> List[str]:
        """验证字符串配置"""
        errors = []
        
        if rule.required and not value:
            errors.append("配置项为必填项")
        
        if value:
            if rule.min_length and len(value) < rule.min_length:
                errors.append(f"长度不能小于 {rule.min_length}")
            
            if rule.max_length and len(value) > rule.max_length:
                errors.append(f"长度不能大于 {rule.max_length}")
            
            if rule.pattern:
                import re
                if not re.match(rule.pattern, value):
                    errors.append(f"格式不符合要求: {rule.pattern}")
            
            if rule.allowed_values and value not in rule.allowed_values:
                errors.append(f"值必须在 {rule.allowed_values} 中")
        
        return errors
    
    @staticmethod
    def validate_number(value: float, rule: ValidationRule) -> List[str]:
        """验证数字配置"""
        errors = []
        
        if rule.required and value is None:
            errors.append("配置项为必填项")
        
        if value is not None:
            if rule.min_value and value < rule.min_value:
                errors.append(f"值不能小于 {rule.min_value}")
            
            if rule.max_value and value > rule.max_value:
                errors.append(f"值不能大于 {rule.max_value}")
            
            if rule.allowed_values and value not in rule.allowed_values:
                errors.append(f"值必须在 {rule.allowed_values} 中")
        
        return errors


class SecretManager:
    """密钥管理器"""
    
    def __init__(self, secrets_file: str = ".secrets.json"):
        self.secrets_file = Path(secrets_file)
        self.secrets_cache = {}
        self._load_secrets()
    
    def _load_secrets(self):
        """加载密钥文件"""
        if self.secrets_file.exists():
            try:
                with open(self.secrets_file, 'r') as f:
                    self.secrets_cache = json.load(f)
            except Exception as e:
                print(f"警告: 无法加载密钥文件: {e}")
                self.secrets_cache = {}
    
    def _save_secrets(self):
        """保存密钥文件"""
        try:
            with open(self.secrets_file, 'w') as f:
                json.dump(self.secrets_cache, f, indent=2)
        except Exception as e:
            print(f"警告: 无法保存密钥文件: {e}")
    
    def get_secret(self, key: str) -> Optional[str]:
        """获取密钥"""
        return self.secrets_cache.get(key)
    
    def set_secret(self, key: str, value: str):
        """设置密钥"""
        self.secrets_cache[key] = value
        self._save_secrets()
    
    def generate_api_key(self, key_name: str, length: int = 32) -> str:
        """生成API密钥"""
        api_key = secrets.token_urlsafe(length)
        self.set_secret(key_name, api_key)
        return api_key
    
    def mask_sensitive_value(self, value: str) -> str:
        """隐藏敏感值"""
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]


class ConfigurationManager:
    """配置管理器"""
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        self.environment = environment
        self.config_data: Dict[str, ConfigValue] = {}
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.secret_manager = SecretManager()
        self.config_files = []
        
        # 定义配置验证规则
        self._define_validation_rules()
        
        # 加载配置
        self._load_configuration()
    
    def _define_validation_rules(self):
        """定义配置验证规则"""
        self.validation_rules = {
            "database_url": ValidationRule(
                required=True,
                min_length=10,
                pattern=r"^postgresql://.+"
            ),
            "redis_url": ValidationRule(
                required=False,
                pattern=r"^redis://.+"
            ),
            "api_port": ValidationRule(
                required=True,
                min_value=1024,
                max_value=65535
            ),
            "log_level": ValidationRule(
                required=True,
                allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            ),
            "cache_ttl": ValidationRule(
                required=False,
                min_value=0,
                max_value=86400
            ),
            "max_connections": ValidationRule(
                required=False,
                min_value=1,
                max_value=1000
            )
        }
    
    def _load_configuration(self):
        """加载配置"""
        # 1. 加载默认配置
        self._load_default_config()
        
        # 2. 加载环境特定配置
        self._load_environment_config()
        
        # 3. 加载配置文件
        self._load_config_files()
        
        # 4. 加载环境变量
        self._load_environment_variables()
        
        # 5. 加载密钥
        self._load_secrets()
        
        # 6. 验证配置
        self._validate_configuration()
    
    def _load_default_config(self):
        """加载默认配置"""
        default_config = {
            "api_port": 8000,
            "log_level": "INFO",
            "debug_mode": False,
            "cache_ttl": 300,
            "max_connections": 100,
            "request_timeout": 30,
            "enable_metrics": True,
            "enable_tracing": False
        }
        
        for key, value in default_config.items():
            self.config_data[key] = ConfigValue(
                value=value,
                source=ConfigSource.DEFAULT,
                timestamp="default"
            )
    
    def _load_environment_config(self):
        """加载环境特定配置"""
        env_configs = {
            Environment.DEVELOPMENT: {
                "log_level": "DEBUG",
                "debug_mode": True,
                "enable_tracing": True
            },
            Environment.STAGING: {
                "log_level": "INFO",
                "debug_mode": False,
                "enable_metrics": True
            },
            Environment.PRODUCTION: {
                "log_level": "WARNING",
                "debug_mode": False,
                "enable_metrics": True,
                "max_connections": 200
            },
            Environment.TESTING: {
                "log_level": "DEBUG",
                "debug_mode": True,
                "enable_tracing": True,
                "cache_ttl": 0
            }
        }
        
        if self.environment in env_configs:
            for key, value in env_configs[self.environment].items():
                self.config_data[key] = ConfigValue(
                    value=value,
                    source=ConfigSource.DEFAULT,
                    timestamp=f"env_{self.environment.value}"
                )
    
    def _load_config_files(self):
        """加载配置文件"""
        config_files = [
            "config/default.yaml",
            f"config/{self.environment.value}.yaml",
            "config/local.yaml"
        ]
        
        for config_file in config_files:
            file_path = Path(config_file)
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        file_config = yaml.safe_load(f) or {}
                    
                    for key, value in file_config.items():
                        self.config_data[key] = ConfigValue(
                            value=value,
                            source=ConfigSource.FILE,
                            timestamp=config_file
                        )
                    
                    self.config_files.append(config_file)
                    print(f"✓ 加载配置文件: {config_file}")
                    
                except Exception as e:
                    print(f"✗ 无法加载配置文件 {config_file}: {e}")
    
    def _load_environment_variables(self):
        """加载环境变量"""
        env_mappings = {
            "DATABASE_URL": "database_url",
            "REDIS_URL": "redis_url",
            "API_PORT": "api_port",
            "LOG_LEVEL": "log_level",
            "DEBUG_MODE": "debug_mode",
            "CACHE_TTL": "cache_ttl",
            "MAX_CONNECTIONS": "max_connections"
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # 类型转换
                if config_key in ["api_port", "cache_ttl", "max_connections"]:
                    try:
                        value = int(value)
                    except ValueError:
                        print(f"警告: 环境变量 {env_var} 的值 '{value}' 无法转换为整数")
                        continue
                elif config_key == "debug_mode":
                    value = value.lower() in ["true", "1", "yes"]
                
                self.config_data[config_key] = ConfigValue(
                    value=value,
                    source=ConfigSource.ENV_VARS,
                    timestamp=env_var
                )
    
    def _load_secrets(self):
        """加载密钥"""
        secret_keys = ["api_key", "database_password", "redis_password"]
        
        for key in secret_keys:
            secret_value = self.secret_manager.get_secret(key)
            if secret_value:
                self.config_data[key] = ConfigValue(
                    value=secret_value,
                    source=ConfigSource.SECRETS,
                    timestamp="secrets_file",
                    sensitive=True
                )
    
    def _validate_configuration(self):
        """验证配置"""
        errors = []
        
        for key, rule in self.validation_rules.items():
            if key in self.config_data:
                value = self.config_data[key].value
                
                if isinstance(value, str):
                    validation_errors = ConfigValidator.validate_string(value, rule)
                elif isinstance(value, (int, float)):
                    validation_errors = ConfigValidator.validate_number(value, rule)
                else:
                    validation_errors = []
                
                if validation_errors:
                    errors.append(f"{key}: {'; '.join(validation_errors)}")
            elif rule.required:
                errors.append(f"{key}: 缺少必填配置项")
        
        if errors:
            error_msg = "配置验证失败:\n" + "\n".join(errors)
            raise ValueError(error_msg)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        if key in self.config_data:
            return self.config_data[key].value
        return default
    
    def get_with_source(self, key: str) -> Optional[ConfigValue]:
        """获取配置值及其来源"""
        return self.config_data.get(key)
    
    def set(self, key: str, value: Any, source: ConfigSource = ConfigSource.DEFAULT):
        """设置配置值"""
        self.config_data[key] = ConfigValue(
            value=value,
            source=source,
            timestamp="manual"
        )
    
    def generate_config_summary(self) -> Dict[str, Any]:
        """生成配置摘要"""
        summary = {
            "environment": self.environment.value,
            "total_config_items": len(self.config_data),
            "config_files_loaded": self.config_files,
            "config_summary": {}
        }
        
        for key, config_value in self.config_data.items():
            display_value = config_value.value
            if config_value.sensitive:
                display_value = self.secret_manager.mask_sensitive_value(str(display_value))
            
            summary["config_summary"][key] = {
                "value": display_value,
                "source": config_value.source.value,
                "sensitive": config_value.sensitive
            }
        
        return summary
    
    def export_config(self, format: str = "json") -> str:
        """导出配置"""
        config_dict = {}
        
        for key, config_value in self.config_data.items():
            if not config_value.sensitive:
                config_dict[key] = config_value.value
        
        if format == "json":
            return json.dumps(config_dict, indent=2, ensure_ascii=False)
        elif format == "yaml":
            return yaml.dump(config_dict, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的格式: {format}")


def demo_config_and_env():
    """演示配置管理与环境适配"""
    print("=== 配置管理与环境适配演示 ===\n")
    
    # 设置环境变量用于演示
    os.environ["API_PORT"] = "8080"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    # 创建配置文件目录
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # 创建默认配置文件
    default_config = {
        "database_url": "postgresql://localhost:5432/mydb",
        "redis_url": "redis://localhost:6379",
        "request_timeout": 30,
        "enable_metrics": True
    }
    
    with open(config_dir / "default.yaml", "w") as f:
        yaml.dump(default_config, f)
    
    # 创建开发环境配置文件
    dev_config = {
        "debug_mode": True,
        "enable_tracing": True,
        "cache_ttl": 60
    }
    
    with open(config_dir / "development.yaml", "w") as f:
        yaml.dump(dev_config, f)
    
    # 初始化配置管理器
    print("🔧 初始化配置管理器 (开发环境)...")
    config_manager = ConfigurationManager(Environment.DEVELOPMENT)
    
    # 生成配置摘要
    print("📋 生成配置摘要...")