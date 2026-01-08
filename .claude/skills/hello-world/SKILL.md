---
name: hello-world
description: 我的第一个 Skill - 打印问候语。用于演示 Agent Skills 标准的基本结构。
version: "1.0.0"
metadata:
  author: Your Name
  difficulty: beginner
---

# Hello World Skill

这是一个简单的示例 Skill，用于演示 Agent Skills 的基本结构。

## 使用方法

调用此 Skill 向用户打印问候语:

```
/hello-world
```

自定义问候对象:

```
/hello-world --name Alice
```

## 功能说明

当用户调用 `/hello-world` 时，Claude 会:
1. 读取用户提供的 name 参数（默认为 "World"）
2. 生成友好的问候消息
3. 添加欢迎信息和表情符号

## 示例

**示例 1**: 基本用法
```
用户: /hello-world
Claude: ✨ Hello, World!
        🎉 Welcome to Anthropic Skills!
```

**示例 2**: 自定义名字
```
用户: /hello-world --name Alice
Claude: ✨ Hello, Alice!
        🎉 Welcome to Anthropic Skills!
```

## 参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| name | string | 否 | World | 要问候的名字 |

## 最佳实践

- 保持问候语简洁友好
- 使用适当的表情符号增强视觉效果
- 确保输出格式一致
