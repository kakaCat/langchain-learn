#!/usr/bin/env python3
"""
15 - Claude Code Style File Operations Demo

演示 Claude Code 的文件操作能力（模拟真实工具）：
1. Read - 读取文件（支持偏移和限制，处理大文件）
2. Write - 写入文件（覆盖检查，安全保护）
3. Edit - 精确编辑（基于字符串替换，避免误修改）
4. Glob - 模式匹配查找文件（支持递归搜索）

这些是 Claude Code 最核心的文件操作工具。
"""

from __future__ import annotations

import glob as glob_module
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_environment() -> None:
    load_dotenv(
        dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False
    )


# ============================================================================
# Claude Code 风格的文件操作工具
# ============================================================================


class FileOperations:
    """模拟 Claude Code 的文件操作工具"""

    @staticmethod
    def read(file_path: str, offset: int = 0, limit: Optional[int] = None) -> str:
        """
        读取文件内容（支持偏移和限制）

        Args:
            file_path: 文件路径
            offset: 起始行号（从0开始）
            limit: 读取行数（None表示读取全部）

        Returns:
            文件内容（带行号，cat -n 格式）
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # 应用偏移和限制
            start = offset
            end = offset + limit if limit else len(lines)
            selected_lines = lines[start:end]

            # 格式化输出（带行号）
            result = ""
            for i, line in enumerate(selected_lines, start=start + 1):
                # 移除原有换行符，添加行号
                result += f"{i:6d}\t{line}"

            info = f"✓ 读取成功: {file_path}\n"
            info += f"  总行数: {len(lines)}, 显示: {start + 1}-{start + len(selected_lines)}\n"
            info += f"  文件大小: {os.path.getsize(file_path)} bytes\n"
            info += "\n" + result

            return info

        except FileNotFoundError:
            return f"✗ 文件不存在: {file_path}"
        except Exception as e:
            return f"✗ 读取失败: {str(e)}"

    @staticmethod
    def write(file_path: str, content: str, overwrite: bool = False) -> str:
        """
        写入文件（带覆盖检查）

        Args:
            file_path: 文件路径
            content: 要写入的内容
            overwrite: 是否允许覆盖已存在的文件

        Returns:
            操作结果消息
        """
        try:
            # 检查文件是否存在
            if os.path.exists(file_path) and not overwrite:
                return (
                    f"✗ 文件已存在: {file_path}\n"
                    f"  提示: 设置 overwrite=True 以覆盖文件"
                )

            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

            # 写入文件
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            lines = content.count("\n") + 1
            return (
                f"✓ 写入成功: {file_path}\n"
                f"  行数: {lines}\n"
                f"  字符数: {len(content)}\n"
                f"  操作: {'覆盖' if overwrite and os.path.exists(file_path) else '新建'}"
            )

        except Exception as e:
            return f"✗ 写入失败: {str(e)}"

    @staticmethod
    def edit(
        file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """
        精确编辑文件（基于字符串替换）

        Args:
            file_path: 文件路径
            old_string: 要替换的字符串
            new_string: 新字符串
            replace_all: 是否替换所有匹配（False时只替换第一个）

        Returns:
            操作结果消息
        """
        try:
            # 读取文件
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 检查是否存在
            if old_string not in content:
                return (
                    f"✗ 未找到匹配的字符串\n"
                    f"  文件: {file_path}\n"
                    f"  查找: {old_string[:50]}..."
                )

            # 检查唯一性（如果不是 replace_all）
            if not replace_all and content.count(old_string) > 1:
                return (
                    f"✗ 找到多个匹配（{content.count(old_string)} 个）\n"
                    f"  提示: 提供更长的上下文以确保唯一性，或设置 replace_all=True"
                )

            # 执行替换
            if replace_all:
                new_content = content.replace(old_string, new_string)
                count = content.count(old_string)
            else:
                new_content = content.replace(old_string, new_string, 1)
                count = 1

            # 写回文件
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return (
                f"✓ 编辑成功: {file_path}\n"
                f"  替换次数: {count}\n"
                f"  修改前: {len(content)} 字符\n"
                f"  修改后: {len(new_content)} 字符\n"
                f"  差异: {len(new_content) - len(content):+d} 字符"
            )

        except FileNotFoundError:
            return f"✗ 文件不存在: {file_path}"
        except Exception as e:
            return f"✗ 编辑失败: {str(e)}"

    @staticmethod
    def glob(pattern: str, path: str = ".", recursive: bool = True) -> str:
        """
        使用 glob 模式查找文件

        Args:
            pattern: glob 模式（如 '*.py', '**/*.txt'）
            path: 搜索路径
            recursive: 是否递归搜索

        Returns:
            匹配的文件列表
        """
        try:
            # 构造完整模式
            if "**" in pattern:
                recursive = True

            search_pattern = os.path.join(path, pattern)
            matches = sorted(glob_module.glob(search_pattern, recursive=recursive))

            if matches:
                result = f"✓ 找到 {len(matches)} 个文件:\n"

                # 按文件大小排序（可选）
                file_info = []
                for m in matches:
                    if os.path.isfile(m):
                        size = os.path.getsize(m)
                        file_info.append((m, size))

                # 显示文件列表
                for i, (filepath, size) in enumerate(file_info[:20], 1):
                    size_str = f"{size:>8d} bytes" if size < 1024 else f"{size/1024:>7.1f} KB"
                    result += f"  {i:2d}. {filepath:<60s} {size_str}\n"

                if len(matches) > 20:
                    result += f"  ... 还有 {len(matches) - 20} 个文件\n"

                # 统计信息
                total_size = sum(size for _, size in file_info)
                result += f"\n总计: {len(matches)} 个文件, {total_size / 1024:.1f} KB"

                return result
            else:
                return f"✗ 未找到匹配 '{pattern}' 的文件（路径: {path}）"

        except Exception as e:
            return f"✗ 搜索失败: {str(e)}"


# ============================================================================
# 演示示例
# ============================================================================


def demo_read():
    """演示读取文件"""
    print("\n" + "=" * 80)
    print("示例 1: Read - 读取文件")
    print("=" * 80)

    ops = FileOperations()

    # 读取当前脚本自身
    script_path = __file__

    print("\n1.1 读取前 20 行:")
    print(ops.read(script_path, offset=0, limit=20))

    print("\n1.2 读取第 21-40 行:")
    print(ops.read(script_path, offset=20, limit=20))

    print("\n1.3 读取不存在的文件:")
    print(ops.read("nonexistent_file.txt"))


def demo_write():
    """演示写入文件"""
    print("\n" + "=" * 80)
    print("示例 2: Write - 写入文件")
    print("=" * 80)

    ops = FileOperations()

    # 测试内容
    content = """# Test File
This is a test file created by Claude Code demo.

## Features
- File operations
- Read/Write/Edit
- Glob pattern matching

Date: 2025-11-30
"""

    print("\n2.1 写入新文件:")
    print(ops.write("test_output/demo.txt", content))

    print("\n2.2 尝试覆盖（不允许）:")
    print(ops.write("test_output/demo.txt", "New content", overwrite=False))

    print("\n2.3 覆盖文件（允许）:")
    print(ops.write("test_output/demo.txt", "New content\n", overwrite=True))


def demo_edit():
    """演示编辑文件"""
    print("\n" + "=" * 80)
    print("示例 3: Edit - 精确编辑")
    print("=" * 80)

    ops = FileOperations()

    # 先创建一个测试文件
    test_content = """def hello():
    print("Hello, World!")
    return True

def goodbye():
    print("Goodbye!")
    return False
"""

    ops.write("test_output/edit_demo.py", test_content, overwrite=True)

    print("\n3.1 替换字符串（单次）:")
    result = ops.edit(
        "test_output/edit_demo.py", old_string='print("Hello, World!")', new_string='print("Hello, Claude Code!")'
    )
    print(result)

    print("\n3.2 读取修改后的文件:")
    print(ops.read("test_output/edit_demo.py"))

    print("\n3.3 替换所有匹配:")
    ops.write("test_output/edit_demo.py", test_content, overwrite=True)  # 重置
    result = ops.edit(
        "test_output/edit_demo.py", old_string="return", new_string="# return", replace_all=True
    )
    print(result)


def demo_glob():
    """演示 Glob 文件搜索"""
    print("\n" + "=" * 80)
    print("示例 4: Glob - 文件搜索")
    print("=" * 80)

    ops = FileOperations()

    print("\n4.1 查找当前目录的所有 Python 文件:")
    print(ops.glob("*.py", path="."))

    print("\n4.2 递归查找所有 Python 文件:")
    print(ops.glob("**/*.py", path="."))

    print("\n4.3 查找 Markdown 文件:")
    print(ops.glob("*.md", path="."))

    print("\n4.4 查找测试文件:")
    print(ops.glob("test_*.py", path="."))


def demo_combined():
    """演示组合使用"""
    print("\n" + "=" * 80)
    print("示例 5: 组合使用 - 实际工作流")
    print("=" * 80)

    ops = FileOperations()

    print("\n任务: 批量重命名文件中的函数")
    print("步骤:")
    print("  1. 使用 Glob 查找所有 Python 文件")
    print("  2. 使用 Read 检查文件内容")
    print("  3. 使用 Edit 修改函数名")
    print("  4. 使用 Read 验证修改结果")

    # 1. 查找文件
    print("\n[步骤 1] 查找文件:")
    print(ops.glob("*demo*.py", path=".", recursive=False))

    # 2. 读取文件
    print("\n[步骤 2] 读取当前文件的前 30 行:")
    print(ops.read(__file__, offset=0, limit=30))

    # 3-4. 编辑和验证（演示，不实际修改）
    print("\n[步骤 3-4] 编辑操作（演示）")
    print("  示例: 将函数 demo_read 重命名为 demo_read_file")
    print("  实际操作会使用 edit() 方法")


# ============================================================================
# 主入口
# ============================================================================


def main():
    """运行所有演示"""
    print("\n" + "=" * 80)
    print("Claude Code Style - 文件操作演示")
    print("=" * 80)
    print("\n这个演示展示了 Claude Code 的核心文件操作工具：")
    print("  - Read: 读取文件（支持偏移和限制）")
    print("  - Write: 写入文件（带覆盖检查）")
    print("  - Edit: 精确编辑（基于字符串替换）")
    print("  - Glob: 模式匹配查找文件\n")

    demos = [
        ("Read - 读取文件", demo_read),
        ("Write - 写入文件", demo_write),
        ("Edit - 精确编辑", demo_edit),
        ("Glob - 文件搜索", demo_glob),
        ("组合使用", demo_combined),
    ]

    print("选择要运行的示例：")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  {len(demos) + 1}. 运行所有示例")
    print("  0. 退出")

    try:
        choice = input("\n请输入选择 (0-6): ").strip()

        if choice == "0":
            print("退出演示")
            return
        elif choice == str(len(demos) + 1):
            for name, demo_func in demos:
                demo_func()
        elif choice.isdigit() and 1 <= int(choice) <= len(demos):
            demos[int(choice) - 1][1]()
        else:
            print("无效选择")

    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n执行出错: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
