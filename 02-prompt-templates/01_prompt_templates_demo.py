#!/usr/bin/env python3
"""
Module 2: Prompt Templates Demo (占位)
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 从当前模块目录加载 .env，避免在仓库根运行时找不到配置
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)
def main() -> None:
    template = "请写一篇关于{product}的{style}风格的广告文案。"
    prompt = PromptTemplate.from_template(template)
    final_prompt = prompt.format(product="智能手机", style="科幻")
    print(final_prompt)


if __name__ == "__main__":
    main()