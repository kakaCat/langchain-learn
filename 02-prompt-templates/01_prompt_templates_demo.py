#!/usr/bin/env python3
"""
Module 2: Prompt Templates Demo (占位)
"""

from __future__ import annotations
from langchain_core.prompts import PromptTemplate


def main() -> None:
    template = "请写一篇关于{product}的{style}风格的广告文案。"
    prompt = PromptTemplate.from_template(template)
    final_prompt = prompt.format(product="智能手机", style="科幻")
    print(final_prompt)


if __name__ == "__main__":
    main()