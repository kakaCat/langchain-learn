#!/usr/bin/env python3

from __future__ import annotations

import os
from typing import Optional
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 从当前模块目录加载 .env，避免在仓库根运行时找不到配置
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)
def main() -> None:

    system_template = "你是一位专业的{domain}专家。"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "请总结以下文本：{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    final_messages = chat_prompt.format_prompt(domain="金融", text="...一段很长的金融报道...").to_messages()
    print(final_messages)



if __name__ == "__main__":
    main()