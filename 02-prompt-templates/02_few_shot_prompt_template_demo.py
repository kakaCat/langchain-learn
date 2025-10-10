

from __future__ import annotations

import os
from typing import Optional
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain import FewShotPromptTemplate, PromptTemplate
# 从当前模块目录加载 .env，避免在仓库根运行时找不到配置
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)
def main() -> None:
    # 1. 创建一些例子
    examples = [
        {
            "input": "高兴",
            "output": "难过"
        },
        {
            "input": "高",
            "output": "低"
        }
    ]
    # 2. 指定如何格式化每个例子
    example_formatter_template = "输入：{input}\n输出：{output}"
    example_prompt = PromptTemplate.from_template(example_formatter_template)
    # 3. 创建 FewShotPromptTemplate
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="请给出以下词语的反义词：",
        suffix="输入：{query}\n输出：",
        input_variables=["query"]
    )
    final_messages = few_shot_prompt.format(query="热")
    print(final_messages)



if __name__ == "__main__":
    main()