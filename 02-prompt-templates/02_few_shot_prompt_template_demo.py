

from __future__ import annotations

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

def main() -> None:
    # 1. 创建广告文案例子
    examples = [
        {
            "product": "智能手机",
            "style": "科幻",
            "copy": "穿越时空的智能体验！量子芯片带来前所未有的运算速度，全息投影让沟通触手可及。这不是手机，这是通往未来的钥匙。"
        },
        {
            "product": "智能手表", 
            "style": "科技",
            "copy": "24小时健康守护者！精准监测心率血氧，AI算法预测健康风险。你的私人健康管家，时刻守护你的每一刻。"
        },
        {
            "product": "电动汽车",
            "style": "环保", 
            "copy": "零排放，零妥协！清洁能源驱动未来，智能续航消除里程焦虑。为地球减负，为生活加速。"
        }
    ]
    # 2. 指定如何格式化每个例子
    example_formatter_template = "示例：\n产品：{product}\n风格：{style}\n文案：{copy}"
    example_prompt = PromptTemplate.from_template(example_formatter_template)
    # 3. 创建 FewShotPromptTemplate
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="请根据以下示例，生成指定产品的广告文案：",
        suffix="现在请生成：\n产品：{product}\n风格：{style}\n文案：",
        input_variables=["product", "style"]
    )
    final_messages = few_shot_prompt.format(product="智能家居系统", style="温馨")
    print(final_messages)



if __name__ == "__main__":
    main()