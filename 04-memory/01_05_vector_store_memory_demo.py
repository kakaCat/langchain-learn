from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts([""], embeddings)  # 初始空向量库
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 创建向量存储记忆
memory = VectorStoreRetrieverMemory(
    retriever=retriever,
    memory_key="chat_history",
    input_key="input",
    output_key="output"
)

# 创建提示模板
prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="根据历史对话回答问题：\n{chat_history}\nHuman: {input}\nAI:"
)

# 创建链
llm = ChatOpenAI()
chain = LLMChain(llm=llm, prompt=prompt, memory=memory, output_key="output")

# 使用链，会自动将对话存入向量库
response = chain.run({"input": "你好，我叫张三。"})
print(response)
response = chain.run({"input": "我在微软工作。"})
print(response)
response = chain.run({"input": "我在哪里工作？"})  # 会通过向量搜索找到相关历史
print(response)