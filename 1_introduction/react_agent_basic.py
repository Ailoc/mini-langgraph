from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()
llm = ChatOpenAI(model="ep-m-20250905173140-5fh2w")

agent = create_agent(model=llm, tools=[], system_prompt="你是一个乐于助人的助手。")
query = "现在的时间是几点？"
result = agent.invoke({"message": [{"role": "user", "content": query}]})
print(result)

# result = llm.invoke("现在的时间是几点？")
# print(result.content)