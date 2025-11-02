from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
llm = ChatOpenAI(model="doubao-seed-1-6-251015")
tavily_search = TavilySearch(max_results=3)

@tool
def getNowTime():
    '''获取当前的系统时间，格式为 YYYY-MM-DD HH:MM:SS'''
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [getNowTime, tavily_search]
agent_runnable = llm.bind_tools(tools=tools)
# 工具映射
tools_by_name = {tool.name: tool for tool in tools}