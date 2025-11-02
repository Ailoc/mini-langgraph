from langgraph.graph import END, StateGraph, add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from typing import TypedDict,Annotated
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
load_dotenv()
# memory = MemorySaver()
sqlite_conn = sqlite3.connect("basic_chatbot.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

tavily_search = TavilySearch(max_results=3)
tools = [tavily_search]
llm = ChatOpenAI(model="doubao-seed-1-6-251015")
llm_with_tools = llm.bind_tools(tools=tools)

class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chat_bot(state: BasicChatState):
    return {"messages": llm_with_tools.invoke(state["messages"])}

def tool_router(state: BasicChatState):
    last_messages = state["messages"][-1]
    if hasattr(last_messages, "tool_calls") and len(last_messages.tool_calls) > 0:
        return "tool_node"
    return END

tool_node = ToolNode(tools=tools)

graph = StateGraph(BasicChatState)
graph.add_node("chat_bot", chat_bot)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("chat_bot")

graph.add_edge("tool_node", "chat_bot")

graph.add_conditional_edges(
    "chat_bot",
    tool_router,
    {"tool_node": "tool_node", END: END}
)
app = graph.compile(checkpointer=memory)
config = {"configurable": {
    "thread_id": "user1"
}}

while True:
    input_text = input("😺: ")
    if input_text in ["exit", "quit"]:
        break
    state = {"messages": [HumanMessage(content=input_text)]}
    result = app.invoke(state, config=config)
    print("🤖:", result["messages"][-1].content)