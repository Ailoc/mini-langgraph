import sys
import os

from langchain_core.messages import ToolMessage, BaseMessage
from langgraph.prebuilt import ToolNode

# 将当前文件所在的目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

# from langgraph.prebuilt import ToolNode
from agent_reason_runnable import agent_runnable, tools
from react_state import AgentState

def reason_node(state: AgentState):
    # Agent from create_agent expects {"messages": [...]} as input
    response = agent_runnable.invoke(state["messages"])
    new_calls = len(getattr(response, "tool_calls", []))
    return {"messages": [response], "tool_calls_count": state.get("tool_calls_count", 0) + new_calls}

action_node = ToolNode(tools=tools, handle_tool_errors=True)
