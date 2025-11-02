import sys
import os
from typing import Literal

from langchain.tools.tool_node import tools_condition
from langchain_core.messages import HumanMessage

# 将当前文件所在的目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import END, StateGraph
from nodes import reason_node, action_node
from react_state import AgentState

REASON_NODE = "reason_node"
ACTION_NODE = "action_node"

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    count = state.get("tool_calls_count", 0)
    if count >= 4:
        return END
    return tools_condition(state=state)     # 判断最后一条消息是否有 tool_calls

graph = StateGraph(AgentState)

graph.add_node(REASON_NODE, reason_node)
graph.add_node(ACTION_NODE, action_node)
graph.set_entry_point(REASON_NODE)

graph.add_conditional_edges(
    REASON_NODE,
    should_continue,
    {"tools": ACTION_NODE, END: END}
)
graph.add_edge(ACTION_NODE, REASON_NODE)

app = graph.compile()
inputs = {"messages": [HumanMessage(content="神州21号载人飞船已经发射到现在已经过去了几个小时？")], "tool_calls_count": 0}

for chunk in app.stream(inputs, stream_mode="values"):
    messages = chunk["messages"][-1]
    if messages.content:
        print(f"🐽: {messages.content}")
    if hasattr(messages, "tool_calls") and messages.tool_calls:
        for call in messages.tool_calls:
            print(f"🛠️ Tool Call: {call['name']} with input {call['args']}")

print("Final Result:")
final_state = app.invoke(inputs)
print("\n"+"="*50)
print(f"Total Tool Calls: {final_state.get('tool_calls_count', 0)}")
print(f"Final Answer: {final_state['messages'][-1].content}")