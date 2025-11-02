import sys
import os
# 将当前文件所在的目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import END, StateGraph
from chains import first_responder_chain, reviser_chain
from schema import GraphState, AnswerQuestion
from execute_tools import getSearchResults
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid


def firstNode(state: GraphState):
    res = first_responder_chain.invoke({"history": state.history})
    res_json = res.model_dump_json(indent=2)
    # 只返回需要更新的字段，langgraph 会根据 add_messages 自动追加
    return {"history": [AIMessage(content=res_json)]}

def reviserNode(state: GraphState):
    res = reviser_chain.invoke({"history": state.history})
    res_json = res.model_dump_json(indent=2)
    # 只返回需要更新的字段
    return {"history": [AIMessage(content=res_json)]}

graph = StateGraph(GraphState)
graph.add_node("first_response", firstNode)
graph.add_node("get_search_results", getSearchResults)
graph.add_node("revise_response", reviserNode)

def get_continue(state: GraphState):
    # 统计历史消息中工具消息的条目
    tool_message_count = sum(1 for msg in state.history if isinstance(msg, ToolMessage))
    if tool_message_count >= 3:
        return END
    return "get_search_results"


graph.set_entry_point("first_response")
graph.add_edge("first_response", "get_search_results")
graph.add_edge("get_search_results", "revise_response")
graph.add_conditional_edges("revise_response", get_continue)

app = graph.compile()

result = app.invoke({"history": [HumanMessage(content="请讲一下langgraph的基本使用方法，并给出完整示例代码进行举例说明。使用中文回答")]})
print("Final Result:")

print(result["history"][-1].content)
