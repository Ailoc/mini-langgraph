import sys
import os

# 将当前文件所在的目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import END, StateGraph
from chains import generate_chain, reflection_chain

class GraphState(TypedDict):
    """Represents the state of our graph."""
    history: List[BaseMessage]

GENERATE = "generate"
REFLECT = "reflect"

def generate_node(state: GraphState):
    """
    Generates a poem based on the current history.
    """
    result = generate_chain.invoke({"history": state["history"]})
    return {"history": state["history"] + [result]}

def reflect_node(state: GraphState):
    """
    Reflects on the generated poem and provides feedback.
    """
    res = reflection_chain.invoke({"history": state["history"]})
    return {"history": state["history"] + [HumanMessage(content=res.content)]}

def should_continue(state: GraphState):
    """
    Determines whether to continue with reflection or end the process.
    """
    if len(state["history"]) > 4:
        return END
    return REFLECT

graph = StateGraph(GraphState)
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)
graph.add_conditional_edges(source=GENERATE, path=should_continue)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()
print(app.get_graph().draw_mermaid())  # pip install grandalf
app.get_graph().print_ascii()

result = app.invoke({"history": [HumanMessage(content="请创作一首关于春天的诗词。")]})
print("Final Result:")
for msg in result["history"]:
    print(msg.content)