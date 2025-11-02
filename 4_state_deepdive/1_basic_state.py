from typing import TypedDict
from langgraph.graph import END, StateGraph


class SimpleState(TypedDict):
    count: int

def increment(state: SimpleState) -> SimpleState:
    return {"count": state["count"] + 1}

def should_continue(state: SimpleState):
    if state["count"] >= 5:
        return "stop"
    return "continue"

graph = StateGraph(SimpleState)
graph.add_node("increment", increment)
graph.set_entry_point("increment")

graph.add_conditional_edges(
    "increment",
    should_continue,
    {"stop": END, "continue": "increment"}
)
app = graph.compile()
state = {"count": 0}
result = app.invoke(state)
print(result)