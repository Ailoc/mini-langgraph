from typing import TypedDict,List, Annotated
from langgraph.graph import END, StateGraph
import operator

class SimpleState(TypedDict):
    count: int
    sum: Annotated[int, operator.add]
    history: Annotated[List[int], operator.concat]

def increment(state: SimpleState) -> SimpleState:
    new_count = state["count"] + 1
    new_sum = new_count
    history = [new_count]
    return {"count": new_count, "sum": new_sum, "history": history}

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
state = {"count": 0, "sum": 0, "history": []}
result = app.invoke(state)
print(result)