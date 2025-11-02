from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, add_messages, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(model="doubao-seed-1-6-251015")

def generate(state: State):
    return {"messages": llm.invoke(state["messages"])}

def get_human_decision(state: State):
    last_message = state["messages"][-1]
    print("AI:", last_message.content)
    decision = input("对于上述内容，您觉得是否需要进一步改进? (yes/no): ").strip().lower()
    if decision == "yes":
        return "continue"
    return "stop"

def collection_feedback(state: State):
    feedback = input("Please provide your feedback: ")
    return {"messages": [HumanMessage(content=feedback)]}

def post(state: State):
    last_message = state["messages"][-1]
    print("Final AI Message:", last_message.content)
    return state

graph = StateGraph(State)
graph.add_node("generate", generate)
graph.add_node("collection_feedback", collection_feedback)
graph.add_node("post", post)
graph.set_entry_point("generate")

graph.add_conditional_edges(
    "generate",
    get_human_decision,
    {"continue": "collection_feedback", "stop": "post"}
)
graph.add_edge("collection_feedback", "generate")
graph.add_edge("post", END)

app = graph.compile()
initial_state = {"messages": [HumanMessage(content="给我讲一个笑话.")]}
final_state = app.invoke(initial_state)
print(final_state["messages"][-1].content)