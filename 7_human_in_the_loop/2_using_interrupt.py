from langgraph.graph import StateGraph, END, START, add_messages
from langgraph.types import Command, interrupt
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
memory = MemorySaver()
llm = ChatOpenAI(model="doubao-seed-1-6-251015")
class State(TypedDict):
    topic: str
    generated_messages: Annotated[List[str], add_messages]
    human_feedback: Annotated[List[str], add_messages]

def model(state: State):
    topic = state["topic"]
    feedback = state["human_feedback"] if "human_feedback" in state else []
    prompt = f'''
        故事主题: {topic}。
        如果有用户反馈，请参考反馈来改进故事: 
        目前的反馈为：{feedback[-1] if feedback else "No feedback yet"}。
    '''
    response = llm.invoke([
        SystemMessage(content="你是一个讲故事的AI。可以根据用户给定的主题讲一个有趣的故事"),
        HumanMessage(content=prompt)
    ])
    generated_messages = response.content
    print(generated_messages)
    return {"generated_messages": [generated_messages]}

def human_node(state: State):
    generated_messages = state["generated_messages"][-1]
    user_feedback = interrupt({
        "generate_messages": generated_messages,
        "message": "请提供您对以下故事的反馈，或者输入end结束:\n"
    })
    if user_feedback.lower() == "end":
        return Command(update={"human_feedback": state["human_feedback"]}, goto=END)
    return Command(update={"human_feedback": state["human_feedback"] + [user_feedback]}, goto="model")

graph = StateGraph(State)
graph.add_node("model", model)
graph.add_node("human_node", human_node)
graph.set_entry_point("model")
graph.add_edge("model", "human_node")

app = graph.compile(checkpointer=memory)
initial_state = {"topic": "勇敢的小猫咪", "generated_messages": [], "human_feedback": []}

thread_config = {"configurable": {
    "thread_id": "user1"
}}

for chunk in app.stream(initial_state, config=thread_config):
    for node_id, value in chunk.items():
        if node_id == "__interrupt__":
            while True:
                user_feedback = input("请提供您对以下故事的反馈，或者输入(end)结束：")
                app.invoke(Command(resume=user_feedback), config=thread_config)   # 继续传递中断信息
                if user_feedback.lower() == "end":
                    break