from typing import TypedDict,Annotated,Optional
from langchain_core.runnables import RunnableConfig
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, add_messages
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from uuid import uuid4

from langchain_core.messages import HumanMessage, AIMessageChunk
import json

from starlette.middleware.cors import CORSMiddleware

load_dotenv()

llm = ChatOpenAI(model="deepseek-v3-1-terminus", streaming=True)
# llm = ChatDeepSeek(model="deepseek-chat", streaming=True)
tavily = TavilySearch(max_results=5)
tools = [tavily]
memory = MemorySaver()
llm_with_tools = llm.bind_tools(tools)

from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.prebuilt import ToolNode
class State(TypedDict):
    messages: Annotated[list, add_messages]

async def model(state: State):
    result = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [result]}

async def tool_router(state: State):
    last_messages = state["messages"][-1]
    if hasattr(last_messages, "tool_calls") and len(last_messages.tool_calls) > 0:
        return "tool_node"
    return END

tool_node = ToolNode(tools=tools)
graph = StateGraph(State)
graph.add_node("chat_bot", model)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("chat_bot")

graph.add_edge("tool_node", "chat_bot")

graph.add_conditional_edges(
    "chat_bot",
    tool_router,
    {"tool_node": "tool_node", END: END}
)
app_graph = graph.compile(checkpointer=memory)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)

def serialise_ai_message_chunk(chunk):
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError("Chunk must be of type AIMessageChunk")

async def generate_chat_response(message: str, checkpoint_id: Optional[str] = None):
    is_new_conversation = checkpoint_id is None
    if is_new_conversation:
        new_checkpoint_id = str(uuid4())
        config: RunnableConfig = {
            "configurable": {
                "thread_id": new_checkpoint_id
            }
        }

        events = app_graph.astream_events(
            input={
                "messages": [HumanMessage(content=message)]
            },
            version="v2",
            config=config,
        )
        yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{new_checkpoint_id}\"}}\n\n"
    else:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": checkpoint_id
            }
        }
        events = app_graph.astream_events(
            input={
                "messages": [HumanMessage(content=message)]
            },
            version="v2",
            config=config,
        )
    async for event in events:
        event_type = event["event"]
        if event_type == "on_chat_model_stream":
            chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])
            safe_content = chunk_content.replace("'", "\\'").replace("\n", "\\n")
            yield f"data: {{\"type\": \"content\", \"content\": \"{safe_content}\"}}\n\n"
        elif event_type == "on_chat_model_end":
            tool_calls = event["data"]["output"].tool_calls if hasattr(event["data"]["output"], "tool_calls") else []
            search_calls = [call for call in tool_calls if call["name"] == "tavily_search"]

            if search_calls:
                search_query = search_calls[0]["args"].get("query", "")
                safe_query = search_query.replace("'", "\\'").replace("\n", "\\n")
                yield f"data: {{\"type\": \"search_start\",  \"query\": \"{safe_query}\"}}\n\n"
        elif event_type == "on_tool_end" and event["name"] == "tavily_search":
            output = event["data"]["output"]
            if isinstance(output, list):
                urls = []
                for item in output:
                    if isinstance(item, dict) and "url" in item:
                        urls.append(item["url"])
                urls_json = json.dumps(urls)
                yield f"data: {{\"type\": \"search_results\",  \"urls\": {urls_json}}}\n\n"
    yield f"data: {{\"type\": \"end\"}}\n\n"

@app.get("/chat_stream/{message}")
async def chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
    return  StreamingResponse(
        generate_chat_response(message, checkpoint_id),
        media_type="text/event-stream"
    )


























