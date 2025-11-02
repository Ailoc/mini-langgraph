import os
import sys
# 将当前文件所在的目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from schema import AnswerQuestion, GraphState
from langchain_tavily import TavilySearch
from langchain_core.messages import AIMessage, ToolMessage
from dotenv import load_dotenv
import uuid

load_dotenv()
tavily_search = TavilySearch(max_results=3)

def getSearchResults(state: GraphState):
    last_answer = state.history[-1]
    if not isinstance(last_answer, AIMessage):
        raise ValueError("Last message in history is not an AIMessage.")
    answer_content = last_answer.content
    answer_data = AnswerQuestion.model_validate_json(answer_content)
    search_queries = answer_data.search_queries
    all_results = []
    for query in search_queries:
        results = tavily_search.invoke({"query": query})
        # 创建 ToolMessage，并生成 tool_call_id
        tool_message = ToolMessage(content=str(results), tool_call_id=str(uuid.uuid4()))
        all_results.append(tool_message)
    # 只返回需要更新的字段
    return {"history": all_results}
