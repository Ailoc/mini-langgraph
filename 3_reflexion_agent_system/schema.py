from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class Reflection(BaseModel):
    missing: str = Field(description="在你针对用户的回答中发现的缺失的信息，这些信息对于提供完整和准确的回答是必要的。")
    superfluous: str = Field(description="在你针对用户的回答中发现的多余的信息，这些信息并不直接相关或必要。")

class AnswerQuestion(BaseModel):
    '''回答用户的问题'''
    answer: str = Field(description="针对用户问题的完整回答。")
    search_queries: List[str] = Field(description="1-3个为完善回答而提出的搜索查询列表。")
    reflection: Reflection = Field(description="对回答的反思，指出其中缺失的信息。")

class ReviseAnswer(AnswerQuestion):
    '''改进后之前的回答到新的回答'''
    references: List[str] = Field(description="用于支持改进回答的参考网页链接列表。")

class GraphState(BaseModel):
    '''图的状态,保存历史消息'''
    history: Annotated[List[BaseMessage], add_messages] = Field(description="与用户和代理的对话历史消息列表。")
