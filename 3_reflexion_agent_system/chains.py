import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schema import AnswerQuestion, ReviseAnswer
# from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from dotenv import load_dotenv

load_dotenv()

import datetime

llm = ChatOpenAI(model="doubao-seed-1-6-251015", max_tokens=60000)
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", '''
        你是一个专业的代码教学助理，擅长通过分步骤讲解帮助用户理解编程概念和解决方案,并能给出实际的编程代码示例。
        今天是 {current_date}。
        1. {first_instruction}
        2. 对你的回复内容进行评价，思考你的回复是否完整且易于理解，并尽可能改进讲述方式。
        3. 编程相关的代码往往具有时效性，请确保你的回答基于最新的信息和最佳实践。
        5. 在你思考完后，列出1-3个搜索关键词，这些搜索关键词不包含在反思中，但可以帮助你找到更好的信息来改进你的回复。
        '''),
        MessagesPlaceholder(variable_name="history"),
        ("system", "回答用户的问题并按照特定的格式进行回复。"),
    ]
).partial(current_date=datetime.datetime.now().strftime("%Y年%m月%d日"))

first_responder_prompt = actor_prompt_template.partial(
    first_instruction="阅读用户的问题并提供一个详细且易于理解的回答，确保涵盖所有相关方面。"
)
revise_instruction = '''请你根据最新的信息和最佳实践，改进你的回答，使其更加完整且易于理解。
    - 你应该根据之前你对于回复内容的评价进行进一步改进，增加相应的信息和细节。
    - 你必须包含对于网页内容的引用，以支持你的改进。在回答末尾增加引用部分，列出你参考的网页链接，每个链接前使用类似于[1]的编号进行标注。
    - 你应该尽可能删除冗余的信息，确保回答简洁明了。
    - 你应该针对你生成的内容进一步提出1-3个搜索关键词，这些搜索关键词不包含在反思中，但可以帮助你找到更好的信息来改进你的回复。
'''
reviser_chain = actor_prompt_template.partial(
    first_instruction=revise_instruction
) | llm.with_structured_output(ReviseAnswer)

# pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])
first_responder_chain = first_responder_prompt | llm.with_structured_output(AnswerQuestion)




# response = first_responder_chain.invoke({
#     "history": [
#         ("human", "请讲一下langgraph的基本使用方法，并给出完整示例代码进行举例说明。")
#     ]
# })
# print(response)