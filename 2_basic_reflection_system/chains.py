from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
generate_prompts = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个诗词创作助手，可以根据用户的要求创作诗词。如果用户对你的诗词不满意，你需要根据用户的反馈进行修改，直到用户满意为止。"),
        MessagesPlaceholder(variable_name="history"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个诗词鉴赏专家，可以为用户创作的诗词提供详细的反馈和改进建议。包括内容、意境、用词等方面。"),
        MessagesPlaceholder(variable_name="history"),
    ]
)
llm = ChatOpenAI(model="ep-m-20250905173140-5fh2w")

generate_chain = generate_prompts | llm
reflection_chain = reflection_prompt | llm