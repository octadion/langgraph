from langchain_openai import ChatOpenAI
from agent.graph import build_supervisor
from dotenv import load_dotenv
load_dotenv()
OPENAI_MODEL = "gpt-3.5-turbo"
chat_model = ChatOpenAI(model=OPENAI_MODEL, max_tokens=512, temperature=0, streaming=True)

sql_llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=512, temperature=0, streaming=True)

super_graph = build_supervisor(sql_llm, chat_model)