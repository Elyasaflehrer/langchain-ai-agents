from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from graph.consts import MODEL

llm = ChatOpenAI(temperature=0, model=MODEL)
client = Client()

prompt = client.pull_prompt("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()