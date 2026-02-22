import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searchin for {query}")
    return tavily.search(query=query)

llm = ChatOpenAI(model="gpt-5")
tools = [search]
agent = create_agent(model=llm,tools=tools)
def main():
    print("Hello from langchain-ai-agents!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 jobs postings for an ai engineer using langchain in the bay area on linkedin and list their details")})
    print(result)


if __name__ == "__main__":
    main()
