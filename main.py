import os

from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="The url of the source")
    job_title:str = Field(description="The title of the job")

class Titles(BaseModel):
    """Schema for a titles used by the agent"""
    job_title:str = Field(description="The title of the job")

class AgentResponse(BaseModel):
    """Scheme for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list, description="List of sources used to genereted the answer")
    titles:List[Titles] = Field(default_factory=List,description="List of jobs Titles used to genereted the answer")


load_dotenv()

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm,tools=tools,response_format=AgentResponse)
def main():
    print("Hello from langchain-ai-agents!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 jobs postings for an ai engineer using langchain in the bay area on linkedin and list their details")})
    print(result)


if __name__ == "__main__":
    main()
