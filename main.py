import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello ReAct LangGraph with Function Calling!")


if __name__ == "__main__":
    main()
