import os

from dotenv import load_dotenv
from graph.graph import app
load_dotenv()

if __name__ == "__main__":
    print("Hello from langchain-ai-agents!")
    result = app.invoke({"question": "what is the agent memory?"})
    print(result.get("generation"))
