import asyncio

from langchain_mcp_adapters.client import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client

load_dotenv()

llm = ChatOpenAI()

stdio_server_params = StdioServerParameters(
    command="python",
    args=["/home/elyasaf/workstation/langchain-ai-agents/servers/math_server_math.py"],
)

async def main():
    async with stdio_client(stdio_server_params) as (reader, writer):
        async with ClientSession(read_stream=reader, write_stream=writer) as session:
            await session.initialize()
            print("Session initialized")
            tools = await load_mcp_tools(session)
            
            agent = create_agent(model = llm,tools = tools,)

            result = await agent.ainvoke({"messages": [HumanMessage(content="What is 2 + 2 * 4?")]})
            print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())