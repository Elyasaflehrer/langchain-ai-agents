from itertools import chain
import os
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

from chains import generate_chain, reflect_chain

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

REFLECT = "reflect"
GENERATE = "generate"
MESSAGES = "messages"

def generation_node(state: MessageGraph):
    return {MESSAGES: [generate_chain.invoke({MESSAGES: state[MESSAGES]})]}

def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({MESSAGES: state[MESSAGES]})
    return {MESSAGES: [HumanMessage(content=res.content)]}


def main():
    print("Hello from langchain-ai-agents!")


    builder = StateGraph(state_schema=MessageGraph)
    builder.add_node(GENERATE, generation_node)
    builder.add_node(REFLECT, reflection_node)

    builder.set_entry_point(GENERATE)


    def should_continue(state: MessageGraph):
        if len(state[MESSAGES]) > 6 :
            return END
        return REFLECT

    builder.add_conditional_edges(GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT})
    builder.add_edge(REFLECT,GENERATE)

    graph = builder.compile()
    print(graph.get_graph().draw_mermaid())

    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)

if __name__ == "__main__":
    main()

