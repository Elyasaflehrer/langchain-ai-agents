from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state_graph import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATION---")
    question = state["question"]
    documents = state["documents"]
    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}