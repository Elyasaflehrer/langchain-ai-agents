from typing import Any, Dict
from graph.state_graph import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("--RETRIVER--")
    question = state["question"]
    documents = retriever.invoke(question)
    return {
        "documents": documents,
        "question": question,
    }