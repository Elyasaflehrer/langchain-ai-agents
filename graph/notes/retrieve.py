from typing import Any, Dict
from graph.state import GraphStage
from ingestion import retriever

def retrieve(state: GraphStage) -> Dict[str, Any]:
    print("--RETRIVER--")
    question = state["question"]
    documents = retriever.invoke(question)
    return {
        "documents": documents,
        "question": question,
    }