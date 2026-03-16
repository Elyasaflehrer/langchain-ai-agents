from dotenv import load_dotenv
load_dotenv()

from graph.chains.generation import generation_chain


from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever
from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
from graph.chains.route import RouteQuery, question_router
def test_retrieval_grader_answer_yes():
    question = "agent memory"
    documents = retriever.invoke(question)
    doc_txt = documents[1].page_content
    res: GradeDocuments = retrieval_grader.invoke({"question": question, "document": doc_txt})

    assert res.binary_score.lower() == "yes"

def test_retrieval_grader_answer_no() -> None:
    question = "how to make pizza"
    documents = retriever.invoke(question)
    doc_txt = documents[0].page_content
    res: GradeDocuments = retrieval_grader.invoke({"question": question, "document": doc_txt})

    assert res.binary_score.lower() == "no"

def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    documents = retriever.invoke(question)

    generation = generation_chain.invoke({"context": documents, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    assert res.binary_score

def test_hallucination_grader_answer_no() -> None:
    question = "how to make pizza"
    documents = retriever.invoke(question)
    generation = generation_chain.invoke({"context": documents, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    assert not res.binary_score

def test_router_to_vectorstore() -> None:
    question = "agent memory"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "how to make pizza"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"

