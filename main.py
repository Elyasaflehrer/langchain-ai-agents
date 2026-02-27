import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter

load_dotenv()

def main():
    print("Retrieving...")
    query = "what is Pinecone in machine learning?"

    print("Initializing components...")
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()
    vectorestore = PineconeVectorStore(
        embedding=embeddings, index_name=os.environ["INDEX_NAME"]
    )

    retriever = vectorestore.as_retriever(search_kwargs={"k": 3})

    prompt_template = ChatPromptTemplate.from_template(
            """Answer the question based only this on the following context"
            {context}

            Question: {question}

            Provide a detailed answer:"""
            )
    # result = retrieval_chain_without_lcel(query ,retriever, prompt_template=prompt_template, llm=llm)
    # print(f"{result}")
    llm_to_invoke = create_retrievel_chain_with_lcel(retriever=retriever, prompt_template=prompt_template, llm=llm)
    result_lcel = llm_to_invoke.invoke({"question": query})
    print(f"{result_lcel}")


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str,retriever: VectorStoreRetriever , prompt_template: ChatPromptTemplate, llm: ChatOpenAI):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitaions:
    - Manual step-by-step support
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrice relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    #3 Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4 Invode LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5 Return the content
    return response.content

def create_retrievel_chain_with_lcel(retriever: VectorStoreRetriever, prompt_template: ChatPromptTemplate, llm: ChatOpenAI):
    """
    Creat a retriecal chain using LCEL (LangChain Expresstion Language)
    Returns a chain that can be invoked with {"question: "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain




if __name__ == "__main__":
    main()
