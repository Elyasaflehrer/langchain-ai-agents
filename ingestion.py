import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
load_dotenv()

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

def ingestion():
    print("Hello from langchain-ai-agents!")

    loader = TextLoader("/home/elyasaf/workstation/langchain-ai-agents/mediumblog1.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents=document)
    print(f"created {len(texts)} chunks")

    # index_name = os.environ['INDEX_NAME']
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("ingestiong...")
    PineconeVectorStore.from_documents(
        texts, embedding=embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")

if __name__ == "__main__":
    ingestion()
