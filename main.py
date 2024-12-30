import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from src.preprocess import create_documents
from src.pipeline import query_rag
from src.prompts import PROMPT_TEMPLATE_1
load_dotenv('keys.env')

QUERY = input("Ask anything related to Web3, Crypto and Blockchain: ")

if __name__ == '__main__':

    # Initialize Hugging Face embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create documnents
    documents = create_documents('credentials.json', os.getenv('SHEET_ID'))

    # Initialize database
    db = Chroma.from_documents(
        documents = documents,
        collection_name="cointelegraph",
        embedding=embedding_model,
        persist_directory="chroma",
        create_collection_if_not_exists=True
    )

    # Query RAG
    formatted_response, response_text = query_rag(
        QUERY, db, PROMPT_TEMPLATE_1, 'llama3.2:1b')

    print(response_text)
    print("\n")
    print(formatted_response)
