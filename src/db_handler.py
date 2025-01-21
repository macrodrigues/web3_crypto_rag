from uuid import uuid4
import shutil
import logging
import sys
import time
import os
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from src.preprocess import create_documents
from langchain_chroma import Chroma
from chromadb.config import Settings

load_dotenv('keys.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('update_documents.log')
    ]
)
logger = logging.getLogger(__name__)

def save_to_chromadb(documents, db):
    """ Add documents to chroma db vector database"""

    # Create UUIDs for each document
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Add new documents to the chroma db
    db.add_documents(documents=documents, ids=uuids)

def update_chromadb():
    """Update ChromaDB with new documents by deleting existing
      collection and creating a new one."""
    try:
        # Initialize Hugging Face embeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Number of documents to add
        num_docs = 400

        # Create UUIDs for each document
        ids = [str(i) for i in range(1, num_docs + 1)]

        # Create new documents
        documents = create_documents(
            'credentials.json', os.getenv("SHEET_ID"))

        # Create temporary directory for new database
        if os.path.exists("chroma"):
            # Load existing database
            db = Chroma(
                collection_name="cointelegraph",
                embedding_function=embedding_model,
                persist_directory="chroma"
            )

            db.update_documents(
                ids=ids,
                documents=documents[:num_docs],
            )

        else:
            # Create fresh directory
            os.makedirs('chroma')

            # Initialize ChromaDB
            db =Chroma.from_documents(
                ids=ids,
                documents=documents[:num_docs],
                collection_name="cointelegraph",
                embedding=embedding_model,
                persist_directory='chroma'
        )

        logger.info(
            "Successfully refreshed ChromaDB collection with new documents")
        
        return db
        
    except Exception as e:
        logger.error("Error updating ChromaDB: %s", str(e))
        raise      
