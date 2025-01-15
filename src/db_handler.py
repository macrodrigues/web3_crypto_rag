from uuid import uuid4
import shutil
import logging
import sys
import os
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from src.preprocess import create_documents
from langchain_chroma import Chroma

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

    # Create UUIDs for eah document
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Add new documents to the chroma db
    db.add_documents(documents=documents, ids=uuids)

def update_chromadb():
    """Update ChromaDB with new documents by deleting existing
      collection and creating a new one."""
    try:
        # Create temporary directory for new database
        temp_dir = "chroma_temp"
        os.makedirs(temp_dir, exist_ok=True)

        # Initialize Hugging Face embeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create new documents
        documents = create_documents(
            'credentials.json', os.getenv("SHEET_ID"))

        # Create new database in temporary location
        new_db = Chroma.from_documents(
            documents=documents,
            collection_name="cointelegraph", 
            embedding=embedding_model,
            persist_directory=temp_dir
        )

        # If successful, replace old database with new one
        if os.path.exists("chroma"):
            shutil.rmtree("chroma")
        shutil.move(temp_dir, "chroma")

        logger.info(
            "Successfully refreshed ChromaDB collection with new documents")
        
        return True
        
    except Exception as e:
        logger.error("Error updating ChromaDB: %s", str(e))
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise      
