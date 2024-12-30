""""Flask web application for Web3 & Cryptocurrency Expert Chatbot."""
# pylint: disable=W0718
import logging
import sys
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.preprocess import create_documents
from src.pipeline import query_rag
from src.prompts import PROMPT_TEMPLATE_1
load_dotenv('keys.env')

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# Initialize global variables for database and embedding model
db = None
embedding_model = None

def initialize_chatbot():
    """Initialize ChromaDB and embeddings."""
    global db, embedding_model  # Use global variables for shared state

    # Initialize Hugging Face embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create documents
    documents = create_documents(
        'credentials.json', os.getenv("SHEET_ID"))

    # Initialize database
    db = Chroma.from_documents(
        documents=documents,
        collection_name="cointelegraph",
        embedding=embedding_model,
        persist_directory="chroma",
        create_collection_if_not_exists=True
    )
    logger.info("Successfully created ChromaDB collection")

@app.route('/')
def home():
    """Render the chat interface."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error("Error serving home page: %s", str(e))
        return "Internal server error", 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages using RAG system."""
    try:
        if db is None or embedding_model is None:
            logger.error("Chatbot components not properly initialized")
            return jsonify({'error': 'Chatbot not initialized properly'}), 503

        data = request.get_json()
        if not data or 'message' not in data:
            logger.warning("Invalid request: missing message")
            return jsonify({'error': 'No message provided'}), 400

        user_message = data['message'].strip()
        logger.info("Processing chat message: %s...", user_message[:50])

        # Use RAG system to generate response
        formatted_response, response_text = query_rag(
            user_message, db, PROMPT_TEMPLATE_1, 'llama3.2')

        if response_text is None:
            logger.error("Failed to generate response")
            return jsonify({'error': 'Failed to generate response'}), 500

        logger.info("Successfully generated response")
        return jsonify({
            'response': response_text,
            'formatted_response': formatted_response
        })

    except Exception as e:
        logger.error("Error processing chat message: %s", str(e))
        return jsonify(
            {'error': 'An error occurred while processing your request'}), 500

if __name__ == '__main__':
    try:
        # Initialize chatbot components
        logger.info("Starting application initialization...")
        initialize_chatbot()
        logger.info("Chatbot initialization completed successfully")

        # Start Flask server
        logger.info("Starting Flask server on port 5000...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error("Critical server error: %s", str(e))
        sys.exit(1)
