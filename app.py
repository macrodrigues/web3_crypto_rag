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
from flask_apscheduler import APScheduler
from datetime import datetime
from src.db_handler import update_chromadb
from src.preprocess import create_documents

class Config:
    SCHEDULER_API_ENABLED = True

scheduler = APScheduler()

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
app.config.from_object(Config())
CORS(app)

# Initialize global variables for database and embedding model
db = None
embedding_model = None
last_update = None

def update_db_task():
    """Background task to update the database"""
    global db, last_update
    try:
        logger.info("Starting scheduled database update...")
        
        # Update the database
        update_chromadb()
        
        # Reinitialize the database connection
        initialize_chatbot()
        
        last_update = datetime.now()
        logger.info("Scheduled database update completed successfully")
    except Exception as e:
        logger.error(f"Error in scheduled database update: {str(e)}")

def initialize_chatbot():
    """Initialize ChromaDB and embeddings."""
    global db, embedding_model, last_update

    try:
        # Initialize Hugging Face embeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load existing database
        db = Chroma(
            collection_name="cointelegraph",
            embedding_function=embedding_model,
            persist_directory="chroma"
        )
        
        if not last_update:
            last_update = datetime.now()
            
        logger.info("Successfully loaded ChromaDB collection")
    except Exception as e:
        logger.error("Error initializing ChromaDB: %s", str(e))
        raise

def clear_log_files():
    """Clear all log files at startup"""
    log_files = ['app.log', 'update_documents.log']
    try:
        for log_file in log_files:
            if os.path.exists(log_file):
                with open(log_file, 'w') as f:
                    f.truncate(0)
                logger.info(f"Cleared {log_file}")
    except Exception as e:
        logger.error(f"Error clearing log files: {str(e)}")

@app.route('/')
def home():
    """Render the chat interface."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error("Error serving home page: %s", str(e))
        return "Internal server error", 500
    
@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """Clear the chat history."""
    try:
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error("Error clearing chat: %s", str(e))
        return jsonify({'error': 'Failed to clear chat'}), 500    

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
        formatted_response= query_rag(
            user_message, db, PROMPT_TEMPLATE_1)

        logger.info("Successfully generated response")
        return jsonify({
            'response': formatted_response,
        })

    except Exception as e:
        logger.error("Error processing chat message: %s", str(e))
        return jsonify(
            {'error': 'An error occurred while processing your request'}), 500

@app.route('/force_update', methods=['POST'])
def force_update():
    """Endpoint to force a database update"""
    try:
        update_db_task()
        return jsonify(
            {'status': 'success', 'message': 'Database updated successfully'})
    except Exception as e:
        logger.error(f"Error forcing database update: {str(e)}")
        return jsonify({'error': 'Failed to update database'}), 500

@app.route('/db_status')
def db_status():
    """Get the database status and last update time"""
    return jsonify({
        'last_update': last_update.isoformat() if last_update else None,
        'status': 'active' if db is not None else 'inactive'
    })

if __name__ == '__main__':
    try:
        # Clear log files at startup
        clear_log_files()
        
        # Initialize chatbot components
        logger.info("Starting application initialization...")
        initialize_chatbot()
        
        # Initialize scheduler
        scheduler.init_app(app)
        
        # Schedule database updates every 10 minutes
        scheduler.add_job(
            id='update_db',
            func=update_db_task,
            trigger='interval',
            days=3,
            next_run_time=datetime.now()  # Run once immediately
        )
        
        scheduler.start()
        
        logger.info(
            "Chatbot initialization and "
            "scheduler setup completed successfully")

        # Start Flask server
        logger.info("Starting Flask server on port 5000...")
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except Exception as e:
        logger.error("Critical server error: %s", str(e))
        sys.exit(1)
