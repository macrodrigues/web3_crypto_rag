from uuid import uuid4

def save_to_chromadb(documents, db):
    """ Add documents to chroma db vector database"""

    # Create UUIDs for eah document
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Add new documents to the chroma db
    db.add_documents(documents=documents, ids=uuids)
