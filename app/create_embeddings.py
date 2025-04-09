from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import torch
import glob
from tqdm import tqdm
from typing import List
from langchain.schema import Document
import shutil

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight but effective
PERSIST_DIR = "../incorp_db"  # Where Chroma will store data

def get_embedding_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def create_vector_db(documents: List[Document]):
    """Creates and persists Chroma DB with embeddings"""
    # 1. Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": 'cuda'},
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    
    # 2. Create Vector Store
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    
    # 3. Persist to disk
    vector_db.persist()
    print(f"Vector DB created at {os.path.abspath(PERSIST_DIR)}")
    return vector_db

if __name__ == "__main__":
    # Connect with your previous code
    from process_knowledgebase import process_immigration_doc 
    md_files = glob.glob("../knowledge_base/*.md")
    # Process all files and create DB
    all_chunks = []
    for file_path in tqdm(md_files):
        all_chunks.extend(process_immigration_doc(file_path))
    
    db = create_vector_db(all_chunks)
    
    # Test retrieval
    results = db.similarity_search("What are EP salary requirements?", k=3)
    for doc in results:
        print(f"\nFrom {doc.metadata['source']}:\n{doc.page_content}...")
