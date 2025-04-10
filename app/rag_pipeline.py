from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI  # Or your preferred LLM
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Load Vector Store
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory="../incorp_db",
        embedding_function=embeddings
    )
    print(f"✅ Loaded vector store with {vector_db._collection.count()} documents")
    return vector_db

# 2. Create RAG Chain
def create_rag_chain(vector_db):
    # Define prompt template
    prompt_template = """
        You are an AI assistant working for InCorp Asia, a provider of business solutions in Asia. You have access to detailed internal documentation about services like immigration, incorporation, tax, and compliance.

        Use the context below to answer the question as best as you can. If the answer can be inferred, explain it clearly. Only say "I don’t know" if the answer is completely unavailable.

        Context:
        {context}

        ---

        Question: {question}
        Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Initialize LLM (replace with your API key)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.6,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Create retrieval chain
    retriever = vector_db.as_retriever(
        search_type = "mmr",
        search_kwargs={"k": 12, "fetch_k": 20}
    )
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    return rag_chain

# 3. Test Function
def test_rag():
    vector_db = load_vector_store()
    rag_chain = create_rag_chain(vector_db)
    
    while True:
        question = input("\n Question:")
        response = rag_chain.invoke(question)
        print(f" Response: {response.content}")

if __name__ == "__main__":
    test_rag()
