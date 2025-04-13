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
    print(f"Loaded vector store with {vector_db._collection.count()} documents")
    return vector_db

# 2. Create RAG Chain
def create_rag_chain(vector_db):
    # Define prompt template
    prompt_template = """
        You are an AI assistant for InCorp Asia (business solutions: immigration, incorporation, tax, compliance, etc.).

        **Decision Tree:**
        1. If asked about jobs: "We currently don't have job openings at InCorp Asia."
        2. If user shares personal info: "Thank you for sharing your details!"
        3. If query is COMPLETELY unrelated to the company and context (for example travel(but answer for visas), contacting, etc): "I'm sorry, I can't assist with this as it's unrelated to our services."
        4. If query is relevant but context is missing: return "<SERVICE_FALLBACK>"
        5. Else: Answer based on context

        **Context:**
        {context}

        **Question:** {question}

        **Answer (EXACTLY ONE):**
        - Predefined response (rules 1-3)
        - "<SERVICE_FALLBACK>" (rule 4)
        - Contextual answer (rule 5)
        
        `Note: we donot have the service of connecting to specialists of the company so dont mention that`

        **Answer:**
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
