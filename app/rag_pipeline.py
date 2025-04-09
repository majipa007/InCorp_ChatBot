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
    Answer the question based only on the following context:
    {context}
    
    Question: {question}
    
    Answer in markdown format with clear sections if needed. 
    If the Context is not relevant to the Question, say "Sorry I donot have the information related to that".
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Initialize LLM (replace with your API key)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Create retrieval chain
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
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
