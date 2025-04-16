from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Import your existing modules
from rag_pipeline import load_vector_store, create_rag_chain
from lead_capture import LeadInfo, LeadCapture

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="InCorp Chatbot API")

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage
sessions = {}

# Function for fallback responses
async def fall_back(history: List[Dict], question: str, llm):
    prompt = f"""
        **Role**: You are the Fallback Specialist for InCorp Asia's chatbot. 
        The main AI couldn't answer this query about our services.

        **Current Query That Needs Fallback**:
        "{question}"

        **Your Task**:
        1. Provide a GENERAL but helpful response on the basis of the prompts provided.
        2. Never say "according to your documents" or "in the context"
        3. Just provide a good reply to the query dont respond with something staring with "Okay i understand".
 
        `Note: we donot have the service of connecting to specialists of the company so dont mention that`

        **Answer**:
    """
    response = await llm.ainvoke(prompt)
    return response

# Initialize session function
async def init_session(session_id: str = None):
    """Initialize a new chat session"""
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.6,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Load RAG chain
    vector_db = load_vector_store()
    rag_chain = create_rag_chain(vector_db)
    
    # Create session data structure
    sessions[session_id] = {
        "rag_chain": rag_chain,
        "history": [],
        "full_history": [],
        "lead_capture": LeadCapture(llm),
        "llm": llm,
        "created_at": datetime.now()
    }
    
    return session_id

# Get or create session
async def get_or_create_session(session_id: str = None):
    """Get or create a session"""
    if session_id and session_id in sessions:
        return session_id
    
    # Initialize new session
    return await init_session(session_id)

@app.get("/chat/{session_id}")
async def chat(
    session_id: Optional[str] = None, 
    message: str = Query(..., description="User message")
):
    """
    Process a chat message and return a response
    Uses path parameter for session ID and query parameter for message
    """
    # Get or create session
    session_id = await get_or_create_session(session_id)
    session = sessions[session_id]
    
    # Get session components
    lead_capture = session["lead_capture"]
    history = session["history"]
    full_history = session["full_history"]
    rag_chain = session["rag_chain"]
    llm = session["llm"]
    
    # Extract lead info from message (keeping this functionality)
    await lead_capture.extract_info_from_message(message, history)
    
    # Increment question counter
    lead_capture.increment_question()
    
    # Format conversation history for context
    chat_history = "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in history])
    full_query = f"Chat History:\n{chat_history}\n\nNew Question: {message}"
    
    # Get response from RAG chain
    response = await rag_chain.ainvoke(full_query)
    content = response.content

    # Check if we should use fall back llm
    if "<SERVICE_FALLBACK>" in content:
        response = await fall_back(chat_history, message, llm)
        content = response.content
    
    # Check if we should request lead info
    if lead_capture.should_request_info():
        content += lead_capture.get_info_request_message()
    
    # Store in the session history
    history.append({"user": message, "ai": content})
    full_history.append({"users": message, "ai": content})
    
    # Only keep the last 3 messages in immediate history (as in original code)
    session["history"] = history[-3:]
    
    return {
        "message": content,
        "session_id": session_id
    }

@app.get("/new-session")
async def create_new_session():
    """Create a new chat session"""
    session_id = await init_session()
    return {"session_id": session_id}

@app.get("/healthcheck")
async def healthcheck():
    """Simple health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
