import chainlit as cl
import os 
from rag_pipeline import load_vector_store, create_rag_chain
from typing import List, Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv  
from datetime import datetime 
from lead_capture import LeadInfo, LeadCapture
import psycopg2
from psycopg2.extras import Json
import hashlib

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "database",
    "user": "postgres",
    "password": "postgres",
    "port": "5432"
}

def get_lead_id(name: str, email: str) -> str:
    """
        Generate sha256 hash as lead ID
    """
    return hashlib.sha256(f"{name}{email}".encode()).hexdigest()

def store_lead(lead_info: LeadInfo, chat_history: List[Dict], chat_id: str):
    """
        Upsert lead data with chat history
    """
    print("chat id: \n\n"+chat_id)
    try:
        # lead_id = get_lead_id(lead_info.name, lead_info.email)
        
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chats (id, name, email, phone, chat)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        email = EXCLUDED.email,
                        phone = EXCLUDED.phone,
                        chat = EXCLUDED.chat,
                        last_updated = NOW()
                """, (
                    chat_id,
                    lead_info.name,
                    lead_info.email,
                    lead_info.phone,
                    Json(chat_history)
                ))
                conn.commit()
                
    except Exception as e:
        print(f"Failed to store lead: {e}")
        raise


@cl.on_chat_start
async def init_chat():
    """
        Initialization function whenever the chainlit is initialized in any browser session.
    """

    # Initialize the same LLM used in RAG pipeline
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.6,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Send welcome message
    await cl.Message(
        content="Hi! I'm InCorp's immigration assistant. Ask me about visas, PR, or work passes.",
    ).send()
    
    # Load RAG chain (using the same LLM instance)
    vector_db = load_vector_store()
    cl.user_session.set("rag_chain", create_rag_chain(vector_db))
    
    # Initialize chat history
    cl.user_session.set("history", [])

    # Initialize full history
    cl.user_session.set("full_history", [])
    
    # Initialize lead capture with the LLM
    cl.user_session.set("lead_capture", LeadCapture(llm))

@cl.on_message
async def main(message: cl.Message):
    """
        The function that gets called whenever a message is send in the chainlit ui.
    """
    # Getting the current session objects
    rag_chain = cl.user_session.get("rag_chain")
    history: List[Dict] = cl.user_session.get("history")
    full_history: List[Dict] = cl.user_session.get("full_history")
    lead_capture: LeadCapture = cl.user_session.get("lead_capture")
    
    # Process message for lead info
    info_updated = await lead_capture.extract_info_from_message(message.content, history)
    
    # Show typing indicator
    msg = cl.Message(content="")
    await msg.send()
    
    # Increment question counter
    lead_capture.increment_question()
    
    # Format conversation history for context
    chat_history = "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in history])
    full_query = f"Chat History:\n{chat_history}\n\nNew Question: {message.content}"
    
    # Get response from RAG chain
    response = await rag_chain.ainvoke(full_query)
    content = response.content
    
    # Check if we should request lead info
    if lead_capture.should_request_info():
        content += lead_capture.get_info_request_message() 

    # Update message
    msg.content = content
    await msg.update()
    
    # Store in the session history
    history.append({"user": message.content, "ai": content})
    full_history.append({"users": message.content, "ai":content})
    cl.user_session.set("full_history", full_history)
    cl.user_session.set("history", history[-3:])
    
    # if lead_capture.info_captured:
    store_lead(lead_capture.lead_info, full_history, cl.user_session.get("id"))

    # Save updated lead capture
    cl.user_session.set("lead_capture", lead_capture)
