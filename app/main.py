import chainlit as cl
import os 
from rag_pipeline import load_vector_store, create_rag_chain
from typing import List, Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv  
from datetime import datetime 
from lead_capture import LeadInfo, LeadCapture


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
    
    # Add confirmation if lead info was captured
    if info_updated and lead_capture.info_captured:
        store_lead(lead_capture.lead_info)
    
    # Update message
    msg.content = content
    await msg.update()
    
    # Store in the session history
    history.append({"user": message.content, "ai": content})
    cl.user_session.set("history", history[-3:])
    
    # Save updated lead capture
    cl.user_session.set("lead_capture", lead_capture)

def store_lead(lead_info: LeadInfo):
    """Store lead information"""
    lead_data = {
        "name": lead_info.name,
        "email": lead_info.email,
        "phone": lead_info.phone,
        "timestamp": datetime.now().isoformat()
    }
    print(f"Lead captured: {lead_data}")
