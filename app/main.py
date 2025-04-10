import chainlit as cl
from rag_pipeline import load_vector_store, create_rag_chain
from typing import List, Dict, Optional
import re
from pydantic import BaseModel, Field, validator

class LeadInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    
    @validator('email')
    def validate_email(cls, v):
        if v is None:
            return v
        # Basic email validation with regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if not re.match(email_pattern, v):
            raise ValueError('Invalid email format')
        return v
    
    @validator('phone')
    def validate_phone(cls, v):
        if v is None:
            return v
        # Remove any non-digit characters except '+'
        phone_cleaned = ''.join(c for c in v if c.isdigit() or c == '+')
        if phone_cleaned and not phone_cleaned.startswith('+'):
            phone_cleaned = '+' + phone_cleaned
        return phone_cleaned
    
    def is_complete(self):
        """Check if minimum required fields are present"""
        return self.name is not None and self.email is not None
    
    def missing_fields(self):
        """Return list of missing required fields"""
        missing = []
        if self.name is None:
            missing.append("name")
        if self.email is None:
            missing.append("email")
        return missing

class LeadCapture:
    def __init__(self):
        self.lead_info = LeadInfo()
        self.questions_asked = 0
        self.info_requested = False
        self.info_captured = False
        self.confirmation_sent = False
    
    def extract_info_from_message(self, message_content: str) -> bool:
        """Extract information from message and return True if any new info extracted"""
        updated = False
        
        # Email extraction
        if self.lead_info.email is None:
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            email_match = re.search(email_pattern, message_content)
            if email_match:
                try:
                    # Use model validator
                    email = email_match.group(0)
                    self.lead_info = self.lead_info.copy(update={"email": email})
                    updated = True
                    print(email)
                except ValueError:
                    # Invalid email format, ignore
                    pass
        
        # Phone extraction
        if self.lead_info.phone is None:
            phone_pattern = r'\b(\+\d{1,3}[-.\s]?)?(\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
            phone_match = re.search(phone_pattern, message_content)
            if phone_match:
                try:
                    phone = phone_match.group(0)
                    self.lead_info = self.lead_info.copy(update={"phone": phone})
                    updated = True
                    print(phone)
                except ValueError:
                    pass
        
        # Name extraction with improved patterns
        if self.lead_info.name is None:
            name_patterns = [
                r'(?:my name is|I am|I\'m) ([A-Z][a-z]+(?: [A-Z][a-z]+){0,2})',
                r'(?:this is|name:) ([A-Z][a-z]+(?: [A-Z][a-z]+){0,2})',
                r'(?:name|I\'m|I am) (?:called|known as) ([A-Z][a-z]+(?: [A-Z][a-z]+){0,2})'
            ]
            
            # Direct response to query about name
            if "what's your name" in message_content.lower() or "what is your name" in message_content.lower():
                name_patterns.append(r'^([A-Z][a-z]+(?: [A-Z][a-z]+){0,2})$')
            
            for pattern in name_patterns:
                name_match = re.search(pattern, message_content)
                if name_match:
                    name = name_match.group(1)
                    self.lead_info = self.lead_info.copy(update={"name": name})
                    updated = True
                    print(name)
                    break
                    
            # Look for name in form format
            if "name:" in message_content.lower():
                lines = message_content.split("\n")
                for line in lines:
                    if line.lower().startswith("name:"):
                        potential_name = line.split(":", 1)[1].strip()
                        if potential_name and len(potential_name.split()) <= 3:
                            self.lead_info = self.lead_info.copy(update={"name": potential_name})
                            updated = True
                            break
        
        # Check if all required info captured
        if self.lead_info.is_complete() and not self.info_captured:
            self.info_captured = True
            updated = True
            
        return updated
    
    def increment_question(self):
        self.questions_asked += 1
    
    def should_request_info(self):
        return (self.questions_asked >= 2 and 
                not self.info_requested and 
                not self.info_captured)
    
    def get_info_request_message(self):
        missing = self.lead_info.missing_fields()
        
        if "name" in missing and "email" in missing:
            return ("\n\nI'd be happy to continue helping you with your immigration questions. "
                   "To provide you with the most personalized assistance and keep you updated on relevant immigration changes, "
                   "could you share your name and email? A phone number is optional but helpful if you'd like a consultant to follow up.")
        elif "name" in missing:
            return ("\n\nTo personalize your experience better, could you share your name with me?")
        elif "email" in missing:
            return ("\n\nTo keep you updated on any immigration changes relevant to your situation, "
                   "could you share your email address?")
        return ""
    
    def get_confirmation_message(self):
        if not self.confirmation_sent and self.info_captured:
            self.confirmation_sent = True
            message = f"\n\nThank you for sharing your contact information"
            if self.lead_info.name:
                message += f", {self.lead_info.name}"
            message += "! I've saved your details and will continue assisting you with your immigration queries."
            return message
        return ""

@cl.on_chat_start
async def init_chat():
    # Send welcome message
    await cl.Message(
        content="Hi! I'm InCorp's immigration assistant. Ask me about visas, PR, or work passes.",
    ).send()
    # Load RAG chain once per session
    vector_db = load_vector_store()
    cl.user_session.set("rag_chain", create_rag_chain(vector_db))
    # Store the chat history
    cl.user_session.set("history", [])
    # Initialize lead capture
    cl.user_session.set("lead_capture", LeadCapture())

@cl.on_message
async def main(message: cl.Message):
    # Getting the current session objects
    rag_chain = cl.user_session.get("rag_chain")
    history: List[Dict] = cl.user_session.get("history")
    lead_capture: LeadCapture = cl.user_session.get("lead_capture")
    
    # Process message for lead info extraction
    info_updated = lead_capture.extract_info_from_message(message.content)
    
    # Show typing indicator
    msg = cl.Message(content="")
    await msg.send()
    
    # Increment question counter
    lead_capture.increment_question()
    
    # Format conversation history for context
    chat_history = "\n".join(
        [f"User: {h['user']}\nAI: {h['ai']}" for h in history]
    )
    full_query = f"Chat History:\n{chat_history}\n\nNew Question: {message.content}"
    
    # Get response from RAG chain
    response = await rag_chain.ainvoke(full_query)
    content = response.content
    
    # Check if we should request lead info
    if lead_capture.should_request_info():
        lead_capture.info_requested = True
        content += lead_capture.get_info_request_message()
    
    # Add confirmation if lead info was just captured
    if info_updated and lead_capture.info_captured:
        content += lead_capture.get_confirmation_message()
    
    # Update message
    msg.content = content
    await msg.update()
    
    # Store in the session history
    history.append({"user": message.content, "ai": content})
    cl.user_session.set("history", history[-3:])
    
    # Save updated lead capture instance back to session
    cl.user_session.set("lead_capture", lead_capture)
