import json
from typing import List, Dict, Optional
from pydantic import BaseModel

class LeadInfo(BaseModel):
    """
        A class to store the Lead Information of every user.

        This calss stores the name, email and phone of the user.
    """
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    
    def is_complete(self):
        """
            Returns if the Lead Information is completed or not.
        """
        return self.name is not None and self.email is not None
    
    def missing_fields(self):
        """
            Stores the missing fields in an array and returns it.
        """
        missing = []
        if self.name is None:
            missing.append("name")
        if self.email is None:
            missing.append("email")
        return missing

class LeadCapture:
    """ 
        A class that performs the lead capture from the text provided by the user until the lead capture is completed.
        
        Attributes:
            lead_info (LeadInfo) : stores lead information.
            questions_asked (boolean) : counts the number of questions asked for throwing the lead capture request. 
            info_captured (boolane) : flag that says if all the required info is captured or not.
    """
    def __init__(self, llm):
        """
            Initializes a new Lead Capture object.

            Args:
                llm : the LLM model that is used to perform the lead capture.
        """
        self.lead_info = LeadInfo()
        self.questions_asked = 0
        self.info_captured = False
        self.llm = llm
    
    async def extract_info_from_message(self, message_content: str, history: List[Dict]) -> bool:
        """
            A function that perform the extraction from the message content.
            
            Args:
                message_content: The message entered by the user.
                history: The User and Chat bot's history of conversation.
        """

        if self.info_captured: 
            return False
            
        context = ""
        if history:
            last_msg = history[-1]["ai"] if history else ""
            if "could you share your name" in last_msg.lower() or "email" in last_msg.lower():
                context = f"Context: {last_msg}\n"

        prompt = f"""
        Extract contact details from this message. Return ONLY PURE JSON:
        {{
            "name": "extracted name or null",
            "email": "extracted email or null", 
            "phone": "extracted phone or null"
        }}
        {context}Message: "{message_content}"
        """
        response = await self.llm.ainvoke(prompt)
        print(response.content[7:-3])
        try:
            response = await self.llm.ainvoke(prompt)
            extracted_info = json.loads(response.content[7:-3])
            
            updated = False
            for field in ["name", "email", "phone"]:
                if extracted_info.get(field) and getattr(self.lead_info, field) is None:
                    setattr(self.lead_info, field, extracted_info[field])
                    updated = True
            
            if self.lead_info.is_complete() and not self.info_captured:
                self.info_captured = True
                updated = True                
            return updated
                
        except Exception as e:
            print(f"Info extraction error: {e}")
            return False

    def increment_question(self):
        """
            A function that adds the questions_asked attribute of the lead capture object.
        """
        self.questions_asked += 1
    
    def should_request_info(self):
        """
            A function that returns boolean for requesting the lead capture info 
            based on the no of questions_asked and info_captured.
        """
        return (self.questions_asked >= 2 and not self.info_captured)
    
    def get_info_request_message(self):
        """
            A function that sends the chat request to get the required lead info based on what is avilable.
        """
        missing = self.lead_info.missing_fields()
        
        if "name" in missing and "email" in missing:
            return ("\n\n`I'd be happy to continue helping you with your immigration questions. "
                   "To provide you with the most personalized assistance and keep you updated on relevant immigration changes, "
                   "could you share your name and email? A phone number is optional but helpful if you'd like a consultant to follow up.`")
        elif "name" in missing:
            return ("\n\n`To personalize your experience better, could you share your name with me?`")
        elif "email" in missing:
            return ("\n\n`To keep you updated on any immigration changes relevant to your situation, "
                   "could you share your email address?`")
        return ""
