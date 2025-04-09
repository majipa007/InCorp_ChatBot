import chainlit as cl
from rag_pipeline import load_vector_store, create_rag_chain
from typing import List

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
    

@cl.on_message
async def main(message: cl.Message):
    # Getting the current session objects
    rag_chain = cl.user_session.get("rag_chain")
    history: List[Dict] = cl.user_session.get("history")
 
    # Show typing indicator
    msg = cl.Message(content="")
    await msg.send()

    # Format conversation history for context
    chat_history = "\n".join(
        [f"User: {h['user']}\nAI: {h['ai']}" for h in history]
    )
    full_query = f"Chat History:\n{chat_history}\n\nNew Question: {message.content}"

    # Get response
    response = await rag_chain.ainvoke(full_query)
    
    # Update message
    msg.content = response.content
    await msg.update()

    # Store in the session history
    history.append({"user": message.content, "ai": response.content})
    cl.user_session.set("history", history[-3:])
