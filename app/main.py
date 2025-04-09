import chainlit as cl
from rag_pipeline import load_vector_store, create_rag_chain

@cl.on_chat_start
async def init_chat():
    # Send welcome message
    await cl.Message(
        content="Hi! I'm InCorp's immigration assistant. Ask me about visas, PR, or work passes.",
    ).send()


    # Load RAG chain once per session
    vector_db = load_vector_store()
    cl.user_session.set("rag_chain", create_rag_chain(vector_db))
    

@cl.on_message
async def main(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    
    # Show typing indicator
    msg = cl.Message(content="")
    await msg.send()
    
    # Get response
    response = await rag_chain.ainvoke(message.content)
    
    # Update message
    msg.content = response.content
    await msg.update()
