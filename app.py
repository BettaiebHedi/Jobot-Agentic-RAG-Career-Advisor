import chainlit as cl
from utils.orchestrator import orchestrator_smart
from utils.llm import initialize_llm
from utils.config import text_to_audio
from typing import List
import logging



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





@cl.on_chat_start
async def start():
    try:
        llm = initialize_llm()

        cl.user_session.set("llm", llm)
        cl.user_session.set("conversation", [])


    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        await cl.Message(content="Failed to initialize assistant").send()

@cl.on_message  
async def main(message: cl.Message):
    try:
        llm = cl.user_session.get("llm")
        if not llm:
            raise ValueError("LLM not initialized")
        history = cl.user_session.get("conversation", [])
            
        if message.content.lower() == "history":
            # history = cl.user_session.get("conversation", [])
            history_text = "\n".join(history) or "No history yet."
            await cl.Message(content=history_text).send()
            return
        
        
        
        msg = cl.Message(content="Processing...")
        await msg.send()

        combined_prompt = message.content

        if message.elements:
            for element in message.elements:
                if element.mime == "application/pdf":
                    file_path = element.path
                    combined_prompt  += f"\n[CV_PATH]: {file_path}"
        
        response = await orchestrator_smart(combined_prompt, llm)
        



        if response and "Final Answer:" in response:
            response = response.split("Final Answer:")[-1].strip()
    
        audio_file = "response.mp3"
        text_to_audio(response,audio_file)

        msg = cl.Message(
            content=response,
            elements=[cl.Audio(name="Audio", path=audio_file)]
            )
        await msg.send()



        history.append(f"User: {message.content}")
        history.append(f"Assistant: {response}")
        cl.user_session.set("conversation", history)
        
    except Exception as e:
        logger.error(f"Message error: {str(e)}")
        await cl.Message(content=f"Error: {str(e)}").send()