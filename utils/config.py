import os
from langchain.memory import ConversationBufferWindowMemory
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login
import torch
from gtts import gTTS

def text_to_audio(text, filename="response.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    return filename


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


login(token="HF_TOKEN")


_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


_memory = ConversationBufferWindowMemory(
    k=2,
    memory_key="chat_history",
    return_messages=True
)



def get_embeddings():
    """Get the pre-configured embeddings instance"""
    return _embeddings

def get_memory():
    """Get the pre-configured memory instance"""
    return _memory