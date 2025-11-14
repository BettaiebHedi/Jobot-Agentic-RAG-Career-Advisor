from langchain_community.llms import VLLMOpenAI


def initialize_summarizer():
    return VLLMOpenAI(
    openai_api_key="EMPTY", 
    openai_api_base="http://summarizer-service:8000/v1",  
    model_name="/models/phi-3-mini-4k-instruct",  
    temperature=0.0,
    top_p=1.0,
    max_tokens=500,
    presence_penalty=1.2,  
    )