from langchain_community.llms import VLLMOpenAI


def initialize_llm():
    return VLLMOpenAI(
    openai_api_key="EMPTY", 
    openai_api_base="http://vllm-service:8000/v1",  
    model_name="/models/llama3.1-8b",  
    temperature=0.0,
    top_p=0.9,
    max_tokens=3000,
    presence_penalty=1.2, 
    )


