from langchain.agents import Tool, initialize_agent
from langchain.agents import AgentType

from tools.search import web_search_rich
from utils.config import get_memory
def create_job_mztching_agent(llm):
    tools = [
        
        web_search_rich
    ]
    
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        return_intermediate_steps=False,
        memory=None,
        return_only_outputs=True,
        max_iterations=3,
        early_stopping_method="generate"
        # max_iterations=3
    )