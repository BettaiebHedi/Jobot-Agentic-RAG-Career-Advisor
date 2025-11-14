from langchain.agents import AgentExecutor, initialize_agent
from langchain.agents import AgentType
from tools.documents import *

from utils.config import get_memory

from tools.search import web_search_rich



def create_information_agent(llm):
    tools = [
        retriever_tool_startup_tunisia,
        retriever_tool_freelance_tunisia,
        retriever_tool_certifications,
        retriever_tool_international_labor_market,
        retriever_tool_freelance,
        retriever_tool_Code_de_travail,
        retriever_tool_startup,
        retriever_tool_resume,
        web_search_rich
    ]
    
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=None,
        return_intermediate_steps=False,
        handle_parsing_errors=True,
        return_only_outputs=True,
        max_iterations=1,
        early_stopping_method="generate"
    )