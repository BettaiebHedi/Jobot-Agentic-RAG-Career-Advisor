from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, Tool
from langchain.agents import initialize_agent
from agents.information import *
from agents.cv_enhancer import *
from agents.job_matching import *
from agents.motivation import *
from utils.config import get_memory
import re
import logging
from langchain.schema import BaseMessage
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents.react.output_parser import ReActOutputParser



logger = logging.getLogger(__name__)


from langchain.schema import BaseMessage

from langgraph.graph import StateGraph, END
from langchain.output_parsers import ResponseSchema, StructuredOutputParser, OutputFixingParser




async def router(state):

    response_schemas = [
        ResponseSchema(name="route", description="One of: motivation_letter, ask_cv, ask_offer, information, job_matching, enhance_resume, resume_enquire, unknown")
    ]


    llm = state["llm"]
    base_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    output_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
    format_instructions = output_parser.get_format_instructions()

   

    system_prompt =f"""
    You are an intelligent router agent that reasons step by step to decide what to do.  

    For each user request classify it strictly into JSON format:
    - "motivation_letter": if the user asks to generate a cover letter and provides both a resume (CV) and a job offer (or a link to a job offer).
    - "ask_cv": if the user wants a cover letter but did not provide a CV.
    - "ask_offer": if the user wants a cover letter but did not provide a job offer.
    - "information": if the user asks a question or requests research/information.
    - "job_matching": if the user wants to find job opportunities.
    - "enhance_resume": if the user wants to improve, edit, or get recommendations for their resume.
    - "resume_enquire": if the user asks a question or requests research about resumes/CVs but has not provided a CV and a job offer.
    - "unknown": if you don’t know what to do OR if the query is ambiguous.

    Rules:
      - Respond **ONLY** with one JSON object: {{ "route": "<one of the above>" }}
      - Do not add any text, explanation, or commentary.
      - Do not include reasoning, apologies, or repeated keys.

    {format_instructions}
    
     """

    route = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["input_text"])
    ])

    logger.info(f"RAW route output: {route}")
    try:
        parsed = await output_parser.aparse(route)
        route_clean = parsed["route"].lower()
    except Exception as e:
        logger.error(f"Parsing failed: {e}, raw route: {route}")
        route_clean = "unknown"
    
    return {"route": route_clean, "llm": state["llm"],"input_text": state["input_text"],"chat_history": state.get("chat_history", [])}


async def memory_node(state):
    past_context = get_memory().load_memory_variables({})
    state["chat_history"] = past_context.get("chat_history", [])
    return state

async def update_memory_node(state):
    get_memory().save_context(
        {"input": state.get("input_text", "")},
        {"output": state.get("output", "")}
    )
    return {
        "input_text": None,
        "output": state.get("output", ""),
        "llm": state["llm"],
        "chat_history": state["chat_history"]
    }



AGENT_SYSTEM_PROMPTS = {
    "information": (
    "You are an assistant that answers user questions by using retrieved documents. "
    "Respond with ONLY one clear, summarized final answer. "
    "Output must start directly with 'Final Answer: ...'. "
    "Do not include Thought, Action, Observation, AI:, apologies, instructions to the user, "
    "or requests like 'Type yes to continue'. "
    "Never ask follow-up questions unless explicitly requested."),
    "motivation_letter": "You are an expert in creating CV-adapted cover letters. Be professional, persuasive, and use only information from the CV and job offer.",
    "job_matching": "You are a career/job matching assistant. Suggest relevant opportunities based on user input.",
    "enhance_resume": "You are a resume enhancement assistant. Suggest improvements and optimizations."
}




# Information Agent
async def run_information_agent(state):
    agent = create_information_agent(state["llm"])
    system_prompt = AGENT_SYSTEM_PROMPTS.get("information", "")
    
    result = await agent.arun({
        "input": state["input_text"],
        "chat_history": state.get("chat_history", [])
    })
    logger.info(f"agent information Response type : {type(result)}, value: {result}")
    

    return {
        "input_text": state.get("input_text", ""),
        "output": result,
        "llm": state.get("llm"),  
        "chat_history": state.get("chat_history", [])
    }

# Motivation Letter Agent
async def run_motivation_agent(state):
    agent = create_motivation_agent(state["llm"])
    system_prompt = AGENT_SYSTEM_PROMPTS.get("motivation_letter", "")
    result = await agent.arun([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["input_text"])
    ])
    logger.info(f"agent motivation Response type : {type(result)}, value: {result}")
    

    return {
        "input_text": state.get("input_text", ""),
        "output": result,
        "llm": state.get("llm"),  
        "chat_history": state.get("chat_history", [])
    }

# Job Matching Agent
async def run_job_matching_agent(state):
    agent = create_job_mztching_agent(state["llm"])
    system_prompt = AGENT_SYSTEM_PROMPTS.get("job_matching", "")
    result = await agent.arun([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["input_text"])
    ])
    logger.info(f"agent job matching Response type : {type(result)}, value: {result}")
    

    return {
        "input_text": state.get("input_text", ""),
        "output": result,
        "llm": state.get("llm"),  
        "chat_history": state.get("chat_history", [])
    }

# CV Enhancer Agent
async def run_cv_enhancer_agent(state):
    agent = create_cv_enhancer_agent(state["llm"])
    system_prompt = AGENT_SYSTEM_PROMPTS.get("enhance_resume", "")
    result = await agent.arun([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["input_text"])
    ])
    logger.info(f"agent cv enhancer Response type : {type(result)}, value: {result}")
    
    return {
        "input_text": state.get("input_text", ""),
        "output": result,
        "llm": state.get("llm"),  
        "chat_history": state.get("chat_history", [])
    }




# ---- Graph Setup ----
graph = StateGraph(dict)

graph.add_node("memory", memory_node)
graph.add_node("update_memory", update_memory_node)

# Router node
graph.add_node("router", router)

graph.add_node("information", run_information_agent)
graph.add_node("motivation_letter", run_motivation_agent)
graph.add_node("job_matching", run_job_matching_agent)
graph.add_node("enhance_resume", run_cv_enhancer_agent)


# Routing rules
graph.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "motivation_letter": "motivation_letter",
        "ask_cv": "motivation_letter",     
        "ask_offer": "motivation_letter",  
        "information": "information",
        "resume_enquire": "information",   
        "job_matching": "job_matching",
        "enhance_resume": "enhance_resume",
        "unknown": END
    }
)

# After agent completes -> update memory -> loop back to memory
graph.add_edge("memory", "router")
graph.add_edge("information", "update_memory")
graph.add_edge("motivation_letter", "update_memory")
graph.add_edge("job_matching", "update_memory")
graph.add_edge("enhance_resume", "update_memory")
graph.add_edge("update_memory", END)

graph.set_entry_point("memory")
workflow = graph.compile()




async def orchestrator_smart(input_text: str, llm):
    result = await workflow.ainvoke({
        "input_text": input_text,
        "llm": llm},
        stop=["Final Answer:"]
        )
    
    logger.info(f"Response type : {type(result)}, value: {result}")

    output_text = result.get("output", str(result))

    match = re.search(r"AI:\s*(.*)", output_text, flags=re.DOTALL)
    if match:
        output_text = match.group(1).strip()

    logger.info(f"output : {type(output_text)}, value: {output_text}")

    return output_text
