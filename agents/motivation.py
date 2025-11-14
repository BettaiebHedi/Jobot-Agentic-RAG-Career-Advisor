from langchain.agents import Tool, initialize_agent
from langchain.agents import AgentType
from tools.documents import fetch_pdf_content_tool
from utils.config import get_memory
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from tenacity import retry, wait_exponential, stop_after_attempt
from bs4 import BeautifulSoup
import json
import ast
import httpx


def create_motivation_agent(llm):
    async def extract_job_information(url: str) -> str:
        """Extracts job details from URLs"""
        if url.startswith("url ="):
           url = url.split("=", 1)[1].strip().strip('"')
        else:
           url = url.strip()

        
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
        prompt = f"""
Voici une offre d'emploi extraite du site :

{text}

Retourne un JSON contenant **toutes** les informations suivantes :
- titre
- entreprise
- compétences (liste)
- qualifications (liste)
- responsabilités (liste)
- lieu
- salaire (si mentionné)

**Ne retourne rien d'autre que ce JSON brut.**
"""

        result = await llm.ainvoke(prompt)
        if isinstance(result, dict) and "output" in result:
            result_content = result["output"].strip()
        elif hasattr(result, "content"):
            result_content = result.content.strip()
        elif isinstance(result, str):
            result_content = result.strip()
        else:
            result_content = str(result)

        try:
            job_info = json.loads(result_content)
            return json.dumps(job_info, indent=2)
        except json.JSONDecodeError:
            return f"Erreur: le modèle n'a pas renvoyé un JSON valide. Voici ce que j'ai reçu :\n\n{result_content}"

    async def generate_cover_letter_tool(input_str: str) -> str:
        prompt = f"""
Tu es un expert en rédaction de lettres de motivation sur mesure.

À partir des deux blocs d’informations ci-dessous — le premier étant un CV, le second une offre d'emploi — rédige une lettre de motivation convaincante, claire et engageante, en t'appuyant **uniquement sur les éléments réellement présents dans le CV**.

Ne fais aucune supposition ni invention sur le parcours du candidat. Si une information n’apparaît pas dans le CV, ne l’utilise pas.

Ta lettre doit :
- Être fluide, professionnelle et authentique.
- Mettre en avant les expériences et compétences pertinentes pour l’offre.
- Montrer une motivation sincère et bien alignée avec la mission de l’entreprise.
- Être structurée naturellement (introduction, paragraphes cohérents, conclusion).
- Ne contenir **aucun encadré, ni titre, ni commentaire** : uniquement le texte de la lettre.

Voici les informations :

{input_str}

Génère uniquement le texte de la lettre de motivation.
"""
        result = await llm.ainvoke(prompt)
        if isinstance(result, dict) and "output" in result:
           return result["output"].strip()
        elif hasattr(result, "content"):
           return result.content.strip()
        elif isinstance(result, str):
           return result.strip()
        else:
           return str(result)
        

    tools = [
        fetch_pdf_content_tool,
        Tool(
            name="extract_job_info",
            func=None,
            coroutine=extract_job_information,
            description="Extracts job details from URLs"
        ),
        Tool(
            name="generate_cover_letter",
            func=None,
            coroutine=generate_cover_letter_tool,
            description="Generates a cover letter from a CV and job description. Input should be a dict with 'cv' and 'job'."
        )
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=None,
        return_intermediate_steps=False,
        handle_parsing_errors=True,
        return_only_outputs=True,
        max_iterations=3,
        early_stopping_method="generate"
        # max_iterations=3
    )
