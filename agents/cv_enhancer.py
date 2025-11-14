from langchain.agents import Tool, initialize_agent
from langchain.agents import AgentType
from tools.documents import fetch_pdf_content_tool
from utils.config import get_memory
from langchain.tools import tool
import httpx
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from bs4 import BeautifulSoup
import json




def create_cv_enhancer_agent(llm):
    
    async def enhance_cv_tool(input_str: str) -> str:
        prompt = f"""
Tu es un expert en rédaction et optimisation de CV.

Ta mission est de réécrire ce CV pour le rendre plus professionnel, clair, structuré et percutant, **sans inventer de nouvelles expériences ou compétences**.

Ta version améliorée doit :
- Mettre en valeur les expériences et compétences clés déjà présentes.
- Uniformiser la présentation et la formulation.
- Optimiser chaque section pour maximiser l’impact auprès des recruteurs.
- Éliminer les formulations vagues, les répétitions et les éléments inutiles.
- Garder un ton professionnel, authentique et fluide.
- Respecter les informations données : **ne rien ajouter qui ne figure pas dans le CV d'origine**.

Voici le contenu du CV :

{input_str}

Génère uniquement le contenu du CV optimisé, sans encadré ni commentaire.
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

    async def extract_job_info_tool(url: str) -> str:
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

    tools = [
        fetch_pdf_content_tool,
        Tool(
            name="extract_job_info",
            func=None,
            coroutine=extract_job_info_tool,
            description="Extracts job details from URLs"
        ),
        Tool(
            name="cv_enhancer",
            func=None,
            coroutine=enhance_cv_tool,
            description="Enhances CV structure and clarity"
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
