import os
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import TavilySearchResults
from langchain.tools import tool

def clean_scrape(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        html = requests.get(url, headers=headers, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        text = "\n".join(paragraphs[:50])  
        return text
    except Exception:
        return ""


def is_valid_domain(url: str) -> bool:
    blacklist = ["pinterest", "youtube", "facebook", "instagram", "tiktok", "ads","inkedin","indeed","glassdoor"]
    return not any(bad in url for bad in blacklist)


os.environ["TAVILY_API_KEY"] = "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"
tavily = TavilySearchResults()



@tool("web_search_rich", return_direct=False)
def web_search_rich(query: str) -> str:
    """
    Performs a web search using Tavily and scrapes the first valid pages.
    Returns merged clean content from multiple sources.
    """
    results = tavily.run(query)

    if not results or not isinstance(results, list):
        return "No results found."

    merged_output = []
    for r in results:
        if "url" not in r:
            continue
        url = r["url"]
        if not is_valid_domain(url):
            continue

        content = clean_scrape(url)
        if content.strip():
            merged_output.append(f"\n--- Source: {url} ---\n{content}")

    return "\n\n".join(merged_output) if merged_output else "No valid content found."

