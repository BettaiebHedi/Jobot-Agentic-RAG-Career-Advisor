from langchain.tools import Tool
from langchain_community.document_loaders import PyMuPDFLoader
from data.docs import *
from sentence_transformers import CrossEncoder
from utils.summarizer import initialize_summarizer
from langchain.schema import SystemMessage, HumanMessage
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter

llm_summarizer=initialize_summarizer()
SUMMARIZER_SYSTEM_PROMPT = """
You are a highly skilled document summarizer. 
Your task is to take the given text and produce a **clear, concise, and coherent paragraph** summarizing the most important information. 
Ignore any irrelevant, repetitive, or off-topic details. 
Focus only on the key points and main ideas, and make the summary easy to read and understand.
Keep the summary under 150 words.
"""
async def summarize_document(text: str, llm=llm_summarizer) -> str:

    messages = [
        SystemMessage(content=SUMMARIZER_SYSTEM_PROMPT),
        HumanMessage(content=text)
    ]

    summary = await llm.ainvoke(messages)
    return summary.content if hasattr(summary, "content") else str(summary)

async def summarize_long_text(text: str, llm=llm_summarizer) -> str:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    chunk_summaries = []
    for chunk in chunks:
        summary = await summarize_document(chunk, llm=llm)
        chunk_summaries.append(summary)
    combined_summary_text = "\n".join(chunk_summaries)
    final_summary = await summarize_document(combined_summary_text, llm=llm)

    return final_summary

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def fetch_pdf_content_func(pdf_path: str) -> str:
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()[0]
    return data.page_content

fetch_pdf_content_tool = Tool(
    name="fetch_pdf_content",
    func=fetch_pdf_content_func,
    description="Fetches text content from PDF files"
)


MAX_CHARS = 4000 
def retrieve_documents_freelance(query: str) -> str:
    docs = vectordb_freelance.similarity_search(query, k=30, ranker_type="weighted", ranker_params={"weights": [0.6, 0.4]})
    if docs :
        query_doc_pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(query_doc_pairs)
        reranked_docs = [
           doc for _, doc in sorted(
               zip(scores, docs),
               key=lambda x: x[0],
               reverse=True
             )
        ]
        top_docs = reranked_docs[:2]
        combined_text = "\n".join([doc.page_content for doc in top_docs])
        # --- TRUNCATE ---
        truncated_text = combined_text[:MAX_CHARS]
        summary = asyncio.run(summarize_document(truncated_text, llm=llm_summarizer))
        return summary
    else:
        return "Could not find relevant information."
    

def retrieve_documents_freelance_tunisia(query: str) -> str:
    docs = vectordb_freelance_tunisia.similarity_search(query, k=30, ranker_type="weighted", ranker_params={"weights": [0.6, 0.4]})
    if docs :
        query_doc_pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(query_doc_pairs)
        reranked_docs = [
           doc for _, doc in sorted(
               zip(scores, docs),
               key=lambda x: x[0],
               reverse=True
             )
        ]
        top_docs = reranked_docs[:2]
        combined_text = "\n".join([doc.page_content for doc in top_docs])
        truncated_text = combined_text[:MAX_CHARS]
        # Summarize
        summary = asyncio.run(summarize_document(truncated_text, llm=llm_summarizer))
        return summary
    else:
        return "Could not find relevant information."
    

def retrieve_documents_startup(query: str) -> str:
    docs = vectordb_startup.similarity_search(query, k=30, ranker_type="weighted", ranker_params={"weights": [0.6, 0.4]})
    if docs :
        query_doc_pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(query_doc_pairs)
        reranked_docs = [
           doc for _, doc in sorted(
               zip(scores, docs),
               key=lambda x: x[0],
               reverse=True
             )
        ]
        top_docs = reranked_docs[:2]
        combined_text = "\n".join([doc.page_content for doc in top_docs])
        
        truncated_text = combined_text[:MAX_CHARS]
        # Summarize
        summary = asyncio.run(summarize_document(truncated_text, llm=llm_summarizer))
        return summary
    else:
        return "Could not find relevant information."
    

def retrieve_documents_startup_tunisia(query: str) -> str:
    docs = vectordb_startup_tunisia.similarity_search(query, k=30, ranker_type="weighted", ranker_params={"weights": [0.6, 0.4]})
    if docs :
        query_doc_pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(query_doc_pairs)
        reranked_docs = [
           doc for _, doc in sorted(
               zip(scores, docs),
               key=lambda x: x[0],
               reverse=True
             )
        ]
        top_docs = reranked_docs[:2]
        combined_text = "\n".join([doc.page_content for doc in top_docs])
    
        truncated_text = combined_text[:MAX_CHARS]
        # Summarize
        summary = asyncio.run(summarize_document(truncated_text, llm=llm_summarizer))
        return summary
    else:
        return "Could not find relevant information."



def retrieve_documents_certifications(query: str) -> str:
    docs = vectordb_certifications.similarity_search(query, k=30, ranker_type="weighted", ranker_params={"weights": [0.6, 0.4]})
    if docs :
        query_doc_pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(query_doc_pairs)
        reranked_docs = [
           doc for _, doc in sorted(
               zip(scores, docs),
               key=lambda x: x[0],
               reverse=True
             )
        ]
        top_docs = reranked_docs[:2]
        combined_text = "\n".join([doc.page_content for doc in top_docs])
        
        truncated_text = combined_text[:MAX_CHARS]
        # Summarize
        summary = asyncio.run(summarize_document(truncated_text, llm=llm_summarizer))
        return summary
    else:
        return "Could not find relevant information."



def retrieve_documents_Code_de_travail(query: str) -> str:
    docs = vectordb_Code_de_travail.similarity_search(query, k=30, ranker_type="weighted", ranker_params={"weights": [0.6, 0.4]})
    if docs :
        query_doc_pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(query_doc_pairs)
        reranked_docs = [
           doc for _, doc in sorted(
               zip(scores, docs),
               key=lambda x: x[0],
               reverse=True
             )
        ]
        top_docs = reranked_docs[:2]
        combined_text = "\n".join([doc.page_content for doc in top_docs])
       
        truncated_text = combined_text[:MAX_CHARS]
        # Summarize
        summary = asyncio.run(summarize_document(truncated_text, llm=llm_summarizer))
        return summary
    else:
        return "Could not find relevant information."



def retrieve_documents_cv_enhancement(query: str) -> str:
    docs = vectordb_cv_enhancement.similarity_search(query, k=30, ranker_type="weighted", ranker_params={"weights": [0.6, 0.4]})
    if docs :
        query_doc_pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(query_doc_pairs)
        reranked_docs = [
           doc for _, doc in sorted(
               zip(scores, docs),
               key=lambda x: x[0],
               reverse=True
             )
        ]
        top_docs = reranked_docs[:2]
        combined_text = "\n".join([doc.page_content for doc in top_docs])
       
        truncated_text = combined_text[:MAX_CHARS]
        # Summarize
        summary = asyncio.run(summarize_document(truncated_text, llm=llm_summarizer))
        return summary
    else:
        return "Could not find relevant information."



def retrieve_documents_international_labor_market(query: str) -> str:
    docs = vectordb_international_labor_market.similarity_search(query, k=30, ranker_type="weighted", ranker_params={"weights": [0.6, 0.4]})
    if docs :
        query_doc_pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(query_doc_pairs)
        reranked_docs = [
           doc for _, doc in sorted(
               zip(scores, docs),
               key=lambda x: x[0],
               reverse=True
             )
        ]
        top_docs = reranked_docs[:2]
        combined_text = "\n".join([doc.page_content for doc in top_docs])
        
        truncated_text = combined_text[:MAX_CHARS]
        # Summarize
        summary = asyncio.run(summarize_document(truncated_text, llm=llm_summarizer))
        return summary
    else:
        return "Could not find relevant information."



def retrieve_documents_tunisian_labor_market(query: str) -> str:
    docs = vectordb_tunisian_labor_market.similarity_search(query, k=30, ranker_type="weighted", ranker_params={"weights": [0.6, 0.4]})
    if docs :
        query_doc_pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(query_doc_pairs)
        reranked_docs = [
           doc for _, doc in sorted(
               zip(scores, docs),
               key=lambda x: x[0],
               reverse=True
             )
        ]
        top_docs = reranked_docs[:2]
        combined_text = "\n".join([doc.page_content for doc in top_docs])
        
        truncated_text = combined_text[:MAX_CHARS]
        # Summarize
        summary = asyncio.run(summarize_document(truncated_text, llm=llm_summarizer))
        return summary
    else:
        return "Could not find relevant information."


# Create the Retriever Tools
retriever_tool_freelance = Tool(
    name="international freelance retriever",
    func=retrieve_documents_freelance,
    description="Retrieve relevant documents from the vector store related to freelancing internationally based on semantic similarity."
)
retriever_tool_freelance_tunisia = Tool(
    name="tunisian freelance retriever",
    func=retrieve_documents_freelance_tunisia,
    description="Retrieve relevant documents from the vector store related to freelancing in tunisia based on semantic similarity."
)
retriever_tool_international_labor_market = Tool(
    name="international labor market retriever",
    func=retrieve_documents_international_labor_market,
    description="Retrieve relevant documents from the vector store related to international labor market based on semantic similarity."
)
retriever_tool_certifications = Tool(
    name="certifications retriever",
    func=retrieve_documents_certifications,
    description="Retrieve relevant documents from the vector store related to certifications based on semantic similarity."
)
retriever_tool_startup = Tool(
    name="international startup retriever",
    func=retrieve_documents_startup,
    description="Retrieve relevant documents from the vector store related to startup internationally based on semantic similarity."
)
retriever_tool_startup_tunisia = Tool(
    name="tunisian startup retriever",
    func=retrieve_documents_startup,
    description="Retrieve relevant documents from the vector store related to startup in tunisia based on semantic similarity."
)
retriever_tool_resume = Tool(
    name="resume retriever",
    func=retrieve_documents_cv_enhancement,
    description="Retrieve relevant documents from the vector store related to resume enhancement based on semantic similarity."
)
retriever_tool_Code_de_travail = Tool(
    name="labor code retriever",
    func=retrieve_documents_Code_de_travail,
    description="Retrieve relevant documents from the vector store related to the Tunisian labor code (Code du Travail en Tunisie) based on semantic similarity."
)
