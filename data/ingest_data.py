
import json
from utils.config import get_embeddings
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain.schema import Document
from pymilvus import connections
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os





# Connect to Milvus
uri = f"http://{os.environ.get('MILVUS_HOST', 'localhost')}:{os.environ.get('MILVUS_PORT', '19530')}"
connections.connect(
    alias="default",
    uri=uri,
    user=os.environ.get("MILVUS_USER", "admin"),
    password=os.environ.get("MILVUS_PASSWORD", "admin123")
)

embedding_model = get_embeddings()
splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)

def safe_document(text):
    if len(text) > 65000:
        return [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    return [Document(page_content=text)]

def generate_docs(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        loaded_chunks = json.load(f)
    docs_processed = [chunk[0] for chunk in loaded_chunks]
    split_docs = []
    for text in docs_processed:
        split_docs.extend(safe_document(text))
    return split_docs

DATASETS = {
    "freelance_hybrid": "data/freelance.json",
    "freelance_tunisia_hybrid": "data/freelance_tunisia.json",
    "startup_tunisia_hybrid": "data/startup_tunisia.json",
    "startup_hybrid": "data/startup.json",
    "Code_de_travail_hybrid": "data/Code_Travail.json",
    "cv_enhancement_hybrid": "data/Cv_enhancement.json",
    "international_labor_market_hybrid": "data/International_labor_Market.json",
    "tunisian_labor_market_hybrid": "data/Tunisian_labor_market.json",
}

for collection_name, json_file in DATASETS.items():
    print(f"Ingesting {collection_name} ...")
    docs = generate_docs(json_file)
    vectordb = Milvus.from_documents(
        documents=docs,
        embedding=embedding_model,
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        collection_name=collection_name,
        connection_args={"uri": uri, "user": os.environ.get("MILVUS_USER", "admin"),"password": os.environ.get("MILVUS_PASSWORD", "admin123")},
        consistency_level="Strong",
        drop_old=True
    )
    print(f"Done: {collection_name} -> {len(docs)} docs")


def generate_docs_for_certifications(doc_name):
    with open(doc_name, "r", encoding="utf-8") as f:
        loaded_chunks = json.load(f)
    
    docs_processed = []
    for chunk in loaded_chunks:
        if isinstance(chunk, dict):
            # Convert all key-value pairs into a string
            full_text = "\n".join(f"{key}: {value}" for key, value in chunk.items())
            docs_processed.append(Document(page_content=full_text))
        else:
            # Handle non-dict entries, just convert to string
            docs_processed.append(Document(page_content=str(chunk)))
    
    return docs_processed


vectordb_certifications = Milvus.from_documents(
    documents=generate_docs_for_certifications("data/Certifications.json"),
    embedding=embedding_model,
    builtin_function=BM25BuiltInFunction(),
    vector_field=["dense", "sparse"],
    collection_name="certifications_hybrid",
    connection_args={"uri": uri,"user": os.environ.get("MILVUS_USER", "admin"),"password": os.environ.get("MILVUS_PASSWORD", "admin123")},
    consistency_level="Strong",
    drop_old=True
)
