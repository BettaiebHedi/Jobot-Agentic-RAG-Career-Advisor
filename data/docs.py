import json
from langchain.docstore.document import Document
from langchain_community.vectorstores import Milvus
from pymilvus import connections
from langchain_milvus import Milvus, BM25BuiltInFunction
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from utils.config import get_embeddings
from langchain.schema import Document
import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter


uri = f"http://{os.environ.get('MILVUS_HOST', 'standalone')}:{os.environ.get('MILVUS_PORT', '19530')}"

connections.connect(
    alias="default",
    uri=uri,
    user=os.environ.get("MILVUS_USER", "admin"),
    password=os.environ.get("MILVUS_PASSWORD", "admin123")
)


embedding_model=get_embeddings()


vectordb_freelance = Milvus(
    embedding_function=embedding_model,
    collection_name="freelance_hybrid",
    # vector_field="dense",
    # sparse_vector_field="sparse",
    builtin_function=BM25BuiltInFunction(), 
    vector_field=["dense", "sparse"],
    connection_args={"uri": uri, "user": os.environ.get("MILVUS_USER", "admin"), "password": os.environ.get("MILVUS_PASSWORD", "admin123")}
)



vectordb_freelance_tunisia = Milvus(
    embedding_function=embedding_model,
    collection_name="freelance_tunisia_hybrid",
    # vector_field="dense",
    # sparse_vector_field="sparse",
    builtin_function=BM25BuiltInFunction(), 
    vector_field=["dense", "sparse"],
    connection_args={"uri": uri, "user": os.environ.get("MILVUS_USER", "admin"), "password": os.environ.get("MILVUS_PASSWORD", "admin123")}
)




vectordb_startup_tunisia = Milvus(
    embedding_function=embedding_model,
    collection_name="startup_tunisia_hybrid",
    # vector_field="dense",
    # sparse_vector_field="sparse",
    builtin_function=BM25BuiltInFunction(), 
    vector_field=["dense", "sparse"],
    connection_args={"uri": uri, "user": os.environ.get("MILVUS_USER", "admin"), "password": os.environ.get("MILVUS_PASSWORD", "admin123")}
)




vectordb_startup = Milvus(
    embedding_function=embedding_model,
    collection_name="startup_hybrid",
    # vector_field="dense",
    # sparse_vector_field="sparse",
    builtin_function=BM25BuiltInFunction(), 
    vector_field=["dense", "sparse"],
    connection_args={"uri": uri, "user": os.environ.get("MILVUS_USER", "admin"), "password": os.environ.get("MILVUS_PASSWORD", "admin123")}
)




vectordb_certifications = Milvus(
    embedding_function=embedding_model,
    collection_name="certifications_hybrid",
    # vector_field="dense",
    # sparse_vector_field="sparse",
    builtin_function=BM25BuiltInFunction(), 
    vector_field=["dense", "sparse"],
    connection_args={"uri": uri, "user": os.environ.get("MILVUS_USER", "admin"), "password": os.environ.get("MILVUS_PASSWORD", "admin123")}
)



vectordb_Code_de_travail = Milvus(
    embedding_function=embedding_model,
    collection_name="Code_de_travail_hybrid",
    # vector_field="dense",
    # sparse_vector_field="sparse",
    builtin_function=BM25BuiltInFunction(), 
    vector_field=["dense", "sparse"],
    connection_args={"uri": uri, "user": os.environ.get("MILVUS_USER", "admin"), "password": os.environ.get("MILVUS_PASSWORD", "admin123")}
)




vectordb_cv_enhancement = Milvus(
    embedding_function=embedding_model,
    collection_name="cv_enhancement_hybrid",
    # vector_field="dense",
    # sparse_vector_field="sparse",
    builtin_function=BM25BuiltInFunction(), 
    vector_field=["dense", "sparse"],
    connection_args={"uri": uri, "user": os.environ.get("MILVUS_USER", "admin"), "password": os.environ.get("MILVUS_PASSWORD", "admin123")}
)




vectordb_international_labor_market = Milvus(
    embedding_function=embedding_model,
    collection_name="international_labor_market_hybrid",
    # vector_field="dense",
    # sparse_vector_field="sparse",
    builtin_function=BM25BuiltInFunction(), 
    vector_field=["dense", "sparse"],
    connection_args={"uri": uri, "user": os.environ.get("MILVUS_USER", "admin"), "password": os.environ.get("MILVUS_PASSWORD", "admin123")}
)




vectordb_tunisian_labor_market = Milvus(
    embedding_function=embedding_model,
    collection_name="tunisian_labor_market_hybrid",
    # vector_field="dense",
    # sparse_vector_field="sparse",
    builtin_function=BM25BuiltInFunction(), 
    vector_field=["dense", "sparse"],
    connection_args={"uri": uri, "user": os.environ.get("MILVUS_USER", "admin"), "password": os.environ.get("MILVUS_PASSWORD", "admin123")}
)