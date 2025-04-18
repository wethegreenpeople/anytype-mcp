from collections import Counter
import json
import logging
import os
import time
from typing import Dict, Any

from fastmcp import FastMCP
from platformdirs import user_data_dir
from anytype_api.anytype_store import AnyTypeStore
from utils.anytype_authenticator import AnytypeAuthenticator
import chromadb
import nltk
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from utils.embedding_utils import EmbeddingUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Constants
OLLAMA_MODEL = "mxbai-embed-large"
TOKENIZER = "punkt_tab"

# App-specific info
app_name = "anytype-mcp"
author = "John Singh"  # Used on Windows

# Cross-platform directories for app data
persist_dir = user_data_dir(app_name, author)
chroma_dir = os.path.join(persist_dir, "chroma-data")
nltk_dir = os.path.join(persist_dir, "nltk-data")
config_path = os.path.join(persist_dir, "anytype_config.json")
cache_path = os.path.join(persist_dir, "embed_cache.json")

# Ensure directories exist
os.makedirs(chroma_dir, exist_ok=True)
os.makedirs(nltk_dir, exist_ok=True)

# Initialize FastMCP
mcp = FastMCP("anytype", dependencies=[
    "chromadb",
    "httpx",
    "nltk",
    "ollama",
    "platformdirs",
    "sentence-transformers",
])

# Initialize NLTK tokenizer
nltk.download(TOKENIZER, download_dir=nltk_dir)
nltk.data.path.append(nltk_dir)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=chroma_dir)
embedding_function = OllamaEmbeddingFunction(model_name=OLLAMA_MODEL)
collection = client.get_or_create_collection(name="anytype_pages", embedding_function=embedding_function)

# Utils
anytype_auth = AnytypeAuthenticator(AnyTypeStore(None, None), config_path)
embedding_utils = EmbeddingUtils(client, collection, cache_path)

@mcp.tool()
async def ingest_documents() -> Dict[str, Any]:
    """
    Ingest anytype documents from the API into the vector store for RAG, semantic searches, and for additional information in other tools on this MCP server.
    Ingestion could have been done prior, so don't start the ingestion process unless explicitly asked to
    
    Returns:
        Ingestion summary
    """
    documents = []
    offset = 0
    logger.info("Starting anytype ingestion")
    store = anytype_auth.get_authenticated_store()
    while (True):
        results = []
        results = (await store.query_documents_async(offset, 50, "")).get("data", [])
        documents.extend(results)

        if len(results) != 50: 
            break

        offset += 50

    embedding_utils.ingest_anytype_documents(documents)
    
    return {
        "status": "success",
        "documents_ingested": len(documents)
    }

@mcp.tool()
async def query_documents(
    query: str
) -> Dict[str, Any]:
    """
    Perform a semantic search and RAG query on the ingested anytype documents.
    Lower similarity scores are better matches.

    Args:
        query: The semantic search query string
        metadata_filter: Optional dictionary of metadata fields to filter with regex

    Example:
        await query_documents("programming concepts", {"tags": r"programming"})

    Returns:
        Dictionary containing the answer and retrieved document references
    """
    # Prepare the base query
    query_kwargs = {
        "query_texts": [query],
        "n_results": 10,
        "include": ["metadatas", "distances", "documents"]
    }

    results = collection.query(**query_kwargs)

    threshold = 0.4
    store = anytype_auth.get_authenticated_store()

    seen_ids = set()
    final_references = []

    for i, (doc_text, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        doc_id = metadata.get("document_id")
        space_id = metadata.get("space_id")
        title = metadata.get("title")

        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        if i == 0 or distance <= threshold:
            # First best match or under threshold: include full doc
            full_doc = await store.get_document_async(doc_id, space_id)
            final_references.append({
                "id": doc_id,
                "title": title,
                "link": f"anytype://object?objectId={doc_id}&spaceId={space_id}",
                "similarity_score": distance,
                "content": embedding_utils.flatten_anytype_blocks(full_doc["object"]),
                "metadatas": metadata
            })
        else:
            # Otherwise include just the chunk
            final_references.append({
                "id": doc_id,
                "title": title,
                "link": f"anytype://object?objectId={doc_id}&spaceId={space_id}",
                "similarity_score": distance,
                "chunk": doc_text,
                "metadatas": metadata
            })

    return {
        "status": "success",
        "references": final_references
    }

def get_stats():
    stats = {
        "collection_name": collection.name,
        "embedding_model": getattr(embedding_function, "model_name", "unknown")
    }

    if collection.count() == 0:
        return {
            "status": "success",
            "message": "No ingestion started"
        }
    try:
        all_docs = collection.get(include=["metadatas"], limit=100_000)

        metadatas_raw = all_docs.get("metadatas", [])
        
        # Handle both nested or flat metadata list structures
        if isinstance(metadatas_raw, list) and all(isinstance(m, list) for m in metadatas_raw):
            metadatas = metadatas_raw[0]
        else:
            metadatas = metadatas_raw

        doc_ids = [
            meta.get("document_id")
            for meta in metadatas
            if isinstance(meta, dict) and meta.get("document_id")
        ]

        if not doc_ids:
            stats.update({
                "total_chunks": 0,
                "unique_documents": 0,
                "avg_chunks_per_document": 0.0
            })
            return stats

        count_per_doc = Counter(doc_ids)

        stats.update({
            "total_chunks": len(doc_ids),
            "unique_documents": len(count_per_doc),
            "avg_chunks_per_document": round(sum(count_per_doc.values()) / len(count_per_doc), 2),
            "chroma_dir": chroma_dir,
            "nltk_dir": nltk_dir
        })
        return stats

    except Exception as e:
        logger.exception("Failed to get ingestion stats")
        return {
            "status": "error",
            "message": str(e)
        }

@mcp.tool()
async def get_ingestion_stats() -> Dict[str, Any]:
    """
    Get statistics about the Chroma DB ingestion process.
    
    Returns:
        Stats such as total chunks, unique documents, and model used.
    """
    return get_stats()

@mcp.tool()
async def clear_ingestion() -> Dict[str, Any]:
    """
    Clear previous ingestion
    
    Returns:
        Stats such as total chunks, unique documents, and model used.
    """
    global collection
    client.delete_collection(name="anytype_pages")
    collection = client.get_or_create_collection(name="anytype_pages", embedding_function=embedding_function)
    return get_stats()
    


@mcp.tool()
async def get_object(space_id: str, object_id: str) -> str:
    """Get the contents of a single anytype object

        Can be used to get extra information about objects that are stored in, or related to other objects. Can also be used to get the full document from only a chunk
        
        Args:
            space_id: The space ID of the object
            object_id: The object's ID
    """
    store = anytype_auth.get_authenticated_store()
    return await store.get_document_async(object_id, space_id)

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')