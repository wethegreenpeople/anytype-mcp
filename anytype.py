import logging
import os
from typing import List, Optional, Dict, Any

from fastmcp import FastMCP, Context
from platformdirs import user_data_dir
from anytype_store import AnyTypeStore
from fastmcp.prompts.base import UserMessage, AssistantMessage, Message
from anytype_authenticator import AnytypeAuthenticator
import chromadb
import nltk
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from nltk.tokenize import sent_tokenize
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# App-specific info
app_name = "anytype-mcp"
author = "John Singh"  # Used on Windows

# Cross-platform directory for app data
persist_dir = user_data_dir(app_name, author)
chroma_dir = os.path.join(persist_dir, "chroma-data")
nltk_dir = os.path.join(persist_dir, "nltk-data")

# Ensure directories exist
os.makedirs(chroma_dir, exist_ok=True)
os.makedirs(nltk_dir, exist_ok=True)

mcp = FastMCP("anytype")
anytype_auth = AnytypeAuthenticator(AnyTypeStore(None, None))
nltk.download("punkt_tab", download_dir=nltk_dir)

OLLAMA_MODEL = "mxbai-embed-large"

client = chromadb.PersistentClient(path=chroma_dir)
embedding_function = OllamaEmbeddingFunction(model_name=OLLAMA_MODEL)
collection = client.get_or_create_collection(name="anytype_pages", embedding_function=embedding_function)

def flatten_anytype_blocks(page: str) -> str:
    snippet = page.get("snippet", "")
    blocks = page.get("blocks", [])
    block_texts = [
        block.get("text", {}).get("text", "")
        for block in blocks
        if "text" in block and block["text"].get("text")
    ]
    return "\n".join([snippet] + block_texts).strip()

def split_into_sentences(text: str) -> list[str]:
    return sent_tokenize(text)

def chunk_sentences(sentences, max_len=500):
    chunks = []
    buffer = ""

    for sent in sentences:
        if len(buffer) + len(sent) <= max_len:
            buffer += " " + sent
        else:
            chunks.append(buffer.strip())
            buffer = sent

    if buffer:
        chunks.append(buffer.strip())

    return chunks

def ingest_anytype_documents(documents):
    for doc in documents:
        doc_id = doc["id"]
        title = doc.get("name", "")
        raw_text = flatten_anytype_blocks(doc)
        setenences = split_into_sentences(raw_text)
        chunks = chunk_sentences(setenences)

        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                metadatas=[{
                    "document_id": doc_id,
                    "chunk_index": i,
                    "space_id": doc.get("space_id"),
                    "title": title
                }],
                ids=[f"{doc_id}_{i}"]
            )

@mcp.tool()
async def ingest_documents() -> Dict[str, Any]:
    """
    Ingest anytype documents from the API into the vector store for RAG, semantic searches, and for additional information in other tools on this MCP server.
    
    Returns:
        Ingestion summary
    """
    documents = []
    offset = 0
    logger.info("Starting anytype ingestion")
    while (True):
        results = []
        results = (await store.query_documents_async(offset, 50, "")).get("data", [])
        documents.extend(results)

        if len(results) != 50: 
            break

        offset += 50

    ingest_anytype_documents(documents)
    
    return {
        "status": "success",
        "documents_ingested": len(documents)
    }

@mcp.tool()
async def query_documents(query: str) -> Dict[str, Any]:
    """
    Perform a semantic search and RAG query on the ingested anytype documents. Lower similarity scores are better matches
    
    Args:
        query: Search query string
    
    Returns:
        Dictionary containing the answer and retrieved document references
    """
    results = collection.query(
        query_texts=[query],
        n_results=5,
        include=["metadatas", "distances", "documents"]
    )

    threshold = 0.3
    all_references = [
        {
            "id": metadata.get("document_id"),
            "title": metadata.get("title"),
            "link": f"anytype://object?objectId={metadata.get('document_id')}&spaceId={metadata.get('space_id')}",
            "similarity_score": distance,
            "content": flatten_anytype_blocks((await store.get_document_async(metadata.get('document_id'), metadata.get('space_id')))["object"])
        }
        for doc_text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]
    
    filtered_references = [ref for ref in all_references if ref["similarity_score"] <= threshold]
    
    final_references = filtered_references if filtered_references else all_references
    
    return {
        "references": final_references,
        "status": "success"
    }

@mcp.tool()
async def get_ingestion_stats() -> Dict[str, Any]:
    """
    Get statistics about the Chroma DB ingestion process.
    
    Returns:
        Stats such as total chunks, unique documents, and model used.
    """
    from collections import Counter

    stats = {
        "collection_name": collection.name,
        "embedding_model": getattr(embedding_function, "model_name", "unknown")
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

@mcp.resource("anytype://{space_id}//{object_id}")
async def get_object(space_id: str, object_id: str) -> str:
    """Get the contents of a single anytype object"""
    store = anytype_auth.get_authenticated_store()
    return await store.get_document_async(object_id, space_id)

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')