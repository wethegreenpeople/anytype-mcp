from collections import Counter
import json
import logging
import os
import time
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
logging.getLogger("httpx").setLevel(logging.WARNING)

# App-specific info
app_name = "anytype-mcp"
author = "John Singh"  # Used on Windows

# Cross-platform directory for app data
persist_dir = user_data_dir(app_name, author)
chroma_dir = os.path.join(persist_dir, "chroma-data")
nltk_dir = os.path.join(persist_dir, "nltk-data")
config_path = os.path.join(persist_dir, "anytype_config.json")
cache_path = os.path.join(persist_dir, "embed_cache.json")

# Ensure directories exist
os.makedirs(chroma_dir, exist_ok=True)
os.makedirs(nltk_dir, exist_ok=True)

mcp = FastMCP("anytype", dependencies=[
    "chromadb>=1.0.4",
    "httpx>=0.28.1",
    "mcp[cli]>=1.6.0",
    "nltk>=3.9.1",
    "ollama>=0.4.7",
    "platformdirs>=4.3.7",
    "sentence-transformers>=4.0.2",
])
anytype_auth = AnytypeAuthenticator(AnyTypeStore(None, None), config_path)
nltk.download("punkt_tab", download_dir=nltk_dir)
nltk.data.path.append(nltk_dir)

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


def load_ingest_cache():
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}

def save_ingest_cache(cache):
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

def sanitize_metadata_value(v):
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(map(str, v))  # in case we missed a list
    return v

def sanitize_metadata(metadata: dict) -> dict:
    return {k: sanitize_metadata_value(v) for k, v in metadata.items()}

def extract_metadata(obj):
    """
    Extract all relevant metadata from an Anytype object's properties
    """
    metadata = {}
    
    for prop in obj.get('properties', []):
        prop_id = prop.get('id')
        prop_name = prop.get('name')
        prop_format = prop.get('format')
        
        # Handle multi-select properties
        if prop_format == 'multi_select':
            metadata[prop_id] = [
                tag.get('name', '') for tag in prop.get('multi_select', [])
            ]
        
        # Handle date properties
        elif prop_format == 'date':
            metadata[prop_id] = prop.get('date')
        
        # Handle text properties
        elif prop_format == 'text':
            metadata[prop_id] = prop.get('text')
        
        # Handle object properties
        elif prop_format == 'object':
            metadata[prop_id] = prop.get('object')
        
        # Add the property name as a key as well for more readable queries
        if prop_name and prop_name != prop_id:
            metadata[prop_name] = metadata.get(prop_id)
    
    return metadata

def ingest_anytype_documents(documents):
    total_docs = len(documents)
    print(f"Starting ingestion of {total_docs} documents...")

    ingest_cache = load_ingest_cache()
    start_time = time.time()
    updated_cache = ingest_cache.copy()

    for idx, doc in enumerate(documents):
        obj = doc
        
        doc_id = obj.get("id")
        title = obj.get("name", "")

        # Extract metadata from properties
        properties = {prop['id']: prop for prop in obj.get('properties', [])}
        
        # Extract dates
        created_date = properties.get('created_date', {}).get('date')
        last_modified_date = properties.get('last_modified_date', {}).get('date')

        if collection.count() > 0 and doc_id in ingest_cache and ingest_cache[doc_id] == last_modified_date:
            print(f"Skipping '{title}' (unchanged)")
            continue

        # Extract tags
        tags_prop = next((prop for prop in obj.get('properties', []) if prop['id'] == 'tag'), None)
        tags = tags_prop.get('multi_select', []) if tags_prop else []
        tags = [tag.get('name', '') for tag in tags]

        # Extract space_id
        space_id = obj.get('space_id')

        # Flatten text content
        raw_text = flatten_anytype_blocks(obj)
        full_text = f"{raw_text}"
        sentences = split_into_sentences(full_text)
        chunks = chunk_sentences(sentences)

        collection.delete(where={"document_id": doc_id})
        metadata = extract_metadata(obj)
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                 metadatas=[sanitize_metadata({
                    "document_id": doc_id,
                    "chunk_index": i,
                    "space_id": space_id,
                    "title": title,
                    **metadata  # This unpacks all the extracted metadata
                })],
                ids=[f"{doc_id}_{i}"]
            )

        updated_cache[doc_id] = last_modified_date

        # Log progress
        if (idx + 1) % 10 == 0 or (idx + 1) == total_docs:
            elapsed = time.time() - start_time
            docs_done = idx + 1
            docs_left = total_docs - docs_done
            avg_time_per_doc = elapsed / docs_done
            est_time_left = avg_time_per_doc * docs_left

            print(f"Ingested {docs_done}/{total_docs} docs "
                  f"({(docs_done / total_docs) * 100:.1f}%) | "
                  f"Elapsed: {elapsed:.1f}s | "
                  f"ETA: {est_time_left:.1f}s")

    save_ingest_cache(updated_cache)
    print("Ingest cache updated!")

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

    ingest_anytype_documents(documents)
    
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
                "content": flatten_anytype_blocks(full_doc["object"]),
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