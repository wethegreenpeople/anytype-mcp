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

from utils.embedding_utils import EmbeddingUtils
from endpoints.tools.ingestion_tools import IngestionTools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Constants
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

# Utils
anytype_auth = AnytypeAuthenticator(AnyTypeStore(None, None), config_path)
embedding_utils = EmbeddingUtils(chroma_dir, cache_path)

# Initialize the ingestion tools with the required dependencies
ingestion_tools = IngestionTools(mcp, anytype_auth, embedding_utils)

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

    # Use the collection from embedding_utils
    results = embedding_utils.collection.query(**query_kwargs)

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