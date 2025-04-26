import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the module search path
parent_dir = str(Path(__file__).parent.parent.absolute())
sys.path.append(parent_dir)

import nltk
from platformdirs import user_data_dir
from fastmcp import FastMCP
from utils.anytype_authenticator import AnytypeAuthenticator
from anytype_api.anytype_store import AnyTypeStore
from utils.embedding_utils import EmbeddingUtils
from endpoints.tools.ingestion_tools import IngestionTools

async def main():
    # Now run ingestion
    persist_dir = user_data_dir("anytype-mcp", "John Singh")
    config_path = os.path.join(persist_dir, "anytype_config.json")
    chroma_dir = os.path.join(persist_dir, "chroma-data")
    cache_path = os.path.join(persist_dir, "embed_cache.json")
    nltk_dir = os.path.join(persist_dir, "nltk-data")

    # Create directories if they don't exist
    os.makedirs(chroma_dir, exist_ok=True)
    os.makedirs(nltk_dir, exist_ok=True)
    
    # Set up auth
    auth = AnytypeAuthenticator(AnyTypeStore(None, None), config_path)
    if auth.config.get('app_token') == None:
        await auth.get_challenge_id_async()
        print("Secret Code: ")
        await auth.get_token_async(input())
    
    # Set up nltk
    tokenizer_path = os.path.join(nltk_dir, "tokenizers", "punkt_tab")
    if not os.path.exists(tokenizer_path):
        print(f"Downloading NLTK tokenizer to {nltk_dir}")
        nltk.download("punkt_tab", download_dir=nltk_dir)
    else:
        print(f"NLTK tokenizer already exists at {tokenizer_path}")
    nltk.data.path.append(nltk_dir)
    
    # Initialize dependencies
    mcp = FastMCP("anytype")
    embedding_utils = EmbeddingUtils(chroma_dir, cache_path)
    
    # Create and use IngestionTools
    ingestion_tools = IngestionTools(mcp, auth, embedding_utils)
    
    # Run ingestion
    result = await ingestion_tools.ingest_documents()
    print(f"Done ingesting docs! Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())