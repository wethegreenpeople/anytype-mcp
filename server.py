import logging
import os
from typing import Optional

from fastmcp import FastMCP
from platformdirs import user_data_dir
from anytype_api.anytype_store import AnyTypeStore
from utils.anytype_authenticator import AnytypeAuthenticator
import nltk

from utils.embedding_utils import EmbeddingUtils
from endpoints.tools import IngestionTools
from endpoints.tools import AnyTypeTools

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

# Initialize the MCP tools with the required dependencies
ingestion_tools = IngestionTools(mcp, anytype_auth, embedding_utils)
anytype_tools = AnyTypeTools(mcp, anytype_auth, embedding_utils)

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')