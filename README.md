# Anytype MCP Server

A Model Context Protocol (MCP) server for Anytype that enables semantic search and RAG (Retrieval-Augmented Generation) capabilities over your Anytype documents.

## Features

- Semantic search across your Anytype documents
- Automatic chunking of documents for improved search accuracy
- Full metadata support including tags, dates, and custom properties

## Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) - Modern Python package installer
- [Ollama](https://ollama.ai) - For running the embedding model locally
- [Anytype](https://anytype.io) desktop application

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wethegreenpeople/anytype-mcp.git
cd anytype-mcp
```

2. Install dependencies:
```bash
uv sync
```

3. Install the mxbai-embed-large model with Ollama:
```bash
ollama pull mxbai-embed-large
```

4. Optional: Run initial document ingestion (this can be done later from the client, but depending on your machine and how many notes you have, this might take a few minutes):
```bash
python scripts/ingest_only.py
```

5. Optional: Install the server in Claude:
```bash
fastmcp install server.py
```

## Manual Configuration for Claude Desktop

If you want to manually add the Anytype MCP server to your Claude Desktop application, you'll need to modify the `claude_desktop_config.json` file. This file is typically located at:

- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

Add the following configuration to the `mcpServers` section of your `claude_desktop_config.json`:

```json
"anytype": {
  "command": "uv",
  "args": [
    "run",
    "--with",
    "chromadb",
    "--with",
    "fastmcp",
    "--with",
    "httpx",
    "--with",
    "nltk",
    "--with",
    "ollama",
    "--with",
    "platformdirs",
    "--with",
    "sentence-transformers",
    "fastmcp",
    "run",
    "[PATH_TO_ANYTYPE_MCP]/server.py"
  ]
}
```

Replace `[PATH_TO_ANYTYPE_MCP]` with the absolute path to your anytype-mcp directory. For example:
- Windows: `C:\\Users\\username\\Documents\\anytype-mcp\\server.py`
- macOS/Linux: `/home/username/anytype-mcp/server.py`

Make sure to use double backslashes (`\\`) for Windows paths in the JSON file.

## Authentication

The server handles authentication with Anytype via a challenge-based flow:
1. First run requires obtaining a challenge ID 
2. You'll need to provide a secret code from Anytype
3. The server stores the authentication token for future sessions

## Usage

The server provides several tools that can be used within Claude:

- `ingest_documents`: Ingest or update Anytype documents into the vector store
- `query_anytype_documents`: Perform semantic search across your documents
- `get_anytype_object`: Retrieve a specific Anytype object by ID
- `get_ingestion_stats`: View statistics about ingested documents
- `clear_ingestion`: Clear the vector store and start fresh

## Technical Details

### Embedding Model

The `mxbai-embed-large` model is used for generating embeddings in this project. It's fetched from Ollama and configured in the `EmbeddingUtils` class. If you prefer to use a different embedding model, you can modify the `OLLAMA_MODEL` variable in `utils/embedding_utils.py`.

### Data Storage

The application uses platformdirs for cross-platform data storage:
- ChromaDB data: Stored in a platform-specific user data directory
- NLTK data: Downloaded and stored locally
- Authentication tokens: Securely stored in a configuration file

### NLTK and Tokenization

This project uses the NLTK library for sentence tokenization, specifically the `punkt_tab` tokenizer. The tokenizer is automatically downloaded to the application's data directory during initialization.

## License

## Contributing

