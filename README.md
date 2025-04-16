# Anytype MCP Server

A Model Context Protocol (MCP) server for Anytype that enables semantic search and RAG capabilities over your Anytype documents.

## Features

- Semantic search across your Anytype documents
- Automatic chunking of documents for improved search accuracy
- Full metadata support including tags, dates, and custom properties
- Built with FastMCP for efficient communication
- Uses ChromaDB for vector storage and Ollama for embeddings

## Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) - Modern Python package installer
- [Ollama](https://ollama.ai) - For running the embedding model locally
- [Anytype](https://anytype.io) desktop application

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/anytype-mcp.git
cd anytype-mcp
```

2. Install dependencies:
```bash
uv sync
```

3. Run initial document ingestion:
```bash
python ingest_only.py
```

4. Install the server in Claude:
```bash
fastmcp install anytype.py
```

## Usage

The server provides several tools that can be used within Claude:

- `ingest_documents`: Ingest or update Anytype documents into the vector store
- `query_documents`: Perform semantic search across your documents
- `get_object`: Retrieve a specific Anytype object by ID
- `get_ingestion_stats`: View statistics about ingested documents
- `clear_ingestion`: Clear the vector store and start fresh

## License



## Contributing

