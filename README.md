# Anytype MCP Server

A Model Context Protocol (MCP) server for Anytype that enables semantic search and RAG capabilities over your Anytype documents.

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
git clone https://github.com/your-username/anytype-mcp.git
cd anytype-mcp
```

2. Install dependencies:
```bash
uv sync
```

3. Install the mxbai-embed-large model with Ollama:
```bash
ollama pull mxbai/embed-large
```

4. Optional: Run initial document ingestion (this can be done from the client but depending on your machine and how many notes you have, this might take a few minutes):
```bash
python ingest_only.py
```

5. Optional: Install the server in Claude:
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

## Additional Notes

### Using `mxbai-embed-large` from Ollama
The `mxbai-embed-large` model is used for generating embeddings in this project. It is fetched from Ollama and configured in the code. If you prefer to use a different embedding model, you can modify the `OLLAMA_MODEL` variable in `anytype.py` to point to your desired model.

### NLTK and `punkt_tab`
This project uses the NLTK library for sentence tokenization, specifically the `punkt_tab` tokenizer. The tokenizer is automatically downloaded to the application's data directory during the initial setup. If you prefer to use a different tokenizer or library, you can replace the `nltk` dependency and update the `split_into_sentences` function in `anytype.py` accordingly.

## License

## Contributing

