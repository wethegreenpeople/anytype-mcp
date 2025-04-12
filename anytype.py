import asyncio
import os
import logging
from typing import List, Optional, Dict, Any

from fastmcp import FastMCP, Context
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores.utils import filter_complex_metadata
from anytype_store import AnyTypeStore
from langchain_ollama import OllamaEmbeddings
from fastmcp.prompts.base import UserMessage, AssistantMessage, Message
from anytype_authenticator import AnytypeAuthenticator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("anytype")
anytype_auth = AnytypeAuthenticator(AnyTypeStore(None, None))

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap=100
)

# Prompt template for RAG
rag_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant answering questions based on the uploaded documents.
    Context:
    {context}
    
    Question:
    {question}
    
    Answer concisely and accurately in three sentences or less. If you are not confident in any answer based on the provided context, say "I'm not sure about that" exactly, and then provide your best guess.
    """
)

# Global vector store and retriever
vector_store = None
document_store = None
retriever = None

@mcp.tool()
async def ingest_documents() -> Dict[str, Any]:
    """
    Ingest anytype documents from the API into the vector store for RAG, semantic searches, and for additional information in other tools on this MCP server.
    
    Returns:
        Ingestion summary
    """
    global vector_store, document_store, retriever
    documents = []
    offset = 0
    store = anytype_auth.get_authenticated_store()
    while (True):
        results = []
        results = (await store.get_documents_async(offset, 50)).get("data", [])
        documents.extend(results)

        if len(results) != 50: 
            break

        offset += 50
    
    # Convert Anytype documents to Langchain Documents
    docs = []
    for page in documents:
        page_id = page.get("id", "")
        title = page.get("name", "")
        snippet = page.get("snippet", "")

        # Extract visible text from blocks
        blocks = page.get("blocks", [])
        block_texts = [
            block.get("text", {}).get("text", "")
            for block in blocks
            if "text" in block and block["text"].get("text")
        ]
        full_content = "\n".join([snippet] + block_texts).strip()

        # Extract metadata
        tags = [
            tag["name"] for tag in page.get("details", []) 
            if tag["id"] == "tags"
            for tag in tag.get("details", {}).get("tags", [])
        ]

        author = next((
            detail["details"]["details"].get("name", "Unknown")
            for detail in page.get("details", [])
            if detail["id"] == "created_by"
        ), "Unknown")

        created = next((
            detail["details"].get("created_date")
            for detail in page.get("details", [])
            if detail["id"] == "created_date"
        ), None)

        docs.append(Document(
            page_content=full_content,
            metadata={
                "id": page_id,
                "title": title,
                "author": author,
                "created_date": created,
                "tags": tags,
                "space_id": page.get("space_id")
            }
        ))
    
    # Filter and preprocess documents
    docs = filter_complex_metadata(docs)
    
    # Initialize vector store and retriever
    vector_store = Chroma(
        collection_name="full_documents",
        embedding_function=embeddings
    )
    
    document_store = InMemoryStore()
    
    retriever = ParentDocumentRetriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.20},
        vectorstore=vector_store,
        docstore=document_store,
        child_splitter=text_splitter,
    )
    
    # Add documents to the retriever
    retriever.add_documents(docs)
    
    return {
        "status": "success",
        "documents_ingested": len(docs),
        "collection_name": "full_documents"
    }

@mcp.tool()
async def query_documents(query: str) -> Dict[str, Any]:
    """
    Perform a semantic search and RAG query on the ingested anytype documents.
    
    Args:
        query: Search query string
    
    Returns:
        Dictionary containing the answer and retrieved document references
    """
    global retriever, vector_store, document_store
    
    # Validate vector store and retriever
    if not retriever or not vector_store or not document_store:
        return {
            "error": "No documents have been ingested. Please use ingest_documents first.",
            "status": "error"
        }
    
    # Retrieve relevant documents
    logger.info(f"Retrieving context for query: {query}")
    retrieved_docs = retriever.invoke(query)
    
    if not retrieved_docs:
        return {
            "answer": "No relevant context found in the documents to answer your question.",
            "references": [],
            "status": "no_context"
        }
    
    references = [
        {
            "id": doc.metadata.get('id'),
            "title": doc.metadata.get('title', 'Untitled'),
            "link": f"anytype://object?objectId={doc.metadata.get('id')}&spaceId={doc.metadata.get('space_id')}",
            "similarity score": retriever.vectorstore.similarity_search_with_relevance_scores(query, k=1)[0][1],
            "content": "\n\n".join(doc.page_content for doc in retrieved_docs)
        }
        for doc in retrieved_docs
    ]
    
    return {
        "references": references,
        "status": "success"
    }

@mcp.tool()
async def clear_document_store() -> Dict[str, str]:
    """
    Clear the current vector store and retriever.
    
    Returns:
        Status of the clearing operation
    """
    global vector_store, document_store, retriever
    
    vector_store = None
    document_store = None
    retriever = None
    
    return {
        "status": "success",
        "message": "Document store and retriever have been cleared."
    }

@mcp.resource("rag://model-info")
def get_rag_model_info() -> Dict[str, Any]:
    """
    Provide information about the RAG models being used.
    
    Returns:
        Dictionary with model details
    """
    return {
        "embedding_model": "mxbai-embed-large",
        "embedding_dimension": len(embeddings.embed_query("test"))
    }

@mcp.prompt()
async def document_q_a(query: str) -> list[Message]:
    """Ask a question and provide the context of your anytype documents for the response"""

    global retriever, vector_store, document_store
    
    # Validate vector store and retriever
    if not retriever or not vector_store or not document_store:
        return {
            "error": "No documents have been ingested. Please use ingest_documents first.",
            "status": "error"
        }
    
    # Retrieve relevant documents
    logger.info(f"Retrieving context for query: {query}")
    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    if not retrieved_docs:
        return [AssistantMessage("No documents were retrieved for your query")]

    return[
        UserMessage(f"You are a helpful assistant answering questions based on the uploaded document.\n\n Context: {context} \n\nQuestion: {query} \n\nAnswer concisely and accurately in three sentences or less. If you are not confident in any answer based on the provided context, say 'I'm not sure about that' exactly, and then provide your best guess at the answer.")
    ]

@mcp.resource("anytype://{space_id}//{object_id}")
async def get_object(space_id: str, object_id: str) -> str:
    """Get the contents of a single anytype object"""
    store = anytype_auth.get_authenticated_store()
    return await store.get_document_async(object_id, space_id)

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')