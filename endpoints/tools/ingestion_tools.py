import logging
from typing import Dict, List, Optional, TypedDict, Union
from collections import Counter
from fastmcp import FastMCP
from utils import EmbeddingUtils
from utils.anytype_authenticator import AnytypeAuthenticator
from anytype_api.anytype_store import AnyTypeStore
import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

class IngestionStatus(TypedDict):
    status: str
    documents_ingested: int

class StatsResponse(TypedDict, total=False):
    status: str
    message: Optional[str]
    collection_name: str
    embedding_model: str
    total_chunks: int
    unique_documents: int
    avg_chunks_per_document: float
    chroma_dir: str
    nltk_dir: Optional[str]

class IngestionTools():
    def __init__(self, mcp: FastMCP, anytype_auth: AnytypeAuthenticator, embedding_utils: EmbeddingUtils):
        self.mcp = mcp
        self.anytype_auth = anytype_auth
        self.embedding_utils = embedding_utils
        self.collection = self.embedding_utils.collection
        self.client = self.embedding_utils.client
        self.embedding_function = self.embedding_utils.embedding_function
        
        # Register methods as tools
        self.register_tools()

    @property
    def chroma_dir(self) -> str:
        return self.embedding_utils.chroma_dir

    @property
    def nltk_dir(self) -> Optional[str]:
        # This would need to be passed in from server.py
        # For now we'll return None as it's only used for reporting
        return None
        
    def register_tools(self) -> None:
        """Register class methods as MCP tools"""
        self.mcp.tool()(self.ingest_documents)
        self.mcp.tool()(self.get_ingestion_stats)
        self.mcp.tool()(self.clear_ingestion)

    async def ingest_documents(self) -> IngestionStatus:
        """
        Ingest anytype documents from the API into the vector store for RAG, semantic searches, and for additional information in other tools on this MCP server.
        Ingestion could have been done prior, so don't start the ingestion process unless explicitly asked to
        
        Returns:
            Ingestion summary
        """
        documents: List[Dict[str, object]] = []
        offset: int = 0
        logger.info("Starting anytype ingestion")
        store: AnyTypeStore = self.anytype_auth.get_authenticated_store()
        while (True):
            results: List[Dict[str, object]] = []
            results = (await store.query_documents_async(offset, 50, "")).get("data", [])
            documents.extend(results)

            if len(results) != 50: 
                break

            offset += 50

        self.embedding_utils.ingest_anytype_documents(documents)
        
        return {
            "status": "success",
            "documents_ingested": len(documents)
        }
    
    def get_stats(self) -> StatsResponse:
        stats: Dict[str, Union[str, int, float, None]] = {
            "collection_name": self.collection.name,
            "embedding_model": getattr(self.embedding_function, "model_name", "unknown")
        }

        if self.collection.count() == 0:
            return {
                "status": "success",
                "message": "No ingestion started"
            }
        try:
            all_docs = self.collection.get(include=["metadatas"], limit=100_000)

            metadatas_raw = all_docs.get("metadatas", [])
            
            # Handle both nested or flat metadata list structures
            if isinstance(metadatas_raw, list) and all(isinstance(m, list) for m in metadatas_raw):
                metadatas = metadatas_raw[0]
            else:
                metadatas = metadatas_raw

            doc_ids: List[str] = [
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
                return stats  # type: ignore

            count_per_doc = Counter(doc_ids)

            stats.update({
                "total_chunks": len(doc_ids),
                "unique_documents": len(count_per_doc),
                "avg_chunks_per_document": round(sum(count_per_doc.values()) / len(count_per_doc), 2),
                "chroma_dir": self.chroma_dir,
                "nltk_dir": self.nltk_dir
            })
            return stats  # type: ignore

        except Exception as e:
            logger.exception("Failed to get ingestion stats")
            return {
                "status": "error",
                "message": str(e)
            }

    async def get_ingestion_stats(self) -> StatsResponse:
        """
        Get statistics about the Chroma DB ingestion process.
        
        Returns:
            Stats such as total chunks, unique documents, and model used.
        """
        return self.get_stats()

    async def clear_ingestion(self) -> StatsResponse:
        """
        Clear previous ingestion
        
        Returns:
            Stats such as total chunks, unique documents, and model used.
        """
        self.client.delete_collection(name="anytype_pages")
        self.collection = self.client.get_or_create_collection(name="anytype_pages", embedding_function=self.embedding_function)
        return self.get_stats()