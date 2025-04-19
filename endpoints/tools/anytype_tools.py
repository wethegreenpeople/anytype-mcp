import logging
from typing import Dict, List, TypedDict, Union, Set, Optional
from fastmcp import FastMCP
from utils import EmbeddingUtils
from utils.anytype_authenticator import AnytypeAuthenticator
from anytype_api.anytype_store import AnyTypeStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

class DocumentMetadata(TypedDict, total=False):
    document_id: str
    space_id: str
    title: str
    chunk_index: int

class DocumentReference(TypedDict, total=False):
    id: str
    title: str
    link: str
    similarity_score: float
    content: Optional[str]
    chunk: Optional[str]
    metadatas: DocumentMetadata

class QueryResponse(TypedDict):
    status: str
    references: List[DocumentReference]

class AnyTypeTools():
    def __init__(self, mcp: FastMCP, anytype_auth: AnytypeAuthenticator, embedding_utils: EmbeddingUtils):
        self.mcp = mcp
        self.anytype_auth = anytype_auth
        self.embedding_utils = embedding_utils
        
        # Register methods as tools
        self.register_tools()

    def register_tools(self) -> None:
        """Register class methods as MCP tools"""
        self.mcp.tool()(self.query_anytype_documents)
        self.mcp.tool()(self.get_anytype_object)

    async def query_anytype_documents(
        self,
        query: str,
        results_limit: int = 5
    ) -> QueryResponse:
        """
        Perform a semantic search and RAG query on the ingested anytype documents.
        Lower similarity scores are better matches.

        Args:
            query: The semantic search query string
            results_limit: How many results to return from the query (default 5)

        Example:
            await query_documents("programming concepts", {"tags": r"programming"})

        Returns:
            Dictionary containing the answer and retrieved document references
        """
        # Prepare the base query
        query_kwargs: Dict[str, Union[List[str], int, List[str]]] = {
            "query_texts": [query],
            "n_results": results_limit,
            "include": ["metadatas", "distances", "documents"]
        }

        # Use the collection from embedding_utils
        results: Dict[str, List] = self.embedding_utils.collection.query(**query_kwargs)

        threshold: float = 0.4
        store: AnyTypeStore = self.anytype_auth.get_authenticated_store()

        seen_ids: Set[str] = set()
        final_references: List[DocumentReference] = []

        for i, (doc_text, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            doc_id: str = metadata.get("document_id", "")
            space_id: str = metadata.get("space_id", "")
            title: str = metadata.get("title", "")

            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            if i == 0 or distance <= threshold:
                # First best match or under threshold: include full doc
                full_doc: Dict[str, object] = await store.get_document_async(doc_id, space_id)
                final_references.append({
                    "id": doc_id,
                    "title": title,
                    "link": f"anytype://object?objectId={doc_id}&spaceId={space_id}",
                    "similarity_score": distance,
                    "content": self.embedding_utils.flatten_anytype_blocks(full_doc["object"]),
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

    async def get_anytype_object(self, space_id: str, object_id: str) -> Dict[str, object]:
        """Get the contents of a single anytype object

            Can be used to get extra information about objects that are stored in, or related to other objects. Can also be used to get the full document from only a chunk
            
            Args:
                space_id: The space ID of the object
                object_id: The object's ID
        """
        store: AnyTypeStore = self.anytype_auth.get_authenticated_store()
        return await store.get_document_async(object_id, space_id)