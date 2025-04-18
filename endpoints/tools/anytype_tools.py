import logging
from typing import Any, Dict
from utils import EmbeddingUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

class AnyTypeTools():
    def __init__(self, mcp, anytype_auth, embedding_utils: EmbeddingUtils):
        self.mcp = mcp
        self.anytype_auth = anytype_auth
        self.embedding_utils = embedding_utils
        
        # Register methods as tools
        self.register_tools()

    def register_tools(self):
        """Register class methods as MCP tools"""
        self.mcp.tool()(self.query_anytype_documents)
        self.mcp.tool()(self.get_anytype_object)

    async def query_anytype_documents(
        self,
        query: str,
        results_limit: int = 5
    ) -> Dict[str, Any]:
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
        query_kwargs = {
            "query_texts": [query],
            "n_results": results_limit,
            "include": ["metadatas", "distances", "documents"]
        }

        # Use the collection from embedding_utils
        results = self.embedding_utils.collection.query(**query_kwargs)

        threshold = 0.4
        store = self.anytype_auth.get_authenticated_store()

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

    async def get_anytype_object(self, space_id: str, object_id: str) -> str:
        """Get the contents of a single anytype object

            Can be used to get extra information about objects that are stored in, or related to other objects. Can also be used to get the full document from only a chunk
            
            Args:
                space_id: The space ID of the object
                object_id: The object's ID
        """
        store = self.anytype_auth.get_authenticated_store()
        return await store.get_document_async(object_id, space_id)