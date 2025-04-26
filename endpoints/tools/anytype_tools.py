import logging
import re
from typing import Dict, List, TypedDict, Union, Set, Optional, Tuple, Any
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

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and filter metadata to remove duplicate and technical fields
        before returning to the user.
        
        Technical fields include internal IDs, redundant timestamps, and system identifiers
        that aren't meaningful to end users. This creates a more user-friendly
        response by prioritizing human-readable information.
        
        Args:
            metadata: The raw metadata dictionary from Anytype
            
        Returns:
            A cleaned metadata dictionary with only useful fields
        """
        if not metadata:
            return {}
            
        # Fields to keep even if they might be considered technical
        essential_fields = {
            "document_id", "space_id", "title"
        }
        
        # Fields to exclude completely (technical or internal use only)
        exclude_fields = {
            "chunk_index", "is_title_record"
        }
        
        # Fields that might have both ID and human-readable name
        # Format: {human_readable_name: id}
        duplicate_fields = {
            "Created Date": "created_date",
            "Last modified date": "last_modified_date",
            "Created By": "created_by",
            "Last modified by": "last_modified_by",
            "Due date": "due_date",
            "Tag": "tag",
            "Last opened date": "last_opened_date",
            "Hub": "679c2c12d1eba95760d9115f",
            "Concept": "679c3231d1eba95760d91180"
        }
        
        # Create a new filtered metadata dictionary
        cleaned = {}
        
        # Track which fields we've already processed to avoid duplicates
        processed_ids = set()
        
        # First pass: add essential fields and human-readable fields
        for key, value in metadata.items():
            # Skip excluded fields
            if key in exclude_fields:
                continue
                
            # Always keep essential fields
            if key in essential_fields:
                cleaned[key] = value
                processed_ids.add(key)
                continue
                
            # For hexadecimal IDs and UUID-like strings, skip unless essential
            if re.match(r'^[0-9a-f]{24,}$', str(key)) or re.match(r'^[0-9a-f]{8}-', str(key)):
                # These are likely internal IDs, skip
                continue
                
            # If this is a human-readable field with a duplicate technical ID
            if key in duplicate_fields:
                tech_id = duplicate_fields[key]
                # Mark the technical ID as processed so we don't add it later
                processed_ids.add(tech_id)
                # Use the human-readable name
                cleaned[key] = value
                continue
                
            # For all other fields, keep them if not already processed
            if key not in processed_ids:
                cleaned[key] = value
        
        return cleaned

    def _score_result(self, query: str, doc_title: str, metadata: dict, distance: float) -> float:
        """
        Calculate a compound score taking into account multiple relevance factors.
        
        This score helps rank results beyond simple vector similarity by considering:
        - Semantic similarity (distance from vector embedding)
        - Title match (exact or partial)
        - Metadata relevance (tags, dates, etc.)
        
        Lower scores indicate better matches, which will be prioritized in results.
        
        Args:
            query: The user's search query
            doc_title: The title of the document
            metadata: The document's metadata
            distance: The raw semantic distance score from vector similarity
            
        Returns:
            A compound score where lower values indicate higher relevance
        """
        # Start with the original distance score
        compound_score = distance
        
        # Boost for exact or partial title match
        if query.lower() == doc_title.lower():
            compound_score *= 0.3  # Strong boost for exact title match
        elif query.lower() in doc_title.lower():
            compound_score *= 0.5  # Moderate boost for partial title match
        
        # Boost documents with matching tags
        tags = metadata.get("tag", "")
        if isinstance(tags, str) and tags:
            for tag in tags.lower().split(","):
                if tag.strip() and tag.strip() in query.lower():
                    compound_score *= 0.8  # Small boost for matching tag
                
        # Boost for recently modified documents
        if "last_modified_date" in metadata:
            # Already handled by the ranking mechanism, but could add additional logic
            pass
        
        # Generic pattern matching for numbered items (lectures, assignments, chapters, etc.)
        # This matches patterns like "Lecture 3", "Chapter 2", "Assignment 5", etc.
        query_patterns = re.findall(r'(\w+)\s*(\d+)', query.lower())
        title_patterns = re.findall(r'(\w+)\s*(\d+)', doc_title.lower())
        
        if query_patterns and title_patterns:
            for q_type, q_num in query_patterns:
                for t_type, t_num in title_patterns:
                    # If both the type (lecture, chapter, etc.) and number match
                    if q_type == t_type and q_num == t_num:
                        compound_score *= 0.3  # Strong boost for matching numbered items
                    # If just the number matches and it's significant
                    elif q_num == t_num and len(q_num) > 0:
                        compound_score *= 0.6  # Moderate boost for matching numbers
        
        return compound_score

    def _get_dynamic_threshold(self, distances: List[float]) -> float:
        """
        Determine a dynamic threshold based on the distribution of distances.
        
        Instead of using a fixed cutoff, this adapts to different queries and document sets.
        This threshold is used to decide which documents should be returned with full content
        versus just chunks, optimizing for both relevance and performance.
        
        Args:
            distances: List of semantic distance scores from search results
            
        Returns:
            A floating point threshold value
        """
        if not distances:
            return 0.4  # Fallback to default
            
        # If we have few results, be more lenient
        if len(distances) <= 2:
            return max(distances) + 0.05
            
        # Sort distances (lower is better)
        sorted_distances = sorted(distances)
        
        # If there's a clear best result, use a tighter threshold
        if len(sorted_distances) >= 2 and sorted_distances[1] - sorted_distances[0] > 0.1:
            return sorted_distances[0] + 0.1
            
        # Otherwise use median + some margin
        if len(sorted_distances) >= 3:
            median = sorted_distances[len(sorted_distances) // 2]
            return min(median * 1.2, 0.5)  # Cap at 0.5
            
        # Default fallback
        return 0.4

    def _keyword_search(self, query: str, results_limit: int) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Perform a keyword-based search to complement semantic search.
        
        This method searches for exact matches of query terms in document text and titles,
        providing results even when semantic similarity might miss literal matches.
        
        Args:
            query: The search query string
            results_limit: Maximum number of results to return
            
        Returns:
            List of tuples containing (document text, metadata dictionary, score)
            where lower scores indicate better matches
        """
        try:
            # Simple implementation using collection.get and filtering
            all_docs = self.embedding_utils.collection.get(
                include=["metadatas", "documents"],
                limit=100  # Get a larger set to filter from
            )
            
            results = []
            
            # Prepare search terms
            search_terms = query.lower().split()
            
            for i, (doc_text, metadata) in enumerate(zip(
                all_docs["documents"],
                all_docs["metadatas"]
            )):
                # Skip if no document text
                if not doc_text:
                    continue
                    
                # Calculate match score based on term frequency
                matches = 0
                for term in search_terms:
                    if term in doc_text.lower() or term in metadata.get("title", "").lower():
                        matches += 1
                
                if matches > 0:
                    # Create a pseudo-distance (lower is better)
                    # Convert to a scale similar to embedding distances
                    score = 1.0 - (matches / len(search_terms))
                    
                    # Add exact match boost
                    if query.lower() in doc_text.lower():
                        score *= 0.7
                    
                    # Title exact match is even better
                    if query.lower() in metadata.get("title", "").lower():
                        score *= 0.5
                    
                    results.append((doc_text, metadata, score))
            
            # Sort by score (ascending) and limit
            results.sort(key=lambda x: x[2])
            return results[:results_limit]
            
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return []

    async def query_anytype_documents(
        self,
        query: str,
        results_limit: int = 5
    ) -> QueryResponse:
        """
        Perform a semantic search and RAG query on the ingested anytype documents.
        
        This method combines vector similarity search with keyword matching to find
        the most relevant Anytype documents based on the user's query. The search uses
        both semantic understanding (RAG - Retrieval Augmented Generation) and exact
        term matching for comprehensive results.
        
        Args:
            query: The semantic search query string
            results_limit: How many results to return from the query (default 5)
        
        Returns:
            Dictionary containing retrieved document references with the following fields:
            - status: Success or error status
            - references: List of document references with metadata, content, and similarity scores
              where lower similarity scores indicate better matches
        
        Example:
            await query_anytype_documents("programming concepts", 5)
        """
        # Prepare the base query for semantic search
        query_kwargs: Dict[str, Union[List[str], int, List[str]]] = {
            "query_texts": [query],
            "n_results": max(results_limit * 2, 10),  # Get more results for filtering
            "include": ["metadatas", "distances", "documents"]
        }

        # Use the collection from embedding_utils for semantic search
        semantic_results: Dict[str, List] = self.embedding_utils.collection.query(**query_kwargs)
        
        # Also perform keyword search
        keyword_results = self._keyword_search(query, results_limit)
        
        # Combine results from both approaches
        combined_results = []
        
        # Process semantic search results
        if "documents" in semantic_results and semantic_results["documents"]:
            for i, (doc_text, metadata, distance) in enumerate(zip(
                semantic_results["documents"][0],
                semantic_results["metadatas"][0],
                semantic_results["distances"][0]
            )):
                # Apply custom scoring
                adjusted_score = self._score_result(query, metadata.get("title", ""), metadata, distance)
                combined_results.append((doc_text, metadata, adjusted_score, "semantic"))
        
        # Add keyword results
        for doc_text, metadata, score in keyword_results:
            combined_results.append((doc_text, metadata, score, "keyword"))
        
        # Remove duplicates (prefer semantic results if same document)
        seen_doc_ids = set()
        filtered_results = []
        
        for doc_text, metadata, score, source in combined_results:
            doc_id = metadata.get("document_id", "")
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                filtered_results.append((doc_text, metadata, score, source))
        
        # Sort by adjusted score
        filtered_results.sort(key=lambda x: x[2])
        
        # Calculate dynamic threshold based on distribution of scores
        semantic_distances = [r[2] for r in filtered_results if r[3] == "semantic"]
        threshold = self._get_dynamic_threshold(semantic_distances)
        
        logger.debug(f"Dynamic threshold: {threshold}")
        
        # Limit to requested number of results
        top_results = filtered_results[:results_limit]
        
        # Fetch full documents where needed
        store: AnyTypeStore = self.anytype_auth.get_authenticated_store()
        final_references: List[DocumentReference] = []
        
        for doc_text, metadata, score, source in top_results:
            doc_id: str = metadata.get("document_id", "")
            space_id: str = metadata.get("space_id", "")
            title: str = metadata.get("title", "")
            
            # Clean the metadata before adding to results
            cleaned_metadata = self._clean_metadata(metadata)
            
            if score <= threshold or len(top_results) <= 1:
                # Under threshold or only result: include full doc
                try:
                    full_doc: Dict[str, object] = await store.get_document_async(doc_id, space_id)
                    final_references.append({
                        "id": doc_id,
                        "title": title,
                        "link": f"anytype://object?objectId={doc_id}&spaceId={space_id}",
                        "similarity_score": score,
                        "content": self.embedding_utils.flatten_anytype_blocks(full_doc["object"]),
                        "metadatas": cleaned_metadata
                    })
                except Exception as e:
                    logger.warning(f"Failed to fetch full document {doc_id}: {e}")
                    # Fallback to chunk
                    final_references.append({
                        "id": doc_id,
                        "title": title,
                        "link": f"anytype://object?objectId={doc_id}&spaceId={space_id}",
                        "similarity_score": score,
                        "chunk": doc_text,
                        "metadatas": cleaned_metadata
                    })
            else:
                # Otherwise include just the chunk
                final_references.append({
                    "id": doc_id,
                    "title": title,
                    "link": f"anytype://object?objectId={doc_id}&spaceId={space_id}",
                    "similarity_score": score,
                    "chunk": doc_text,
                    "metadatas": cleaned_metadata
                })

        return {
            "status": "success",
            "references": final_references
        }

    async def get_anytype_object(self, space_id: str, object_id: str) -> Dict[str, object]:
        """
        Get the contents of a single anytype object.
        
        This method retrieves the full content and structure of an Anytype object
        by its ID and space ID. It's useful for accessing complete information about 
        an object when you only have a reference or a chunk from a search result.
        
        Args:
            space_id: The space ID of the object
            object_id: The object's ID
            
        Returns:
            Dictionary containing the complete Anytype object data with its 
            content, metadata, and relationships
        """
        store: AnyTypeStore = self.anytype_auth.get_authenticated_store()
        return await store.get_document_async(object_id, space_id)