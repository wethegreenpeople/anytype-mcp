import json
import logging
import os
import time
import re
from typing import Dict, List, Optional, Union, Set, TypedDict, cast
import chromadb
from nltk.tokenize import sent_tokenize
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from chromadb.api.models.Collection import Collection


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

class AnytypeBlock(TypedDict, total=False):
    text: Dict[str, str]

class AnytypeObject(TypedDict, total=False):
    snippet: str
    blocks: List[AnytypeBlock]
    properties: List[Dict[str, object]]
    id: str
    name: str
    space_id: str

class Property(TypedDict, total=False):
    id: str
    name: str
    format: str
    multi_select: List[Dict[str, str]]
    date: str
    text: str
    object: object

class EmbeddingUtils:
    def __init__(self, chroma_dir: str, cache_path: str) -> None:
        OLLAMA_MODEL = "mxbai-embed-large"

        self.chroma_dir = chroma_dir
        self.client = chromadb.PersistentClient(path=chroma_dir)

        # Initialize ChromaDB client
        self.embedding_function = OllamaEmbeddingFunction(model_name=OLLAMA_MODEL)
        self.collection: Collection = self.client.get_or_create_collection(name="anytype_pages", embedding_function=self.embedding_function)
        self.cache_path = cache_path

        self.cache: Dict[str, str] = self._load_cache()
    
    def _load_cache(self) -> Dict[str, str]:
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                return json.load(f)
        return {}
    
    def flatten_anytype_blocks(self, page: AnytypeObject) -> str:
        """
        Convert Anytype blocks to plain text, preserving important document structure
        """
        snippet = page.get("snippet", "")
        title = page.get("name", "")
        blocks = page.get("blocks", [])
        
        # Start with title and snippet for better context
        lines = []
        if title:
            lines.append(f"# {title}")
        if snippet:
            lines.append(snippet)
            
        # Process each block
        for block in blocks:
            if "text" in block and block["text"].get("text"):
                block_text = block["text"].get("text", "")
                block_style = block["text"].get("style", "")
                
                # Format based on style to preserve structure
                if block_style == "Title":
                    lines.append(f"# {block_text}")
                elif block_style == "Header1":
                    lines.append(f"## {block_text}")
                elif block_style == "Header2" or block_style == "Header3":
                    lines.append(f"### {block_text}")
                elif block_style == "Numbered" or block_style == "Marked":
                    lines.append(f"- {block_text}")
                elif block_style == "Code":
                    lines.append(f"```\n{block_text}\n```")
                else:
                    lines.append(block_text)
        
        return "\n\n".join(lines).strip()

    def _split_into_semantic_chunks(self, text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into semantic chunks based on headers and paragraph structures
        with a maximum size and overlap between chunks for context continuity
        """
        # If text is small enough, return as is
        if len(text) <= max_chunk_size:
            return [text]
        
        # Split into sections by headers
        header_pattern = r'(?:^|\n)(#{1,3}\s+.*?)(?=\n|$)'
        sections = re.split(header_pattern, text)
        
        # Remove empty sections
        sections = [s.strip() for s in sections if s.strip()]
        
        # Further split large sections into paragraphs
        chunks = []
        current_chunk = ""
        
        for section in sections:
            # If it's a header or small section, try to keep it with its content
            if section.startswith('#') or len(section) < max_chunk_size // 2:
                if len(current_chunk) + len(section) + 2 <= max_chunk_size:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += "\n\n"
                    current_chunk += section
                else:
                    # Start a new chunk
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = section
            else:
                # For larger sections, split by paragraphs
                paragraphs = section.split("\n\n")
                
                for paragraph in paragraphs:
                    if len(paragraph.strip()) == 0:
                        continue
                        
                    if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
                        # Add to current chunk
                        if current_chunk:
                            current_chunk += "\n\n"
                        current_chunk += paragraph
                    else:
                        # Start a new chunk
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        # Handle paragraphs that are larger than max_chunk_size
                        if len(paragraph) > max_chunk_size:
                            # First try to split by sentences
                            sentences = sent_tokenize(paragraph)
                            
                            inner_chunk = ""
                            for sentence in sentences:
                                if len(inner_chunk) + len(sentence) + 2 <= max_chunk_size:
                                    if inner_chunk:
                                        inner_chunk += " "
                                    inner_chunk += sentence
                                else:
                                    if inner_chunk:
                                        chunks.append(inner_chunk)
                                    inner_chunk = sentence
                            
                            if inner_chunk:
                                current_chunk = inner_chunk
                            else:
                                current_chunk = ""
                        else:
                            current_chunk = paragraph
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)
        
        # Apply overlap between chunks for better context
        if len(chunks) > 1 and overlap > 0:
            overlapped_chunks = []
            
            for i, chunk in enumerate(chunks):
                if i > 0:
                    # Get end of previous chunk to create overlap
                    prev_chunk = chunks[i-1]
                    overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
                    
                    # Only add overlap if it won't exceed max_chunk_size
                    if len(chunk) + len(overlap_text) + 2 <= max_chunk_size * 1.2:  # Allow 20% overflow for overlap
                        chunk = f"{overlap_text}\n\n{chunk}"
                
                overlapped_chunks.append(chunk)
            
            chunks = overlapped_chunks
            
        # Add title context to each chunk if available
        if text.startswith("# "):
            title_line = text.split("\n")[0]
            chunks = [f"{title_line}\n\n{chunk}" if not chunk.startswith("# ") else chunk for chunk in chunks]
        
        return chunks

    def _extract_metadata(self, obj: AnytypeObject) -> Dict[str, object]:
        """
        Extract all relevant metadata from an Anytype object's properties
        """
        metadata: Dict[str, object] = {}
        
        for prop in obj.get('properties', []):
            prop_id = prop.get('id')
            prop_name = prop.get('name')
            prop_format = prop.get('format')
            
            if not prop_id:
                continue
                
            # Handle multi-select properties
            if prop_format == 'multi_select':
                metadata[prop_id] = [
                    tag.get('name', '') for tag in prop.get('multi_select', [])
                ]
            
            # Handle date properties
            elif prop_format == 'date':
                metadata[prop_id] = prop.get('date', '')
            
            # Handle text properties
            elif prop_format == 'text':
                metadata[prop_id] = prop.get('text', '')
            
            # Handle object properties
            elif prop_format == 'object':
                metadata[prop_id] = prop.get('object', {})
            
            # Add the property name as a key as well for more readable queries
            if prop_name and prop_name != prop_id:
                metadata[prop_name] = metadata.get(prop_id, '')
        
        return metadata

    def _sanitize_metadata_value(self, v: object) -> Union[str, int, float, bool]:
        if v is None:
            return ""
        if isinstance(v, list):
            return ", ".join(map(str, v))  # in case we missed a list
        if isinstance(v, dict):
            return json.dumps(v)  # Convert dictionaries to JSON strings
        return v

    def _sanitize_metadata(self, metadata: Dict[str, object]) -> Dict[str, object]:
        return {k: self._sanitize_metadata_value(v) for k, v in metadata.items()}

    def _save_ingest_cache(self, updated_cache: Dict[str, str]) -> None:
        with open(self.cache_path, "w") as f:
            json.dump(updated_cache, f, indent=2)

    def ingest_anytype_documents(self, documents: List[AnytypeObject]) -> None:
        total_docs = len(documents)
        logger.info(f"Starting ingestion of {total_docs} documents...")

        start_time = time.time()
        updated_cache = self.cache.copy()
        skipped_docs = 0
        processed_docs = 0

        for idx, doc in enumerate(documents):
            obj = doc
            
            doc_id = obj.get("id", "")
            title = obj.get("name", "")

            # Extract metadata from properties
            properties = {prop['id']: cast(Property, prop) for prop in obj.get('properties', []) if 'id' in prop}
            
            # Extract dates
            created_date = properties.get('created_date', {}).get('date', '')
            last_modified_date = properties.get('last_modified_date', {}).get('date', '')

            if self.collection.count() > 0 and doc_id in self.cache and self.cache[doc_id] == last_modified_date:
                skipped_docs += 1
                logger.debug(f"Skipping '{title}' (unchanged)")
                continue

            processed_docs += 1
            # Extract tags
            tags_prop = next((prop for prop in obj.get('properties', []) if prop.get('id') == 'tag'), None)
            tags = tags_prop.get('multi_select', []) if tags_prop else []
            tags = [tag.get('name', '') for tag in tags]

            # Extract space_id
            space_id = obj.get('space_id', '')

            # Flatten text content with improved formatting
            raw_text = self.flatten_anytype_blocks(obj)
            
            # Split into semantic chunks that preserve document structure
            chunks = self._split_into_semantic_chunks(raw_text)

            # Create special metadata for the document title to improve searching
            title_metadata = {
                "document_id": doc_id,
                "chunk_index": -1,  # Special index for title
                "space_id": space_id,
                "title": title,
                "is_title_record": True,
                **self._extract_metadata(obj)
            }

            # Delete existing entries for this document
            self.collection.delete(where={"document_id": doc_id})
            
            # Add the title as a special record with extra weight
            self.collection.add(
                documents=[f"Title: {title}"],
                metadatas=[self._sanitize_metadata(title_metadata)],
                ids=[f"{doc_id}_title"]
            )

            # Add each semantic chunk
            metadata = self._extract_metadata(obj) 
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "document_id": doc_id,
                    "chunk_index": i,
                    "space_id": space_id,
                    "title": title,
                    "is_title_record": False,
                    **metadata
                }
                
                self.collection.add(
                    documents=[chunk],
                    metadatas=[self._sanitize_metadata(chunk_metadata)],
                    ids=[f"{doc_id}_{i}"]
                )

            updated_cache[doc_id] = last_modified_date

            # Log progress
            if (idx + 1) % 10 == 0 or (idx + 1) == total_docs:
                elapsed = time.time() - start_time
                docs_done = idx + 1
                docs_left = total_docs - docs_done
                
                if processed_docs > 0:
                    # Calculate rate and ETA based on processed docs only
                    avg_time_per_doc = elapsed / processed_docs
                    est_time_left = avg_time_per_doc * (total_docs - skipped_docs - processed_docs)
                    processing_rate = processed_docs / elapsed if elapsed > 0 else 0
                    
                    logger.info(
                        f"Progress: {docs_done}/{total_docs} docs "
                        f"({(docs_done / total_docs) * 100:.1f}%) | "
                        f"Processed: {processed_docs} | "
                        f"Skipped: {skipped_docs} | "
                        f"Rate: {processing_rate:.2f} docs/sec | "
                        f"Elapsed: {elapsed:.1f}s | "
                        f"ETA: {est_time_left:.1f}s"
                    )
                else:
                    # If we haven't processed any docs yet, can't calculate per-doc stats
                    logger.info(
                        f"Progress: {docs_done}/{total_docs} docs "
                        f"({(docs_done / total_docs) * 100:.1f}%) | "
                        f"Processed: {processed_docs} | "
                        f"Skipped: {skipped_docs} | "
                        f"Elapsed: {elapsed:.1f}s"
                    )

        self._save_ingest_cache(updated_cache)
        self.cache = updated_cache
        
        # Final summary
        total_elapsed = time.time() - start_time
        logger.info(
            f"Ingestion complete! "
            f"Total: {total_docs} | "
            f"Processed: {processed_docs} | "
            f"Skipped: {skipped_docs} | "
            f"Time: {total_elapsed:.1f}s"
        )