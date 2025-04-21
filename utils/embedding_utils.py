import json
import logging
import os
import time
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
        snippet = page.get("snippet", "")
        blocks = page.get("blocks", [])
        block_texts = [
            block.get("text", {}).get("text", "")
            for block in blocks
            if "text" in block and block["text"].get("text")
        ]
        return "\n".join([snippet] + block_texts).strip()

    def _split_into_sentences(self, text: str) -> List[str]:
        return sent_tokenize(text)

    def _chunk_sentences(self, sentences: List[str], max_len: int = 500) -> List[str]:
        chunks: List[str] = []
        buffer = ""

        for sent in sentences:
            if len(buffer) + len(sent) <= max_len:
                buffer += " " + sent
            else:
                chunks.append(buffer.strip())
                buffer = sent

        if buffer:
            chunks.append(buffer.strip())

        return chunks

    def _save_ingest_cache(self, updated_cache: Dict[str, str]) -> None:
        with open(self.cache_path, "w") as f:
            json.dump(updated_cache, f, indent=2)

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

    def ingest_anytype_documents(self, documents: List[AnytypeObject]) -> None:
        total_docs = len(documents)
        logger.info(f"Starting ingestion of {total_docs} documents...")

        start_time = time.time()
        updated_cache = self.cache.copy()

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
                logger.debug(f"Skipping '{title}' (unchanged)")
                continue

            # Extract tags
            tags_prop = next((prop for prop in obj.get('properties', []) if prop.get('id') == 'tag'), None)
            tags = tags_prop.get('multi_select', []) if tags_prop else []
            tags = [tag.get('name', '') for tag in tags]

            # Extract space_id
            space_id = obj.get('space_id', '')

            # Flatten text content
            raw_text = self.flatten_anytype_blocks(obj)
            full_text = f"{raw_text}"
            sentences = self._split_into_sentences(full_text)
            chunks = self._chunk_sentences(sentences)

            self.collection.delete(where={"document_id": doc_id})
            metadata = self._extract_metadata(obj)
            for i, chunk in enumerate(chunks):
                self.collection.add(
                    documents=[chunk],
                    metadatas=[self._sanitize_metadata({
                        "document_id": doc_id,
                        "chunk_index": i,
                        "space_id": space_id,
                        "title": title,
                        **metadata  # This unpacks all the extracted metadata
                    })],
                    ids=[f"{doc_id}_{i}"]
                )

            updated_cache[doc_id] = last_modified_date

            # Log progress
            if (idx + 1) % 10 == 0 or (idx + 1) == total_docs:
                elapsed = time.time() - start_time
                docs_done = idx + 1
                docs_left = total_docs - docs_done
                avg_time_per_doc = elapsed / docs_done
                est_time_left = avg_time_per_doc * docs_left

                logger.info(f"Ingested {docs_done}/{total_docs} docs "
                    f"({(docs_done / total_docs) * 100:.1f}%) | "
                    f"Elapsed: {elapsed:.1f}s | "
                    f"ETA: {est_time_left:.1f}s")

        self._save_ingest_cache(updated_cache)
        self.cache = updated_cache
        logger.info("Ingest cache updated!")