"""
RAG Utilities for Causal Impact Knowledge Base

This module provides:
- Hierarchical chunking by H1/H2 headers with metadata inheritance
- Google text-embedding-004 integration
- ChromaDB vector store management
- Semantic search with context
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

import chromadb
from chromadb.config import Settings
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Base paths
BASE_DIR = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
CHROMA_DB_DIR = KNOWLEDGE_BASE_DIR / "chroma_db"

# Embedding model
EMBEDDING_MODEL = "models/text-embedding-004"


class GoogleEmbeddingFunction:
    """Custom embedding function using Google's text-embedding-004 model."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Google embedding function.
        
        Args:
            api_key: Optional Google API key. If not provided, uses GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found. Set it as environment variable or pass directly.")
        genai.configure(api_key=self.api_key)
        
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            input: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in input:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query (uses different task type).
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector
        """
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']


class HierarchicalChunker:
    """
    Chunk markdown documents by H1 and H2 headers with metadata inheritance.
    
    Structure:
    - H1 headers define major sections (parent context)
    - H2 headers define chunks (inherit H1 context)
    - Content under H1 but before first H2 is captured
    - Long sections are split with recursive character splitting
    """
    
    def __init__(self, max_chunk_size: int = 1500, overlap: int = 100):
        """
        Initialize chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk before splitting
            overlap: Character overlap when splitting long sections
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        
    def chunk_document(self, content: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Chunk a markdown document into hierarchical sections.
        
        Args:
            content: Markdown content
            source_file: Source filename for metadata
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        chunks = []
        
        # Split by H1 headers
        h1_pattern = r'^# (.+)$'
        h1_sections = re.split(h1_pattern, content, flags=re.MULTILINE)
        
        # First element is content before any H1 (usually empty)
        if h1_sections[0].strip():
            chunks.extend(self._process_section(
                content=h1_sections[0],
                h1_title="Introduction",
                source_file=source_file
            ))
        
        # Process H1 sections (pairs of title, content)
        for i in range(1, len(h1_sections), 2):
            if i + 1 < len(h1_sections):
                h1_title = h1_sections[i].strip()
                h1_content = h1_sections[i + 1]
                chunks.extend(self._process_section(
                    content=h1_content,
                    h1_title=h1_title,
                    source_file=source_file
                ))
        
        return chunks
    
    def _process_section(self, content: str, h1_title: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Process an H1 section, splitting by H2 headers.
        
        Args:
            content: Section content
            h1_title: H1 header title
            source_file: Source filename
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # Split by H2 headers
        h2_pattern = r'^## (.+)$'
        h2_sections = re.split(h2_pattern, content, flags=re.MULTILINE)
        
        # Content before first H2 (introduction to the H1 section)
        if h2_sections[0].strip():
            intro_chunks = self._create_chunks(
                content=h2_sections[0].strip(),
                h1_title=h1_title,
                h2_title="Overview",
                source_file=source_file
            )
            chunks.extend(intro_chunks)
        
        # Process H2 sections
        for i in range(1, len(h2_sections), 2):
            if i + 1 < len(h2_sections):
                h2_title = h2_sections[i].strip()
                h2_content = h2_sections[i + 1].strip()
                
                if h2_content:
                    h2_chunks = self._create_chunks(
                        content=h2_content,
                        h1_title=h1_title,
                        h2_title=h2_title,
                        source_file=source_file
                    )
                    chunks.extend(h2_chunks)
        
        return chunks
    
    def _create_chunks(
        self, 
        content: str, 
        h1_title: str, 
        h2_title: str, 
        source_file: str
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from content, splitting if necessary.
        
        Args:
            content: Text content
            h1_title: Parent H1 title
            h2_title: H2 section title
            source_file: Source filename
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # If content is short enough, create single chunk
        if len(content) <= self.max_chunk_size:
            chunk_id = self._generate_chunk_id(source_file, h1_title, h2_title, 0)
            chunks.append({
                "id": chunk_id,
                "content": content,
                "metadata": {
                    "source_file": source_file,
                    "h1_title": h1_title,
                    "h2_title": h2_title,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "parent_context": f"{h1_title} > {h2_title}"
                }
            })
        else:
            # Split long content
            split_contents = self._recursive_split(content)
            total_chunks = len(split_contents)
            
            for idx, split_content in enumerate(split_contents):
                chunk_id = self._generate_chunk_id(source_file, h1_title, h2_title, idx)
                chunks.append({
                    "id": chunk_id,
                    "content": split_content,
                    "metadata": {
                        "source_file": source_file,
                        "h1_title": h1_title,
                        "h2_title": h2_title,
                        "chunk_index": idx,
                        "total_chunks": total_chunks,
                        "parent_context": f"{h1_title} > {h2_title}"
                    }
                })
        
        return chunks
    
    def _recursive_split(self, content: str) -> List[str]:
        """
        Recursively split content by paragraphs, then sentences if needed.
        
        Args:
            content: Text to split
            
        Returns:
            List of content pieces
        """
        # Try splitting by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', content)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.max_chunk_size:
                current_chunk += ("\n\n" + para if current_chunk else para)
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If single paragraph is too long, split by sentences
                if len(para) > self.max_chunk_size:
                    sentence_chunks = self._split_by_sentences(para)
                    chunks.extend(sentence_chunks[:-1])
                    current_chunk = sentence_chunks[-1] if sentence_chunks else ""
                else:
                    current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.max_chunk_size:
                current_chunk += (" " + sentence if current_chunk else sentence)
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _generate_chunk_id(self, source_file: str, h1: str, h2: str, index: int) -> str:
        """Generate a unique chunk ID."""
        content = f"{source_file}:{h1}:{h2}:{index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class KnowledgeBaseManager:
    """
    Manage the ChromaDB vector store for the causal impact knowledge base.
    """
    
    def __init__(
        self, 
        collection_name: str = "causal_impact_kb",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize knowledge base manager.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(CHROMA_DB_DIR)
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Initialize embedding function
        self.embedding_fn = GoogleEmbeddingFunction()
        
        # Initialize chunker
        self.chunker = HierarchicalChunker()
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create new one."""
        try:
            collection = self.client.get_collection(
                name=self.collection_name
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Causal Impact Best Practices Knowledge Base"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def index_documents(self, docs_directory: Optional[str] = None, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Index all markdown documents in the knowledge base directory.
        
        Args:
            docs_directory: Directory containing markdown files
            force_reindex: If True, delete existing and reindex all
            
        Returns:
            Dict with indexing statistics
        """
        docs_dir = Path(docs_directory) if docs_directory else KNOWLEDGE_BASE_DIR
        
        if force_reindex:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Causal Impact Best Practices Knowledge Base"}
            )
            logger.info("Deleted existing collection for reindexing")
        
        # Find all markdown files
        md_files = list(docs_dir.glob("*.md"))
        
        stats = {
            "files_processed": 0,
            "total_chunks": 0,
            "chunks_by_file": {}
        }
        
        all_chunks = []
        
        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8")
                chunks = self.chunker.chunk_document(content, md_file.name)
                all_chunks.extend(chunks)
                
                stats["files_processed"] += 1
                stats["chunks_by_file"][md_file.name] = len(chunks)
                
                logger.info(f"Processed {md_file.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {md_file.name}: {e}")
        
        # Add all chunks to collection
        if all_chunks:
            ids = [c["id"] for c in all_chunks]
            documents = [c["content"] for c in all_chunks]
            metadatas = [c["metadata"] for c in all_chunks]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} chunks...")
            embeddings = self.embedding_fn(documents)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            stats["total_chunks"] = len(all_chunks)
            logger.info(f"Indexed {len(all_chunks)} chunks to ChromaDB")
        
        return stats
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results with content and metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_fn.embed_query(query)
        
        # Build query args
        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": n_results
        }
        
        if filter_metadata:
            query_args["where"] = filter_metadata
        
        # Execute search
        results = self.collection.query(**query_args)
        
        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                result = {
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "id": results["ids"][0][i] if results["ids"] else None
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def search_with_context(
        self, 
        query: str, 
        n_results: int = 5
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Search and return results with formatted context string.
        
        Args:
            query: Search query
            n_results: Number of results
            
        Returns:
            Tuple of (results list, formatted context string)
        """
        results = self.search(query, n_results)
        
        # Build context string
        context_parts = []
        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            context_parts.append(
                f"[Source {i}: {metadata.get('source_file', 'unknown')} > "
                f"{metadata.get('h1_title', '')} > {metadata.get('h2_title', '')}]\n"
                f"{result['content']}"
            )
        
        context_string = "\n\n---\n\n".join(context_parts)
        
        return results, context_string
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()
        
        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "persist_directory": self.persist_directory
        }


# Convenience functions for tools integration

_kb_manager: Optional[KnowledgeBaseManager] = None


def get_kb_manager() -> KnowledgeBaseManager:
    """Get or create the singleton KnowledgeBaseManager instance."""
    global _kb_manager
    if _kb_manager is None:
        _kb_manager = KnowledgeBaseManager()
    return _kb_manager


def search_knowledge_base(query: str, n_results: int = 5) -> str:
    """
    Search the causal impact knowledge base.
    
    This is the main function to be used as an agent tool.
    
    Args:
        query: Natural language question or search terms
        n_results: Number of results to return (default 5)
        
    Returns:
        Formatted string with search results and context
    """
    try:
        kb = get_kb_manager()
        results, context = kb.search_with_context(query, n_results)
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        return f"Found {len(results)} relevant sections:\n\n{context}"
    except Exception as e:
        logger.exception("Error searching knowledge base")
        return f"Error searching knowledge base: {str(e)}"


def index_knowledge_base(force_reindex: bool = False) -> str:
    """
    Index or reindex the knowledge base documents.
    
    Args:
        force_reindex: If True, delete existing index and reindex all
        
    Returns:
        String with indexing statistics
    """
    try:
        kb = get_kb_manager()
        stats = kb.index_documents(force_reindex=force_reindex)
        
        return (
            f"Indexing complete!\n"
            f"Files processed: {stats['files_processed']}\n"
            f"Total chunks: {stats['total_chunks']}\n"
            f"Chunks by file:\n" + 
            "\n".join(f"  - {f}: {c}" for f, c in stats['chunks_by_file'].items())
        )
    except Exception as e:
        logger.exception("Error indexing knowledge base")
        return f"Error indexing knowledge base: {str(e)}"


def get_knowledge_base_stats() -> str:
    """
    Get statistics about the knowledge base.
    
    Returns:
        String with collection statistics
    """
    try:
        kb = get_kb_manager()
        stats = kb.get_collection_stats()
        
        return (
            f"Knowledge Base Statistics:\n"
            f"  Collection: {stats['collection_name']}\n"
            f"  Total documents: {stats['total_documents']}\n"
            f"  Storage: {stats['persist_directory']}"
        )
    except Exception as e:
        return f"Error getting stats: {str(e)}"
