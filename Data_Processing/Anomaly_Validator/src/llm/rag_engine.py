"""
RAG (Retrieval-Augmented Generation) Engine for Web-Scraped Content

Uses embeddings to retrieve only the most relevant chunks of web-scraped data
before passing to the LLM, optimizing token usage and improving relevance.
"""

import os
import time
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """Represents a chunk of text with its embedding and metadata."""
    text: str
    embedding: np.ndarray
    source: str  # URL or document identifier
    chunk_id: int
    metadata: Dict[str, Any]


class RAGEngine:
    """
    RAG engine for efficient context retrieval from large web-scraped data.
    
    Uses Gemini embeddings to create vector representations of scraped content,
    then retrieves only the most relevant chunks for LLM analysis.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model: str = "models/embedding-001",
        chunk_size: int = 1000,  # Balanced chunk size for both APIs
        chunk_overlap: int = 100,  # Overlap for context
        top_k: int = 5,
        batch_size: int = 10,  # Texts per batch (A4F token limit: 8192 total)
        requests_per_minute: int = 4,  # Conservative for A4F's 5 RPM limit
        use_a4f: bool = False,  # Try Gemini first, fallback to A4F
        a4f_fallback: bool = True  # Enable A4F as fallback
    ):
        """
        Initialize RAG engine with Gemini (primary) and A4F (fallback).
        
        Args:
            api_key: Gemini API key
            embedding_model: Embedding model name
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
            top_k: Number of top relevant chunks to retrieve
            batch_size: Number of texts to embed per API request
            requests_per_minute: Rate limit for API calls
            use_a4f: If True, use A4F as primary; if False, use Gemini
            a4f_fallback: Enable A4F as automatic fallback
        """
        self.use_a4f = use_a4f
        self.a4f_fallback = a4f_fallback
        self.gemini_quota_exhausted = False
        
        # Always set up Gemini (primary or fallback)
        primary_key = api_key or os.getenv('GEMINI_API_KEY')
        if primary_key:
            self.api_keys = [primary_key]
            
            # Collect all Gemini API keys
            fallback_key = os.getenv('GEMINI_API_KEY_FALLBACK')
            if fallback_key:
                fallback_keys = [k.strip() for k in fallback_key.split(',') if k.strip()]
                self.api_keys.extend(fallback_keys)
            
            for i in range(2, 10):
                extra_key = os.getenv(f'GEMINI_API_KEY_{i}')
                if extra_key:
                    self.api_keys.append(extra_key.strip())
            
            genai.configure(api_key=self.api_keys[0])
            self.embedding_model = embedding_model
            self.embedding_dim = 768
        else:
            self.api_keys = []
        
        # Setup A4F (primary or fallback) with parallel support
        self.a4f_api_keys = []
        primary_a4f = os.getenv('A4F_API_KEY')
        if primary_a4f:
            self.a4f_api_keys.append(primary_a4f)
        
        # Add additional A4F keys for parallel processing
        for i in range(2, 10):
            extra_a4f_key = os.getenv(f'A4F_API_KEY_{i}')
            if extra_a4f_key:
                self.a4f_api_keys.append(extra_a4f_key)
        
        self.a4f_api_key = self.a4f_api_keys[0] if self.a4f_api_keys else None
        if self.a4f_api_key:
            self.a4f_url = "https://api.a4f.co/v1/embeddings"
            self.a4f_model = "provider-6/qwen3-embedding-4b"
            self.a4f_dim = 2560
        
        if use_a4f and not self.a4f_api_key:
            raise ValueError("A4F API key required (set A4F_API_KEY env var)")
        
        if not use_a4f and not self.api_keys:
            raise ValueError("Gemini API key required for RAG engine")
        
        # Print configuration
        if use_a4f:
            print(f"✓ RAG Engine initialized with A4F API (Primary, PARALLEL mode):")
            print(f"  - Model: {self.a4f_model}")
            print(f"  - Embedding dim: {self.a4f_dim}")
            print(f"  - API keys: {len(self.a4f_api_keys)} keys for parallel processing")
            print(f"  - Batch size: 5 chunks/request per key")
            if self.a4f_fallback and self.api_keys:
                print(f"  - Fallback: Gemini ({len(self.api_keys)} keys) if A4F fails")
        else:
            print(f"✓ RAG Engine initialized with Gemini API (Primary, PARALLEL mode):")
            print(f"  - Model: {self.embedding_model}")
            print(f"  - API keys: {len(self.api_keys)} keys for parallel processing")
            if self.a4f_fallback and len(self.a4f_api_keys) > 0:
                print(f"  - Fallback: A4F API ({len(self.a4f_api_keys)} keys, parallel) if Gemini quota exhausted")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.batch_size = batch_size
        self.requests_per_minute = requests_per_minute
        self.min_delay = 60.0 / requests_per_minute
        self.last_request_time = 0
        
        # Storage for document chunks
        self.chunks: List[DocumentChunk] = []
        
        print(f"  - Chunk size: {chunk_size} chars")
        print(f"  - Delay: {self.min_delay:.2f}s between requests")
    
    def chunk_text(self, text: str, source: str, metadata: Dict[str, Any] = None) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            source: Source identifier (URL, filename, etc.)
            metadata: Additional metadata
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.5:  # At least 50% of chunk
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        
        return chunks
    
    def _embed_single_a4f(self, texts: List[str], api_key: str, key_id: int) -> tuple:
        """
        Embed texts with a specific A4F API key.
        
        Args:
            texts: List of texts to embed (batch of 5)
            api_key: A4F API key to use
            key_id: ID of the key (for logging)
            
        Returns:
            Tuple of (list of embeddings, key_id, error)
        """
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # A4F supports array of strings
            payload = {
                "model": self.a4f_model,
                "input": texts if len(texts) > 1 else texts[0]
            }
            
            # Use shorter timeout and disable retries to fail fast
            response = requests.post(
                self.a4f_url, 
                json=payload, 
                headers=headers, 
                timeout=(5, 15)  # (connect timeout, read timeout)
            )
            
            if not response.ok:
                error_msg = f"{response.status_code}: {response.text[:200]}"
                return ([np.zeros(self.a4f_dim) for _ in texts], key_id, error_msg)
            
            result = response.json()
            
            if 'data' not in result:
                return ([np.zeros(self.a4f_dim) for _ in texts], key_id, "Invalid response format")
            
            embeddings_data = sorted(result['data'], key=lambda x: x.get('index', 0))
            embeddings = [np.array(item['embedding']) for item in embeddings_data]
            
            return (embeddings, key_id, None)
            
        except Exception as e:
            return ([np.zeros(self.a4f_dim) for _ in texts], key_id, str(e)[:100])
    
    def _embed_batch_a4f_parallel(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed batch using A4F API with PARALLEL processing across multiple keys.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        num_keys = len(self.a4f_api_keys)
        # Limit parallelism to avoid overwhelming API server
        max_workers = min(3, num_keys)  # Max 3 concurrent requests
        print(f"  → Parallel A4F embedding: {len(texts)} chunks across {num_keys} API keys ({max_workers} workers)")
        
        # Split texts into mini-batches of 5 (A4F token limit)
        batch_size = 5
        mini_batches = []
        for i in range(0, len(texts), batch_size):
            mini_batches.append(texts[i:i+batch_size])
        
        embeddings_dict = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Distribute mini-batches across API keys
            for batch_idx, mini_batch in enumerate(mini_batches):
                key_index = batch_idx % num_keys
                api_key = self.a4f_api_keys[key_index]
                future = executor.submit(self._embed_single_a4f, mini_batch, api_key, key_index)
                futures.append((future, batch_idx, mini_batch))
            
            # Collect results
            completed = 0
            errors = 0
            error_messages = []
            
            for future, batch_idx, mini_batch in futures:
                try:
                    batch_embeddings, key_id, error = future.result(timeout=30)
                    
                    # Store embeddings with their original indices
                    start_idx = batch_idx * batch_size
                    for i, emb in enumerate(batch_embeddings):
                        embeddings_dict[start_idx + i] = emb
                    
                    if error:
                        errors += len(mini_batch)
                        if len(error_messages) < 3:
                            error_messages.append(f"Key {key_id}: {error}")
                    else:
                        completed += len(mini_batch)
                    
                    if (completed + errors) % 20 == 0:
                        print(f"    ✓ {completed + errors}/{len(texts)} chunks processed ({errors} errors)")
                        
                except Exception as e:
                    start_idx = batch_idx * batch_size
                    for i in range(len(mini_batch)):
                        embeddings_dict[start_idx + i] = np.zeros(self.a4f_dim)
                    errors += len(mini_batch)
                    if len(error_messages) < 3:
                        error_messages.append(f"Exception: {str(e)[:100]}")
        
        # Return embeddings in original order
        embeddings = [embeddings_dict[i] for i in range(len(texts))]
        print(f"  ✓ Parallel A4F embedding complete: {completed}/{len(texts)} successful")
        
        if error_messages:
            print(f"  ⚠️  Sample errors:")
            for err_msg in error_messages:
                print(f"     - {err_msg}")
        
        return embeddings
    
    def _embed_batch_a4f(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed batch using A4F API (uses parallel if multiple keys available).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Use parallel processing if we have multiple A4F keys
        if len(self.a4f_api_keys) > 1:
            return self._embed_batch_a4f_parallel(texts)
        
        # Fallback to single-key processing
        batch_embeddings, _, error = self._embed_single_a4f(texts, self.a4f_api_key, 0)
        if error:
            print(f"\n⚠️  A4F error: {error}")
        return batch_embeddings
    
    def _embed_single_gemini(self, text: str, api_key: str, key_id: int) -> tuple:
        """
        Embed a single text with a specific Gemini API key.
        
        Args:
            text: Text to embed
            api_key: Gemini API key to use
            key_id: ID of the key (for logging)
            
        Returns:
            Tuple of (embedding vector, key_id)
        """
        try:
            # Configure this specific key (thread-safe)
            import google.generativeai as genai_thread
            genai_thread.configure(api_key=api_key)
            
            # Add small delay to respect rate limits (6s for 10 RPM)
            time.sleep(6)
            
            result = genai_thread.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            return (np.array(result['embedding']), key_id, None)
            
        except Exception as e:
            return (np.zeros(self.embedding_dim), key_id, str(e)[:100])
    
    def _embed_batch_gemini_parallel(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed batch using Gemini API with PARALLEL processing across multiple keys.
        
        Distributes chunks across all available API keys and processes them simultaneously.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors in original order
        """
        if not texts:
            return []
        
        num_keys = len(self.api_keys)
        print(f"  → Parallel embedding: {len(texts)} chunks across {num_keys} API keys")
        
        # Distribute texts across API keys
        embeddings_dict = {}  # {index: embedding}
        
        with ThreadPoolExecutor(max_workers=num_keys) as executor:
            futures = []
            
            # Submit all embedding tasks in parallel
            for idx, text in enumerate(texts):
                key_index = idx % num_keys  # Round-robin distribution
                api_key = self.api_keys[key_index]
                future = executor.submit(self._embed_single_gemini, text, api_key, key_index)
                futures.append((future, idx))
            
            # Collect results as they complete
            completed = 0
            errors = 0
            error_messages = []
            for future, original_idx in futures:
                try:
                    embedding, key_id, error = future.result(timeout=30)
                    embeddings_dict[original_idx] = embedding
                    
                    if error:
                        errors += 1
                        if len(error_messages) < 3:  # Keep first 3 error messages
                            error_messages.append(f"Key {key_id}: {error}")
                    else:
                        completed += 1
                    
                    if (completed + errors) % 10 == 0:
                        print(f"    ✓ {completed + errors}/{len(texts)} chunks processed ({errors} errors)")
                        
                except Exception as e:
                    embeddings_dict[original_idx] = np.zeros(self.embedding_dim)
                    errors += 1
                    if len(error_messages) < 3:
                        error_messages.append(f"Exception: {str(e)[:100]}")
        
        # Return embeddings in original order
        embeddings = [embeddings_dict[i] for i in range(len(texts))]
        print(f"  ✓ Parallel embedding complete: {completed}/{len(texts)} successful")
        
        # Print sample errors if any
        if error_messages:
            print(f"  ⚠️  Sample errors:")
            for err_msg in error_messages:
                print(f"     - {err_msg}")
        
        return embeddings
    
    def _embed_batch_gemini(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed batch using Gemini API (uses parallel processing if multiple keys available).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Use parallel processing if we have multiple keys
        if len(self.api_keys) > 1:
            return self._embed_batch_gemini_parallel(texts)
        
        # Fallback to sequential for single key (original implementation)
        embeddings = []
        for text in texts:
            try:
                elapsed = time.sleep(self.min_delay) if hasattr(self, 'last_request_time') else None
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(np.array(result['embedding']))
            except Exception as e:
                print(f"\n⚠️  Gemini error: {str(e)[:100]}")
                embeddings.append(np.zeros(self.embedding_dim))
        
        return embeddings
    
    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed batch of texts using configured API with automatic fallback.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (zero vectors if all APIs exhausted)
        """
        # Try primary API first
        if self.use_a4f:
            result = self._embed_batch_a4f(texts)
            # Check if all embeddings are zero (indicating failure)
            if self.a4f_fallback and self.api_keys and all(np.all(emb == 0) for emb in result):
                print(f"  🔄 A4F failed, falling back to Gemini...")
                gemini_result = self._embed_batch_gemini(texts)
                # Update embedding dimension to Gemini's
                self.embedding_dim = 768
                return gemini_result
            return result
        else:
            result = self._embed_batch_gemini(texts)
            # Check if quota exhausted (all embeddings zero)
            success_count = sum(1 for emb in result if not np.all(emb == 0))
            if self.a4f_fallback and self.a4f_api_key and success_count == 0:
                if not self.gemini_quota_exhausted:
                    print(f"  🔄 Gemini quota exhausted, switching to A4F fallback...")
                    self.gemini_quota_exhausted = True
                # Update embedding dimension to A4F's
                self.embedding_dim = self.a4f_dim
                # Split into smaller batches for A4F (stricter token limit)
                # A4F has 8192 token limit per request, so use batch size of 5
                if len(texts) > 5:
                    all_embeddings = []
                    for i in range(0, len(texts), 5):
                        batch = texts[i:i+5]
                        batch_embeddings = self._embed_batch_a4f(batch)
                        all_embeddings.extend(batch_embeddings)
                        # Check if this batch also failed (all zeros)
                        if all(np.all(emb == 0) for emb in batch_embeddings):
                            print(f"  ⚠️  All embedding APIs exhausted, returning zero vectors")
                            # Fill remaining with zeros and return immediately
                            remaining = len(texts) - len(all_embeddings)
                            all_embeddings.extend([np.zeros(self.embedding_dim) for _ in range(remaining)])
                            return all_embeddings
                        if i + 5 < len(texts):  # Rate limiting between sub-batches
                            time.sleep(1)
                    return all_embeddings
                else:
                    return self._embed_batch_a4f(texts)
            return result
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text using Gemini with API key fallback.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        results = self._embed_batch([text])
        return results[0] if results else np.zeros(self.embedding_dim)
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = 'text',
        source_key: str = 'source'
    ):
        """
        Add documents to RAG knowledge base with BATCHED embeddings.
        
        Args:
            documents: List of documents with text and metadata
            text_key: Key for text content in document dict
            source_key: Key for source identifier
        """
        print(f"📚 Adding {len(documents)} documents to RAG knowledge base...")
        
        # Step 1: Collect ALL chunks from ALL documents first
        pending_chunks = []
        chunk_texts = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc.get(text_key, '')
            source = doc.get(source_key, f'doc_{doc_idx}')
            metadata = {k: v for k, v in doc.items() if k not in [text_key, source_key]}
            
            # Chunk the document
            text_chunks = self.chunk_text(text, source, metadata)
            
            # Collect chunk info (but don't embed yet)
            for chunk_idx, chunk_text in enumerate(text_chunks):
                if len(chunk_text.strip()) < 50:  # Skip very short chunks
                    continue
                
                pending_chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'chunk_id': chunk_idx,
                    'metadata': metadata
                })
                chunk_texts.append(chunk_text)
        
        total_chunks = len(chunk_texts)
        print(f"  → {total_chunks} chunks to embed")
        
        # Step 2: Batch embed ALL chunks at once
        if total_chunks == 0:
            print("  ⚠️  No chunks to embed")
            return
        
        print(f"  → Embedding in batches of {self.batch_size} (rate limit: {self.requests_per_minute} RPM)")
        
        all_embeddings = []
        num_batches = (total_chunks + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_chunks)
            batch = chunk_texts[start_idx:end_idx]
            
            print(f"  → Batch {batch_idx + 1}/{num_batches}: Embedding {len(batch)} chunks...", end='', flush=True)
            
            # Batch embed with rate limiting
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            print(f" ✓")
            
            # Add delay between batches to respect A4F 5 RPM limit (12s per request)
            if batch_idx < num_batches - 1:  # Don't delay after last batch
                time.sleep(13)
        
        # Step 3: Create DocumentChunk objects with embeddings
        for chunk_info, embedding in zip(pending_chunks, all_embeddings):
            chunk = DocumentChunk(
                text=chunk_info['text'],
                embedding=embedding,
                source=chunk_info['source'],
                chunk_id=chunk_info['chunk_id'],
                metadata=chunk_info['metadata']
            )
            self.chunks.append(chunk)
        
        print(f"✓ Added {len(self.chunks)} chunks with embeddings (used {num_batches} API requests)")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks for a query using cosine similarity.
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve (uses default if None)
            min_similarity: Minimum cosine similarity threshold
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if not self.chunks:
            print("⚠️  No documents in RAG knowledge base")
            return []
        
        top_k = top_k or self.top_k
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Calculate cosine similarity with all chunks
        similarities = []
        for chunk in self.chunks:
            # Cosine similarity
            similarity = np.dot(query_embedding, chunk.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding) + 1e-10
            )
            
            if similarity >= min_similarity:
                similarities.append({
                    'text': chunk.text,
                    'source': chunk.source,
                    'chunk_id': chunk.chunk_id,
                    'similarity': float(similarity),
                    'metadata': chunk.metadata
                })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"🔍 Retrieved {len(similarities[:top_k])} relevant chunks (from {len(self.chunks)} total)")
        
        return similarities[:top_k]
    
    def get_context_for_query(
        self,
        query: str,
        max_context_length: int = 3000,
        include_sources: bool = True
    ) -> str:
        """
        Get formatted context string for LLM from relevant chunks.
        
        Args:
            query: Query to retrieve context for
            max_context_length: Maximum characters in context
            include_sources: Whether to include source citations
            
        Returns:
            Formatted context string ready for LLM
        """
        relevant_chunks = self.retrieve(query)
        
        if not relevant_chunks:
            return "No relevant information found."
        
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(relevant_chunks):
            # Format chunk with source citation
            if include_sources:
                chunk_text = f"[Source: {chunk['source']}]\n{chunk['text']}\n"
            else:
                chunk_text = f"{chunk['text']}\n"
            
            # Check if adding this chunk exceeds limit
            if current_length + len(chunk_text) > max_context_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        context = "\n---\n".join(context_parts)
        
        return f"""**Relevant Information from Web Research:**

{context}

**Note:** The above information was retrieved from {len(context_parts)} sources and is ranked by relevance to the query."""
    
    def clear(self):
        """Clear all stored chunks."""
        self.chunks = []
        print("✓ RAG knowledge base cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG knowledge base."""
        unique_sources = set(chunk.source for chunk in self.chunks)
        
        return {
            'total_chunks': len(self.chunks),
            'unique_sources': len(unique_sources),
            'chunk_size': self.chunk_size,
            'top_k': self.top_k,
            'sources': list(unique_sources)
        }


class HybridRAG:
    """
    Hybrid RAG that combines multiple retrieval strategies.
    
    Uses both semantic similarity (embeddings) and keyword matching
    for more robust retrieval.
    """
    
    def __init__(self, rag_engine: RAGEngine, keyword_weight: float = 0.3):
        """
        Initialize hybrid RAG.
        
        Args:
            rag_engine: Base RAG engine
            keyword_weight: Weight for keyword matching (0-1)
        """
        self.rag_engine = rag_engine
        self.keyword_weight = keyword_weight
        self.semantic_weight = 1.0 - keyword_weight
    
    def _keyword_score(self, query: str, text: str) -> float:
        """Calculate simple keyword match score."""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
        
        matches = len(query_words & text_words)
        return matches / len(query_words)
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve using hybrid semantic + keyword approach.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of relevant chunks with combined scores
        """
        # Get semantic results
        semantic_results = self.rag_engine.retrieve(query, top_k=top_k * 2)
        
        # Re-rank with keyword matching
        for result in semantic_results:
            keyword_score = self._keyword_score(query, result['text'])
            semantic_score = result['similarity']
            
            # Combined score
            result['hybrid_score'] = (
                self.semantic_weight * semantic_score +
                self.keyword_weight * keyword_score
            )
            result['keyword_score'] = keyword_score
        
        # Re-sort by hybrid score
        semantic_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return semantic_results[:top_k or self.rag_engine.top_k]


if __name__ == "__main__":
    # Test RAG engine
    print("Testing RAG Engine...")
    
    try:
        # Initialize RAG
        rag = RAGEngine(chunk_size=300, top_k=3)
        
        # Add sample documents (simulating web-scraped content)
        sample_docs = [
            {
                'text': "Adani Ports announces $5 billion acquisition of new terminal in Mumbai. "
                       "The transaction is expected to close in Q4 2024 and will be funded through "
                       "a combination of debt and equity. This represents the largest investment in "
                       "port infrastructure in India this year.",
                'source': 'https://adaniports.com/press-release-2024-10'
            },
            {
                'text': "Stock market rebounds 12% after Federal Reserve announces emergency rate cut. "
                       "The unprecedented move aims to stabilize financial markets amid growing concerns "
                       "about economic slowdown. Market analysts predict continued volatility.",
                'source': 'https://reuters.com/markets/fed-rate-cut'
            },
            {
                'text': "Adani Group quarterly results show strong growth across energy and infrastructure "
                       "segments. Revenue increased 23% year-over-year driven by new project completions.",
                'source': 'https://adani.com/investor-relations/quarterly-results'
            }
        ]
        
        rag.add_documents(sample_docs)
        
        # Test retrieval
        query = "Adani Ports large transaction $5 billion"
        context = rag.get_context_for_query(query, max_context_length=500)
        
        print(f"\n🔍 Query: {query}")
        print(f"\n📄 Retrieved Context:\n{context}")
        
        # Test stats
        stats = rag.get_stats()
        print(f"\n📊 RAG Stats: {stats}")
        
        print("\n✓ RAG Engine tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

