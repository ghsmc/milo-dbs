#!/usr/bin/env python3
"""
Optimized ML model loader that avoids TensorFlow timeouts
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
import numpy as np

# Set environment variables before importing TensorFlow/sentence-transformers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations that can cause hangs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedEmbeddingModel:
    """Optimized embedding model that loads efficiently"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = 'models'):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _load_model_with_timeout(self, timeout_seconds: int = 60):
        """Load model with timeout to prevent hangs"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Model loading timed out after {timeout_seconds} seconds")
        
        # Set alarm for timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            logger.info(f"Loading model {self.model_name} with {timeout_seconds}s timeout...")
            
            # Import here to delay loading
            from sentence_transformers import SentenceTransformer
            
            # Load model with specific cache directory
            model = SentenceTransformer(
                self.model_name, 
                cache_folder=self.cache_dir,
                device='cpu'  # Force CPU to avoid GPU initialization issues
            )
            
            signal.alarm(0)  # Cancel alarm
            logger.info(f"Successfully loaded model {self.model_name}")
            return model
            
        except TimeoutError:
            signal.alarm(0)  # Cancel alarm
            logger.error(f"Model loading timed out after {timeout_seconds} seconds")
            raise
        except Exception as e:
            signal.alarm(0)  # Cancel alarm
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load(self, max_retries: int = 2, timeout_seconds: int = 60):
        """Load the model with retries and timeout"""
        if self.model is not None:
            return self.model
            
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading attempt {attempt + 1}/{max_retries}")
                self.model = self._load_model_with_timeout(timeout_seconds)
                return self.model
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("All loading attempts failed")
                    raise
                    
                # Wait before retry
                time.sleep(2)
        
        return None
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        if self.model is None:
            self.load()
            
        if not texts:
            return np.array([])
            
        # Process in batches to avoid memory issues
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.append(embeddings)
            
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text"""
        return self.encode([text])[0] if text else np.array([])


class OptimizedVectorStore:
    """Optimized vector store using FAISS"""
    
    def __init__(self, dimension: int = 384):  # all-MiniLM-L6-v2 dimension
        self.dimension = dimension
        self.index = None
        self.texts = []
        self.metadata = []
        self._init_faiss()
        
    def _init_faiss(self):
        """Initialize FAISS index"""
        try:
            import faiss
            # Use CPU index for stability
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
            logger.info(f"Initialized FAISS index with dimension {self.dimension}")
        except ImportError:
            logger.error("FAISS not installed. Install with: pip install faiss-cpu")
            raise
    
    def add_vectors(self, embeddings: np.ndarray, texts: List[str], metadata: List[Dict[str, Any]]):
        """Add vectors to the index"""
        if len(embeddings) == 0:
            return
            
        # Normalize embeddings for cosine similarity
        import faiss
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} vectors to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        if self.index.ntotal == 0:
            return []
            
        # Normalize query
        import faiss
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append({
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'similarity_score': float(score)
                })
        
        return results
    
    def save(self, filepath: str):
        """Save index to file"""
        if self.index is not None:
            import faiss
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata
            with open(f"{filepath}.json", 'w') as f:
                json.dump({
                    'texts': self.texts,
                    'metadata': self.metadata,
                    'dimension': self.dimension
                }, f)
            
            logger.info(f"Saved vector store to {filepath}")
    
    def load(self, filepath: str):
        """Load index from file"""
        try:
            import faiss
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            with open(f"{filepath}.json", 'r') as f:
                data = json.load(f)
                self.texts = data['texts']
                self.metadata = data['metadata']
                self.dimension = data['dimension']
            
            logger.info(f"Loaded vector store from {filepath}")
            return True
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"Could not load vector store: {e}")
            return False


class SemanticSearchEngine:
    """Complete semantic search engine with optimized loading"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = OptimizedEmbeddingModel(
            model_name=config.get('embedding_model', 'all-MiniLM-L6-v2'),
            cache_dir=config.get('model_dir', 'models')
        )
        self.vector_store = OptimizedVectorStore()
        self.is_loaded = False
    
    def initialize(self):
        """Initialize the search engine"""
        logger.info("Initializing semantic search engine...")
        
        try:
            # Load embedding model
            self.embedding_model.load(max_retries=2, timeout_seconds=60)
            
            # Try to load existing vector store
            store_path = os.path.join(self.config.get('model_dir', 'models'), 'vector_store')
            if not self.vector_store.load(store_path):
                logger.info("No existing vector store found. Will need to build embeddings.")
            
            self.is_loaded = True
            logger.info("Semantic search engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic search engine: {e}")
            raise
    
    def build_embeddings(self, texts: List[str], metadata: List[Dict[str, Any]], batch_size: int = 32):
        """Build embeddings for a collection of texts"""
        if not self.is_loaded:
            self.initialize()
        
        logger.info(f"Building embeddings for {len(texts)} texts...")
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(batch_texts, batch_size=batch_size)
            
            # Add to vector store
            self.vector_store.add_vectors(embeddings, batch_texts, batch_metadata)
        
        # Save vector store
        store_path = os.path.join(self.config.get('model_dir', 'models'), 'vector_store')
        self.vector_store.save(store_path)
        
        logger.info("Embedding building completed")
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search"""
        if not self.is_loaded:
            self.initialize()
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode_single(query)
        
        # Search
        results = self.vector_store.search(query_embedding, k)
        
        return results


# Test function
def test_optimized_loader():
    """Test the optimized loader"""
    print("Testing optimized ML loader...")
    
    config = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'model_dir': 'models'
    }
    
    try:
        # Test model loading
        embedding_model = OptimizedEmbeddingModel(config['embedding_model'], config['model_dir'])
        embedding_model.load(timeout_seconds=60)
        
        # Test encoding
        test_texts = ["Software Engineer at Google", "Data Scientist at Meta"]
        embeddings = embedding_model.encode(test_texts)
        print(f"✓ Generated embeddings shape: {embeddings.shape}")
        
        # Test vector store
        vector_store = OptimizedVectorStore()
        metadata = [{'id': i} for i in range(len(test_texts))]
        vector_store.add_vectors(embeddings, test_texts, metadata)
        
        # Test search
        query_embedding = embedding_model.encode_single("Engineer")
        results = vector_store.search(query_embedding, k=2)
        print(f"✓ Search returned {len(results)} results")
        
        print("✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_optimized_loader()