"""
Semantic Layer Module
Implements sentence embeddings and vector similarity search for contextual understanding
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import pickle
import logging
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import faiss
import warnings
warnings.filterwarnings('ignore')

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")

# Try to import pinecone
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("pinecone-client not installed. Install with: pip install pinecone-client")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SemanticMatch:
    """Semantic similarity match result"""
    person_id: str
    profile_text: str
    similarity_score: float
    embedding_vector: Optional[np.ndarray] = None


@dataclass
class ProfileEmbedding:
    """Profile with its embedding vector"""
    person_id: str
    profile_text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]


class ProfileTextBuilder:
    """Builds structured text representations of profiles for embedding"""
    
    def __init__(self):
        self.text_templates = {
            'full': """
            Name: {name}
            Title: {current_title}
            Company: {current_company}
            Location: {location}
            Education: {education}
            Experience: {experience}
            Skills: {skills}
            """,
            
            'professional': """
            {current_title} at {current_company}
            Location: {location}
            Skills: {skills}
            Background: {experience}
            """,
            
            'compact': """
            {current_title} at {current_company}, {location}
            Skills: {skills}
            """,
            
            'career_focused': """
            Career: {current_title} at {current_company}
            Previous roles: {experience}
            Specializations: {skills}
            Education: {education}
            Location: {location}
            """
        }
    
    def build_profile_text(self, profile_data: Dict[str, Any], 
                          template: str = 'professional') -> str:
        """Build structured text representation of a profile"""
        
        # Extract and clean data
        name = profile_data.get('name', '')
        current_title = profile_data.get('current_title', 'Unknown Title')
        current_company = profile_data.get('current_company', 'Unknown Company')
        location = profile_data.get('location', 'Unknown Location')
        
        # Process education
        education_parts = []
        if profile_data.get('degree'):
            education_parts.append(profile_data['degree'])
        if profile_data.get('major'):
            education_parts.append(profile_data['major'])
        if profile_data.get('graduation_year'):
            education_parts.append(str(profile_data['graduation_year']))
        education = ' '.join(education_parts) if education_parts else 'Education not specified'
        
        # Process experience
        experience_list = profile_data.get('experience', [])
        if experience_list and experience_list[0] is not None:
            experience_texts = []
            for exp in experience_list[:3]:  # Top 3 experiences
                if exp and isinstance(exp, dict):
                    exp_text = f"{exp.get('title', '')} at {exp.get('company', '')}"
                    if exp_text.strip() != ' at ':
                        experience_texts.append(exp_text)
            experience = '; '.join(experience_texts) if experience_texts else 'Experience not specified'
        else:
            experience = 'Experience not specified'
        
        # Process skills
        skills_list = profile_data.get('skills', [])
        if skills_list and skills_list[0] is not None:
            skills = ', '.join([skill for skill in skills_list[:10] if skill])  # Top 10 skills
        else:
            skills = 'Skills not specified'
        
        # Build text using template
        template_str = self.text_templates.get(template, self.text_templates['professional'])
        
        profile_text = template_str.format(
            name=name,
            current_title=current_title,
            current_company=current_company,
            location=location,
            education=education,
            experience=experience,
            skills=skills
        )
        
        # Clean up the text
        profile_text = ' '.join(profile_text.split())  # Remove extra whitespace
        profile_text = profile_text.replace('  ', ' ')  # Remove double spaces
        
        return profile_text.strip()
    
    def build_query_text(self, query_components: Dict[str, Any]) -> str:
        """Build text representation of a search query for embedding"""
        query_parts = []
        
        # Add role/title information
        if query_components.get('titles'):
            query_parts.append(f"Role: {' '.join(query_components['titles'])}")
        
        # Add company information
        if query_components.get('companies'):
            query_parts.append(f"Company: {' '.join(query_components['companies'])}")
        
        # Add location information
        if query_components.get('locations'):
            query_parts.append(f"Location: {' '.join(query_components['locations'])}")
        
        # Add skills
        if query_components.get('skills'):
            query_parts.append(f"Skills: {' '.join(query_components['skills'])}")
        
        # Add keywords
        if query_components.get('keywords'):
            query_parts.append(f"Keywords: {' '.join(query_components['keywords'])}")
        
        # Add seniority
        if query_components.get('seniority'):
            query_parts.append(f"Level: {' '.join(query_components['seniority'])}")
        
        return ' '.join(query_parts)


class EmbeddingGenerator:
    """Generates embeddings using sentence transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default for MiniLM
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {self.model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            self.model = None
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not self.model:
            raise ValueError("Model not loaded. Install sentence-transformers.")
        
        if not texts:
            return np.array([])
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.generate_embeddings([text])[0]
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return np.dot(embedding1, embedding2)
    
    def find_similar_texts(self, query_embedding: np.ndarray, 
                          candidate_embeddings: np.ndarray,
                          top_k: int = 10) -> List[Tuple[int, float]]:
        """Find most similar texts using cosine similarity"""
        similarities = np.dot(candidate_embeddings, query_embedding)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]


class FAISSVectorStore:
    """FAISS-based vector store for fast similarity search"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.id_to_person = {}
        self.person_to_id = {}
        self.metadata = {}
        self.next_id = 0
        
    def build_index(self, embeddings: np.ndarray, 
                   person_ids: List[str],
                   metadata: List[Dict[str, Any]] = None):
        """Build FAISS index from embeddings"""
        n_vectors = embeddings.shape[0]
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Build ID mappings
        for i, person_id in enumerate(person_ids):
            self.id_to_person[i] = person_id
            self.person_to_id[person_id] = i
        
        # Store metadata
        if metadata:
            self.metadata = {person_ids[i]: metadata[i] for i in range(len(person_ids))}
        
        logger.info(f"Built FAISS index with {n_vectors} vectors")
    
    def search(self, query_embedding: np.ndarray, 
               top_k: int = 10) -> List[SemanticMatch]:
        """Search for similar vectors"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Search
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Convert results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == -1:  # No more results
                break
                
            person_id = self.id_to_person[idx]
            metadata = self.metadata.get(person_id, {})
            
            result = SemanticMatch(
                person_id=person_id,
                profile_text=metadata.get('profile_text', ''),
                similarity_score=float(similarity)
            )
            results.append(result)
        
        return results
    
    def save_index(self, filepath: str):
        """Save FAISS index to file"""
        faiss.write_index(self.index, filepath)
        
        # Save metadata
        metadata_filepath = filepath.replace('.index', '_metadata.pkl')
        with open(metadata_filepath, 'wb') as f:
            pickle.dump({
                'id_to_person': self.id_to_person,
                'person_to_id': self.person_to_id,
                'metadata': self.metadata,
                'next_id': self.next_id
            }, f)
        
        logger.info(f"Saved FAISS index to {filepath}")
    
    def load_index(self, filepath: str):
        """Load FAISS index from file"""
        self.index = faiss.read_index(filepath)
        
        # Load metadata
        metadata_filepath = filepath.replace('.index', '_metadata.pkl')
        with open(metadata_filepath, 'rb') as f:
            data = pickle.load(f)
            self.id_to_person = data['id_to_person']
            self.person_to_id = data['person_to_id']
            self.metadata = data['metadata']
            self.next_id = data['next_id']
        
        logger.info(f"Loaded FAISS index from {filepath}")


class PineconeVectorStore:
    """Pinecone-based vector store for scalable similarity search"""
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.index = None
        
        if PINECONE_AVAILABLE:
            self._initialize_pinecone()
        else:
            logger.error("pinecone-client not available. Install with: pip install pinecone-client")
    
    def _initialize_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            # Create index if it doesn't exist
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=384,  # Default for MiniLM
                    metric='cosine'
                )
            
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self.index = None
    
    def upsert_embeddings(self, embeddings: np.ndarray, 
                         person_ids: List[str],
                         metadata: List[Dict[str, Any]] = None):
        """Upsert embeddings to Pinecone"""
        if not self.index:
            raise ValueError("Pinecone index not initialized")
        
        # Prepare vectors for upsert
        vectors = []
        for i, (person_id, embedding) in enumerate(zip(person_ids, embeddings)):
            vector_metadata = metadata[i] if metadata else {}
            vectors.append({
                'id': person_id,
                'values': embedding.tolist(),
                'metadata': vector_metadata
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            
        logger.info(f"Upserted {len(vectors)} embeddings to Pinecone")
    
    def search(self, query_embedding: np.ndarray, 
               top_k: int = 10,
               filter_dict: Dict[str, Any] = None) -> List[SemanticMatch]:
        """Search for similar vectors in Pinecone"""
        if not self.index:
            raise ValueError("Pinecone index not initialized")
        
        # Search
        response = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )
        
        # Convert results
        results = []
        for match in response['matches']:
            result = SemanticMatch(
                person_id=match['id'],
                profile_text=match['metadata'].get('profile_text', ''),
                similarity_score=match['score']
            )
            results.append(result)
        
        return results


class SemanticSearchEngine:
    """Main semantic search engine"""
    
    def __init__(self, db_config: Dict[str, str], 
                 vector_store_type: str = 'faiss',
                 model_name: str = 'all-MiniLM-L6-v2',
                 use_multi_aspect: bool = True):
        self.db_config = db_config
        self.vector_store_type = vector_store_type
        self.model_name = model_name
        self.use_multi_aspect = use_multi_aspect
        
        # Initialize components
        self.text_builder = ProfileTextBuilder()
        self.embedding_generator = EmbeddingGenerator(model_name)
        self.vector_store = None
        
        # Multi-aspect components (new)
        if use_multi_aspect:
            from multi_aspect_embeddings import MultiAspectEmbedder, MultiAspectVectorStore
            self.multi_aspect_embedder = MultiAspectEmbedder(model_name)
            self.multi_aspect_store = MultiAspectVectorStore(self.embedding_generator.embedding_dim)
        
        # Initialize vector store
        if vector_store_type == 'faiss':
            self.vector_store = FAISSVectorStore(self.embedding_generator.embedding_dim)
        elif vector_store_type == 'pinecone':
            # Would need Pinecone credentials
            pass
    
    def build_embeddings_database(self, batch_size: int = 1000):
        """Build embeddings for all profiles in database"""
        conn = psycopg2.connect(**self.db_config)
        
        try:
            # Get all profiles with their data
            query = """
                SELECT 
                    a.person_id,
                    a.name,
                    a.current_title,
                    a.current_company,
                    a.location,
                    a.degree,
                    a.major,
                    a.graduation_year,
                    array_agg(DISTINCT s.skill) as skills,
                    array_agg(DISTINCT jsonb_build_object(
                        'title', e.title,
                        'company', e.company,
                        'start_date', e.start_date,
                        'end_date', e.end_date
                    )) as experience
                FROM alumni a
                LEFT JOIN skills s ON a.person_id = s.person_id
                LEFT JOIN experience e ON a.person_id = e.person_id
                GROUP BY a.person_id, a.name, a.current_title, a.current_company,
                         a.location, a.degree, a.major, a.graduation_year
            """
            
            df = pd.read_sql(query, conn)
            
            # Process in batches
            all_embeddings = []
            all_person_ids = []
            all_metadata = []
            
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i + batch_size]
                
                # Build profile texts
                batch_texts = []
                batch_metadata = []
                
                for _, row in batch_df.iterrows():
                    profile_text = self.text_builder.build_profile_text(row.to_dict())
                    batch_texts.append(profile_text)
                    
                    metadata = {
                        'profile_text': profile_text,
                        'name': row['name'],
                        'current_title': row['current_title'],
                        'current_company': row['current_company'],
                        'location': row['location']
                    }
                    batch_metadata.append(metadata)
                
                # Generate embeddings
                batch_embeddings = self.embedding_generator.generate_embeddings(batch_texts)
                
                # Accumulate results
                all_embeddings.append(batch_embeddings)
                all_person_ids.extend(batch_df['person_id'].tolist())
                all_metadata.extend(batch_metadata)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
            
            # Combine all embeddings
            final_embeddings = np.vstack(all_embeddings)
            
            # Build vector store
            self.vector_store.build_index(final_embeddings, all_person_ids, all_metadata)
            
            # Save to database
            self._save_embeddings_to_db(final_embeddings, all_person_ids, all_metadata)
            
            logger.info(f"Built embeddings for {len(all_person_ids)} profiles")
            
        finally:
            conn.close()
    
    def _save_embeddings_to_db(self, embeddings: np.ndarray, 
                              person_ids: List[str], 
                              metadata: List[Dict[str, Any]]):
        """Save embeddings to database"""
        conn = psycopg2.connect(**self.db_config)
        
        try:
            with conn.cursor() as cur:
                # Create embeddings table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS profile_embeddings (
                        person_id VARCHAR(50) PRIMARY KEY REFERENCES alumni(person_id),
                        profile_text TEXT,
                        embedding BYTEA,
                        model_name VARCHAR(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Prepare data for insertion
                data_to_insert = []
                for i, person_id in enumerate(person_ids):
                    embedding_bytes = embeddings[i].tobytes()
                    profile_text = metadata[i]['profile_text']
                    
                    data_to_insert.append((
                        person_id,
                        profile_text,
                        embedding_bytes,
                        self.model_name
                    ))
                
                # Batch insert
                execute_batch(cur, """
                    INSERT INTO profile_embeddings (person_id, profile_text, embedding, model_name)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (person_id) DO UPDATE
                    SET profile_text = EXCLUDED.profile_text,
                        embedding = EXCLUDED.embedding,
                        model_name = EXCLUDED.model_name,
                        created_at = CURRENT_TIMESTAMP
                """, data_to_insert)
                
            conn.commit()
            logger.info("Saved embeddings to database")
            
        finally:
            conn.close()
    
    def build_multi_aspect_embeddings(self, batch_size: int = 1000):
        """Build multi-aspect embeddings for all profiles"""
        if not self.use_multi_aspect:
            logger.warning("Multi-aspect embeddings disabled")
            return
        
        conn = psycopg2.connect(**self.db_config)
        
        try:
            # Get all profiles with their data
            query = """
                SELECT 
                    a.person_id,
                    a.name,
                    a.current_title,
                    a.current_company,
                    a.location,
                    a.degree,
                    a.major,
                    a.graduation_year,
                    a.education_school,
                    array_agg(DISTINCT s.skill) FILTER (WHERE s.skill IS NOT NULL) as skills,
                    array_agg(DISTINCT jsonb_build_object(
                        'title', e.title,
                        'company', e.company,
                        'description', e.description,
                        'start_date', e.start_date,
                        'end_date', e.end_date
                    )) FILTER (WHERE e.title IS NOT NULL) as experience
                FROM alumni a
                LEFT JOIN skills s ON a.person_id = s.person_id
                LEFT JOIN experience e ON a.person_id = e.person_id
                GROUP BY a.person_id, a.name, a.current_title, a.current_company,
                         a.location, a.degree, a.major, a.graduation_year, a.education_school
                LIMIT 5000  -- Process in chunks for large datasets
            """
            
            df = pd.read_sql(query, conn)
            
            # Process in batches
            all_profiles = []
            
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i + batch_size]
                
                # Generate multi-aspect embeddings for batch
                for _, row in batch_df.iterrows():
                    try:
                        profile = self.multi_aspect_embedder.generate_profile_embeddings(row.to_dict())
                        all_profiles.append(profile)
                    except Exception as e:
                        logger.warning(f"Failed to generate embeddings for {row['person_id']}: {e}")
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
            
            # Save to database
            self.multi_aspect_embedder.save_profile_embeddings(all_profiles, self.db_config)
            
            # Build vector indexes
            self.multi_aspect_store.build_indexes(all_profiles)
            
            logger.info(f"Built multi-aspect embeddings for {len(all_profiles)} profiles")
            
        finally:
            conn.close()
    
    def multi_aspect_search(self, query: str, 
                           aspect_weights: Dict[str, float] = None,
                           top_k: int = 10):
        """Perform multi-aspect semantic search"""
        if not self.use_multi_aspect:
            raise ValueError("Multi-aspect search not enabled")
        
        # Load profiles if not already loaded
        profiles = self.multi_aspect_embedder.load_profile_embeddings(self.db_config)
        
        # Perform search
        results = self.multi_aspect_embedder.search_similar_profiles(
            query, profiles, aspect_weights, top_k
        )
        
        # Convert to SemanticMatch format for compatibility
        semantic_matches = []
        for result in results:
            match = SemanticMatch(
                person_id=result.person_id,
                profile_text=result.profile_summary,
                similarity_score=result.overall_similarity
            )
            semantic_matches.append(match)
        
        return semantic_matches
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[SemanticMatch]:
        """Perform semantic search"""
        if not self.vector_store or not self.vector_store.index:
            raise ValueError("Vector store not initialized. Call build_embeddings_database first.")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_single_embedding(query)
        
        # Search
        results = self.vector_store.search(query_embedding, top_k)
        
        return results
    
    def hybrid_search(self, query_components: Dict[str, Any], 
                     top_k: int = 10,
                     semantic_weight: float = 0.3) -> List[SemanticMatch]:
        """Combine semantic and keyword search"""
        # Build semantic query
        semantic_query = self.text_builder.build_query_text(query_components)
        
        # Get semantic results
        semantic_results = self.semantic_search(semantic_query, top_k * 2)
        
        # TODO: Combine with keyword search results
        # For now, return semantic results
        return semantic_results[:top_k]
    
    def save_vector_store(self, filepath: str):
        """Save vector store to file"""
        if isinstance(self.vector_store, FAISSVectorStore):
            self.vector_store.save_index(filepath)
        else:
            logger.warning("Vector store saving not implemented for this type")
    
    def load_vector_store(self, filepath: str):
        """Load vector store from file"""
        if isinstance(self.vector_store, FAISSVectorStore):
            self.vector_store.load_index(filepath)
        else:
            logger.warning("Vector store loading not implemented for this type")
    
    def get_similar_profiles(self, person_id: str, top_k: int = 5) -> List[SemanticMatch]:
        """Find profiles similar to a given person"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        # Get the person's embedding
        if person_id not in self.vector_store.person_to_id:
            raise ValueError(f"Person {person_id} not found in vector store")
        
        # Get embedding from vector store
        vector_id = self.vector_store.person_to_id[person_id]
        person_embedding = self.vector_store.index.reconstruct(vector_id)
        
        # Search for similar profiles
        results = self.vector_store.search(person_embedding, top_k + 1)  # +1 to exclude self
        
        # Remove the person themselves from results
        filtered_results = [r for r in results if r.person_id != person_id]
        
        return filtered_results[:top_k]


# Example usage
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'yale_alumni',
        'user': 'postgres',
        'password': 'password',
        'port': 5432
    }
    
    # Initialize semantic search engine
    semantic_engine = SemanticSearchEngine(db_config)
    
    # Build embeddings database
    # semantic_engine.build_embeddings_database()
    
    # Test semantic search
    test_queries = [
        "software engineer with machine learning experience",
        "investment banking analyst at Goldman Sachs",
        "product manager at tech startup in San Francisco",
        "consultant with finance background"
    ]
    
    print("Semantic Search Examples:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            results = semantic_engine.semantic_search(query, top_k=3)
            for i, result in enumerate(results):
                print(f"  {i+1}. {result.person_id}: {result.similarity_score:.3f}")
                print(f"     {result.profile_text[:100]}...")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test profile building
    text_builder = ProfileTextBuilder()
    sample_profile = {
        'name': 'John Doe',
        'current_title': 'Senior Software Engineer',
        'current_company': 'Google',
        'location': 'San Francisco, CA',
        'degree': 'BS Computer Science',
        'graduation_year': 2018,
        'skills': ['Python', 'Machine Learning', 'TensorFlow'],
        'experience': [
            {'title': 'Software Engineer', 'company': 'Facebook'},
            {'title': 'Intern', 'company': 'Microsoft'}
        ]
    }
    
    profile_text = text_builder.build_profile_text(sample_profile)
    print(f"\nSample profile text:\n{profile_text}")
    
    # Test query text building
    query_components = {
        'titles': ['software engineer'],
        'companies': ['Google', 'Facebook'],
        'skills': ['machine learning', 'python'],
        'locations': ['San Francisco']
    }
    
    query_text = text_builder.build_query_text(query_components)
    print(f"\nQuery text: {query_text}")