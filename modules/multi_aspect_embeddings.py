"""
Multi-Aspect Embeddings System
Implements separate embeddings for different profile aspects as per feedback #8
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")

# Try to import faiss
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("faiss not installed. Install with: pip install faiss-cpu")

logger = logging.getLogger(__name__)


@dataclass
class AspectEmbedding:
    """Individual aspect embedding"""
    aspect_type: str  # 'current_role', 'experience', 'skills', 'education'
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'aspect_type': self.aspect_type,
            'text': self.text,
            'embedding': self.embedding.tolist(),
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary"""
        return cls(
            aspect_type=data['aspect_type'],
            text=data['text'],
            embedding=np.array(data['embedding']),
            metadata=data.get('metadata', {})
        )


@dataclass
class MultiAspectProfile:
    """Profile with multiple aspect embeddings"""
    person_id: str
    aspects: Dict[str, AspectEmbedding]
    created_at: datetime
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'person_id': self.person_id,
            'aspects': {k: v.to_dict() for k, v in self.aspects.items()},
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary"""
        return cls(
            person_id=data['person_id'],
            aspects={k: AspectEmbedding.from_dict(v) for k, v in data['aspects'].items()},
            created_at=datetime.fromisoformat(data['created_at'])
        )


@dataclass
class MultiAspectSearchResult:
    """Search result with aspect-specific scores"""
    person_id: str
    overall_similarity: float
    aspect_scores: Dict[str, float]
    matched_aspects: List[str]
    profile_summary: str


class MultiAspectEmbedder:
    """
    Generates separate embeddings for different profile aspects
    Addresses feedback #8: Multiple embeddings per profile for better semantic matching
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default for MiniLM
        
        # Define aspect weights for search
        self.default_aspect_weights = {
            'current_role': 0.4,    # Most important for current relevance
            'experience': 0.3,      # Historical context
            'skills': 0.2,          # Technical/functional capabilities
            'education': 0.1        # Background context
        }
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            logger.error("sentence-transformers not available")
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded model {self.model_name} with dimension {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            self.model = None
    
    def extract_aspects_from_profile(self, profile_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract different aspects from profile data as text"""
        aspects = {}
        
        # 1. Current Role Aspect
        current_role_parts = []
        if profile_data.get('current_title'):
            current_role_parts.append(profile_data['current_title'])
        if profile_data.get('current_company'):
            current_role_parts.append(f"at {profile_data['current_company']}")
        if profile_data.get('location'):
            current_role_parts.append(f"in {profile_data['location']}")
        
        if current_role_parts:
            aspects['current_role'] = ' '.join(current_role_parts)
        
        # 2. Experience Aspect (aggregate past roles)
        experience_texts = []
        experience_list = profile_data.get('experience', [])
        if experience_list and experience_list[0] is not None:
            for exp in experience_list[:5]:  # Top 5 experiences
                if exp and isinstance(exp, dict):
                    exp_parts = []
                    if exp.get('title'):
                        exp_parts.append(exp['title'])
                    if exp.get('company'):
                        exp_parts.append(f"at {exp['company']}")
                    if exp.get('description'):
                        # Limit description length
                        desc = exp['description'][:200] + '...' if len(exp['description']) > 200 else exp['description']
                        exp_parts.append(desc)
                    
                    if exp_parts:
                        experience_texts.append(' '.join(exp_parts))
        
        if experience_texts:
            aspects['experience'] = '. '.join(experience_texts)
        
        # 3. Skills Aspect
        skills_list = profile_data.get('skills', [])
        if skills_list and skills_list[0] is not None:
            # Group skills by type if possible
            skills_text = ', '.join([skill for skill in skills_list[:15] if skill])
            if skills_text:
                aspects['skills'] = f"Skills and expertise: {skills_text}"
        
        # 4. Education Aspect
        education_parts = []
        if profile_data.get('degree'):
            education_parts.append(profile_data['degree'])
        if profile_data.get('major'):
            education_parts.append(f"in {profile_data['major']}")
        if profile_data.get('education_school'):
            education_parts.append(f"from {profile_data['education_school']}")
        if profile_data.get('graduation_year'):
            education_parts.append(f"({profile_data['graduation_year']})")
        
        if education_parts:
            aspects['education'] = ' '.join(education_parts)
        
        return aspects
    
    def generate_profile_embeddings(self, profile_data: Dict[str, Any]) -> MultiAspectProfile:
        """Generate embeddings for all aspects of a profile"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Extract text for each aspect
        aspect_texts = self.extract_aspects_from_profile(profile_data)
        
        # Generate embeddings for each aspect
        aspect_embeddings = {}
        for aspect_type, text in aspect_texts.items():
            if text and text.strip():
                try:
                    embedding = self.model.encode(text, normalize_embeddings=True)
                    
                    # Create metadata for this aspect
                    metadata = {
                        'text_length': len(text),
                        'generated_at': datetime.now().isoformat()
                    }
                    
                    # Add aspect-specific metadata
                    if aspect_type == 'experience':
                        exp_list = profile_data.get('experience', [])
                        metadata['experience_count'] = len([e for e in exp_list if e]) if exp_list else 0
                    elif aspect_type == 'skills':
                        skills_list = profile_data.get('skills', [])
                        metadata['skills_count'] = len([s for s in skills_list if s]) if skills_list else 0
                    
                    aspect_embeddings[aspect_type] = AspectEmbedding(
                        aspect_type=aspect_type,
                        text=text,
                        embedding=embedding,
                        metadata=metadata
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for aspect {aspect_type}: {e}")
        
        return MultiAspectProfile(
            person_id=profile_data['person_id'],
            aspects=aspect_embeddings,
            created_at=datetime.now()
        )
    
    def search_similar_profiles(self, 
                              query_text: str,
                              profiles: List[MultiAspectProfile],
                              aspect_weights: Dict[str, float] = None,
                              top_k: int = 10) -> List[MultiAspectSearchResult]:
        """
        Search for similar profiles using multi-aspect embeddings
        
        Args:
            query_text: Search query
            profiles: List of profiles with embeddings
            aspect_weights: Weights for different aspects
            top_k: Number of results to return
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Use default weights if not provided
        if aspect_weights is None:
            aspect_weights = self.default_aspect_weights.copy()
        
        # Generate query embedding
        query_embedding = self.model.encode(query_text, normalize_embeddings=True)
        
        # Calculate similarities for each profile
        results = []
        for profile in profiles:
            aspect_scores = {}
            matched_aspects = []
            
            # Calculate similarity for each aspect
            for aspect_type, aspect_embedding in profile.aspects.items():
                similarity = np.dot(query_embedding, aspect_embedding.embedding)
                aspect_scores[aspect_type] = float(similarity)
                
                # Consider an aspect "matched" if similarity > threshold
                if similarity > 0.5:  # Threshold can be tuned
                    matched_aspects.append(aspect_type)
            
            # Calculate weighted overall similarity
            overall_similarity = 0.0
            total_weight = 0.0
            
            for aspect_type, weight in aspect_weights.items():
                if aspect_type in aspect_scores:
                    overall_similarity += aspect_scores[aspect_type] * weight
                    total_weight += weight
            
            # Normalize by total weight used
            if total_weight > 0:
                overall_similarity /= total_weight
            
            # Create summary of matched content
            profile_summary = self._create_profile_summary(profile, matched_aspects)
            
            results.append(MultiAspectSearchResult(
                person_id=profile.person_id,
                overall_similarity=overall_similarity,
                aspect_scores=aspect_scores,
                matched_aspects=matched_aspects,
                profile_summary=profile_summary
            ))
        
        # Sort by overall similarity and return top_k
        results.sort(key=lambda x: x.overall_similarity, reverse=True)
        return results[:top_k]
    
    def _create_profile_summary(self, 
                              profile: MultiAspectProfile, 
                              matched_aspects: List[str]) -> str:
        """Create a summary of the profile focusing on matched aspects"""
        summary_parts = []
        
        # Prioritize matched aspects
        for aspect_type in matched_aspects:
            if aspect_type in profile.aspects:
                aspect = profile.aspects[aspect_type]
                if aspect_type == 'current_role':
                    summary_parts.append(f"Current: {aspect.text}")
                elif aspect_type == 'skills':
                    # Truncate skills for summary
                    skills_short = aspect.text[:100] + '...' if len(aspect.text) > 100 else aspect.text
                    summary_parts.append(skills_short)
                elif aspect_type == 'experience':
                    # Get first sentence of experience
                    exp_short = aspect.text.split('.')[0] + '...' if '.' in aspect.text else aspect.text[:100] + '...'
                    summary_parts.append(f"Experience: {exp_short}")
        
        # Add current role if not already included
        if 'current_role' not in matched_aspects and 'current_role' in profile.aspects:
            summary_parts.insert(0, profile.aspects['current_role'].text)
        
        return ' | '.join(summary_parts[:3])  # Limit to 3 parts
    
    def save_profile_embeddings(self, profiles: List[MultiAspectProfile], db_config: Dict[str, str]):
        """Save multi-aspect embeddings to database"""
        conn = psycopg2.connect(**db_config)
        
        try:
            with conn.cursor() as cur:
                # Create table if not exists
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS multi_aspect_embeddings (
                        person_id VARCHAR(50) PRIMARY KEY,
                        embedding_data JSONB,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Insert or update embeddings
                data_to_insert = []
                for profile in profiles:
                    data_to_insert.append((
                        profile.person_id,
                        json.dumps(profile.to_dict()),
                        profile.created_at
                    ))
                
                execute_batch(cur, """
                    INSERT INTO multi_aspect_embeddings (person_id, embedding_data, created_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (person_id) DO UPDATE SET
                        embedding_data = EXCLUDED.embedding_data,
                        updated_at = CURRENT_TIMESTAMP
                """, data_to_insert)
                
                conn.commit()
                logger.info(f"Saved {len(profiles)} multi-aspect profiles to database")
                
        finally:
            conn.close()
    
    def load_profile_embeddings(self, 
                              db_config: Dict[str, str],
                              person_ids: List[str] = None) -> List[MultiAspectProfile]:
        """Load multi-aspect embeddings from database"""
        conn = psycopg2.connect(**db_config)
        profiles = []
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if person_ids:
                    # Load specific profiles
                    cur.execute("""
                        SELECT person_id, embedding_data
                        FROM multi_aspect_embeddings
                        WHERE person_id = ANY(%s)
                    """, (person_ids,))
                else:
                    # Load all profiles
                    cur.execute("""
                        SELECT person_id, embedding_data
                        FROM multi_aspect_embeddings
                    """)
                
                rows = cur.fetchall()
                
                for row in rows:
                    try:
                        embedding_data = json.loads(row['embedding_data'])
                        profile = MultiAspectProfile.from_dict(embedding_data)
                        profiles.append(profile)
                    except Exception as e:
                        logger.warning(f"Failed to load profile {row['person_id']}: {e}")
                
                logger.info(f"Loaded {len(profiles)} multi-aspect profiles from database")
                
        finally:
            conn.close()
        
        return profiles


class MultiAspectVectorStore:
    """Vector store for multi-aspect embeddings using FAISS"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.aspect_indexes = {}  # Separate FAISS index for each aspect
        self.person_id_mapping = {}  # Map FAISS IDs to person IDs
        
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available. Vector operations will be limited.")
    
    def build_indexes(self, profiles: List[MultiAspectProfile]):
        """Build FAISS indexes for each aspect type"""
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available")
            return
        
        # Group embeddings by aspect type
        aspect_data = {}
        for profile in profiles:
            for aspect_type, aspect_embedding in profile.aspects.items():
                if aspect_type not in aspect_data:
                    aspect_data[aspect_type] = []
                aspect_data[aspect_type].append((profile.person_id, aspect_embedding.embedding))
        
        # Build separate index for each aspect
        for aspect_type, embeddings_data in aspect_data.items():
            if not embeddings_data:
                continue
            
            # Create FAISS index
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for normalized vectors
            
            # Prepare data
            person_ids = []
            embeddings = []
            for person_id, embedding in embeddings_data:
                person_ids.append(person_id)
                embeddings.append(embedding)
            
            # Add to index
            embeddings_matrix = np.vstack(embeddings).astype(np.float32)
            index.add(embeddings_matrix)
            
            # Store index and mapping
            self.aspect_indexes[aspect_type] = index
            self.person_id_mapping[aspect_type] = person_ids
            
            logger.info(f"Built FAISS index for {aspect_type} with {len(embeddings_data)} vectors")
    
    def search_by_aspect(self, 
                        query_embedding: np.ndarray,
                        aspect_type: str,
                        top_k: int = 20) -> List[Tuple[str, float]]:
        """Search for similar vectors in a specific aspect"""
        if not FAISS_AVAILABLE or aspect_type not in self.aspect_indexes:
            return []
        
        index = self.aspect_indexes[aspect_type]
        person_ids = self.person_id_mapping[aspect_type]
        
        # Search
        query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        similarities, indices = index.search(query_vector, min(top_k, len(person_ids)))
        
        # Convert results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(person_ids):
                results.append((person_ids[idx], float(similarities[0][i])))
        
        return results