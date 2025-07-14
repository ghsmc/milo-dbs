"""
Co-occurrence Analysis Module
Builds term-document matrices, calculates PMI scores, and creates expansion lookup tables
This is the key to automatic query expansion based on actual data patterns
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import math
import logging
import json
import pickle
from dataclasses import dataclass
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CooccurrenceResult:
    """Result of co-occurrence analysis between two terms"""
    term1: str
    term2: str
    cooccurrence_count: int
    term1_count: int
    term2_count: int
    pmi: float  # Pointwise Mutual Information
    npmi: float  # Normalized PMI (-1 to 1)
    lift: float  # Lift score
    confidence: float  # Confidence (term1 -> term2)
    reverse_confidence: float  # Confidence (term2 -> term1)
    

class CooccurrenceAnalyzer:
    """Analyzes co-occurrence patterns in alumni profiles"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.term_doc_matrix = None
        self.term_to_idx = {}
        self.idx_to_term = {}
        self.term_counts = Counter()
        self.doc_count = 0
        self.cooccurrence_cache = {}
        
        # TF-IDF vectorizer for filtering common terms
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=5,  # Term must appear in at least 5 documents
            max_df=0.8,  # Term can't appear in more than 80% of documents
            ngram_range=(1, 2),  # Unigrams and bigrams
            stop_words='english'
        )
        
    def build_term_document_matrix(self):
        """Build term-document matrix from alumni profiles"""
        conn = psycopg2.connect(**self.db_config)
        
        # Get all profiles with their extracted entities
        query = """
            SELECT 
                a.person_id,
                a.normalized_company,
                a.normalized_title,
                a.normalized_location,
                te.normalized_title as extracted_title,
                te.seniority,
                te.role_type,
                te.specialization,
                te.department,
                te.industry_focus,
                le.normalized_location as extracted_location,
                le.metro_area,
                array_agg(DISTINCT s.skill) as skills
            FROM alumni a
            LEFT JOIN title_entities te ON a.person_id = te.person_id
            LEFT JOIN location_entities le ON a.person_id = le.person_id
            LEFT JOIN skills s ON a.person_id = s.person_id
            GROUP BY a.person_id, a.normalized_company, a.normalized_title, 
                     a.normalized_location, te.normalized_title, te.seniority,
                     te.role_type, te.specialization, te.department, 
                     te.industry_focus, le.normalized_location, le.metro_area
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Build documents (each profile is a document)
        documents = []
        doc_terms_list = []
        
        for _, row in df.iterrows():
            # Collect all terms for this profile
            terms = []
            
            # Add company
            if row['normalized_company']:
                terms.append(f"company:{row['normalized_company'].lower()}")
                
            # Add title components
            if row['extracted_title']:
                terms.append(f"title:{row['extracted_title'].lower()}")
            if row['seniority']:
                terms.append(f"seniority:{row['seniority']}")
            if row['role_type']:
                terms.append(f"role:{row['role_type']}")
            if row['specialization']:
                terms.append(f"spec:{row['specialization']}")
            if row['department']:
                terms.append(f"dept:{row['department']}")
            if row['industry_focus']:
                terms.append(f"industry:{row['industry_focus']}")
                
            # Add location
            if row['metro_area']:
                terms.append(f"location:{row['metro_area'].lower()}")
            elif row['extracted_location']:
                terms.append(f"location:{row['extracted_location'].lower()}")
                
            # Add skills
            if row['skills'] and row['skills'][0] is not None:
                for skill in row['skills']:
                    if skill:
                        terms.append(f"skill:{skill.lower()}")
            
            # Create document text for TF-IDF
            doc_text = ' '.join([t.split(':')[1] for t in terms])
            documents.append(doc_text)
            doc_terms_list.append(terms)
            
        self.doc_count = len(documents)
        logger.info(f"Built {self.doc_count} documents for analysis")
        
        # Build vocabulary
        all_terms = set()
        for terms in doc_terms_list:
            all_terms.update(terms)
        
        # Create term indices
        self.term_to_idx = {term: idx for idx, term in enumerate(sorted(all_terms))}
        self.idx_to_term = {idx: term for term, idx in self.term_to_idx.items()}
        vocab_size = len(self.term_to_idx)
        
        logger.info(f"Vocabulary size: {vocab_size}")
        
        # Build sparse term-document matrix
        self.term_doc_matrix = lil_matrix((vocab_size, self.doc_count))
        
        for doc_idx, terms in enumerate(doc_terms_list):
            for term in terms:
                if term in self.term_to_idx:
                    term_idx = self.term_to_idx[term]
                    self.term_doc_matrix[term_idx, doc_idx] = 1
                    self.term_counts[term] += 1
        
        # Convert to CSR format for efficient operations
        self.term_doc_matrix = self.term_doc_matrix.tocsr()
        
        # Compute TF-IDF scores for filtering
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        self.tfidf_features = self.tfidf_vectorizer.get_feature_names_out()
        
        logger.info("Term-document matrix built successfully")
        
    def calculate_pmi_scores(self, min_cooccurrence: int = 5) -> List[CooccurrenceResult]:
        """Calculate PMI scores for all term pairs"""
        results = []
        vocab_size = len(self.term_to_idx)
        
        # Calculate co-occurrence matrix (term x term)
        # Using matrix multiplication: C = M * M^T
        cooccurrence_matrix = self.term_doc_matrix @ self.term_doc_matrix.T
        
        # Convert to COO format for iteration
        cooccurrence_coo = cooccurrence_matrix.tocoo()
        
        # Process each co-occurrence
        for i, j, cooccurrence_count in zip(cooccurrence_coo.row, 
                                           cooccurrence_coo.col, 
                                           cooccurrence_coo.data):
            if i >= j:  # Skip diagonal and duplicate pairs
                continue
                
            if cooccurrence_count < min_cooccurrence:
                continue
                
            term1 = self.idx_to_term[i]
            term2 = self.idx_to_term[j]
            
            term1_count = self.term_counts[term1]
            term2_count = self.term_counts[term2]
            
            # Calculate PMI
            p_term1 = term1_count / self.doc_count
            p_term2 = term2_count / self.doc_count
            p_cooccurrence = cooccurrence_count / self.doc_count
            
            if p_cooccurrence > 0:
                pmi = math.log2(p_cooccurrence / (p_term1 * p_term2))
                
                # Normalized PMI
                npmi = pmi / (-math.log2(p_cooccurrence))
                
                # Lift
                lift = p_cooccurrence / (p_term1 * p_term2)
                
                # Confidence scores
                confidence = cooccurrence_count / term1_count
                reverse_confidence = cooccurrence_count / term2_count
                
                result = CooccurrenceResult(
                    term1=term1,
                    term2=term2,
                    cooccurrence_count=int(cooccurrence_count),
                    term1_count=term1_count,
                    term2_count=term2_count,
                    pmi=pmi,
                    npmi=npmi,
                    lift=lift,
                    confidence=confidence,
                    reverse_confidence=reverse_confidence
                )
                
                results.append(result)
                
                # Cache for fast lookup
                self.cooccurrence_cache[(term1, term2)] = result
                self.cooccurrence_cache[(term2, term1)] = result
        
        # Sort by PMI score
        results.sort(key=lambda x: x.pmi, reverse=True)
        
        logger.info(f"Calculated PMI scores for {len(results)} term pairs")
        
        return results
    
    def build_expansion_tables(self, min_confidence: float = 0.3, 
                             min_lift: float = 2.0) -> Dict[str, List[Tuple[str, float]]]:
        """Build expansion lookup tables for query expansion"""
        expansion_tables = defaultdict(list)
        
        # Get all co-occurrence results
        if not self.cooccurrence_cache:
            self.calculate_pmi_scores()
        
        # Build expansion tables
        for (term1, term2), result in self.cooccurrence_cache.items():
            if result.lift >= min_lift and result.confidence >= min_confidence:
                # Add expansion with confidence weight
                expansion_tables[term1].append((term2, result.confidence))
        
        # Sort expansions by confidence
        for term, expansions in expansion_tables.items():
            expansion_tables[term] = sorted(expansions, key=lambda x: x[1], reverse=True)
        
        logger.info(f"Built expansion tables for {len(expansion_tables)} terms")
        
        return dict(expansion_tables)
    
    def get_term_associations(self, term: str, max_results: int = 10) -> List[Dict[str, any]]:
        """Get top associated terms for a given term"""
        associations = []
        
        # Normalize term
        term_lower = term.lower()
        
        # Check different term types
        term_variants = [
            f"company:{term_lower}",
            f"title:{term_lower}",
            f"role:{term_lower}",
            f"skill:{term_lower}",
            f"location:{term_lower}",
            term_lower  # Raw term
        ]
        
        for variant in term_variants:
            if variant in self.term_to_idx:
                # Get co-occurrences
                for other_term, result in self.cooccurrence_cache.items():
                    if result.term1 == variant:
                        associations.append({
                            'term': result.term2,
                            'cooccurrence': result.cooccurrence_count,
                            'pmi': result.pmi,
                            'confidence': result.confidence,
                            'lift': result.lift
                        })
                    elif result.term2 == variant:
                        associations.append({
                            'term': result.term1,
                            'cooccurrence': result.cooccurrence_count,
                            'pmi': result.pmi,
                            'confidence': result.reverse_confidence,
                            'lift': result.lift
                        })
        
        # Sort by PMI and return top results
        associations.sort(key=lambda x: x['pmi'], reverse=True)
        
        return associations[:max_results]
    
    def discover_synonyms(self, min_similarity: float = 0.7) -> Dict[str, List[str]]:
        """Discover synonyms based on similar co-occurrence patterns"""
        synonyms = defaultdict(list)
        
        # For each term, find terms with similar co-occurrence patterns
        vocab_size = len(self.term_to_idx)
        
        for i in range(vocab_size):
            term1 = self.idx_to_term[i]
            term1_vector = self.term_doc_matrix[i].toarray().flatten()
            
            if np.sum(term1_vector) < 5:  # Skip rare terms
                continue
            
            for j in range(i + 1, vocab_size):
                term2 = self.idx_to_term[j]
                term2_vector = self.term_doc_matrix[j].toarray().flatten()
                
                if np.sum(term2_vector) < 5:  # Skip rare terms
                    continue
                
                # Calculate cosine similarity
                dot_product = np.dot(term1_vector, term2_vector)
                norm1 = np.linalg.norm(term1_vector)
                norm2 = np.linalg.norm(term2_vector)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                    
                    if similarity >= min_similarity:
                        # Check if they're the same type (e.g., both companies)
                        type1 = term1.split(':')[0] if ':' in term1 else 'unknown'
                        type2 = term2.split(':')[0] if ':' in term2 else 'unknown'
                        
                        if type1 == type2:
                            synonyms[term1].append((term2, similarity))
                            synonyms[term2].append((term1, similarity))
        
        # Sort synonyms by similarity
        for term, syns in synonyms.items():
            synonyms[term] = sorted(syns, key=lambda x: x[1], reverse=True)
        
        logger.info(f"Discovered synonyms for {len(synonyms)} terms")
        
        return dict(synonyms)
    
    def save_analysis_results(self, output_dir: str):
        """Save analysis results to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save expansion tables
        expansion_tables = self.build_expansion_tables()
        with open(os.path.join(output_dir, 'expansion_tables.json'), 'w') as f:
            json.dump(expansion_tables, f, indent=2)
        
        # Save top co-occurrences
        top_cooccurrences = []
        for result in sorted(self.cooccurrence_cache.values(), 
                           key=lambda x: x.pmi, reverse=True)[:1000]:
            top_cooccurrences.append({
                'term1': result.term1,
                'term2': result.term2,
                'count': result.cooccurrence_count,
                'pmi': result.pmi,
                'lift': result.lift,
                'confidence': result.confidence
            })
        
        with open(os.path.join(output_dir, 'top_cooccurrences.json'), 'w') as f:
            json.dump(top_cooccurrences, f, indent=2)
        
        # Save term frequencies
        with open(os.path.join(output_dir, 'term_frequencies.json'), 'w') as f:
            json.dump(dict(self.term_counts.most_common(1000)), f, indent=2)
        
        # Save the analyzer object
        with open(os.path.join(output_dir, 'cooccurrence_analyzer.pkl'), 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Analysis results saved to {output_dir}")
    
    def get_expansion_examples(self) -> Dict[str, List[str]]:
        """Get example expansions for common queries"""
        examples = {}
        
        # Common finance-related queries
        test_queries = [
            'ib', 'investment banking', 'goldman', 'pe', 'private equity',
            'vc', 'venture capital', 'consultant', 'mckinsey', 'software engineer',
            'product manager', 'data scientist', 'new york', 'san francisco'
        ]
        
        expansion_tables = self.build_expansion_tables()
        
        for query in test_queries:
            # Try different term types
            expansions = []
            for term_type in ['company:', 'title:', 'role:', 'location:', '']:
                term = f"{term_type}{query.lower()}"
                if term in expansion_tables:
                    expansions.extend([
                        exp[0] for exp in expansion_tables[term][:5]
                    ])
            
            if expansions:
                examples[query] = list(set(expansions))[:10]
        
        return examples


class QueryExpansionIndex:
    """Fast lookup index for query expansion"""
    
    def __init__(self, expansion_tables: Dict[str, List[Tuple[str, float]]]):
        self.expansion_tables = expansion_tables
        self.term_types = ['company', 'title', 'role', 'skill', 'location', 
                          'seniority', 'spec', 'dept', 'industry']
        
    def expand_query(self, query: str, max_expansions: int = 5) -> List[Tuple[str, float]]:
        """Expand a query term using co-occurrence data"""
        query_lower = query.lower()
        expansions = []
        
        # Try exact match first
        if query_lower in self.expansion_tables:
            expansions.extend(self.expansion_tables[query_lower][:max_expansions])
        
        # Try with different prefixes
        for term_type in self.term_types:
            prefixed_term = f"{term_type}:{query_lower}"
            if prefixed_term in self.expansion_tables:
                expansions.extend(self.expansion_tables[prefixed_term][:max_expansions])
        
        # Deduplicate and sort by confidence
        seen = set()
        unique_expansions = []
        for term, conf in sorted(expansions, key=lambda x: x[1], reverse=True):
            if term not in seen:
                seen.add(term)
                unique_expansions.append((term, conf))
        
        return unique_expansions[:max_expansions]
    
    def multi_hop_expansion(self, query: str, max_hops: int = 2, 
                           decay_factor: float = 0.5) -> List[Tuple[str, float]]:
        """Perform multi-hop expansion with decay"""
        expanded_terms = {query: 1.0}
        
        for hop in range(max_hops):
            new_terms = {}
            
            for term, weight in expanded_terms.items():
                # Get expansions for this term
                expansions = self.expand_query(term)
                
                for exp_term, exp_conf in expansions:
                    # Calculate decayed weight
                    new_weight = weight * exp_conf * (decay_factor ** hop)
                    
                    if exp_term in new_terms:
                        new_terms[exp_term] = max(new_terms[exp_term], new_weight)
                    else:
                        new_terms[exp_term] = new_weight
            
            # Add new terms to expanded set
            for term, weight in new_terms.items():
                if term not in expanded_terms:
                    expanded_terms[term] = weight
        
        # Remove original query and sort by weight
        expanded_terms.pop(query, None)
        return sorted(expanded_terms.items(), key=lambda x: x[1], reverse=True)


# Example usage
if __name__ == "__main__":
    db_config = {
        'host': 'localhost',
        'database': 'yale_alumni',
        'user': 'postgres',
        'password': 'password',
        'port': 5432
    }
    
    # Initialize analyzer
    analyzer = CooccurrenceAnalyzer(db_config)
    
    # Build term-document matrix
    analyzer.build_term_document_matrix()
    
    # Calculate PMI scores
    results = analyzer.calculate_pmi_scores()
    
    # Print top co-occurrences
    print("\nTop 20 Co-occurrences by PMI:")
    for result in results[:20]:
        print(f"{result.term1} <-> {result.term2}: "
              f"PMI={result.pmi:.3f}, Lift={result.lift:.2f}, "
              f"Count={result.cooccurrence_count}")
    
    # Get associations for 'goldman sachs'
    print("\nAssociations for 'goldman sachs':")
    associations = analyzer.get_term_associations('goldman sachs')
    for assoc in associations[:10]:
        print(f"  {assoc['term']}: PMI={assoc['pmi']:.3f}, "
              f"Confidence={assoc['confidence']:.3f}")
    
    # Save results
    analyzer.save_analysis_results('cooccurrence_results')
    
    # Test query expansion
    expansion_tables = analyzer.build_expansion_tables()
    index = QueryExpansionIndex(expansion_tables)
    
    print("\nQuery Expansion Examples:")
    for query in ['ib', 'software engineer', 'new york']:
        expansions = index.expand_query(query)
        print(f"\n{query} -> {[t[0] for t in expansions]}")