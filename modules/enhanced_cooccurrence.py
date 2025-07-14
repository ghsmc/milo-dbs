"""
Enhanced Co-occurrence Analysis Module
Implements temporal filtering and improved tokenization as per feedback #3
Prevents associations between meaningful terms and temporal/resume artifacts
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
import re
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


class TemporalTermFilter:
    """
    Filters temporal and resume-specific terms to prevent bad associations
    Addresses feedback #3: Don't associate "ib" with "May"
    """
    
    def __init__(self):
        # Temporal terms that should be filtered
        self.temporal_stopwords = {
            # Months
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
            
            # Common date formats
            'present', 'current', 'ongoing', 'now', 'today',
            
            # Duration terms
            'year', 'years', 'month', 'months', 'week', 'weeks', 'day', 'days',
            
            # Resume boilerplate
            'responsibilities', 'duties', 'tasks', 'role', 'position',
            'responsible', 'managed', 'led', 'oversaw', 'coordinated',
            'worked', 'experience', 'background', 'including', 'such', 'as',
            
            # Common filler words in job descriptions
            'various', 'multiple', 'several', 'different', 'range', 'wide',
            'broad', 'diverse', 'extensive', 'comprehensive',
            
            # Articles and prepositions (extend standard stop words)
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            
            # Numbers as strings (common in dates/durations)
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
            'nine', 'ten', 'eleven', 'twelve', 'twenty', 'thirty',
            
            # Time periods that appear in experience descriptions
            'summer', 'fall', 'spring', 'winter', 'semester', 'quarter',
            'internship', 'intern', 'co-op', 'temporary', 'temp', 'contract',
            'part-time', 'full-time', 'freelance', 'volunteer',
        }
        
        # Date patterns to filter
        self.date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',    # YYYY-MM-DD
            r'\b\d{1,2}\s+\d{4}\b',          # MM YYYY
            r'\b\d{4}\b',                     # Just years (if isolated)
        ]
        
        # Compiled patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.date_patterns]
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing temporal patterns"""
        if not text:
            return ""
        
        # Remove date patterns
        cleaned_text = text
        for pattern in self.compiled_patterns:
            cleaned_text = pattern.sub(' ', cleaned_text)
        
        # Remove extra whitespace
        cleaned_text = ' '.join(cleaned_text.split())
        
        return cleaned_text.lower()
    
    def filter_terms(self, terms: List[str]) -> List[str]:
        """Filter out temporal and boilerplate terms"""
        filtered = []
        
        for term in terms:
            # Clean the term
            cleaned_term = self.clean_text(term)
            
            # Skip if empty after cleaning
            if not cleaned_term or len(cleaned_term) < 2:
                continue
            
            # Skip if it's a temporal stopword
            if cleaned_term in self.temporal_stopwords:
                continue
            
            # Skip if it's just numbers
            if cleaned_term.isdigit():
                continue
            
            # Skip if it's mostly punctuation
            if len(re.sub(r'[^\w\s]', '', cleaned_term)) < len(cleaned_term) * 0.5:
                continue
            
            # Skip if it matches date patterns
            if any(pattern.match(cleaned_term) for pattern in self.compiled_patterns):
                continue
            
            # Keep terms that are prefixed (like "title:", "company:")
            if ':' in term:
                prefix, value = term.split(':', 1)
                cleaned_value = self.clean_text(value)
                if cleaned_value and cleaned_value not in self.temporal_stopwords:
                    filtered.append(f"{prefix}:{cleaned_value}")
            else:
                filtered.append(cleaned_term)
        
        return filtered


class EnhancedCooccurrenceAnalyzer:
    """
    Enhanced co-occurrence analyzer with temporal filtering
    Addresses feedback #3: Better tokenization and field limitation
    """
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.term_doc_matrix = None
        self.term_to_idx = {}
        self.idx_to_term = {}
        self.term_counts = Counter()
        self.doc_count = 0
        self.cooccurrence_cache = {}
        
        # Initialize temporal filter
        self.temporal_filter = TemporalTermFilter()
        
        # Enhanced TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=5,  # Term must appear in at least 5 documents
            max_df=0.7,  # Term can't appear in more than 70% of documents
            ngram_range=(1, 2),  # Unigrams and bigrams
            stop_words='english',
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]{1,}\b'  # At least 2 chars, start with letter
        )
        
        # Field weights for importance
        self.field_weights = {
            'title': 3.0,      # Most important for associations
            'company': 2.0,    # High importance
            'seniority': 2.0,  # High importance 
            'role': 2.0,       # High importance
            'spec': 1.5,       # Medium importance
            'industry': 1.5,   # Medium importance
            'skill': 1.0,      # Standard importance
            'location': 0.5,   # Lower importance
            'dept': 0.5        # Lower importance
        }
    
    def build_term_document_matrix(self):
        """Build term-document matrix with enhanced filtering"""
        conn = psycopg2.connect(**self.db_config)
        
        # IMPORTANT: Only process title and location fields as per feedback #3
        # This prevents association of unrelated terms from description fields
        query = """
            SELECT 
                a.person_id,
                a.current_title,
                a.normalized_company,
                te.normalized_title as extracted_title,
                te.seniority,
                te.role_type,
                te.specialization,
                te.industry_focus,
                le.metro_area
            FROM alumni a
            LEFT JOIN title_entities te ON a.person_id = te.person_id
            LEFT JOIN location_entities le ON a.person_id = le.person_id
            WHERE a.current_title IS NOT NULL  -- Focus on profiles with titles
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Build documents (each profile is a document)
        documents = []
        doc_terms_list = []
        
        for _, row in df.iterrows():
            # Collect terms for this profile (ONLY from title and location)
            raw_terms = []
            
            # Add title components (highest weight)
            if row['extracted_title']:
                title_terms = self.temporal_filter.clean_text(row['extracted_title']).split()
                for term in title_terms:
                    if len(term) > 2:  # Skip very short terms
                        raw_terms.append(f"title:{term}")
            
            # Add seniority (important signal)
            if row['seniority']:
                raw_terms.append(f"seniority:{row['seniority']}")
            
            # Add role type (important signal)
            if row['role_type']:
                raw_terms.append(f"role:{row['role_type']}")
            
            # Add specialization (important signal)
            if row['specialization']:
                raw_terms.append(f"spec:{row['specialization']}")
            
            # Add industry (important signal)
            if row['industry_focus']:
                raw_terms.append(f"industry:{row['industry_focus']}")
            
            # Add company (but cleaned to avoid temporal issues)
            if row['normalized_company']:
                company_clean = self.temporal_filter.clean_text(row['normalized_company'])
                if company_clean:
                    raw_terms.append(f"company:{company_clean}")
            
            # Add location (lowest weight, most filtered)
            if row['metro_area']:
                location_clean = self.temporal_filter.clean_text(row['metro_area'])
                if location_clean and location_clean not in {'area', 'region', 'metro'}:
                    raw_terms.append(f"location:{location_clean}")
            
            # Filter terms through temporal filter
            filtered_terms = self.temporal_filter.filter_terms(raw_terms)
            
            # Skip profiles with too few meaningful terms
            if len(filtered_terms) < 2:
                continue
            
            # Create document text for TF-IDF (extract values after colon)
            doc_text = ' '.join([
                t.split(':', 1)[1] if ':' in t else t 
                for t in filtered_terms
            ])
            
            documents.append(doc_text)
            doc_terms_list.append(filtered_terms)
        
        self.doc_count = len(documents)
        logger.info(f"Built {self.doc_count} documents for analysis")
        
        # Build vocabulary with frequency filtering
        term_freq = Counter()
        for terms in doc_terms_list:
            term_freq.update(terms)
        
        # Filter vocabulary by frequency (remove very rare terms)
        min_freq = max(2, self.doc_count // 1000)  # Dynamic threshold
        filtered_vocab = {
            term for term, count in term_freq.items() 
            if count >= min_freq
        }
        
        logger.info(f"Filtered vocabulary from {len(term_freq)} to {len(filtered_vocab)} terms")
        
        # Create term indices
        self.term_to_idx = {term: idx for idx, term in enumerate(sorted(filtered_vocab))}
        self.idx_to_term = {idx: term for term, idx in self.term_to_idx.items()}
        vocab_size = len(self.term_to_idx)
        
        logger.info(f"Final vocabulary size: {vocab_size}")
        
        # Build sparse term-document matrix with weights
        self.term_doc_matrix = lil_matrix((vocab_size, self.doc_count))
        
        for doc_idx, terms in enumerate(doc_terms_list):
            for term in terms:
                if term in self.term_to_idx:
                    term_idx = self.term_to_idx[term]
                    
                    # Apply field weights
                    weight = 1.0
                    if ':' in term:
                        field_type = term.split(':', 1)[0]
                        weight = self.field_weights.get(field_type, 1.0)
                    
                    self.term_doc_matrix[term_idx, doc_idx] = weight
                    self.term_counts[term] += 1
        
        # Convert to CSR format for efficient operations
        self.term_doc_matrix = self.term_doc_matrix.tocsr()
        
        logger.info("Enhanced term-document matrix built successfully")
    
    def calculate_pmi_scores(self, min_cooccurrence: int = 5) -> List[CooccurrenceResult]:
        """Calculate PMI scores for all term pairs with enhanced filtering"""
        results = []
        vocab_size = len(self.term_to_idx)
        
        # Calculate co-occurrence matrix (term x term)
        # Using matrix multiplication: C = M * M^T
        cooccurrence_matrix = self.term_doc_matrix @ self.term_doc_matrix.T
        
        # Convert to COO format for iteration
        cooccurrence_coo = cooccurrence_matrix.tocoo()
        
        # Process each co-occurrence
        processed_pairs = 0
        for i, j, cooccurrence_count in zip(cooccurrence_coo.row, 
                                           cooccurrence_coo.col, 
                                           cooccurrence_coo.data):
            if i >= j:  # Skip diagonal and duplicate pairs
                continue
            
            if cooccurrence_count < min_cooccurrence:
                continue
            
            term1 = self.idx_to_term[i]
            term2 = self.idx_to_term[j]
            
            # Additional filtering: Don't associate terms from same field type
            if self._should_skip_pair(term1, term2):
                continue
            
            term1_count = self.term_counts[term1]
            term2_count = self.term_counts[term2]
            
            # Calculate PMI
            p_term1 = term1_count / self.doc_count
            p_term2 = term2_count / self.doc_count
            p_cooccurrence = cooccurrence_count / self.doc_count
            
            if p_cooccurrence > 0 and p_term1 > 0 and p_term2 > 0:
                pmi = math.log(p_cooccurrence / (p_term1 * p_term2))
                
                # Normalized PMI (NPMI) scales to [-1, 1]
                npmi = pmi / (-math.log(p_cooccurrence))
                
                # Lift score
                lift = p_cooccurrence / (p_term1 * p_term2)
                
                # Confidence scores
                confidence = cooccurrence_count / term1_count
                reverse_confidence = cooccurrence_count / term2_count
                
                # Apply quality filters
                if lift > 2.0 and confidence > 0.3 and npmi > 0.2:
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
                    processed_pairs += 1
        
        logger.info(f"Found {len(results)} high-quality co-occurrence pairs from {processed_pairs} candidates")
        
        # Sort by lift score (highest first)
        results.sort(key=lambda x: x.lift, reverse=True)
        
        return results
    
    def _should_skip_pair(self, term1: str, term2: str) -> bool:
        """Check if term pair should be skipped"""
        # Don't associate terms from the same field type
        if ':' in term1 and ':' in term2:
            field1 = term1.split(':', 1)[0]
            field2 = term2.split(':', 1)[0]
            
            # Skip pairs from same field (except title-title which can be meaningful)
            if field1 == field2 and field1 != 'title':
                return True
            
            # Skip location-company pairs (often coincidental)
            if {field1, field2} == {'location', 'company'}:
                return True
        
        # Skip if either term is too generic
        generic_terms = {'role:other', 'spec:general', 'industry:general', 'location:other'}
        if term1 in generic_terms or term2 in generic_terms:
            return True
        
        return False
    
    def build_expansion_lookup(self, results: List[CooccurrenceResult],
                             max_expansions_per_term: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Build lookup table for query expansion"""
        expansion_lookup = defaultdict(list)
        
        for result in results:
            # Add bidirectional associations
            expansion_lookup[result.term1].append((result.term2, result.confidence))
            expansion_lookup[result.term2].append((result.term1, result.reverse_confidence))
        
        # Sort and limit expansions per term
        for term in expansion_lookup:
            expansion_lookup[term].sort(key=lambda x: x[1], reverse=True)
            expansion_lookup[term] = expansion_lookup[term][:max_expansions_per_term]
        
        return dict(expansion_lookup)
    
    def save_analysis_results(self, output_dir: str):
        """Save analysis results to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate results
        results = self.calculate_pmi_scores()
        expansion_lookup = self.build_expansion_lookup(results)
        
        # Save co-occurrence results
        results_data = [
            {
                'term1': r.term1,
                'term2': r.term2,
                'cooccurrence_count': r.cooccurrence_count,
                'pmi': r.pmi,
                'npmi': r.npmi,
                'lift': r.lift,
                'confidence': r.confidence
            }
            for r in results
        ]
        
        with open(f"{output_dir}/cooccurrence_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save expansion lookup
        with open(f"{output_dir}/expansion_lookup.json", 'w') as f:
            json.dump(expansion_lookup, f, indent=2)
        
        # Save filtered vocabulary
        vocab_data = {
            'term_to_idx': self.term_to_idx,
            'term_counts': dict(self.term_counts),
            'doc_count': self.doc_count
        }
        
        with open(f"{output_dir}/vocabulary.json", 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        logger.info(f"Saved analysis results to {output_dir}")
        return len(results)
    
    def get_expansion_examples(self) -> List[Dict[str, any]]:
        """Get examples of query expansions for analytics"""
        results = self.calculate_pmi_scores()
        expansion_lookup = self.build_expansion_lookup(results)
        
        # Get top examples for common terms
        examples = []
        test_terms = ['title:analyst', 'title:engineer', 'title:manager', 'seniority:senior']
        
        for term in test_terms:
            if term in expansion_lookup:
                expansions = expansion_lookup[term][:5]  # Top 5
                examples.append({
                    'original_term': term,
                    'expansions': [{'term': exp[0], 'confidence': exp[1]} for exp in expansions]
                })
        
        return examples