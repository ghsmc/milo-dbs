"""
Query Expansion Engine
Implements multi-hop expansions, TF-IDF weighting, and intelligent query parsing
"""

import re
import json
import pickle
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Structured representation of a parsed query"""
    raw_query: str
    entities: Dict[str, List[str]] = field(default_factory=lambda: {
        'companies': [],
        'titles': [],
        'locations': [],
        'skills': [],
        'seniority': [],
        'experience_years': [],
        'education': []
    })
    constraints: Dict[str, Any] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    expanded_terms: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExpansionCandidate:
    """Candidate term for query expansion"""
    term: str
    score: float
    source: str  # 'cooccurrence', 'semantic', 'synonym'
    hop_distance: int
    

class QueryParser:
    """Parses natural language queries into structured components"""
    
    def __init__(self):
        # Common patterns for different entity types
        self.patterns = {
            'company': [
                r'\b(?:at|@|from|with)\s+([A-Z][A-Za-z\s&]+)',
                r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:alumni|alum|people)',
                r'\b(?:work(?:s|ed|ing)?\s+(?:at|for))\s+([A-Z][A-Za-z\s&]+)',
            ],
            'title': [
                r'\b(software\s+engineer(?:ing)?|swe)\b',
                r'\b(product\s+manager|pm)\b',
                r'\b(data\s+scientist|data\s+analyst)\b',
                r'\b(investment\s+bank(?:er|ing)|ib)\b',
                r'\b(private\s+equity|pe)\b',
                r'\b(consult(?:ant|ing))\b',
                r'\b(analyst|associate|vp|vice\s+president|director|manager)\b',
            ],
            'location': [
                r'\b(?:in|from|based\s+in)\s+([A-Z][A-Za-z\s]+)',
                r'\b(new\s+york|nyc|ny|san\s+francisco|sf|bay\s+area|london|hong\s+kong)\b',
            ],
            'seniority': [
                r'\b(junior|senior|lead|principal|staff|entry\s+level|experienced)\b',
                r'\b(intern|associate|manager|director|vp|executive)\b',
            ],
            'experience': [
                r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?experience',
                r'(?:with\s+)?(?:at\s+least\s+)?(\d+)\s*(?:years?|yrs?)',
            ],
            'education': [
                r'\b(yale|harvard|stanford|mit|princeton|columbia)\b',
                r'\b(ba|bs|mba|phd|masters?|bachelors?)\b',
                r'\b(computer\s+science|cs|economics|finance|engineering)\b',
            ]
        }
        
        # Query modifiers and constraints
        self.constraint_patterns = {
            'current': r'\b(?:current|currently|now)\b',
            'former': r'\b(?:former|previously|ex-|past)\b',
            'recent': r'\b(?:recent|recently|last\s+\d+\s+years?)\b',
            'senior_only': r'\b(?:senior|experienced|seasoned)\s+(?:only|professionals?)\b',
        }
        
        # Common abbreviations to expand
        self.abbreviations = {
            'swe': 'software engineer',
            'pm': 'product manager',
            'ib': 'investment banking',
            'pe': 'private equity',
            'vc': 'venture capital',
            'hf': 'hedge fund',
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
        }
        
    def parse(self, query: str) -> ParsedQuery:
        """Parse a natural language query"""
        parsed = ParsedQuery(raw_query=query)
        query_lower = query.lower()
        
        # Expand abbreviations first
        expanded_query = self._expand_abbreviations(query_lower)
        
        # Extract companies
        for pattern in self.patterns['company']:
            matches = re.findall(pattern, query, re.IGNORECASE)
            parsed.entities['companies'].extend([m.strip() for m in matches])
        
        # Extract titles
        for pattern in self.patterns['title']:
            matches = re.findall(pattern, expanded_query, re.IGNORECASE)
            parsed.entities['titles'].extend([m.strip() for m in matches])
        
        # Extract locations
        for pattern in self.patterns['location']:
            matches = re.findall(pattern, query, re.IGNORECASE)
            parsed.entities['locations'].extend([m.strip() for m in matches])
        
        # Extract seniority
        for pattern in self.patterns['seniority']:
            matches = re.findall(pattern, expanded_query, re.IGNORECASE)
            parsed.entities['seniority'].extend([m.strip() for m in matches])
        
        # Extract experience requirements
        for pattern in self.patterns['experience']:
            matches = re.findall(pattern, expanded_query, re.IGNORECASE)
            for match in matches:
                try:
                    years = int(match)
                    parsed.entities['experience_years'].append(years)
                except:
                    pass
        
        # Extract education
        for pattern in self.patterns['education']:
            matches = re.findall(pattern, expanded_query, re.IGNORECASE)
            parsed.entities['education'].extend([m.strip() for m in matches])
        
        # Extract constraints
        for constraint, pattern in self.constraint_patterns.items():
            if re.search(pattern, expanded_query, re.IGNORECASE):
                parsed.constraints[constraint] = True
        
        # Extract remaining keywords
        # Remove stopwords and already extracted entities
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                    'to', 'for', 'of', 'with', 'by', 'from', 'about', 'as',
                    'who', 'what', 'when', 'where', 'how', 'which', 'that',
                    'find', 'search', 'looking', 'need', 'want', 'show', 'get'}
        
        words = expanded_query.split()
        extracted_text = ' '.join([
            item for sublist in parsed.entities.values() 
            for item in sublist
        ]).lower()
        
        for word in words:
            if (word not in stopwords and 
                word not in extracted_text and 
                len(word) > 2):
                parsed.keywords.append(word)
        
        # Deduplicate entities
        for entity_type in parsed.entities:
            parsed.entities[entity_type] = list(set(parsed.entities[entity_type]))
        
        return parsed
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations in text"""
        expanded = text
        for abbr, full in self.abbreviations.items():
            expanded = re.sub(r'\b' + abbr + r'\b', full, expanded, flags=re.IGNORECASE)
        return expanded


class QueryExpansionEngine:
    """Main query expansion engine using multiple techniques"""
    
    def __init__(self, expansion_tables_path: str, tfidf_model_path: Optional[str] = None):
        # Load expansion tables from co-occurrence analysis
        with open(expansion_tables_path, 'r') as f:
            self.expansion_tables = json.load(f)
            
        self.parser = QueryParser()
        
        # TF-IDF model for term weighting
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.tfidf_vocab = None
        
        if tfidf_model_path:
            self._load_tfidf_model(tfidf_model_path)
            
        # Configuration
        self.config = {
            'max_expansions_per_term': 5,
            'max_total_expansions': 20,
            'min_expansion_score': 0.3,
            'decay_factor': 0.5,
            'max_hops': 2,
            'semantic_weight': 0.3,
            'cooccurrence_weight': 0.5,
            'keyword_weight': 0.2
        }
        
    def _load_tfidf_model(self, model_path: str):
        """Load pre-computed TF-IDF model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.tfidf_vectorizer = model_data['vectorizer']
            self.tfidf_matrix = model_data['matrix']
            self.tfidf_vocab = model_data['vocab']
            
    def expand_query(self, query: str) -> ParsedQuery:
        """Expand a natural language query"""
        # Parse the query
        parsed = self.parser.parse(query)
        
        # Collect all terms to expand
        terms_to_expand = []
        
        # Add entities with appropriate prefixes
        for company in parsed.entities['companies']:
            terms_to_expand.append(('company', company))
        
        for title in parsed.entities['titles']:
            terms_to_expand.append(('title', title))
            
        for location in parsed.entities['locations']:
            terms_to_expand.append(('location', location))
            
        for skill in parsed.entities['skills']:
            terms_to_expand.append(('skill', skill))
            
        # Add keywords without prefix
        for keyword in parsed.keywords:
            terms_to_expand.append(('keyword', keyword))
        
        # Perform multi-hop expansion
        all_expansions = {}
        
        for term_type, term in terms_to_expand:
            expansions = self._multi_hop_expansion(term, term_type)
            
            for exp_term, score in expansions:
                if exp_term in all_expansions:
                    all_expansions[exp_term] = max(all_expansions[exp_term], score)
                else:
                    all_expansions[exp_term] = score
        
        # Apply TF-IDF weighting if available
        if self.tfidf_vectorizer:
            all_expansions = self._apply_tfidf_weighting(all_expansions)
        
        # Filter and sort expansions
        filtered_expansions = {
            term: score 
            for term, score in all_expansions.items()
            if score >= self.config['min_expansion_score']
        }
        
        # Keep top expansions
        sorted_expansions = sorted(
            filtered_expansions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.config['max_total_expansions']]
        
        parsed.expanded_terms = dict(sorted_expansions)
        
        return parsed
    
    def _multi_hop_expansion(self, term: str, term_type: str) -> List[Tuple[str, float]]:
        """Perform multi-hop expansion for a single term"""
        expanded = {}
        current_terms = {term: 1.0}
        
        for hop in range(self.config['max_hops']):
            new_terms = {}
            
            for current_term, current_score in current_terms.items():
                # Try with prefix
                prefixed_term = f"{term_type}:{current_term.lower()}"
                
                # Get expansions from co-occurrence
                for lookup_term in [prefixed_term, current_term.lower()]:
                    if lookup_term in self.expansion_tables:
                        expansions = self.expansion_tables[lookup_term]
                        
                        for exp_term, confidence in expansions[:self.config['max_expansions_per_term']]:
                            # Calculate score with decay
                            score = current_score * confidence * (self.config['decay_factor'] ** hop)
                            
                            if exp_term not in expanded:
                                new_terms[exp_term] = score
                            else:
                                new_terms[exp_term] = max(new_terms[exp_term], score)
            
            # Add new terms to expanded set
            for term, score in new_terms.items():
                if term not in expanded:
                    expanded[term] = score
                else:
                    expanded[term] = max(expanded[term], score)
            
            # Update current terms for next hop
            current_terms = new_terms
            
            if not current_terms:  # No more expansions
                break
        
        return list(expanded.items())
    
    def _apply_tfidf_weighting(self, expansions: Dict[str, float]) -> Dict[str, float]:
        """Apply TF-IDF weighting to prevent common terms from dominating"""
        if not self.tfidf_vocab:
            return expansions
            
        weighted_expansions = {}
        
        for term, score in expansions.items():
            # Extract clean term (remove prefix if present)
            clean_term = term.split(':')[-1] if ':' in term else term
            
            # Get IDF weight
            if clean_term in self.tfidf_vocab:
                term_idx = self.tfidf_vocab[clean_term]
                # IDF is stored in the vectorizer
                idf_weight = self.tfidf_vectorizer.idf_[term_idx]
                # Normalize IDF weight (higher IDF = rarer term = higher weight)
                normalized_idf = idf_weight / np.max(self.tfidf_vectorizer.idf_)
                weighted_score = score * (0.5 + 0.5 * normalized_idf)
            else:
                # Keep original score for terms not in vocabulary
                weighted_score = score
                
            weighted_expansions[term] = weighted_score
            
        return weighted_expansions
    
    def generate_search_terms(self, parsed_query: ParsedQuery) -> List[str]:
        """Generate final search terms from parsed and expanded query"""
        search_terms = []
        
        # Add original entities
        for entity_list in parsed_query.entities.values():
            search_terms.extend(entity_list)
        
        # Add keywords
        search_terms.extend(parsed_query.keywords)
        
        # Add top expanded terms
        for term, score in sorted(parsed_query.expanded_terms.items(), 
                                 key=lambda x: x[1], reverse=True)[:10]:
            # Remove prefix for search
            clean_term = term.split(':')[-1] if ':' in term else term
            search_terms.append(clean_term)
        
        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for term in search_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
                
        return unique_terms
    
    def explain_expansion(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Explain how the query was expanded"""
        explanation = {
            'original_query': parsed_query.raw_query,
            'extracted_entities': parsed_query.entities,
            'constraints': parsed_query.constraints,
            'keywords': parsed_query.keywords,
            'expansions': []
        }
        
        # Group expansions by type
        expansion_groups = defaultdict(list)
        for term, score in parsed_query.expanded_terms.items():
            term_type = term.split(':')[0] if ':' in term else 'general'
            clean_term = term.split(':')[-1] if ':' in term else term
            expansion_groups[term_type].append({
                'term': clean_term,
                'score': round(score, 3)
            })
        
        explanation['expansions'] = dict(expansion_groups)
        
        return explanation


class QueryExpansionService:
    """High-level service for query expansion"""
    
    def __init__(self, db_config: Dict[str, str], model_dir: str):
        self.db_config = db_config
        self.model_dir = model_dir
        
        # Load expansion engine
        expansion_tables_path = f"{model_dir}/expansion_tables.json"
        tfidf_model_path = f"{model_dir}/tfidf_model.pkl"
        
        self.engine = QueryExpansionEngine(
            expansion_tables_path,
            tfidf_model_path if os.path.exists(tfidf_model_path) else None
        )
        
    def expand_and_search(self, query: str) -> Dict[str, Any]:
        """Expand query and return search configuration"""
        # Expand the query
        parsed = self.engine.expand_query(query)
        
        # Generate search terms
        search_terms = self.engine.generate_search_terms(parsed)
        
        # Build search configuration
        search_config = {
            'terms': search_terms,
            'filters': self._build_filters(parsed),
            'boost_fields': self._build_boost_fields(parsed),
            'explanation': self.engine.explain_expansion(parsed)
        }
        
        return search_config
    
    def _build_filters(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Build SQL filters from parsed query"""
        filters = {}
        
        # Company filter
        if parsed_query.entities['companies']:
            filters['companies'] = parsed_query.entities['companies']
        
        # Location filter
        if parsed_query.entities['locations']:
            filters['locations'] = parsed_query.entities['locations']
        
        # Seniority filter
        if parsed_query.entities['seniority']:
            filters['seniority_levels'] = parsed_query.entities['seniority']
        
        # Experience filter
        if parsed_query.entities['experience_years']:
            min_years = min(parsed_query.entities['experience_years'])
            filters['min_experience_years'] = min_years
        
        # Constraint filters
        if parsed_query.constraints.get('current'):
            filters['is_current_position'] = True
        
        if parsed_query.constraints.get('recent'):
            filters['recent_only'] = True  # Last 3 years
            
        return filters
    
    def _build_boost_fields(self, parsed_query: ParsedQuery) -> Dict[str, float]:
        """Build field boosts for relevance scoring"""
        boosts = {
            'exact_title_match': 10.0,
            'exact_company_match': 8.0,
            'expanded_match': 3.0,
            'skill_match': 2.0,
            'description_match': 1.0
        }
        
        # Boost recent positions if requested
        if parsed_query.constraints.get('recent'):
            boosts['recency'] = 2.0
            
        # Boost senior positions if seniority mentioned
        if any(s in ['senior', 'lead', 'principal', 'director', 'vp'] 
               for s in parsed_query.entities['seniority']):
            boosts['seniority_level'] = 1.5
            
        return boosts


# Example usage
if __name__ == "__main__":
    import os
    
    # Test query parser
    parser = QueryParser()
    
    test_queries = [
        "software engineers at Google in NYC with 5+ years experience",
        "investment banking analysts from Goldman Sachs who went to Yale",
        "current PMs at startups in SF Bay Area",
        "senior data scientists with ML experience",
        "former McKinsey consultants now in private equity"
    ]
    
    print("Query Parsing Examples:")
    for query in test_queries:
        parsed = parser.parse(query)
        print(f"\nQuery: {query}")
        print(f"Entities: {json.dumps(parsed.entities, indent=2)}")
        print(f"Constraints: {parsed.constraints}")
        print(f"Keywords: {parsed.keywords}")
    
    # Test expansion engine (would need actual data)
    # engine = QueryExpansionEngine('path/to/expansion_tables.json')
    # expanded = engine.expand_query("IB analyst at Goldman")
    # print(f"\nExpanded terms: {expanded.expanded_terms}")