#!/usr/bin/env python3
"""
Enhanced Yale Alumni Search Engine
Uses domain-specific phrase recognition and semantic understanding
Based on analysis of actual 4,165 alumni data
"""

import json
import psycopg2
import sys
import re
import math
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter

@dataclass
class EnhancedSearchResult:
    person_id: int
    graduation_year: int
    name: str
    email: str
    current_position: str
    current_company: str
    industry: str
    function: str
    location: str
    city: str
    state_territory: str
    major: str
    skills: str
    linkedin_url: str
    experience_history: str
    education_history: str
    final_score: float
    score_breakdown: Dict[str, float]
    matched_terms: List[str]
    semantic_matches: List[str]
    explanation: str

class EnhancedQueryProcessor:
    """
    Advanced query processing with phrase recognition and semantic understanding
    """
    
    def __init__(self):
        self.load_enhanced_mappings()
        self.load_phrase_patterns()
        self.load_domain_mappings()
    
    def load_enhanced_mappings(self):
        """Load enhanced mappings created from real alumni data"""
        try:
            with open('enhanced_mappings.json', 'r') as f:
                self.enhanced_mappings = json.load(f)
        except FileNotFoundError:
            # Fallback to built-in mappings based on our analysis
            self.enhanced_mappings = {
                # Financial industry - based on real alumni data
                'wall': ['wall street', 'finance', 'financial', 'investment banking', 'banking', 'goldman', 'morgan'],
                'st': ['street', 'wall street', 'finance', 'financial'], 
                'vc': ['venture capital', 'venture', 'capital', 'investment', 'startup', 'private equity'],
                'banker': ['investment banker', 'banking', 'finance', 'financial', 'investment banking', 'goldman', 'morgan'],
                'investment': ['banking', 'finance', 'financial', 'capital', 'asset management', 'goldman', 'morgan'],
                'banking': ['investment', 'finance', 'financial', 'banker', 'wall street', 'goldman', 'morgan'],
                'goldman': ['goldman sachs', 'investment banking', 'finance', 'wall street', 'banking'],
                'sachs': ['goldman sachs', 'goldman', 'investment banking', 'finance'],
                'morgan': ['morgan stanley', 'jp morgan', 'jpmorgan', 'investment banking', 'finance'],
                'finance': ['financial', 'investment', 'banking', 'capital', 'wall street', 'goldman', 'morgan'],
                'financial': ['finance', 'investment', 'banking', 'capital', 'analyst'],
                
                # Consulting - based on real alumni data  
                'consulting': ['consultant', 'mckinsey', 'bain', 'bcg', 'deloitte', 'strategy'],
                'mckinsey': ['consulting', 'strategy', 'management consulting'],
                'bain': ['consulting', 'strategy', 'management consulting'],
                'bcg': ['boston consulting', 'consulting', 'strategy'],
                
                # Technology - based on real alumni data
                'software': ['engineer', 'developer', 'programming', 'tech', 'google', 'meta'],
                'engineer': ['software', 'developer', 'technical', 'programming', 'google'],
                'data': ['scientist', 'analyst', 'analysis', 'analytics', 'machine learning'],
                'google': ['tech', 'technology', 'software', 'engineer'],
                'meta': ['facebook', 'tech', 'technology', 'software']
            }
    
    def load_phrase_patterns(self):
        """Load important phrase patterns from real alumni data"""
        self.important_phrases = {
            # Financial phrases
            'wall st': 'wall street',
            'wall st.': 'wall street', 
            'wall street': 'wall street',
            'investment bank': 'investment banking',
            'investment banking': 'investment banking',
            'private equity': 'private equity',
            'venture capital': 'venture capital',
            'hedge fund': 'hedge fund',
            'asset management': 'asset management',
            'goldman sachs': 'goldman sachs',
            'morgan stanley': 'morgan stanley',
            'jp morgan': 'jp morgan',
            'jpmorgan': 'jp morgan',
            
            # Tech phrases
            'software engineer': 'software engineer',
            'data scientist': 'data scientist', 
            'machine learning': 'machine learning',
            'artificial intelligence': 'artificial intelligence',
            'product manager': 'product manager',
            
            # Consulting phrases
            'management consulting': 'management consulting',
            'strategy consulting': 'strategy consulting',
            'mckinsey company': 'mckinsey',
            'bain company': 'bain',
            'boston consulting': 'bcg',
            
            # Common combinations that should be preserved
            'vc banker': 'venture capital investment banking',
            'wall st banker': 'wall street investment banking',
            'wall st vc': 'wall street venture capital'
        }
    
    def load_domain_mappings(self):
        """Load domain-specific semantic mappings"""
        self.domain_mappings = {
            'finance': {
                'primary_terms': ['finance', 'financial', 'investment', 'banking', 'capital'],
                'companies': ['goldman sachs', 'morgan stanley', 'jp morgan', 'jpmorgan', 'citadel', 
                            'blackstone', 'kkr', 'apollo', 'bridgewater'],
                'roles': ['investment banker', 'analyst', 'associate', 'vice president', 'managing director',
                         'portfolio manager', 'research analyst', 'trader', 'quant'],
                'keywords': ['wall street', 'private equity', 'venture capital', 'hedge fund', 
                           'asset management', 'securities', 'derivatives', 'equity', 'debt']
            },
            'technology': {
                'primary_terms': ['technology', 'tech', 'software', 'engineer', 'developer'],
                'companies': ['google', 'meta', 'microsoft', 'amazon', 'apple', 'netflix', 'uber'],
                'roles': ['software engineer', 'data scientist', 'product manager', 'engineering manager'],
                'keywords': ['programming', 'coding', 'machine learning', 'ai', 'data analysis']
            },
            'consulting': {
                'primary_terms': ['consulting', 'consultant', 'advisory', 'strategy'],
                'companies': ['mckinsey', 'bain', 'bcg', 'deloitte', 'pwc', 'accenture'],
                'roles': ['consultant', 'senior consultant', 'principal', 'partner'],
                'keywords': ['management consulting', 'strategy consulting', 'business analyst']
            }
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query processing with phrase recognition and semantic understanding"""
        print(f"🧠 Enhanced Query Processing: '{query}'")
        
        # Step 1: Phrase recognition
        recognized_phrases = self.recognize_phrases(query)
        print(f"   📝 Recognized phrases: {recognized_phrases}")
        
        # Step 2: Domain classification
        domain_classification = self.classify_query_domain(query, recognized_phrases)
        print(f"   🎯 Domain classification: {domain_classification}")
        
        # Step 3: Semantic expansion
        semantic_terms = self.expand_semantically(query, recognized_phrases, domain_classification)
        print(f"   🔄 Semantic expansion: {semantic_terms}")
        
        # Step 4: Generate search terms
        search_terms = self.generate_search_terms(query, recognized_phrases, semantic_terms)
        print(f"   ⚡ Final search terms: {search_terms}")
        
        return {
            'original_query': query,
            'recognized_phrases': recognized_phrases,
            'domain_classification': domain_classification,
            'semantic_terms': semantic_terms,
            'search_terms': search_terms
        }
    
    def recognize_phrases(self, query: str) -> List[str]:
        """Recognize important phrases in the query"""
        query_lower = query.lower()
        recognized = []
        
        # Check for exact phrase matches (longest first)
        sorted_phrases = sorted(self.important_phrases.keys(), key=len, reverse=True)
        
        for phrase in sorted_phrases:
            if phrase in query_lower:
                canonical_form = self.important_phrases[phrase]
                if canonical_form not in recognized:
                    recognized.append(canonical_form)
        
        # Special handling for finance phrases
        if any(term in query_lower for term in ['wall st', 'wall street', 'vc', 'banker', 'investment']):
            if 'finance_domain' not in recognized:
                recognized.append('finance_domain')
        
        return recognized
    
    def classify_query_domain(self, query: str, phrases: List[str]) -> Dict[str, float]:
        """Classify query into domain categories with confidence scores"""
        query_text = f"{query} {' '.join(phrases)}".lower()
        domain_scores = {}
        
        for domain, mappings in self.domain_mappings.items():
            score = 0.0
            
            # Check primary terms
            for term in mappings['primary_terms']:
                if term in query_text:
                    score += 3.0
            
            # Check companies
            for company in mappings['companies']:
                if company in query_text:
                    score += 4.0
            
            # Check roles
            for role in mappings['roles']:
                if role in query_text:
                    score += 3.5
            
            # Check keywords
            for keyword in mappings['keywords']:
                if keyword in query_text:
                    score += 2.0
            
            if score > 0:
                domain_scores[domain] = score
        
        # Normalize scores
        total_score = sum(domain_scores.values())
        if total_score > 0:
            domain_scores = {k: v/total_score for k, v in domain_scores.items()}
        
        return domain_scores
    
    def expand_semantically(self, query: str, phrases: List[str], domains: Dict[str, float]) -> Dict[str, float]:
        """Expand query terms using semantic understanding"""
        expanded_terms = {}
        
        # Get original tokens
        original_tokens = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        
        # Expand using enhanced mappings
        for token in original_tokens:
            if token in self.enhanced_mappings:
                for synonym in self.enhanced_mappings[token]:
                    expanded_terms[synonym] = 0.8
        
        # Add phrase-based expansions
        for phrase in phrases:
            phrase_tokens = phrase.split()
            for token in phrase_tokens:
                if token in self.enhanced_mappings:
                    for synonym in self.enhanced_mappings[token]:
                        expanded_terms[synonym] = 0.9  # Higher confidence for phrase-based
        
        # Domain-specific expansion
        for domain, confidence in domains.items():
            if domain in self.domain_mappings:
                domain_data = self.domain_mappings[domain]
                
                # Add domain-specific terms with weighted confidence
                for term_list in [domain_data['companies'], domain_data['roles'], domain_data['keywords']]:
                    for term in term_list:
                        if term not in expanded_terms:
                            expanded_terms[term] = confidence * 0.7
        
        return expanded_terms
    
    def generate_search_terms(self, query: str, phrases: List[str], semantic_terms: Dict[str, float]) -> Dict[str, float]:
        """Generate final weighted search terms"""
        search_terms = {}
        
        # Original query terms (highest weight)
        original_tokens = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        for token in original_tokens:
            if len(token) > 2:
                search_terms[token] = 1.0
        
        # Recognized phrases (very high weight)
        for phrase in phrases:
            search_terms[phrase] = 1.0
            # Also add individual words from phrases
            for word in phrase.split():
                if len(word) > 2:
                    search_terms[word] = 0.9
        
        # Semantic expansion terms (weighted by confidence)
        for term, confidence in semantic_terms.items():
            if term not in search_terms:
                search_terms[term] = confidence
            else:
                # Boost existing terms
                search_terms[term] = max(search_terms[term], confidence)
        
        return search_terms

class EnhancedYaleSearch:
    """
    Enhanced search engine with semantic understanding
    """
    
    def __init__(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        self.db_config = self.config['database']
        self.query_processor = EnhancedQueryProcessor()
    
    def search(self, query: str, limit: int = 10, show_processing: bool = False) -> List[EnhancedSearchResult]:
        """Perform enhanced semantic search"""
        print(f"\n🎓 Enhanced Yale Alumni Search")
        print("=" * 60)
        print(f"Query: '{query}'")
        
        # Process query with enhanced understanding
        query_analysis = self.query_processor.process_query(query)
        
        if show_processing:
            print(f"\n🔍 Query Analysis:")
            print(f"   Phrases: {query_analysis['recognized_phrases']}")
            print(f"   Domains: {query_analysis['domain_classification']}")
            print(f"   Semantic terms: {list(query_analysis['semantic_terms'].keys())[:10]}")
        
        # Execute search with weighted terms
        search_terms = query_analysis['search_terms']
        print(f"\n⚡ Searching with {len(search_terms)} weighted terms")
        
        raw_results = self._execute_enhanced_search(search_terms, limit * 3)
        
        # Apply enhanced scoring
        print(f"📊 Applying enhanced scoring to {len(raw_results)} candidates")
        scored_results = self._apply_enhanced_scoring(raw_results, query_analysis)
        
        # Return top results
        final_results = scored_results[:limit]
        print(f"✅ Returning top {len(final_results)} results")
        
        return final_results
    
    def _execute_enhanced_search(self, search_terms: Dict[str, float], limit: int) -> List[Dict]:
        """Execute search using weighted terms"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        # Build weighted search conditions with domain filtering
        high_weight_terms = [term for term, weight in search_terms.items() if weight >= 0.8]
        medium_weight_terms = [term for term, weight in search_terms.items() if 0.5 <= weight < 0.8]
        
        conditions = []
        params = []
        
        # For finance queries, add strict domain filtering
        is_finance_query = any(term in search_terms for term in ['investment', 'banking', 'finance', 'financial', 'wall street'])
        
        # High-weight terms (must match) - prioritize finance fields for finance queries
        if high_weight_terms:
            high_conditions = []
            for term in high_weight_terms[:10]:  # Limit for performance
                if is_finance_query and term in ['investment', 'banking', 'finance', 'financial', 'wall street', 'goldman', 'morgan']:
                    # For finance terms, focus on finance-relevant fields
                    high_conditions.append("""
                        (LOWER(current_position) LIKE %s OR 
                         LOWER(current_company) LIKE %s OR 
                         LOWER(industry) LIKE %s OR 
                         LOWER(function) LIKE %s)
                    """)
                else:
                    high_conditions.append("""
                        (LOWER(current_position) LIKE %s OR 
                         LOWER(current_company) LIKE %s OR 
                         LOWER(industry) LIKE %s OR 
                         LOWER(skills) LIKE %s)
                    """)
                pattern = f'%{term}%'
                params.extend([pattern] * 4)
            
            if high_conditions:
                conditions.append(f"({' OR '.join(high_conditions)})")
        
        # Medium-weight terms with domain awareness
        if medium_weight_terms:
            medium_conditions = []
            for term in medium_weight_terms[:8]:
                medium_conditions.append("""
                    (LOWER(current_position) LIKE %s OR 
                     LOWER(current_company) LIKE %s OR 
                     LOWER(industry) LIKE %s)
                """)
                pattern = f'%{term}%'
                params.extend([pattern] * 3)
            
            if medium_conditions:
                conditions.append(f"({' OR '.join(medium_conditions)})")
        
        # Add domain-specific filtering for finance queries
        if is_finance_query:
            finance_filter = """
                (LOWER(industry) LIKE '%finance%' OR 
                 LOWER(industry) LIKE '%banking%' OR 
                 LOWER(industry) LIKE '%investment%' OR
                 LOWER(function) LIKE '%finance%' OR
                 LOWER(current_position) LIKE '%investment%' OR
                 LOWER(current_position) LIKE '%banking%' OR
                 LOWER(current_position) LIKE '%analyst%' OR
                 LOWER(current_position) LIKE '%associate%' OR
                 LOWER(current_company) LIKE '%goldman%' OR
                 LOWER(current_company) LIKE '%morgan%' OR
                 LOWER(current_company) LIKE '%jpmorgan%' OR
                 LOWER(current_company) LIKE '%citadel%' OR
                 LOWER(current_company) LIKE '%blackstone%')
            """
            conditions.append(finance_filter)
        
        if not conditions:
            return []
        
        # Build sophisticated scoring SQL
        scoring_terms = list(search_terms.keys())[:5]  # Top 5 terms for scoring
        
        sql = f"""
        WITH scored_results AS (
            SELECT 
                person_id, graduation_year, name, email, city, state_territory,
                current_position, current_company, industry, function, major,
                skills, linkedin_url, experience_history, education_history, location,
                -- Enhanced scoring with domain understanding
                (
                    -- Position match (highest weight)
                    (CASE 
                        WHEN LOWER(current_position) LIKE %s THEN 15
                        WHEN LOWER(current_position) LIKE %s THEN 12
                        WHEN LOWER(current_position) LIKE %s THEN 10
                        ELSE 0 
                    END) +
                    -- Company match (very high weight for finance/tech)
                    (CASE 
                        WHEN LOWER(current_company) LIKE %s THEN 12
                        WHEN LOWER(current_company) LIKE %s THEN 10
                        WHEN LOWER(current_company) LIKE %s THEN 8
                        ELSE 0 
                    END) +
                    -- Industry match (high weight)
                    (CASE 
                        WHEN LOWER(industry) LIKE %s THEN 8
                        WHEN LOWER(industry) LIKE %s THEN 6
                        ELSE 0 
                    END) +
                    -- Skills match (medium weight)
                    (CASE 
                        WHEN LOWER(skills) LIKE %s THEN 6
                        WHEN LOWER(skills) LIKE %s THEN 4
                        ELSE 0 
                    END) +
                    -- Function match (medium weight)
                    (CASE 
                        WHEN LOWER(function) LIKE %s THEN 5
                        ELSE 0 
                    END) +
                    -- Recency bonus (recent grads get boost)
                    (CASE 
                        WHEN graduation_year >= 2020 THEN 3
                        WHEN graduation_year >= 2015 THEN 2
                        WHEN graduation_year >= 2010 THEN 1
                        ELSE 0
                    END)
                ) as relevance_score
            FROM alumni_real
            WHERE {' AND '.join(conditions)}
        )
        SELECT * FROM scored_results
        WHERE relevance_score > 0
        ORDER BY relevance_score DESC, graduation_year DESC, name
        LIMIT %s
        """
        
        # Add scoring parameters (ensure we have enough)
        scoring_params = []
        for i in range(11):  # We need exactly 11 scoring parameters
            if i < len(scoring_terms):
                scoring_params.append(f'%{scoring_terms[i]}%')
            else:
                scoring_params.append('%%')  # Empty pattern as fallback
        
        all_params = scoring_params + params + [limit]
        
        cur.execute(sql, all_params)
        
        results = []
        for row in cur.fetchall():
            results.append({
                'person_id': row[0],
                'graduation_year': row[1] or 0,
                'name': row[2] or '',
                'email': row[3] or '',
                'city': row[4] or '',
                'state_territory': row[5] or '',
                'current_position': row[6] or '',
                'current_company': row[7] or '',
                'industry': row[8] or '',
                'function': row[9] or '',
                'major': row[10] or '',
                'skills': row[11] or '',
                'linkedin_url': row[12] or '',
                'experience_history': row[13] or '',
                'education_history': row[14] or '',
                'location': row[15] or '',
                'base_score': row[16]
            })
        
        conn.close()
        return results
    
    def _apply_enhanced_scoring(self, results: List[Dict], query_analysis: Dict) -> List[EnhancedSearchResult]:
        """Apply enhanced scoring with semantic understanding"""
        
        scored_results = []
        search_terms = query_analysis['search_terms']
        domain_weights = query_analysis['domain_classification']
        
        for result in results:
            # Calculate semantic match score
            semantic_matches = self._find_semantic_matches(result, search_terms)
            matched_terms = self._find_matched_terms(result, search_terms)
            
            # Calculate domain-specific score boost
            domain_boost = self._calculate_domain_boost(result, domain_weights)
            
            # Calculate final score
            base_score = result.get('base_score', 0) / 50.0  # Normalize
            semantic_score = len(semantic_matches) / max(len(search_terms), 1)
            
            # Apply domain boost
            final_score = (base_score * 0.6 + semantic_score * 0.4) * (1.0 + domain_boost)
            final_score = min(final_score, 1.0)
            
            # Create explanation
            explanation = self._create_enhanced_explanation(
                result, matched_terms, semantic_matches, domain_weights
            )
            
            # Score breakdown
            score_breakdown = {
                'base_sql_score': base_score,
                'semantic_match': semantic_score,
                'domain_boost': domain_boost,
                'final_score': final_score
            }
            
            enhanced_result = EnhancedSearchResult(
                person_id=result['person_id'],
                graduation_year=result['graduation_year'],
                name=result['name'],
                email=result['email'],
                current_position=result['current_position'],
                current_company=result['current_company'],
                industry=result['industry'],
                function=result['function'],
                location=result['location'],
                city=result['city'],
                state_territory=result['state_territory'],
                major=result['major'],
                skills=result['skills'],
                linkedin_url=result['linkedin_url'],
                experience_history=result['experience_history'],
                education_history=result['education_history'],
                final_score=final_score,
                score_breakdown=score_breakdown,
                matched_terms=matched_terms,
                semantic_matches=semantic_matches,
                explanation=explanation
            )
            
            scored_results.append(enhanced_result)
        
        # Sort by final score
        scored_results.sort(key=lambda x: x.final_score, reverse=True)
        return scored_results
    
    def _find_semantic_matches(self, result: Dict, search_terms: Dict[str, float]) -> List[str]:
        """Find semantic matches in the result"""
        matches = []
        result_text = f"{result.get('current_position', '')} {result.get('current_company', '')} {result.get('industry', '')} {result.get('skills', '')}".lower()
        
        for term, weight in search_terms.items():
            if term in result_text and weight >= 0.7:
                matches.append(term)
        
        return matches
    
    def _find_matched_terms(self, result: Dict, search_terms: Dict[str, float]) -> List[str]:
        """Find all matched terms"""
        matches = []
        result_text = f"{result.get('current_position', '')} {result.get('current_company', '')} {result.get('industry', '')}".lower()
        
        for term in search_terms.keys():
            if term in result_text:
                matches.append(term)
        
        return matches
    
    def _calculate_domain_boost(self, result: Dict, domain_weights: Dict[str, float]) -> float:
        """Calculate domain-specific boost"""
        boost = 0.0
        result_text = f"{result.get('current_position', '')} {result.get('current_company', '')} {result.get('industry', '')}".lower()
        
        for domain, weight in domain_weights.items():
            domain_mappings = self.query_processor.domain_mappings.get(domain, {})
            
            # Check for domain-specific companies/roles
            for company in domain_mappings.get('companies', []):
                if company in result_text:
                    boost += weight * 0.3
            
            for role in domain_mappings.get('roles', []):
                if role in result_text:
                    boost += weight * 0.2
        
        return min(boost, 0.5)  # Cap boost at 50%
    
    def _create_enhanced_explanation(self, result: Dict, matched_terms: List[str], 
                                   semantic_matches: List[str], domains: Dict[str, float]) -> str:
        """Create detailed explanation"""
        explanations = []
        
        # Check position matches
        position = result.get('current_position', '').lower()
        for term in matched_terms:
            if term in position:
                explanations.append(f"position: '{term}'")
                break
        
        # Check company matches  
        company = result.get('current_company', '').lower()
        for term in matched_terms:
            if term in company:
                explanations.append(f"company: '{term}'")
                break
        
        # Add semantic matches
        if semantic_matches:
            explanations.append(f"semantic: {', '.join(semantic_matches[:2])}")
        
        # Add domain classification
        if domains:
            top_domain = max(domains.keys(), key=lambda k: domains[k])
            explanations.append(f"domain: {top_domain}")
        
        return "; ".join(explanations) if explanations else "general match"
    
    def print_enhanced_results(self, results: List[EnhancedSearchResult]):
        """Print enhanced results with detailed analysis"""
        if not results:
            print("No results found.")
            return
        
        print(f"\n🏆 Enhanced Search Results ({len(results)} found):")
        print("=" * 100)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.name} (Class of {result.graduation_year})")
            print(f"   📧 Email: {result.email}")
            print(f"   💼 Current: {result.current_position} at {result.current_company}")
            print(f"   🏢 Industry: {result.industry}")
            print(f"   🎯 Function: {result.function}")
            print(f"   📍 Location: {result.city}, {result.state_territory}")
            print(f"   🎓 Major: {result.major}")
            
            if result.linkedin_url:
                print(f"   💼 LinkedIn: {result.linkedin_url}")
            
            if result.skills:
                skills_display = result.skills[:80] + "..." if len(result.skills) > 80 else result.skills
                print(f"   💻 Skills: {skills_display}")
            
            print(f"   🎯 Final Score: {result.final_score:.3f}")
            
            # Show enhanced score breakdown
            print(f"   📊 Score Components:")
            print(f"      SQL Base Score: {result.score_breakdown.get('base_sql_score', 0):.3f}")
            print(f"      Semantic Match: {result.score_breakdown.get('semantic_match', 0):.3f}")
            print(f"      Domain Boost: {result.score_breakdown.get('domain_boost', 0):.3f}")
            
            # Show matches
            if result.matched_terms:
                print(f"   🔍 Matched Terms: {', '.join(result.matched_terms[:5])}")
            
            if result.semantic_matches:
                print(f"   🧠 Semantic Matches: {', '.join(result.semantic_matches[:3])}")
            
            print(f"   💡 Explanation: {result.explanation}")
            print(f"   🆔 ID: {result.person_id}")
            print()


def main():
    """Main interface"""
    if len(sys.argv) < 2:
        print("""
Enhanced Yale Alumni Search Engine
(Domain-aware semantic search with phrase recognition)

Usage:
  python enhanced_yale_search.py "search query"                    # Enhanced search
  python enhanced_yale_search.py "search query" --show-processing  # Show query processing

Examples:
  python enhanced_yale_search.py "Wall St. vc banker"
  python enhanced_yale_search.py "investment banking Goldman" --show-processing
  python enhanced_yale_search.py "software engineer machine learning"
  python enhanced_yale_search.py "data scientist startup" --show-processing
        """)
        return
    
    search_engine = EnhancedYaleSearch()
    show_processing = '--show-processing' in sys.argv
    
    # Remove flags from query
    query_parts = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    query = ' '.join(query_parts)
    
    results = search_engine.search(query, limit=10, show_processing=show_processing)
    search_engine.print_enhanced_results(results)

if __name__ == "__main__":
    main()