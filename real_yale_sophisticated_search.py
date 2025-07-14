#!/usr/bin/env python3
"""
Real Yale Alumni Sophisticated Search Engine
Integrates query expansion, co-occurrence analysis, and synonym mapping with real Yale data
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
class ExpansionPath:
    term: str
    confidence: float
    path: List[str]
    hop_distance: int

@dataclass
class QueryExpansionResult:
    original_terms: List[str]
    expanded_terms: Dict[str, float]  # term -> confidence
    expansion_paths: Dict[str, List[str]]
    temporal_filtered: List[str]

@dataclass 
class RealYaleAlumniResult:
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
    expanded_terms_matched: List[str]
    explanation: str

class RealYaleCooccurrenceExpander:
    """
    Build co-occurrence matrix from real Yale alumni data
    """
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.cooccurrence_matrix = {}
        self.term_frequencies = {}
        self._build_cooccurrence_data()
    
    def _build_cooccurrence_data(self):
        """Build co-occurrence matrix from real Yale alumni data"""
        print("📊 Building co-occurrence matrix from real Yale data...")
        
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        # Get all text data for analysis from real alumni
        cur.execute("""
            SELECT current_position, current_company, industry, function, major, skills
            FROM alumni_real 
            WHERE current_position != '' OR current_company != '' OR skills != ''
        """)
        
        rows = cur.fetchall()
        
        # Temporal stopwords (from enhanced_cooccurrence.py)
        temporal_stopwords = {
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
            'sep', 'oct', 'nov', 'dec', '2020', '2021', '2022', '2023', '2024',
            'responsibilities', 'managed', 'led', 'oversaw', 'coordinated',
            'responsible', 'experience', 'years', 'months', 'since', 'from', 'to',
            'and', 'the', 'of', 'at', 'in', 'for', 'with', 'on', 'to', 'a', 'an'
        }
        
        # Build co-occurrence matrix
        term_cooccurrence = defaultdict(lambda: defaultdict(int))
        term_counts = defaultdict(int)
        
        for row in rows:
            # Combine all text fields
            text_parts = []
            for field in row:
                if field and field != '-':
                    text_parts.append(field)
            
            text = ' '.join(text_parts)
            
            # Tokenize and filter
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            tokens = [t for t in tokens if t not in temporal_stopwords]
            
            # Count term frequencies
            for token in tokens:
                term_counts[token] += 1
            
            # Build co-occurrence matrix (within same person's profile)
            unique_tokens = list(set(tokens))
            for i, token1 in enumerate(unique_tokens):
                for token2 in unique_tokens[i+1:]:
                    if token1 != token2:
                        term_cooccurrence[token1][token2] += 1
                        term_cooccurrence[token2][token1] += 1
        
        self.cooccurrence_matrix = dict(term_cooccurrence)
        self.term_frequencies = dict(term_counts)
        
        print(f"   Built matrix with {len(self.term_frequencies)} terms")
        print(f"   Total co-occurrence pairs: {sum(len(v) for v in self.cooccurrence_matrix.values())}")
        
        # Show top co-occurrences as examples
        top_pairs = []
        for term1, related in self.cooccurrence_matrix.items():
            for term2, count in related.items():
                if count > 10:  # High co-occurrence threshold
                    top_pairs.append((term1, term2, count))
        
        top_pairs.sort(key=lambda x: x[2], reverse=True)
        print("   Top co-occurrences:")
        for term1, term2, count in top_pairs[:5]:
            print(f"     '{term1}' ↔ '{term2}' ({count} times)")
        
        conn.close()
    
    def expand_query(self, terms: List[str], max_hops: int = 2, max_expansions: int = 15) -> QueryExpansionResult:
        """
        Expand query using graph-based co-occurrence analysis
        """
        print(f"🕸️ Expanding query: {terms}")
        
        original_terms = [t.lower() for t in terms]
        expanded_terms = {}
        expansion_paths = {}
        
        # Synonym mapping for common Yale/career terms
        synonym_map = {
            # Tech roles
            'software': ['engineer', 'developer', 'programming', 'coding'],
            'engineer': ['developer', 'architect', 'programmer', 'engineering'],
            'developer': ['engineer', 'programmer', 'software', 'development'],
            
            # Data roles
            'data': ['scientist', 'analyst', 'analytics', 'analysis'],
            'scientist': ['researcher', 'analyst', 'research', 'science'],
            'analyst': ['analytics', 'analysis', 'researcher', 'data'],
            
            # Business roles
            'investment': ['banking', 'finance', 'financial', 'investor'],
            'banking': ['finance', 'investment', 'bank', 'financial'],
            'consultant': ['consulting', 'advisory', 'strategy', 'management'],
            'consulting': ['consultant', 'advisory', 'strategy', 'consultancy'],
            
            # Other
            'product': ['manager', 'management', 'pm', 'products'],
            'manager': ['management', 'director', 'lead', 'managing'],
            'marketing': ['brand', 'growth', 'digital', 'market'],
            
            # Companies
            'google': ['alphabet', 'youtube', 'goog'],
            'meta': ['facebook', 'instagram', 'whatsapp'],
            'mckinsey': ['mck', 'consulting'],
            'goldman': ['sachs', 'gs'],
            'sachs': ['goldman', 'gs']
        }
        
        # Add synonyms first
        for term in original_terms:
            if term in synonym_map:
                for synonym in synonym_map[term]:
                    expanded_terms[synonym] = 0.8  # High confidence for synonyms
                    expansion_paths[synonym] = [term, f"synonym:{synonym}"]
        
        # Use Dijkstra-inspired algorithm for multi-hop expansion
        for original_term in original_terms:
            if original_term in self.cooccurrence_matrix:
                # First hop: direct co-occurrences
                for related_term, count in self.cooccurrence_matrix[original_term].items():
                    if count >= 5:  # Minimum co-occurrence threshold for real data
                        # Calculate PMI-based confidence
                        pmi = self._calculate_pmi(original_term, related_term, count)
                        if pmi > 0.1:  # PMI threshold
                            confidence = min(pmi, 1.0)
                            if related_term not in expanded_terms or confidence > expanded_terms[related_term]:
                                expanded_terms[related_term] = confidence
                                expansion_paths[related_term] = [original_term, related_term]
                
                # Second hop: terms related to first-hop terms
                if max_hops > 1:
                    first_hop_terms = list(expanded_terms.keys())[:10]  # Limit for performance
                    for first_hop_term in first_hop_terms:
                        if first_hop_term in self.cooccurrence_matrix:
                            for second_hop_term, count in self.cooccurrence_matrix[first_hop_term].items():
                                if (second_hop_term not in expanded_terms and 
                                    second_hop_term not in original_terms and 
                                    count >= 3):  # Lower threshold for second hop
                                    pmi = self._calculate_pmi(first_hop_term, second_hop_term, count)
                                    if pmi > 0.05:  # Lower threshold for second hop
                                        confidence = min(pmi * 0.7, 1.0)  # Decay for distance
                                        expanded_terms[second_hop_term] = confidence
                                        expansion_paths[second_hop_term] = [original_term, first_hop_term, second_hop_term]
        
        # Sort by confidence and limit results
        sorted_expansions = sorted(expanded_terms.items(), key=lambda x: x[1], reverse=True)
        final_expanded = dict(sorted_expansions[:max_expansions])
        
        print(f"   Found {len(final_expanded)} expansion terms")
        for term, conf in list(final_expanded.items())[:5]:
            path = expansion_paths.get(term, [])
            print(f"     {term} (conf: {conf:.3f}) via: {' → '.join(path)}")
        
        return QueryExpansionResult(
            original_terms=original_terms,
            expanded_terms=final_expanded,
            expansion_paths=expansion_paths,
            temporal_filtered=[]
        )
    
    def _calculate_pmi(self, term1: str, term2: str, cooccurrence_count: int) -> float:
        """Calculate Pointwise Mutual Information"""
        total_profiles = 4165  # Real Yale dataset size
        term1_count = self.term_frequencies.get(term1, 1)
        term2_count = self.term_frequencies.get(term2, 1)
        
        # PMI = log(P(term1, term2) / (P(term1) * P(term2)))
        p_term1_term2 = cooccurrence_count / total_profiles
        p_term1 = term1_count / total_profiles
        p_term2 = term2_count / total_profiles
        
        if p_term1 * p_term2 > 0:
            pmi = math.log(p_term1_term2 / (p_term1 * p_term2))
            return max(pmi / 10.0, 0)  # Normalize to 0-1 range
        return 0.0

class RealYaleSophisticatedSearch:
    """
    Sophisticated search engine for real Yale alumni with query expansion
    """
    
    def __init__(self):
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        self.db_config = self.config['database']
        
        # Initialize components
        print("🚀 Initializing sophisticated real Yale search...")
        self.expander = RealYaleCooccurrenceExpander(self.db_config)
        print("✅ Ready for sophisticated searching!")
    
    def sophisticated_search(self, query: str, limit: int = 10, show_expansion: bool = False) -> List[RealYaleAlumniResult]:
        """
        Perform sophisticated search with query expansion and advanced scoring
        """
        print(f"\n🎓 Sophisticated Real Yale Alumni Search")
        print("=" * 60)
        print(f"Query: '{query}'")
        
        # Step 1: Parse query
        query_terms = re.findall(r'\b[a-zA-Z0-9]{2,}\b', query.lower())
        print(f"📝 Parsed terms: {query_terms}")
        
        # Step 2: Expand query using co-occurrence
        expansion_result = self.expander.expand_query(query_terms, max_hops=2, max_expansions=20)
        
        if show_expansion:
            print(f"\n🔄 Query Expansion Results:")
            print(f"   Original: {expansion_result.original_terms}")
            print(f"   Expanded: {list(expansion_result.expanded_terms.keys())}")
            print(f"   Top expansions with paths:")
            for term, conf in list(expansion_result.expanded_terms.items())[:5]:
                path = expansion_result.expansion_paths.get(term, [])
                print(f"     {term} (conf: {conf:.3f}) → {' → '.join(path)}")
        
        # Step 3: Execute comprehensive search
        all_search_terms = set(expansion_result.original_terms)
        all_search_terms.update(expansion_result.expanded_terms.keys())
        
        print(f"\n⚡ Executing search with {len(all_search_terms)} total terms")
        
        raw_results = self._execute_expanded_search(all_search_terms, expansion_result, limit * 2)
        
        # Step 4: Apply sophisticated scoring
        print(f"📊 Applying sophisticated scoring to {len(raw_results)} candidates")
        
        scored_results = []
        for result in raw_results:
            scores = self._calculate_comprehensive_score(
                result, expansion_result, query_terms
            )
            
            # Find matching expanded terms
            expanded_matches = self._find_expanded_matches(result, expansion_result)
            
            # Create explanation
            explanation = self._create_explanation(result, query_terms, expanded_matches, scores)
            
            alumni_result = RealYaleAlumniResult(
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
                final_score=scores['final_score'],
                score_breakdown=scores,
                expanded_terms_matched=expanded_matches,
                explanation=explanation
            )
            
            scored_results.append(alumni_result)
        
        # Sort by final score
        scored_results.sort(key=lambda x: x.final_score, reverse=True)
        
        print(f"✅ Returning top {min(limit, len(scored_results))} results")
        return scored_results[:limit]
    
    def _execute_expanded_search(self, search_terms: Set[str], expansion_result: QueryExpansionResult, limit: int) -> List[Dict]:
        """Execute search with all expanded terms including sophisticated SQL generation"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        # Build sophisticated search conditions with proper weighting
        search_conditions = []
        params = []
        
        # Group terms by confidence level
        high_confidence_terms = [term for term in expansion_result.original_terms]
        medium_confidence_terms = [term for term, conf in expansion_result.expanded_terms.items() if conf > 0.5]
        low_confidence_terms = [term for term, conf in expansion_result.expanded_terms.items() if conf <= 0.5]
        
        # Build weighted conditions
        weighted_conditions = []
        
        # High confidence (original terms) - must match at least one
        if high_confidence_terms:
            high_conditions = []
            for term in high_confidence_terms:
                if len(term) > 2:
                    high_conditions.append("""
                        (LOWER(current_position) LIKE %s OR 
                         LOWER(current_company) LIKE %s OR 
                         LOWER(industry) LIKE %s OR 
                         LOWER(skills) LIKE %s OR
                         LOWER(major) LIKE %s)
                    """)
                    like_pattern = f'%{term}%'
                    params.extend([like_pattern] * 5)
            
            if high_conditions:
                weighted_conditions.append(f"({' OR '.join(high_conditions)})")
        
        # Medium confidence (strong expansions)
        if medium_confidence_terms:
            medium_conditions = []
            for term in medium_confidence_terms[:10]:  # Limit to top 10
                if len(term) > 2:
                    medium_conditions.append("""
                        (LOWER(current_position) LIKE %s OR 
                         LOWER(current_company) LIKE %s OR 
                         LOWER(skills) LIKE %s)
                    """)
                    like_pattern = f'%{term}%'
                    params.extend([like_pattern] * 3)
            
            if medium_conditions:
                weighted_conditions.append(f"({' OR '.join(medium_conditions)})")
        
        if not weighted_conditions:
            return []
        
        # Build comprehensive SQL with sophisticated scoring
        sql = f"""
        WITH scored_results AS (
            SELECT 
                person_id, graduation_year, name, email, city, state_territory,
                current_position, current_company, industry, function, major,
                skills, linkedin_url, experience_history, education_history, location,
                -- Sophisticated scoring components
                (
                    -- Exact match on position (weight: 10)
                    (CASE 
                        WHEN LOWER(current_position) LIKE %s THEN 10
                        WHEN LOWER(current_position) LIKE %s THEN 8
                        ELSE 0 
                    END) +
                    -- Company match (weight: 8)
                    (CASE 
                        WHEN LOWER(current_company) LIKE %s THEN 8
                        WHEN LOWER(current_company) LIKE %s THEN 6
                        ELSE 0 
                    END) +
                    -- Industry match (weight: 6)
                    (CASE WHEN LOWER(industry) LIKE %s THEN 6 ELSE 0 END) +
                    -- Skills match (weight: 5)
                    (CASE WHEN LOWER(skills) LIKE %s THEN 5 ELSE 0 END) +
                    -- Major match (weight: 4)
                    (CASE WHEN LOWER(major) LIKE %s THEN 4 ELSE 0 END) +
                    -- Experience history match (weight: 3)
                    (CASE WHEN LOWER(experience_history) LIKE %s THEN 3 ELSE 0 END) +
                    -- Recency bonus (weight: 2)
                    (CASE 
                        WHEN graduation_year >= 2020 THEN 2
                        WHEN graduation_year >= 2015 THEN 1
                        ELSE 0
                    END)
                ) as relevance_score
            FROM alumni_real
            WHERE {' AND '.join(weighted_conditions)}
        )
        SELECT * FROM scored_results
        WHERE relevance_score > 0
        ORDER BY relevance_score DESC, graduation_year DESC, name
        LIMIT %s
        """
        
        # Add scoring parameters (using first two original terms for primary scoring)
        scoring_terms = expansion_result.original_terms[:2] if len(expansion_result.original_terms) >= 2 else expansion_result.original_terms
        if len(scoring_terms) < 2:
            scoring_terms = scoring_terms + ['']  # Pad with empty string if needed
        
        scoring_params = []
        for term in scoring_terms:
            scoring_params.append(f'%{term}%' if term else '%%')
        
        # Duplicate for position and company (2 params each)
        scoring_params = scoring_params * 2
        
        # Add single params for industry, skills, major, experience
        first_term = expansion_result.original_terms[0] if expansion_result.original_terms else ''
        for _ in range(4):
            scoring_params.append(f'%{first_term}%')
        
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
    
    def _calculate_comprehensive_score(self, result: Dict, expansion_result: QueryExpansionResult, 
                                     original_terms: List[str]) -> Dict[str, float]:
        """Calculate comprehensive score using all components"""
        
        # Create searchable text
        result_text = ' '.join([
            result.get('name', ''),
            result.get('current_position', ''),
            result.get('current_company', ''),
            result.get('industry', ''),
            result.get('skills', ''),
            result.get('major', ''),
            result.get('experience_history', '')
        ]).lower()
        
        # Calculate exact match score
        exact_matches = sum(1 for term in original_terms if term in result_text)
        exact_match_score = exact_matches / max(len(original_terms), 1)
        
        # Calculate expansion score
        expansion_score = 0.0
        for term, confidence in expansion_result.expanded_terms.items():
            if term in result_text:
                expansion_score += confidence
        expansion_score = min(expansion_score / 3.0, 1.0)  # Normalize
        
        # Calculate semantic score based on field matches
        semantic_score = 0.0
        if any(term in result.get('current_position', '').lower() for term in original_terms):
            semantic_score += 0.4
        if any(term in result.get('current_company', '').lower() for term in original_terms):
            semantic_score += 0.3
        if any(term in result.get('skills', '').lower() for term in original_terms):
            semantic_score += 0.3
        
        # Recency bonus
        recency_score = 0.8
        if result.get('graduation_year', 0) >= 2020:
            recency_score = 1.0
        elif result.get('graduation_year', 0) >= 2015:
            recency_score = 0.9
        
        # Calculate final normalized score
        base_score = result.get('base_score', 0) / 40.0  # Normalize SQL score
        
        final_score = (
            0.3 * base_score +
            0.3 * exact_match_score +
            0.2 * semantic_score +
            0.1 * expansion_score +
            0.1 * recency_score
        )
        
        return {
            'base_score': base_score,
            'exact_match': exact_match_score,
            'semantic_similarity': semantic_score,
            'expansion_boost': expansion_score,
            'recency_factor': recency_score,
            'final_score': min(final_score, 1.0)
        }
    
    def _find_expanded_matches(self, result: Dict, expansion_result: QueryExpansionResult) -> List[str]:
        """Find which expanded terms matched this result"""
        result_text = ' '.join([
            result.get('current_position', ''),
            result.get('current_company', ''),
            result.get('skills', ''),
            result.get('industry', '')
        ]).lower()
        
        matches = []
        for term in expansion_result.expanded_terms.keys():
            if term in result_text:
                matches.append(term)
        
        return matches[:5]  # Limit to top 5 for display
    
    def _create_explanation(self, result: Dict, original_terms: List[str], 
                          expanded_matches: List[str], scores: Dict[str, float]) -> str:
        """Create human-readable explanation"""
        explanations = []
        
        # Check original term matches
        for term in original_terms:
            if term in result.get('current_position', '').lower():
                explanations.append(f"position: '{term}'")
            elif term in result.get('current_company', '').lower():
                explanations.append(f"company: '{term}'")
            elif term in result.get('skills', '').lower():
                explanations.append(f"skills: '{term}'")
        
        # Add expanded term matches
        if expanded_matches:
            explanations.append(f"expanded: {', '.join(expanded_matches[:2])}")
        
        # Add score highlights
        if scores.get('exact_match', 0) > 0.7:
            explanations.append("strong exact match")
        elif scores.get('expansion_boost', 0) > 0.5:
            explanations.append("co-occurrence match")
        
        return "; ".join(explanations) if explanations else "general relevance"
    
    def print_sophisticated_results(self, results: List[RealYaleAlumniResult]):
        """Print results with detailed analysis"""
        if not results:
            print("No results found.")
            return
        
        print(f"\n🏆 Sophisticated Real Yale Alumni Results ({len(results)} found):")
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
            
            # Show skills (truncated)
            if result.skills:
                skills_display = result.skills[:80] + "..." if len(result.skills) > 80 else result.skills
                print(f"   💻 Skills: {skills_display}")
            
            print(f"   🎯 Final Score: {result.final_score:.3f}")
            
            # Show score breakdown
            breakdown = result.score_breakdown
            print(f"   📊 Score Components:")
            print(f"      Base SQL Score: {breakdown.get('base_score', 0):.3f}")
            print(f"      Exact Match: {breakdown.get('exact_match', 0):.3f}")
            print(f"      Semantic Similarity: {breakdown.get('semantic_similarity', 0):.3f}")
            print(f"      Query Expansion: {breakdown.get('expansion_boost', 0):.3f}")
            print(f"      Recency Factor: {breakdown.get('recency_factor', 0):.3f}")
            
            # Show expanded terms that matched
            if result.expanded_terms_matched:
                print(f"   🔄 Expanded Terms Matched: {', '.join(result.expanded_terms_matched)}")
            
            print(f"   💡 Explanation: {result.explanation}")
            print(f"   🆔 ID: {result.person_id}")
            print()


def main():
    """Main interface"""
    if len(sys.argv) < 2:
        print("""
Sophisticated Real Yale Alumni Search Engine
(4,165 Real Alumni with Query Expansion & Co-occurrence Analysis)

Usage:
  python real_yale_sophisticated_search.py "search query"                    # Sophisticated search
  python real_yale_sophisticated_search.py "search query" --show-expansion   # Show expansion details

Examples:
  python real_yale_sophisticated_search.py "software engineer machine learning"
  python real_yale_sophisticated_search.py "investment banking Goldman" --show-expansion
  python real_yale_sophisticated_search.py "data scientist startup" --show-expansion
  python real_yale_sophisticated_search.py "product manager Meta Facebook" --show-expansion
        """)
        return
    
    search_engine = RealYaleSophisticatedSearch()
    show_expansion = '--show-expansion' in sys.argv
    
    # Remove flags from query
    query_parts = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    query = ' '.join(query_parts)
    
    results = search_engine.sophisticated_search(query, limit=10, show_expansion=show_expansion)
    search_engine.print_sophisticated_results(results)

if __name__ == "__main__":
    main()