"""
Search Infrastructure Module
SQL generation, indexing, relevance scoring, and caching for the search engine
"""

import json
import hashlib
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
import redis
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from datetime import datetime, timedelta
import math

# Import the new normalized scoring
from normalized_scoring import NormalizedRelevanceScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Individual search result with relevance score"""
    person_id: str
    name: str
    current_title: str
    current_company: str
    normalized_company: str
    location: str
    graduation_year: Optional[int]
    relevance_score: float
    score_breakdown: Dict[str, float]
    match_highlights: List[str]
    

@dataclass
class SearchResponse:
    """Complete search response with metadata"""
    results: List[SearchResult]
    total_count: int
    query_time_ms: float
    cache_hit: bool
    facets: Dict[str, List[Tuple[str, int]]]
    suggestions: List[str]


class SQLGenerator:
    """Generates optimized SQL queries from parsed search requests"""
    
    def __init__(self):
        self.base_query = """
            SELECT 
                a.person_id,
                a.name,
                a.current_title,
                a.current_company,
                a.normalized_company,
                a.normalized_title,
                a.location,
                a.normalized_location,
                a.graduation_year,
                a.degree,
                a.major,
                te.seniority,
                te.seniority_level,
                te.role_type,
                te.specialization,
                te.department,
                te.industry_focus,
                le.metro_area,
                le.city,
                le.state,
                le.country,
                array_agg(DISTINCT s.skill) as skills,
                array_agg(DISTINCT e.normalized_company) as experience_companies,
                array_agg(DISTINCT e.normalized_title) as experience_titles
            FROM alumni a
            LEFT JOIN title_entities te ON a.person_id = te.person_id
            LEFT JOIN location_entities le ON a.person_id = le.person_id
            LEFT JOIN skills s ON a.person_id = s.person_id
            LEFT JOIN experience e ON a.person_id = e.person_id
            GROUP BY a.person_id, a.name, a.current_title, a.current_company, 
                     a.normalized_company, a.normalized_title, a.location, 
                     a.normalized_location, a.graduation_year, a.degree, a.major,
                     te.seniority, te.seniority_level, te.role_type, te.specialization,
                     te.department, te.industry_focus, le.metro_area, le.city, 
                     le.state, le.country
        """
        
    def generate_search_query(self, search_config: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Generate SQL query from search configuration"""
        where_clauses = []
        parameters = []
        param_counter = 1
        
        filters = search_config.get('filters', {})
        terms = search_config.get('terms', [])
        boost_fields = search_config.get('boost_fields', {})
        
        # Build WHERE clauses
        
        # Company filter
        if 'companies' in filters:
            companies = filters['companies']
            placeholders = ', '.join([f'${param_counter + i}' for i in range(len(companies))])
            where_clauses.append(f"""
                (a.normalized_company = ANY(ARRAY[{placeholders}]) OR 
                 EXISTS(SELECT 1 FROM unnest(experience_companies) AS ec 
                        WHERE ec = ANY(ARRAY[{placeholders}])))
            """)
            parameters.extend(companies)
            parameters.extend(companies)
            param_counter += len(companies) * 2
        
        # Location filter
        if 'locations' in filters:
            locations = filters['locations']
            placeholders = ', '.join([f'${param_counter + i}' for i in range(len(locations))])
            where_clauses.append(f"""
                (le.metro_area = ANY(ARRAY[{placeholders}]) OR 
                 a.normalized_location = ANY(ARRAY[{placeholders}]) OR
                 le.city = ANY(ARRAY[{placeholders}]))
            """)
            parameters.extend(locations)
            parameters.extend(locations)
            parameters.extend(locations)
            param_counter += len(locations) * 3
        
        # Seniority filter
        if 'seniority_levels' in filters:
            seniority_levels = filters['seniority_levels']
            placeholders = ', '.join([f'${param_counter + i}' for i in range(len(seniority_levels))])
            where_clauses.append(f"te.seniority = ANY(ARRAY[{placeholders}])")
            parameters.extend(seniority_levels)
            param_counter += len(seniority_levels)
        
        # Experience filter
        if 'min_experience_years' in filters:
            # This would require calculating years from experience data
            # For now, we'll use a simplified approach
            min_years = filters['min_experience_years']
            where_clauses.append(f"te.seniority_level >= ${param_counter}")
            parameters.append(min_years)
            param_counter += 1
        
        # Text search on terms
        if terms:
            # Build full-text search conditions
            text_conditions = []
            
            # Search in current title
            text_conditions.append(f"""
                to_tsvector('english', COALESCE(a.current_title, '')) @@ 
                plainto_tsquery('english', ${param_counter})
            """)
            
            # Search in normalized title
            text_conditions.append(f"""
                to_tsvector('english', COALESCE(a.normalized_title, '')) @@ 
                plainto_tsquery('english', ${param_counter})
            """)
            
            # Search in company name
            text_conditions.append(f"""
                to_tsvector('english', COALESCE(a.normalized_company, '')) @@ 
                plainto_tsquery('english', ${param_counter})
            """)
            
            # Search in skills
            text_conditions.append(f"""
                EXISTS(SELECT 1 FROM skills s2 
                       WHERE s2.person_id = a.person_id AND 
                       to_tsvector('english', s2.skill) @@ plainto_tsquery('english', ${param_counter}))
            """)
            
            # Search in experience titles
            text_conditions.append(f"""
                EXISTS(SELECT 1 FROM experience e2 
                       WHERE e2.person_id = a.person_id AND 
                       to_tsvector('english', COALESCE(e2.normalized_title, '')) @@ 
                       plainto_tsquery('english', ${param_counter}))
            """)
            
            search_text = ' '.join(terms)
            where_clauses.append(f"({' OR '.join(text_conditions)})")
            parameters.append(search_text)
            param_counter += 1
        
        # Combine WHERE clauses
        where_sql = ''
        if where_clauses:
            where_sql = 'WHERE ' + ' AND '.join(where_clauses)
        
        # Build final query
        query = f"{self.base_query} {where_sql}"
        
        return query, parameters
    
    def generate_count_query(self, search_config: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Generate count query for total results"""
        query, parameters = self.generate_search_query(search_config)
        
        # Replace SELECT clause with COUNT
        count_query = query.replace(
            query.split('FROM')[0],
            'SELECT COUNT(DISTINCT a.person_id)'
        )
        
        return count_query, parameters
    
    def generate_facet_queries(self, search_config: Dict[str, Any]) -> Dict[str, Tuple[str, List[Any]]]:
        """Generate facet queries for search refinement"""
        base_query, base_params = self.generate_search_query(search_config)
        
        facet_queries = {}
        
        # Company facets
        facet_queries['companies'] = (
            f"""
            SELECT a.normalized_company, COUNT(DISTINCT a.person_id) as count
            FROM ({base_query}) sub
            JOIN alumni a ON sub.person_id = a.person_id
            WHERE a.normalized_company IS NOT NULL
            GROUP BY a.normalized_company
            ORDER BY count DESC
            LIMIT 20
            """,
            base_params
        )
        
        # Location facets
        facet_queries['locations'] = (
            f"""
            SELECT le.metro_area, COUNT(DISTINCT a.person_id) as count
            FROM ({base_query}) sub
            JOIN alumni a ON sub.person_id = a.person_id
            LEFT JOIN location_entities le ON a.person_id = le.person_id
            WHERE le.metro_area IS NOT NULL
            GROUP BY le.metro_area
            ORDER BY count DESC
            LIMIT 20
            """,
            base_params
        )
        
        # Seniority facets
        facet_queries['seniority'] = (
            f"""
            SELECT te.seniority, COUNT(DISTINCT a.person_id) as count
            FROM ({base_query}) sub
            JOIN alumni a ON sub.person_id = a.person_id
            LEFT JOIN title_entities te ON a.person_id = te.person_id
            WHERE te.seniority IS NOT NULL
            GROUP BY te.seniority
            ORDER BY count DESC
            LIMIT 10
            """,
            base_params
        )
        
        return facet_queries


class RelevanceScorer:
    """Calculates relevance scores for search results"""
    
    def __init__(self):
        self.score_weights = {
            'exact_title_match': 10.0,
            'exact_company_match': 8.0,
            'normalized_title_match': 6.0,
            'normalized_company_match': 5.0,
            'expanded_term_match': 3.0,
            'skill_match': 2.0,
            'experience_match': 2.0,
            'seniority_match': 1.5,
            'location_match': 1.0,
            'recency_bonus': 2.0,
            'education_match': 1.0
        }
    
    def calculate_relevance(self, result_row: Dict[str, Any], 
                          search_config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Calculate relevance score for a single result"""
        scores = {}
        total_score = 0.0
        
        terms = search_config.get('terms', [])
        filters = search_config.get('filters', {})
        boost_fields = search_config.get('boost_fields', {})
        
        # Exact title match
        if result_row.get('current_title'):
            title_lower = result_row['current_title'].lower()
            for term in terms:
                if term.lower() in title_lower:
                    scores['exact_title_match'] = self.score_weights['exact_title_match']
                    break
        
        # Exact company match
        if result_row.get('current_company'):
            company_lower = result_row['current_company'].lower()
            for term in terms:
                if term.lower() in company_lower:
                    scores['exact_company_match'] = self.score_weights['exact_company_match']
                    break
        
        # Normalized matches
        if result_row.get('normalized_title'):
            norm_title_lower = result_row['normalized_title'].lower()
            for term in terms:
                if term.lower() in norm_title_lower:
                    scores['normalized_title_match'] = self.score_weights['normalized_title_match']
                    break
        
        if result_row.get('normalized_company'):
            norm_company_lower = result_row['normalized_company'].lower()
            for term in terms:
                if term.lower() in norm_company_lower:
                    scores['normalized_company_match'] = self.score_weights['normalized_company_match']
                    break
        
        # Skill matches
        if result_row.get('skills'):
            skills = result_row['skills']
            if skills and skills[0] is not None:
                skill_matches = 0
                for skill in skills:
                    if skill:
                        skill_lower = skill.lower()
                        for term in terms:
                            if term.lower() in skill_lower:
                                skill_matches += 1
                                break
                
                if skill_matches > 0:
                    scores['skill_match'] = self.score_weights['skill_match'] * skill_matches
        
        # Experience matches
        if result_row.get('experience_titles'):
            exp_titles = result_row['experience_titles']
            if exp_titles and exp_titles[0] is not None:
                exp_matches = 0
                for title in exp_titles:
                    if title:
                        title_lower = title.lower()
                        for term in terms:
                            if term.lower() in title_lower:
                                exp_matches += 1
                                break
                
                if exp_matches > 0:
                    scores['experience_match'] = self.score_weights['experience_match'] * exp_matches
        
        # Seniority matching
        if result_row.get('seniority') and 'seniority_levels' in filters:
            if result_row['seniority'] in filters['seniority_levels']:
                scores['seniority_match'] = self.score_weights['seniority_match']
        
        # Location matching
        if result_row.get('metro_area') and 'locations' in filters:
            if result_row['metro_area'] in filters['locations']:
                scores['location_match'] = self.score_weights['location_match']
        
        # Recency bonus (recent graduates or recent position changes)
        if result_row.get('graduation_year'):
            current_year = datetime.now().year
            years_since_graduation = current_year - result_row['graduation_year']
            if years_since_graduation <= 10:  # Recent graduates
                scores['recency_bonus'] = self.score_weights['recency_bonus'] * (10 - years_since_graduation) / 10
        
        # Apply boost fields
        for boost_field, multiplier in boost_fields.items():
            if boost_field in scores:
                scores[boost_field] *= multiplier
        
        # Calculate total score
        total_score = sum(scores.values())
        
        # Apply TF-IDF style normalization
        # Penalize very common terms, boost rare matches
        doc_length = len(str(result_row.get('current_title', '') + 
                            result_row.get('current_company', '') + 
                            result_row.get('normalized_title', '')))
        
        if doc_length > 0:
            total_score = total_score / math.log(1 + doc_length)
        
        return total_score, scores
    
    def generate_highlights(self, result_row: Dict[str, Any], 
                          search_config: Dict[str, Any]) -> List[str]:
        """Generate highlighted match snippets"""
        highlights = []
        terms = search_config.get('terms', [])
        
        # Highlight current title
        if result_row.get('current_title'):
            title = result_row['current_title']
            for term in terms:
                if term.lower() in title.lower():
                    highlighted = title.replace(term, f"<mark>{term}</mark>")
                    highlights.append(f"Title: {highlighted}")
                    break
        
        # Highlight company
        if result_row.get('current_company'):
            company = result_row['current_company']
            for term in terms:
                if term.lower() in company.lower():
                    highlighted = company.replace(term, f"<mark>{term}</mark>")
                    highlights.append(f"Company: {highlighted}")
                    break
        
        # Highlight skills
        if result_row.get('skills'):
            skills = result_row['skills']
            if skills and skills[0] is not None:
                matched_skills = []
                for skill in skills:
                    if skill:
                        for term in terms:
                            if term.lower() in skill.lower():
                                matched_skills.append(skill)
                                break
                
                if matched_skills:
                    highlights.append(f"Skills: {', '.join(matched_skills[:3])}")
        
        return highlights


class SearchCache:
    """Redis-based caching for search results"""
    
    def __init__(self, redis_config: Dict[str, Any]):
        self.redis_client = redis.Redis(**redis_config)
        self.cache_ttl = 3600  # 1 hour
        self.cache_prefix = "search:"
    
    def get_cache_key(self, search_config: Dict[str, Any]) -> str:
        """Generate cache key from search configuration"""
        # Create deterministic hash of search config
        config_str = json.dumps(search_config, sort_keys=True)
        hash_key = hashlib.md5(config_str.encode()).hexdigest()
        return f"{self.cache_prefix}{hash_key}"
    
    def get_cached_results(self, search_config: Dict[str, Any]) -> Optional[SearchResponse]:
        """Get cached search results"""
        cache_key = self.get_cache_key(search_config)
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                # Reconstruct SearchResponse object
                results = [SearchResult(**r) for r in data['results']]
                return SearchResponse(
                    results=results,
                    total_count=data['total_count'],
                    query_time_ms=data['query_time_ms'],
                    cache_hit=True,
                    facets=data['facets'],
                    suggestions=data['suggestions']
                )
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        
        return None
    
    def cache_results(self, search_config: Dict[str, Any], 
                     response: SearchResponse):
        """Cache search results"""
        cache_key = self.get_cache_key(search_config)
        
        try:
            # Convert to serializable format
            cache_data = {
                'results': [r.__dict__ for r in response.results],
                'total_count': response.total_count,
                'query_time_ms': response.query_time_ms,
                'cache_hit': False,
                'facets': response.facets,
                'suggestions': response.suggestions
            }
            
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(cache_data)
            )
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
    
    def invalidate_cache(self, pattern: str = None):
        """Invalidate cache entries"""
        if pattern:
            keys = self.redis_client.keys(f"{self.cache_prefix}{pattern}*")
        else:
            keys = self.redis_client.keys(f"{self.cache_prefix}*")
        
        if keys:
            self.redis_client.delete(*keys)


class SearchEngine:
    """Main search engine orchestrating all components"""
    
    def __init__(self, db_config: Dict[str, str], redis_config: Dict[str, Any]):
        self.db_config = db_config
        self.sql_generator = SQLGenerator()
        self.relevance_scorer = RelevanceScorer()  # Keep old scorer for backward compatibility
        self.normalized_scorer = NormalizedRelevanceScorer()  # New normalized scorer
        self.cache = SearchCache(redis_config)
        self.use_normalized_scoring = True  # Flag to switch between scorers
        
    def search(self, search_config: Dict[str, Any], 
               limit: int = 20, offset: int = 0) -> SearchResponse:
        """Execute search with caching and relevance scoring"""
        start_time = time.time()
        
        # Check cache first
        cached_response = self.cache.get_cached_results(search_config)
        if cached_response:
            logger.info(f"Cache hit for search query")
            return cached_response
        
        # Connect to database
        conn = psycopg2.connect(**self.db_config)
        
        try:
            # Generate and execute search query
            query, params = self.sql_generator.generate_search_query(search_config)
            
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                raw_results = cur.fetchall()
            
            # Calculate relevance scores
            scored_results = []
            for row in raw_results:
                if self.use_normalized_scoring:
                    # Use new normalized scoring system
                    score_result = self.normalized_scorer.calculate_normalized_score(
                        dict(row), 
                        search_config,
                        semantic_similarity=row.get('semantic_score', 0.0),
                        cooccurrence_matches=search_config.get('expanded_terms', {})
                    )
                    relevance_score = score_result['final_score']
                    score_breakdown = score_result['components']
                    score_breakdown.update(score_result.get('modifiers', {}))
                else:
                    # Use legacy scoring for backward compatibility
                    relevance_score, score_breakdown = self.relevance_scorer.calculate_relevance(
                        dict(row), search_config
                    )
                
                highlights = self.relevance_scorer.generate_highlights(
                    dict(row), search_config
                )
                
                result = SearchResult(
                    person_id=row['person_id'],
                    name=row['name'],
                    current_title=row['current_title'] or '',
                    current_company=row['current_company'] or '',
                    normalized_company=row['normalized_company'] or '',
                    location=row['location'] or '',
                    graduation_year=row['graduation_year'],
                    relevance_score=relevance_score,
                    score_breakdown=score_breakdown,
                    match_highlights=highlights
                )
                
                scored_results.append(result)
            
            # Sort by relevance score
            scored_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Apply pagination
            total_count = len(scored_results)
            paginated_results = scored_results[offset:offset + limit]
            
            # Get facets
            facets = self._get_facets(search_config, conn)
            
            # Generate suggestions (could be enhanced with ML)
            suggestions = self._generate_suggestions(search_config, conn)
            
            # Calculate query time
            query_time_ms = (time.time() - start_time) * 1000
            
            # Create response
            response = SearchResponse(
                results=paginated_results,
                total_count=total_count,
                query_time_ms=query_time_ms,
                cache_hit=False,
                facets=facets,
                suggestions=suggestions
            )
            
            # Cache the results
            self.cache.cache_results(search_config, response)
            
            return response
            
        finally:
            conn.close()
    
    def _get_facets(self, search_config: Dict[str, Any], 
                   conn) -> Dict[str, List[Tuple[str, int]]]:
        """Get facet counts for search refinement"""
        facets = {}
        
        facet_queries = self.sql_generator.generate_facet_queries(search_config)
        
        with conn.cursor() as cur:
            for facet_name, (query, params) in facet_queries.items():
                try:
                    cur.execute(query, params)
                    results = cur.fetchall()
                    facets[facet_name] = [(row[0], row[1]) for row in results]
                except Exception as e:
                    logger.warning(f"Facet query error for {facet_name}: {e}")
                    facets[facet_name] = []
        
        return facets
    
    def _generate_suggestions(self, search_config: Dict[str, Any], 
                            conn) -> List[str]:
        """Generate search suggestions"""
        suggestions = []
        
        # Simple suggestions based on common patterns
        terms = search_config.get('terms', [])
        
        if terms:
            # Suggest related companies
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT normalized_company, COUNT(*) as count
                    FROM alumni
                    WHERE normalized_company IS NOT NULL
                    GROUP BY normalized_company
                    ORDER BY count DESC
                    LIMIT 5
                """)
                
                for company, count in cur.fetchall():
                    suggestions.append(f"Add company: {company}")
            
            # Suggest related roles
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT role_type, COUNT(*) as count
                    FROM title_entities
                    WHERE role_type IS NOT NULL
                    GROUP BY role_type
                    ORDER BY count DESC
                    LIMIT 5
                """)
                
                for role, count in cur.fetchall():
                    suggestions.append(f"Add role: {role}")
        
        return suggestions[:5]  # Limit suggestions
    
    def get_search_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get search analytics"""
        # This would track search patterns, popular queries, etc.
        # For now, return basic stats
        
        conn = psycopg2.connect(**self.db_config)
        
        try:
            with conn.cursor() as cur:
                # Total profiles
                cur.execute("SELECT COUNT(*) FROM alumni")
                total_profiles = cur.fetchone()[0]
                
                # Most common companies
                cur.execute("""
                    SELECT normalized_company, COUNT(*) as count
                    FROM alumni
                    WHERE normalized_company IS NOT NULL
                    GROUP BY normalized_company
                    ORDER BY count DESC
                    LIMIT 10
                """)
                top_companies = cur.fetchall()
                
                # Most common roles
                cur.execute("""
                    SELECT role_type, COUNT(*) as count
                    FROM title_entities
                    WHERE role_type IS NOT NULL
                    GROUP BY role_type
                    ORDER BY count DESC
                    LIMIT 10
                """)
                top_roles = cur.fetchall()
                
                return {
                    'total_profiles': total_profiles,
                    'top_companies': top_companies,
                    'top_roles': top_roles,
                    'cache_stats': self._get_cache_stats()
                }
        finally:
            conn.close()
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        try:
            info = self.cache.redis_client.info('stats')
            return {
                'cache_hits': info.get('keyspace_hits', 0),
                'cache_misses': info.get('keyspace_misses', 0),
                'total_keys': len(self.cache.redis_client.keys(f"{self.cache.cache_prefix}*"))
            }
        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    # Database and Redis configuration
    db_config = {
        'host': 'localhost',
        'database': 'yale_alumni',
        'user': 'postgres',
        'password': 'password',
        'port': 5432
    }
    
    redis_config = {
        'host': 'localhost',
        'port': 6379,
        'db': 0,
        'decode_responses': True
    }
    
    # Initialize search engine
    engine = SearchEngine(db_config, redis_config)
    
    # Test search
    search_config = {
        'terms': ['software engineer', 'google', 'machine learning'],
        'filters': {
            'locations': ['San Francisco Bay Area', 'New York'],
            'companies': ['Google', 'Facebook', 'Apple']
        },
        'boost_fields': {
            'exact_title_match': 2.0,
            'recency': 1.5
        }
    }
    
    results = engine.search(search_config, limit=10)
    
    print(f"Found {results.total_count} results in {results.query_time_ms:.2f}ms")
    print(f"Cache hit: {results.cache_hit}")
    
    for i, result in enumerate(results.results[:5]):
        print(f"\n{i+1}. {result.name}")
        print(f"   {result.current_title} at {result.current_company}")
        print(f"   Score: {result.relevance_score:.3f}")
        print(f"   Highlights: {', '.join(result.match_highlights)}")
    
    # Show facets
    print("\nFacets:")
    for facet_name, facet_values in results.facets.items():
        print(f"  {facet_name}: {facet_values[:3]}")
    
    # Show suggestions
    print(f"\nSuggestions: {results.suggestions}")