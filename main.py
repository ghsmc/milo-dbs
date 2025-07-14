"""
Main Yale Alumni Search Engine
Integrates all modules for comprehensive alumni search and networking
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import argparse
import psycopg2

# Add modules to path
sys.path.append('modules')

from data_foundation import DataFoundation
from entity_extraction import EntityExtractionPipeline
from cooccurrence_analysis import CooccurrenceAnalyzer
from enhanced_cooccurrence import EnhancedCooccurrenceAnalyzer  # New enhanced version
from query_expansion import QueryExpansionService
from search_infrastructure import SearchEngine
from semantic_layer import SemanticSearchEngine

# Medium-priority improvements
from ai_entity_extraction import HybridEntityExtractor
from graph_query_expansion import IntegratedQueryExpansion
from enhanced_relationship_mapping import RelationshipGraphBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SearchConfiguration:
    """Configuration for search parameters"""
    enable_semantic_search: bool = True
    enable_query_expansion: bool = True
    enable_multi_aspect_embeddings: bool = True  # New: Multi-aspect embeddings
    enable_enhanced_cooccurrence: bool = True    # New: Enhanced temporal filtering
    use_normalized_scoring: bool = True          # New: Normalized scoring
    
    # Medium-priority features
    enable_ai_entity_extraction: bool = True     # New: AI-enhanced entity extraction
    enable_graph_query_expansion: bool = True    # New: Graph-based query expansion
    enable_enhanced_relationships: bool = True   # New: Enhanced relationship mapping
    ai_confidence_threshold: float = 0.7         # Confidence threshold for AI fallback
    max_results: int = 50
    semantic_weight: float = 0.3
    keyword_weight: float = 0.4
    cooccurrence_weight: float = 0.3
    cache_enabled: bool = True
    

class YaleAlumniSearchEngine:
    """Main search engine orchestrating all components"""
    
    def __init__(self, config_file: str = 'config.json'):
        """Initialize the search engine with configuration"""
        self.config = self._load_config(config_file)
        self.search_config = SearchConfiguration()
        
        # Initialize components
        self.data_foundation = None
        self.entity_extraction = None
        self.cooccurrence_analyzer = None
        self.query_expansion_service = None
        self.search_engine = None
        self.semantic_engine = None
        
        # Initialize all components
        self._initialize_components()
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        default_config = {
            'database': {
                'host': 'localhost',
                'database': 'yale_alumni',
                'user': 'postgres',
                'password': 'password',
                'port': 5432
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'decode_responses': True
            },
            'models': {
                'embedding_model': 'all-MiniLM-L6-v2',
                'model_dir': 'models'
            },
            'data': {
                'input_files': [
                    'OCS_YALE_PEOPLE_5K.xlsx',
                    '361_GPT_COMPANIES.csv'
                ],
                'batch_size': 1000
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge with default config
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
                logger.info("Using default configuration")
        else:
            logger.info(f"Config file {config_file} not found. Using default configuration")
            # Create default config file
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _initialize_components(self):
        """Initialize all search engine components"""
        logger.info("Initializing search engine components...")
        
        # Data foundation
        self.data_foundation = DataFoundation(self.config['database'])
        
        # Entity extraction (with AI enhancement if enabled)
        if self.search_config.enable_ai_entity_extraction:
            self.ai_entity_extractor = HybridEntityExtractor(
                self.config['database'],
                confidence_threshold=self.search_config.ai_confidence_threshold
            )
        
        self.entity_extraction = EntityExtractionPipeline(self.config['database'])
        
        # Co-occurrence analysis (use enhanced version if enabled)
        if self.search_config.enable_enhanced_cooccurrence:
            self.cooccurrence_analyzer = EnhancedCooccurrenceAnalyzer(self.config['database'])
        else:
            self.cooccurrence_analyzer = CooccurrenceAnalyzer(self.config['database'])
        
        # Query expansion service (use graph-based if enabled)
        model_dir = self.config['models']['model_dir']
        if self.search_config.enable_graph_query_expansion:
            self.query_expansion_service = IntegratedQueryExpansion(
                self.config['database'], 
                model_dir
            )
        else:
            self.query_expansion_service = QueryExpansionService(
                self.config['database'], 
                model_dir
            )
        
        # Search engine
        self.search_engine = SearchEngine(
            self.config['database'],
            self.config['redis']
        )
        
        # Semantic search engine (with multi-aspect support)
        self.semantic_engine = SemanticSearchEngine(
            self.config['database'],
            model_name=self.config['models']['embedding_model'],
            use_multi_aspect=self.search_config.enable_multi_aspect_embeddings
        )
        
        # Enhanced relationship mapping (if enabled)
        if self.search_config.enable_enhanced_relationships:
            self.relationship_builder = RelationshipGraphBuilder(self.config['database'])
        
        logger.info("All components initialized successfully")
    
    def setup_database(self):
        """Set up database tables and indexes"""
        logger.info("Setting up database...")
        
        # Create tables
        self.data_foundation.create_tables()
        
        logger.info("Database setup completed")
    
    def ingest_data(self, file_paths: List[str] = None):
        """Ingest and process data files"""
        if file_paths is None:
            file_paths = self.config['data']['input_files']
        
        logger.info(f"Ingesting data from {len(file_paths)} files...")
        
        # Process data files
        quality_report = self.data_foundation.process_data_files(file_paths)
        
        # Extract entities
        logger.info("Extracting entities...")
        freq_stats = self.entity_extraction.build_frequency_dictionaries()
        self.entity_extraction.extract_and_store_entities()
        
        # Build co-occurrence analysis
        logger.info("Building co-occurrence analysis...")
        self.cooccurrence_analyzer.build_term_document_matrix()
        results = self.cooccurrence_analyzer.calculate_pmi_scores()
        
        # Save analysis results
        model_dir = self.config['models']['model_dir']
        os.makedirs(model_dir, exist_ok=True)
        self.cooccurrence_analyzer.save_analysis_results(model_dir)
        
        logger.info("Data ingestion completed")
        return {
            'quality_report': quality_report,
            'frequency_stats': freq_stats,
            'cooccurrence_pairs': len(results)
        }
    
    def build_semantic_index(self):
        """Build semantic search index"""
        logger.info("Building semantic search index...")
        
        # Build embeddings
        self.semantic_engine.build_embeddings_database(
            batch_size=self.config['data']['batch_size']
        )
        
        # Save vector store
        model_dir = self.config['models']['model_dir']
        vector_store_path = os.path.join(model_dir, 'vector_store.index')
        self.semantic_engine.save_vector_store(vector_store_path)
        
        logger.info("Semantic index built successfully")
    
    def build_multi_aspect_index(self):
        """Build multi-aspect semantic embeddings"""
        if not self.search_config.enable_multi_aspect_embeddings:
            logger.warning("Multi-aspect embeddings disabled")
            return
        
        logger.info("Building multi-aspect embeddings...")
        
        # Build multi-aspect embeddings
        self.semantic_engine.build_multi_aspect_embeddings(
            batch_size=self.config['data']['batch_size']
        )
        
        logger.info("Multi-aspect embeddings built successfully")
    
    def build_enhanced_relationships(self):
        """Build enhanced relationship graph with recency/duration weighting"""
        if not self.search_config.enable_enhanced_relationships:
            logger.warning("Enhanced relationships disabled")
            return
        
        logger.info("Building enhanced relationship graph...")
        
        # Build relationship graph
        self.relationship_builder.build_relationship_graph()
        
        logger.info("Enhanced relationship graph built successfully")
    
    def run_ai_entity_extraction(self, batch_size: int = 100):
        """Run AI-enhanced entity extraction on profiles"""
        if not self.search_config.enable_ai_entity_extraction:
            logger.warning("AI entity extraction disabled")
            return
        
        logger.info("Running AI-enhanced entity extraction...")
        
        # Get profiles that need AI extraction (low confidence from rules)
        conn = psycopg2.connect(**self.config['database'])
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT person_id, current_title, current_company
                    FROM alumni 
                    WHERE current_title IS NOT NULL
                    LIMIT %s
                """, (batch_size * 10,))  # Get more than batch size for filtering
                
                profiles = cur.fetchall()
                
        finally:
            conn.close()
        
        # Process in batches
        results = self.ai_entity_extractor.process_batch(
            [{'person_id': p[0], 'current_title': p[1], 'current_company': p[2]} 
             for p in profiles],
            batch_size=batch_size
        )
        
        # Save results
        self.ai_entity_extractor.save_enhanced_extractions(results)
        
        # Get statistics
        stats = self.ai_entity_extractor.get_extraction_stats()
        logger.info(f"AI extraction completed: {stats}")
        
        return stats
    
    def search(self, query: str, filters: Dict[str, Any] = None, 
               limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """Main search function"""
        logger.info(f"Searching: {query}")
        
        # Expand query
        search_config = self.query_expansion_service.expand_and_search(query)
        
        # Apply additional filters
        if filters:
            search_config['filters'].update(filters)
        
        # Perform keyword search
        keyword_results = self.search_engine.search(
            search_config, 
            limit=limit * 2,  # Get more results for merging
            offset=offset
        )
        
        # Perform semantic search if enabled
        semantic_results = []
        if self.search_config.enable_semantic_search:
            try:
                semantic_results = self.semantic_engine.semantic_search(
                    query, 
                    top_k=limit
                )
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")
        
        # Merge results
        final_results = self._merge_search_results(
            keyword_results, 
            semantic_results,
            limit
        )
        
        # Format response
        response = {
            'query': query,
            'results': [self._format_search_result(r) for r in final_results.results],
            'total_count': final_results.total_count,
            'query_time_ms': final_results.query_time_ms,
            'expansion_info': search_config.get('explanation', {}),
            'facets': final_results.facets,
            'suggestions': final_results.suggestions,
            'semantic_enabled': self.search_config.enable_semantic_search
        }
        
        return response
    
    def _merge_search_results(self, keyword_results, semantic_results, limit):
        """Merge keyword and semantic search results"""
        # For now, prioritize keyword results with semantic boost
        # In production, this would be more sophisticated
        
        # Create person_id to semantic score mapping
        semantic_scores = {}
        for result in semantic_results:
            semantic_scores[result.person_id] = result.similarity_score
        
        # Boost keyword results with semantic scores
        for result in keyword_results.results:
            if result.person_id in semantic_scores:
                semantic_boost = semantic_scores[result.person_id] * self.search_config.semantic_weight
                result.relevance_score += semantic_boost
                result.score_breakdown['semantic_boost'] = semantic_boost
        
        # Re-sort and limit
        keyword_results.results.sort(key=lambda x: x.relevance_score, reverse=True)
        keyword_results.results = keyword_results.results[:limit]
        
        return keyword_results
    
    def _format_search_result(self, result) -> Dict[str, Any]:
        """Format search result for API response"""
        return {
            'person_id': result.person_id,
            'name': result.name,
            'current_title': result.current_title,
            'current_company': result.current_company,
            'location': result.location,
            'graduation_year': result.graduation_year,
            'relevance_score': round(result.relevance_score, 3),
            'score_breakdown': {k: round(v, 3) for k, v in result.score_breakdown.items()},
            'highlights': result.match_highlights
        }
    
    def get_similar_profiles(self, person_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find profiles similar to a given person"""
        try:
            results = self.semantic_engine.get_similar_profiles(person_id, limit)
            return [
                {
                    'person_id': r.person_id,
                    'profile_text': r.profile_text,
                    'similarity_score': round(r.similarity_score, 3)
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Failed to get similar profiles: {e}")
            return []
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get search analytics and statistics"""
        analytics = self.search_engine.get_search_analytics()
        
        # Add entity statistics
        entity_stats = self.entity_extraction.get_entity_statistics()
        analytics['entity_statistics'] = entity_stats
        
        # Add co-occurrence examples
        expansion_examples = self.cooccurrence_analyzer.get_expansion_examples()
        analytics['expansion_examples'] = expansion_examples
        
        return analytics
    
    def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        health = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'overall_status': 'healthy'
        }
        
        # Check database connection
        try:
            conn = self.data_foundation.connect_db()
            conn.close()
            health['components']['database'] = 'healthy'
        except Exception as e:
            health['components']['database'] = f'unhealthy: {e}'
            health['overall_status'] = 'unhealthy'
        
        # Check Redis connection
        try:
            self.search_engine.cache.redis_client.ping()
            health['components']['redis'] = 'healthy'
        except Exception as e:
            health['components']['redis'] = f'unhealthy: {e}'
            health['overall_status'] = 'degraded'
        
        # Check vector store
        try:
            if self.semantic_engine.vector_store and self.semantic_engine.vector_store.index:
                health['components']['vector_store'] = 'healthy'
            else:
                health['components']['vector_store'] = 'not_initialized'
        except Exception as e:
            health['components']['vector_store'] = f'unhealthy: {e}'
        
        return health


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Yale Alumni Search Engine')
    parser.add_argument('--setup', action='store_true', help='Setup database')
    parser.add_argument('--ingest', action='store_true', help='Ingest data')
    parser.add_argument('--build-semantic', action='store_true', help='Build semantic index')
    parser.add_argument('--build-multi-aspect', action='store_true', help='Build multi-aspect embeddings')
    parser.add_argument('--build-relationships', action='store_true', help='Build enhanced relationship graph')
    parser.add_argument('--run-ai-extraction', action='store_true', help='Run AI-enhanced entity extraction')
    parser.add_argument('--search', type=str, help='Search query')
    parser.add_argument('--multi-aspect-search', type=str, help='Multi-aspect semantic search')
    parser.add_argument('--similar', type=str, help='Find similar profiles to person_id')
    parser.add_argument('--analytics', action='store_true', help='Show analytics')
    parser.add_argument('--health', action='store_true', help='Health check')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')
    parser.add_argument('--limit', type=int, default=20, help='Result limit')
    
    args = parser.parse_args()
    
    # Initialize search engine
    engine = YaleAlumniSearchEngine(args.config)
    
    if args.setup:
        engine.setup_database()
        print("Database setup completed")
    
    elif args.ingest:
        stats = engine.ingest_data()
        print(f"Data ingestion completed:")
        print(json.dumps(stats, indent=2))
    
    elif args.build_semantic:
        engine.build_semantic_index()
        print("Semantic index built successfully")
    
    elif getattr(args, 'build_multi_aspect', False):
        engine.build_multi_aspect_index()
        print("Multi-aspect embeddings built successfully")
    
    elif getattr(args, 'build_relationships', False):
        engine.build_enhanced_relationships()
        print("Enhanced relationship graph built successfully")
    
    elif getattr(args, 'run_ai_extraction', False):
        stats = engine.run_ai_entity_extraction(batch_size=100)
        print("AI-enhanced entity extraction completed:")
        print(json.dumps(stats, indent=2))
    
    elif args.search:
        results = engine.search(args.search, limit=args.limit)
        print(f"Search Results for: {args.search}")
        print(f"Found {results['total_count']} results in {results['query_time_ms']:.2f}ms")
        
        for i, result in enumerate(results['results'][:10]):
            print(f"\n{i+1}. {result['name']}")
            print(f"   {result['current_title']} at {result['current_company']}")
            print(f"   Location: {result['location']}")
            print(f"   Score: {result['relevance_score']}")
            if result['highlights']:
                print(f"   Highlights: {', '.join(result['highlights'])}")
    
    elif getattr(args, 'multi_aspect_search', None):
        try:
            results = engine.semantic_engine.multi_aspect_search(
                getattr(args, 'multi_aspect_search'), 
                top_k=args.limit
            )
            print(f"Multi-Aspect Search Results for: {getattr(args, 'multi_aspect_search')}")
            
            for i, result in enumerate(results):
                print(f"\n{i+1}. {result.person_id}")
                print(f"   Score: {result.similarity_score:.3f}")
                print(f"   Summary: {result.profile_text[:150]}...")
        except Exception as e:
            print(f"Multi-aspect search failed: {e}")
            print("Make sure multi-aspect embeddings are built first with --build-multi-aspect")
    
    elif args.similar:
        results = engine.get_similar_profiles(args.similar, limit=args.limit)
        print(f"Similar profiles to {args.similar}:")
        
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result['person_id']}")
            print(f"   Score: {result['similarity_score']}")
            print(f"   {result['profile_text'][:100]}...")
    
    elif args.analytics:
        analytics = engine.get_analytics()
        print("Search Analytics:")
        print(json.dumps(analytics, indent=2))
    
    elif args.health:
        health = engine.health_check()
        print("System Health Check:")
        print(json.dumps(health, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()