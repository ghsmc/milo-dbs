"""
Graph-based Query Expansion Module
Implements Dijkstra's algorithm for multi-hop query expansion
Addresses feedback #6: Use Dijkstra's with -log(weight) transformation
"""

import math
import json
import logging
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import heapq
import networkx as nx
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


@dataclass
class ExpansionNode:
    """Node in the expansion graph"""
    term: str
    distance: float
    path: List[str]
    confidence: float


@dataclass
class ExpansionResult:
    """Result of query expansion"""
    original_terms: List[str]
    expanded_terms: Dict[str, float]  # term -> confidence
    expansion_paths: Dict[str, List[str]]  # term -> path from original
    total_expansions: int
    max_hops_used: int


class CooccurrenceGraph:
    """
    Graph representation of term co-occurrences
    Implements Dijkstra's algorithm for finding optimal expansion paths
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.term_frequencies = {}
        self.edge_count = 0
        self.node_count = 0
    
    def build_from_cooccurrence_data(self, cooccurrence_results: List[Dict[str, Any]]):
        """Build graph from co-occurrence analysis results"""
        logger.info("Building co-occurrence graph...")
        
        # Add edges with transformed weights
        for result in cooccurrence_results:
            term1 = result['term1']
            term2 = result['term2']
            confidence = result['confidence']
            lift = result.get('lift', 1.0)
            
            # Only add high-quality connections
            if confidence > 0.3 and lift > 2.0:
                # Transform weight: higher confidence = lower distance
                # Using -log transformation as suggested in feedback
                weight = max(confidence, 0.01)  # Avoid log(0)
                distance = -math.log(weight)
                
                self.graph.add_edge(
                    term1, 
                    term2, 
                    weight=weight,
                    distance=distance,
                    confidence=confidence,
                    lift=lift
                )
                self.edge_count += 1
            
            # Track term frequencies
            self.term_frequencies[term1] = result.get('term1_count', 0)
            self.term_frequencies[term2] = result.get('term2_count', 0)
        
        self.node_count = self.graph.number_of_nodes()
        logger.info(f"Graph built with {self.node_count} nodes and {self.edge_count} edges")
    
    def load_from_json(self, filepath: str):
        """Load graph from saved JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Rebuild graph
        for term1, expansions in data.items():
            for term2, confidence in expansions:
                if confidence > 0:
                    weight = confidence
                    distance = -math.log(max(weight, 0.01))
                    self.graph.add_edge(
                        term1,
                        term2,
                        weight=weight,
                        distance=distance,
                        confidence=confidence
                    )
        
        self.node_count = self.graph.number_of_nodes()
        self.edge_count = self.graph.number_of_edges()


class GraphQueryExpander:
    """
    Query expansion using graph traversal with Dijkstra's algorithm
    Addresses feedback #6 about proper graph-based expansion
    """
    
    def __init__(self, cooccurrence_graph: CooccurrenceGraph):
        self.graph = cooccurrence_graph.graph
        self.term_frequencies = cooccurrence_graph.term_frequencies
        
        # Expansion parameters
        self.max_hops = 2
        self.max_expansions_per_term = 10
        self.min_confidence_threshold = 0.3
        self.decay_factor = 0.8  # Confidence decay per hop
        
        # TF-IDF weight to prevent common terms
        self.total_docs = max(self.term_frequencies.values()) if self.term_frequencies else 1
    
    def expand_query(self, query_terms: List[str], 
                    max_hops: int = None,
                    semantic_context: str = None) -> ExpansionResult:
        """
        Expand query using Dijkstra's algorithm
        
        Args:
            query_terms: List of initial query terms
            max_hops: Maximum expansion hops (default: self.max_hops)
            semantic_context: Optional context for semantic filtering
            
        Returns:
            ExpansionResult with expanded terms and paths
        """
        if max_hops is None:
            max_hops = self.max_hops
        
        # Normalize query terms
        normalized_terms = self._normalize_query_terms(query_terms)
        
        # Run Dijkstra's from each query term
        all_expansions = {}
        all_paths = {}
        max_hops_used = 0
        
        for term in normalized_terms:
            if term not in self.graph:
                logger.debug(f"Term '{term}' not in graph, skipping")
                continue
            
            # Find expansions using Dijkstra's
            expansions, paths = self._dijkstra_expansion(term, max_hops)
            
            # Merge with existing expansions
            for expanded_term, score in expansions.items():
                if expanded_term != term:  # Don't include original term
                    if expanded_term in all_expansions:
                        # Keep maximum score if term reached from multiple sources
                        all_expansions[expanded_term] = max(all_expansions[expanded_term], score)
                    else:
                        all_expansions[expanded_term] = score
                        all_paths[expanded_term] = paths[expanded_term]
                    
                    # Track maximum hops used
                    hop_count = len(paths[expanded_term]) - 1
                    max_hops_used = max(max_hops_used, hop_count)
        
        # Apply TF-IDF weighting to reduce common terms
        weighted_expansions = self._apply_tfidf_weighting(all_expansions)
        
        # Sort and limit expansions
        sorted_expansions = sorted(
            weighted_expansions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.max_expansions_per_term * len(normalized_terms)]
        
        return ExpansionResult(
            original_terms=normalized_terms,
            expanded_terms=dict(sorted_expansions),
            expansion_paths=all_paths,
            total_expansions=len(sorted_expansions),
            max_hops_used=max_hops_used
        )
    
    def _normalize_query_terms(self, query_terms: List[str]) -> List[str]:
        """Normalize query terms to match graph format"""
        normalized = []
        
        for term in query_terms:
            term_lower = term.lower().strip()
            
            # Try different prefixes if not found directly
            if term_lower in self.graph:
                normalized.append(term_lower)
            else:
                # Try with common prefixes
                prefixes = ['title:', 'company:', 'skill:', 'seniority:', 'role:']
                found = False
                
                for prefix in prefixes:
                    prefixed_term = f"{prefix}{term_lower}"
                    if prefixed_term in self.graph:
                        normalized.append(prefixed_term)
                        found = True
                        break
                
                if not found:
                    # Add as-is, might not expand
                    normalized.append(term_lower)
        
        return normalized
    
    def _dijkstra_expansion(self, start_term: str, max_hops: int) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """
        Run Dijkstra's algorithm from a single term
        Returns: (expansions, paths)
        """
        # Initialize distances and paths
        distances = {node: float('inf') for node in self.graph.nodes()}
        distances[start_term] = 0
        paths = {start_term: [start_term]}
        visited = set()
        
        # Priority queue: (distance, node, hop_count)
        pq = [(0, start_term, 0)]
        
        expansions = {}
        
        while pq:
            current_dist, current_node, hop_count = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # Stop if we've exceeded max hops
            if hop_count > max_hops:
                continue
            
            # Add to expansions (convert distance back to confidence)
            if current_node != start_term:
                # confidence = e^(-distance) * decay^hop_count
                confidence = math.exp(-current_dist) * (self.decay_factor ** hop_count)
                
                if confidence >= self.min_confidence_threshold:
                    expansions[current_node] = confidence
            
            # Explore neighbors
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    edge_data = self.graph[current_node][neighbor]
                    edge_distance = edge_data['distance']
                    
                    new_distance = current_dist + edge_distance
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        paths[neighbor] = paths[current_node] + [neighbor]
                        heapq.heappush(pq, (new_distance, neighbor, hop_count + 1))
        
        return expansions, paths
    
    def _apply_tfidf_weighting(self, expansions: Dict[str, float]) -> Dict[str, float]:
        """Apply TF-IDF weighting to reduce common terms"""
        weighted = {}
        
        for term, score in expansions.items():
            # Get term frequency
            tf = self.term_frequencies.get(term, 1)
            
            # Calculate IDF (inverse document frequency)
            # Using smoothed IDF to avoid division by zero
            idf = math.log(self.total_docs / (1 + tf))
            
            # Weight the score
            weighted_score = score * idf
            
            weighted[term] = weighted_score
        
        return weighted
    
    def get_expansion_explanation(self, expansion_result: ExpansionResult) -> str:
        """Generate human-readable explanation of expansions"""
        explanation = []
        
        explanation.append(f"Query Expansion Results:")
        explanation.append(f"Original terms: {', '.join(expansion_result.original_terms)}")
        explanation.append(f"Found {expansion_result.total_expansions} expansions using up to {expansion_result.max_hops_used} hops")
        
        explanation.append("\nTop Expansions:")
        for term, confidence in list(expansion_result.expanded_terms.items())[:10]:
            path = expansion_result.expansion_paths.get(term, [])
            path_str = " → ".join(path) if path else "direct"
            explanation.append(f"  {term}: {confidence:.3f} (path: {path_str})")
        
        return "\n".join(explanation)


class EnhancedQueryExpansionService:
    """
    Enhanced query expansion service using graph-based approach
    Integrates with the existing search infrastructure
    """
    
    def __init__(self, db_config: Dict[str, str], model_dir: str):
        self.db_config = db_config
        self.model_dir = model_dir
        self.cooccurrence_graph = None
        self.query_expander = None
        
        # Load or build graph
        self._initialize_graph()
    
    def _initialize_graph(self):
        """Initialize the co-occurrence graph"""
        # Try to load from saved file first
        expansion_file = f"{self.model_dir}/expansion_lookup.json"
        cooccurrence_file = f"{self.model_dir}/cooccurrence_results.json"
        
        self.cooccurrence_graph = CooccurrenceGraph()
        
        try:
            if os.path.exists(expansion_file):
                logger.info(f"Loading co-occurrence graph from {expansion_file}")
                self.cooccurrence_graph.load_from_json(expansion_file)
            elif os.path.exists(cooccurrence_file):
                logger.info(f"Building graph from {cooccurrence_file}")
                with open(cooccurrence_file, 'r') as f:
                    cooccurrence_data = json.load(f)
                self.cooccurrence_graph.build_from_cooccurrence_data(cooccurrence_data)
            else:
                logger.warning("No co-occurrence data found, building from database")
                self._build_graph_from_database()
            
            self.query_expander = GraphQueryExpander(self.cooccurrence_graph)
            
        except Exception as e:
            logger.error(f"Failed to initialize graph: {e}")
            # Create empty graph as fallback
            self.cooccurrence_graph = CooccurrenceGraph()
            self.query_expander = GraphQueryExpander(self.cooccurrence_graph)
    
    def _build_graph_from_database(self):
        """Build graph from database co-occurrence data"""
        conn = psycopg2.connect(**self.db_config)
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get co-occurrence data
                cur.execute("""
                    SELECT term1, term2, cooccurrence_count, 
                           term1_count, term2_count, pmi, lift, confidence
                    FROM cooccurrence_results
                    WHERE lift > 2.0 AND confidence > 0.3
                    ORDER BY lift DESC
                    LIMIT 10000
                """)
                
                results = cur.fetchall()
                
                if results:
                    self.cooccurrence_graph.build_from_cooccurrence_data(results)
                    logger.info(f"Built graph from {len(results)} co-occurrence pairs")
                else:
                    logger.warning("No co-occurrence data in database")
                    
        finally:
            conn.close()
    
    def expand_query(self, query: str, max_hops: int = 2) -> Dict[str, Any]:
        """
        Expand a search query using graph-based approach
        
        Returns dict with:
            - original_query: str
            - query_terms: List[str]
            - expansions: Dict[str, float]
            - sql_condition: str
            - explanation: str
        """
        # Parse query into terms
        query_terms = self._parse_query(query)
        
        # Expand using graph
        expansion_result = self.query_expander.expand_query(query_terms, max_hops)
        
        # Build SQL condition
        sql_condition = self._build_sql_condition(query_terms, expansion_result.expanded_terms)
        
        # Build response
        return {
            'original_query': query,
            'query_terms': query_terms,
            'expansions': expansion_result.expanded_terms,
            'expansion_paths': expansion_result.expansion_paths,
            'sql_condition': sql_condition,
            'explanation': self.query_expander.get_expansion_explanation(expansion_result)
        }
    
    def _parse_query(self, query: str) -> List[str]:
        """Parse query into terms"""
        # Simple tokenization - in production would use NLP
        terms = []
        
        # Split on common delimiters
        tokens = query.lower().replace(',', ' ').replace(';', ' ').split()
        
        # Filter out stop words
        stop_words = {'and', 'or', 'the', 'a', 'an', 'in', 'at', 'to', 'for', 'with'}
        
        for token in tokens:
            if len(token) > 2 and token not in stop_words:
                terms.append(token)
        
        return terms
    
    def _build_sql_condition(self, original_terms: List[str], 
                           expanded_terms: Dict[str, float]) -> str:
        """Build SQL condition from original and expanded terms"""
        all_terms = set(original_terms)
        
        # Add high-confidence expansions
        for term, confidence in expanded_terms.items():
            if confidence > 0.5:
                # Extract clean term without prefix
                clean_term = term.split(':', 1)[-1] if ':' in term else term
                all_terms.add(clean_term)
        
        # Build SQL condition
        conditions = []
        for term in all_terms:
            # Escape SQL special characters
            escaped_term = term.replace("'", "''")
            conditions.append(f"search_vector @@ plainto_tsquery('english', '{escaped_term}')")
        
        return " OR ".join(conditions) if conditions else "TRUE"
    
    def save_expansion_graph(self, filepath: str):
        """Save the expansion graph to file"""
        if self.cooccurrence_graph and self.cooccurrence_graph.graph:
            # Convert to adjacency list format
            adj_list = {}
            for node in self.cooccurrence_graph.graph.nodes():
                neighbors = []
                for neighbor in self.cooccurrence_graph.graph.neighbors(node):
                    edge_data = self.cooccurrence_graph.graph[node][neighbor]
                    neighbors.append((neighbor, edge_data['confidence']))
                adj_list[node] = neighbors
            
            with open(filepath, 'w') as f:
                json.dump(adj_list, f, indent=2)
            
            logger.info(f"Saved expansion graph to {filepath}")


# Integration with existing system
import os

class IntegratedQueryExpansion:
    """
    Integration layer for the new graph-based query expansion
    Compatible with existing QueryExpansionService interface
    """
    
    def __init__(self, db_config: Dict[str, str], model_dir: str):
        self.enhanced_service = EnhancedQueryExpansionService(db_config, model_dir)
    
    def expand_and_search(self, query: str) -> Dict[str, Any]:
        """
        Expand query and prepare for search
        Maintains compatibility with existing interface
        """
        # Get graph-based expansion
        expansion_result = self.enhanced_service.expand_query(query)
        
        # Format for existing search infrastructure
        search_config = {
            'terms': expansion_result['query_terms'],
            'expanded_terms': expansion_result['expansions'],
            'filters': {},
            'boost_fields': {},
            'explanation': expansion_result['explanation']
        }
        
        return search_config


# Example usage and testing
if __name__ == "__main__":
    # Example co-occurrence data
    sample_cooccurrence = [
        {
            'term1': 'title:analyst',
            'term2': 'company:goldman',
            'confidence': 0.8,
            'lift': 5.2,
            'term1_count': 100,
            'term2_count': 50
        },
        {
            'term1': 'title:analyst',
            'term2': 'seniority:entry',
            'confidence': 0.7,
            'lift': 3.5,
            'term1_count': 100,
            'term2_count': 200
        },
        {
            'term1': 'company:goldman',
            'term2': 'title:investment',
            'confidence': 0.9,
            'lift': 6.0,
            'term1_count': 50,
            'term2_count': 80
        },
        {
            'term1': 'title:investment',
            'term2': 'skill:finance',
            'confidence': 0.85,
            'lift': 4.5,
            'term1_count': 80,
            'term2_count': 120
        }
    ]
    
    # Build graph
    graph = CooccurrenceGraph()
    graph.build_from_cooccurrence_data(sample_cooccurrence)
    
    # Create expander
    expander = GraphQueryExpander(graph)
    
    # Test expansion
    test_query = ['analyst']
    result = expander.expand_query(test_query, max_hops=2)
    
    print("Graph-based Query Expansion Test:")
    print(expander.get_expansion_explanation(result))