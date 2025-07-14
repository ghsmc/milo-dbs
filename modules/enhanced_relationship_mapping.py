"""
Enhanced Relationship Mapping Module
Implements recency and duration weighting for colleague relationships
Addresses feedback #5: Add recency scoring and penalize short durations
"""

import math
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import networkx as nx
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch

logger = logging.getLogger(__name__)


@dataclass
class TimeOverlap:
    """Represents a time overlap between two people at a company"""
    person1_id: str
    person2_id: str
    company: str
    overlap_start: datetime
    overlap_end: datetime
    duration_months: float
    years_ago: float


@dataclass
class RelationshipScore:
    """Scored relationship between two people"""
    person1_id: str
    person2_id: str
    company: str
    base_score: float
    recency_factor: float
    duration_factor: float
    final_score: float
    overlap_details: TimeOverlap
    score_breakdown: Dict[str, float]


@dataclass
class CareerTransition:
    """Career transition pattern"""
    from_company: str
    to_company: str
    from_title: str
    to_title: str
    transition_count: int
    avg_months_between: float
    common_paths: List[Dict[str, Any]]


class EnhancedRelationshipCalculator:
    """
    Enhanced relationship calculator with recency and duration weighting
    Addresses feedback #5 about weighting by recency and penalizing short overlaps
    """
    
    def __init__(self, current_date: datetime = None):
        self.current_date = current_date or datetime.now()
        
        # Scoring parameters
        self.min_overlap_months = 1.0  # Minimum overlap to consider
        self.short_duration_threshold = 3.0  # Months - below this gets penalty
        self.medium_duration_threshold = 12.0  # Months - good overlap
        self.recency_half_life = 5.0  # Years - half-life for recency decay
        
        # Penalties and bonuses
        self.short_duration_penalty = 0.3  # 70% penalty for < 3 months
        self.medium_duration_penalty = 0.6  # 40% penalty for 3-6 months
        self.long_duration_bonus = 1.2  # 20% bonus for > 12 months
        
        # Company tier weights (could be expanded with real data)
        self.company_tiers = {
            'tier1': 1.2,  # Top tier companies (FAANG, Goldman Sachs, etc.)
            'tier2': 1.0,  # Good companies
            'tier3': 0.8   # Other companies
        }
    
    def calculate_relationship_strength(self, overlap: TimeOverlap) -> RelationshipScore:
        """
        Calculate relationship strength with enhanced scoring
        
        Args:
            overlap: TimeOverlap object with details about the shared experience
            
        Returns:
            RelationshipScore with detailed breakdown
        """
        # Base score from overlap duration (normalized to 0-1)
        base_score = min(overlap.duration_months / 12.0, 1.0)
        
        # Duration factor (penalties for short overlaps)
        duration_factor = self._calculate_duration_factor(overlap.duration_months)
        
        # Recency factor (exponential decay)
        recency_factor = self._calculate_recency_factor(overlap.years_ago)
        
        # Company tier factor
        company_factor = self._get_company_tier_weight(overlap.company)
        
        # Calculate final score
        final_score = base_score * duration_factor * recency_factor * company_factor
        
        # Ensure score stays in reasonable bounds
        final_score = min(max(final_score, 0.0), 1.0)
        
        score_breakdown = {
            'base_score': base_score,
            'duration_factor': duration_factor,
            'recency_factor': recency_factor,
            'company_factor': company_factor,
            'raw_final': base_score * duration_factor * recency_factor * company_factor,
            'final_capped': final_score
        }
        
        return RelationshipScore(
            person1_id=overlap.person1_id,
            person2_id=overlap.person2_id,
            company=overlap.company,
            base_score=base_score,
            recency_factor=recency_factor,
            duration_factor=duration_factor,
            final_score=final_score,
            overlap_details=overlap,
            score_breakdown=score_breakdown
        )
    
    def _calculate_duration_factor(self, duration_months: float) -> float:
        """
        Calculate duration factor with penalties for short overlaps
        As per feedback #5: penalize really short durations
        """
        if duration_months < self.short_duration_threshold:
            # Severe penalty for very short overlaps (< 3 months)
            return self.short_duration_penalty
        elif duration_months < 6.0:
            # Medium penalty for short overlaps (3-6 months)
            return self.medium_duration_penalty
        elif duration_months >= self.medium_duration_threshold:
            # Bonus for long overlaps (> 12 months)
            return self.long_duration_bonus
        else:
            # Normal scoring for 6-12 months
            return 1.0
    
    def _calculate_recency_factor(self, years_ago: float) -> float:
        """
        Calculate recency factor using exponential decay
        As per feedback #5: weight by recency
        """
        # Exponential decay: score = e^(-years_ago / half_life)
        return math.exp(-years_ago / self.recency_half_life)
    
    def _get_company_tier_weight(self, company: str) -> float:
        """Get company tier weight (simplified - could use ML classifier)"""
        company_lower = company.lower()
        
        # Tier 1 companies (top tech, finance, consulting)
        tier1_companies = {
            'google', 'apple', 'microsoft', 'amazon', 'meta', 'facebook',
            'goldman sachs', 'morgan stanley', 'jpmorgan', 'blackstone',
            'mckinsey', 'bain', 'bcg', 'deloitte'
        }
        
        # Check if company matches tier 1
        for tier1 in tier1_companies:
            if tier1 in company_lower:
                return self.company_tiers['tier1']
        
        # Default to tier 2
        return self.company_tiers['tier2']


class RelationshipGraphBuilder:
    """Builds weighted relationship graph from experience data"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.calculator = EnhancedRelationshipCalculator()
        self.relationship_graph = nx.Graph()
        self.relationships = []
    
    def build_relationship_graph(self):
        """Build relationship graph from experience data"""
        logger.info("Building enhanced relationship graph...")
        
        # Get overlapping experiences
        overlaps = self._find_experience_overlaps()
        
        # Calculate relationship scores
        for overlap in overlaps:
            relationship = self.calculator.calculate_relationship_strength(overlap)
            self.relationships.append(relationship)
            
            # Add to NetworkX graph
            self.relationship_graph.add_edge(
                relationship.person1_id,
                relationship.person2_id,
                weight=relationship.final_score,
                company=relationship.company,
                duration_months=overlap.duration_months,
                years_ago=overlap.years_ago,
                relationship_data=relationship
            )
        
        logger.info(f"Built relationship graph with {len(self.relationships)} relationships")
        
        # Save to database
        self._save_relationships_to_db()
        
        return self.relationship_graph
    
    def _find_experience_overlaps(self) -> List[TimeOverlap]:
        """Find overlapping experiences between people"""
        conn = psycopg2.connect(**self.db_config)
        overlaps = []
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get all experiences with date ranges
                cur.execute("""
                    SELECT 
                        e1.person_id as person1_id,
                        e2.person_id as person2_id,
                        e1.company,
                        e1.start_date as p1_start,
                        e1.end_date as p1_end,
                        e2.start_date as p2_start,
                        e2.end_date as p2_end
                    FROM experience e1
                    JOIN experience e2 ON e1.company = e2.company
                    WHERE e1.person_id < e2.person_id  -- Avoid duplicates
                      AND e1.start_date IS NOT NULL
                      AND e2.start_date IS NOT NULL
                      AND (
                          -- Check for overlap
                          (e1.start_date <= COALESCE(e2.end_date, CURRENT_DATE) AND
                           COALESCE(e1.end_date, CURRENT_DATE) >= e2.start_date)
                      )
                """)
                
                rows = cur.fetchall()
                
                for row in rows:
                    # Calculate exact overlap period
                    overlap_start = max(row['p1_start'], row['p2_start'])
                    overlap_end = min(
                        row['p1_end'] or self.calculator.current_date.date(),
                        row['p2_end'] or self.calculator.current_date.date()
                    )
                    
                    # Skip if negative overlap (shouldn't happen with query)
                    if overlap_end <= overlap_start:
                        continue
                    
                    # Convert to datetime for calculations
                    if isinstance(overlap_start, str):
                        overlap_start = datetime.strptime(overlap_start, '%Y-%m-%d')
                    elif hasattr(overlap_start, 'date'):
                        overlap_start = datetime.combine(overlap_start, datetime.min.time())
                    
                    if isinstance(overlap_end, str):
                        overlap_end = datetime.strptime(overlap_end, '%Y-%m-%d')
                    elif hasattr(overlap_end, 'date'):
                        overlap_end = datetime.combine(overlap_end, datetime.min.time())
                    
                    # Calculate duration and recency
                    duration_months = (overlap_end - overlap_start).days / 30.44  # Average days per month
                    years_ago = (self.calculator.current_date - overlap_end).days / 365.25
                    
                    # Skip very short overlaps
                    if duration_months < self.calculator.min_overlap_months:
                        continue
                    
                    overlap = TimeOverlap(
                        person1_id=row['person1_id'],
                        person2_id=row['person2_id'],
                        company=row['company'],
                        overlap_start=overlap_start,
                        overlap_end=overlap_end,
                        duration_months=duration_months,
                        years_ago=max(0, years_ago)  # Can't be negative
                    )
                    
                    overlaps.append(overlap)
        
        finally:
            conn.close()
        
        logger.info(f"Found {len(overlaps)} experience overlaps")
        return overlaps
    
    def _save_relationships_to_db(self):
        """Save relationship scores to database"""
        if not self.relationships:
            return
        
        conn = psycopg2.connect(**self.db_config)
        
        try:
            with conn.cursor() as cur:
                # Create enhanced relationships table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS enhanced_relationships (
                        person1_id VARCHAR(50),
                        person2_id VARCHAR(50),
                        company TEXT,
                        base_score FLOAT,
                        recency_factor FLOAT,
                        duration_factor FLOAT,
                        final_score FLOAT,
                        duration_months FLOAT,
                        years_ago FLOAT,
                        overlap_start DATE,
                        overlap_end DATE,
                        score_breakdown JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (person1_id, person2_id, company)
                    )
                """)
                
                # Prepare data for insertion
                data_to_insert = []
                for rel in self.relationships:
                    data_to_insert.append((
                        rel.person1_id,
                        rel.person2_id,
                        rel.company,
                        rel.base_score,
                        rel.recency_factor,
                        rel.duration_factor,
                        rel.final_score,
                        rel.overlap_details.duration_months,
                        rel.overlap_details.years_ago,
                        rel.overlap_details.overlap_start.date(),
                        rel.overlap_details.overlap_end.date(),
                        json.dumps(rel.score_breakdown)
                    ))
                
                # Batch insert
                execute_batch(cur, """
                    INSERT INTO enhanced_relationships 
                    (person1_id, person2_id, company, base_score, recency_factor, 
                     duration_factor, final_score, duration_months, years_ago,
                     overlap_start, overlap_end, score_breakdown)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (person1_id, person2_id, company) DO UPDATE SET
                        base_score = EXCLUDED.base_score,
                        recency_factor = EXCLUDED.recency_factor,
                        duration_factor = EXCLUDED.duration_factor,
                        final_score = EXCLUDED.final_score,
                        duration_months = EXCLUDED.duration_months,
                        years_ago = EXCLUDED.years_ago,
                        overlap_start = EXCLUDED.overlap_start,
                        overlap_end = EXCLUDED.overlap_end,
                        score_breakdown = EXCLUDED.score_breakdown,
                        created_at = CURRENT_TIMESTAMP
                """, data_to_insert)
                
                conn.commit()
                logger.info(f"Saved {len(self.relationships)} enhanced relationships to database")
                
        finally:
            conn.close()
    
    def get_person_connections(self, person_id: str, min_score: float = 0.3) -> List[RelationshipScore]:
        """Get all connections for a person above minimum score"""
        connections = []
        
        for rel in self.relationships:
            if (rel.person1_id == person_id or rel.person2_id == person_id) and rel.final_score >= min_score:
                connections.append(rel)
        
        # Sort by score descending
        connections.sort(key=lambda x: x.final_score, reverse=True)
        return connections
    
    def recommend_connections(self, person_id: str, target_person_id: str) -> List[Dict[str, Any]]:
        """Recommend mutual connections between two people"""
        # Find mutual connections using NetworkX
        if person_id not in self.relationship_graph or target_person_id not in self.relationship_graph:
            return []
        
        # Get common neighbors
        person1_neighbors = set(self.relationship_graph.neighbors(person_id))
        person2_neighbors = set(self.relationship_graph.neighbors(target_person_id))
        mutual_connections = person1_neighbors & person2_neighbors
        
        recommendations = []
        for mutual_id in mutual_connections:
            # Get relationship strengths
            rel1_data = self.relationship_graph[person_id][mutual_id]
            rel2_data = self.relationship_graph[target_person_id][mutual_id]
            
            recommendation = {
                'connector_id': mutual_id,
                'person1_connection_score': rel1_data['weight'],
                'person2_connection_score': rel2_data['weight'],
                'combined_score': (rel1_data['weight'] + rel2_data['weight']) / 2,
                'person1_company': rel1_data['company'],
                'person2_company': rel2_data['company'],
                'years_ago_avg': (rel1_data['years_ago'] + rel2_data['years_ago']) / 2
            }
            recommendations.append(recommendation)
        
        # Sort by combined score
        recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
        return recommendations


class CareerTransitionAnalyzer:
    """Analyzes career transition patterns using Markov chains"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.transition_probabilities = {}
    
    def analyze_career_transitions(self) -> Dict[str, Any]:
        """Analyze career transition patterns"""
        transitions = self._get_career_transitions()
        
        # Build transition matrix
        for transition in transitions:
            from_key = f"{transition['from_company']}|{transition['from_title']}"
            to_key = f"{transition['to_company']}|{transition['to_title']}"
            self.transition_matrix[from_key][to_key] += 1
        
        # Calculate probabilities
        self._calculate_transition_probabilities()
        
        # Find common patterns
        common_patterns = self._find_common_patterns()
        
        return {
            'total_transitions': len(transitions),
            'unique_from_states': len(self.transition_matrix),
            'common_patterns': common_patterns,
            'transition_probabilities': dict(self.transition_probabilities)
        }
    
    def _get_career_transitions(self) -> List[Dict[str, Any]]:
        """Get career transitions from database"""
        conn = psycopg2.connect(**self.db_config)
        transitions = []
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get consecutive experiences for each person
                cur.execute("""
                    WITH ordered_experience AS (
                        SELECT 
                            person_id,
                            title,
                            company,
                            start_date,
                            end_date,
                            ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY start_date) as position
                        FROM experience
                        WHERE start_date IS NOT NULL
                    )
                    SELECT 
                        e1.person_id,
                        e1.title as from_title,
                        e1.company as from_company,
                        e1.end_date as from_end,
                        e2.title as to_title,
                        e2.company as to_company,
                        e2.start_date as to_start
                    FROM ordered_experience e1
                    JOIN ordered_experience e2 ON e1.person_id = e2.person_id 
                                               AND e2.position = e1.position + 1
                    WHERE e1.end_date IS NOT NULL
                      AND e2.start_date IS NOT NULL
                """)
                
                transitions = cur.fetchall()
                
        finally:
            conn.close()
        
        return transitions
    
    def _calculate_transition_probabilities(self):
        """Calculate transition probabilities from counts"""
        for from_state, to_states in self.transition_matrix.items():
            total_transitions = sum(to_states.values())
            
            if total_transitions > 0:
                probabilities = {}
                for to_state, count in to_states.items():
                    probabilities[to_state] = count / total_transitions
                
                self.transition_probabilities[from_state] = probabilities
    
    def _find_common_patterns(self, min_count: int = 5) -> List[CareerTransition]:
        """Find common career transition patterns"""
        patterns = []
        
        for from_state, to_states in self.transition_matrix.items():
            from_company, from_title = from_state.split('|', 1)
            
            for to_state, count in to_states.items():
                if count >= min_count:
                    to_company, to_title = to_state.split('|', 1)
                    
                    pattern = CareerTransition(
                        from_company=from_company,
                        to_company=to_company,
                        from_title=from_title,
                        to_title=to_title,
                        transition_count=count,
                        avg_months_between=0,  # Would calculate from actual data
                        common_paths=[]
                    )
                    patterns.append(pattern)
        
        # Sort by transition count
        patterns.sort(key=lambda x: x.transition_count, reverse=True)
        return patterns[:20]  # Top 20 patterns


# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced relationship calculator
    test_overlap = TimeOverlap(
        person1_id="person_001",
        person2_id="person_002",
        company="Google",
        overlap_start=datetime(2019, 1, 1),
        overlap_end=datetime(2019, 4, 1),  # 3 months - should get penalty
        duration_months=3.0,
        years_ago=2.0
    )
    
    calculator = EnhancedRelationshipCalculator()
    relationship = calculator.calculate_relationship_strength(test_overlap)
    
    print("Enhanced Relationship Scoring Test:")
    print(f"Duration: {test_overlap.duration_months} months")
    print(f"Years ago: {test_overlap.years_ago}")
    print(f"Final score: {relationship.final_score:.3f}")
    print(f"Score breakdown: {relationship.score_breakdown}")
    
    # Test with longer overlap
    long_overlap = TimeOverlap(
        person1_id="person_003",
        person2_id="person_004",
        company="Goldman Sachs",
        overlap_start=datetime(2018, 1, 1),
        overlap_end=datetime(2020, 1, 1),  # 24 months - should get bonus
        duration_months=24.0,
        years_ago=1.0
    )
    
    long_relationship = calculator.calculate_relationship_strength(long_overlap)
    
    print(f"\nLong overlap test:")
    print(f"Duration: {long_overlap.duration_months} months")
    print(f"Final score: {long_relationship.final_score:.3f}")
    print(f"Score breakdown: {long_relationship.score_breakdown}")