"""
Normalized Relevance Scoring System
Implements scoring with 0-1 normalization as per feedback #7
"""

import math
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ScoreComponent:
    """Definition of a score component"""
    weight: float
    description: str
    

class NormalizedRelevanceScorer:
    """
    Calculates relevance scores normalized to 0-1 range
    Addresses feedback #7: Score normalization and component clarification
    """
    
    def __init__(self):
        # Define score components with weights that sum to 1.0
        self.score_components = {
            'exact_match': ScoreComponent(
                weight=0.4,
                description='Exact term matches in title/company fields'
            ),
            'semantic_score': ScoreComponent(
                weight=0.3,
                description='Cosine similarity of profile embeddings'
            ),
            'cooccurrence_score': ScoreComponent(
                weight=0.3,
                description='Strength of term associations from co-occurrence analysis'
            )
        }
        
        # Sub-components for exact match scoring
        self.exact_match_weights = {
            'title': 0.4,
            'company': 0.3,
            'skills': 0.2,
            'experience': 0.1
        }
    
    def calculate_normalized_score(self, 
                                 result_row: Dict[str, Any],
                                 search_config: Dict[str, Any],
                                 semantic_similarity: float = 0.0,
                                 cooccurrence_matches: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Calculate normalized relevance score (0-1 range)
        
        Args:
            result_row: Database row with person data
            search_config: Search configuration with terms and filters
            semantic_similarity: Pre-calculated semantic similarity score (0-1)
            cooccurrence_matches: Dict of expanded terms with confidence scores
            
        Returns:
            Dict with final_score, component scores, and explanation
        """
        raw_scores = {}
        
        # 1. Calculate exact match score (0-1)
        raw_scores['exact_match'] = self._calculate_exact_match_score(
            result_row, search_config
        )
        
        # 2. Use provided semantic score (already 0-1)
        raw_scores['semantic_score'] = semantic_similarity
        
        # 3. Calculate co-occurrence score (0-1)
        raw_scores['cooccurrence_score'] = self._calculate_cooccurrence_score(
            result_row, cooccurrence_matches or {}
        )
        
        # Calculate weighted final score (automatically 0-1)
        final_score = sum(
            raw_scores[comp] * self.score_components[comp].weight
            for comp in self.score_components
        )
        
        # Apply additional modifiers
        modifiers = self._calculate_modifiers(result_row, search_config)
        
        # Apply modifiers while keeping score in 0-1 range
        modified_score = self._apply_modifiers(final_score, modifiers)
        
        return {
            'final_score': round(modified_score, 4),
            'components': {k: round(v, 4) for k, v in raw_scores.items()},
            'modifiers': modifiers,
            'explanation': {
                comp: {
                    'weight': self.score_components[comp].weight,
                    'description': self.score_components[comp].description,
                    'score': round(raw_scores[comp], 4)
                }
                for comp in self.score_components
            }
        }
    
    def _calculate_exact_match_score(self, 
                                   result_row: Dict[str, Any],
                                   search_config: Dict[str, Any]) -> float:
        """Calculate exact match score normalized to 0-1"""
        terms = search_config.get('terms', [])
        if not terms:
            return 0.0
        
        match_scores = {}
        
        # Title matches
        if result_row.get('current_title'):
            title_lower = result_row['current_title'].lower()
            title_matches = sum(1 for term in terms if term.lower() in title_lower)
            match_scores['title'] = min(title_matches / len(terms), 1.0)
        else:
            match_scores['title'] = 0.0
        
        # Company matches
        if result_row.get('current_company'):
            company_lower = result_row['current_company'].lower()
            company_matches = sum(1 for term in terms if term.lower() in company_lower)
            match_scores['company'] = min(company_matches / len(terms), 1.0)
        else:
            match_scores['company'] = 0.0
        
        # Skill matches
        if result_row.get('skills'):
            skills = result_row['skills']
            if skills and skills[0] is not None:
                skill_text = ' '.join(s.lower() for s in skills if s)
                skill_matches = sum(1 for term in terms if term.lower() in skill_text)
                match_scores['skills'] = min(skill_matches / len(terms), 1.0)
            else:
                match_scores['skills'] = 0.0
        else:
            match_scores['skills'] = 0.0
        
        # Experience matches
        if result_row.get('experience_titles'):
            exp_titles = result_row['experience_titles']
            if exp_titles and exp_titles[0] is not None:
                exp_text = ' '.join(t.lower() for t in exp_titles if t)
                exp_matches = sum(1 for term in terms if term.lower() in exp_text)
                match_scores['experience'] = min(exp_matches / len(terms), 1.0)
            else:
                match_scores['experience'] = 0.0
        else:
            match_scores['experience'] = 0.0
        
        # Calculate weighted exact match score
        exact_score = sum(
            match_scores[field] * self.exact_match_weights[field]
            for field in self.exact_match_weights
        )
        
        return exact_score
    
    def _calculate_cooccurrence_score(self,
                                    result_row: Dict[str, Any],
                                    cooccurrence_matches: Dict[str, float]) -> float:
        """Calculate co-occurrence score normalized to 0-1"""
        if not cooccurrence_matches:
            return 0.0
        
        score = 0.0
        fields_to_check = []
        
        # Collect all searchable text
        if result_row.get('current_title'):
            fields_to_check.append(result_row['current_title'].lower())
        if result_row.get('current_company'):
            fields_to_check.append(result_row['current_company'].lower())
        if result_row.get('normalized_title'):
            fields_to_check.append(result_row['normalized_title'].lower())
        if result_row.get('skills'):
            skills = result_row['skills']
            if skills and skills[0] is not None:
                fields_to_check.extend([s.lower() for s in skills if s])
        
        # Check for co-occurrence matches
        matched_terms = set()
        for expanded_term, confidence in cooccurrence_matches.items():
            term_lower = expanded_term.lower()
            for field_text in fields_to_check:
                if term_lower in field_text and expanded_term not in matched_terms:
                    score += confidence
                    matched_terms.add(expanded_term)
                    break
        
        # Normalize by number of expansions (max score = 1.0)
        if len(cooccurrence_matches) > 0:
            score = min(score / len(cooccurrence_matches), 1.0)
        
        return score
    
    def _calculate_modifiers(self, 
                           result_row: Dict[str, Any],
                           search_config: Dict[str, Any]) -> Dict[str, float]:
        """Calculate score modifiers"""
        modifiers = {}
        filters = search_config.get('filters', {})
        
        # Recency modifier (0 to 0.2 bonus)
        if result_row.get('graduation_year'):
            current_year = datetime.now().year
            years_since = current_year - result_row['graduation_year']
            if years_since <= 10:
                modifiers['recency'] = 0.2 * (10 - years_since) / 10
            else:
                modifiers['recency'] = 0.0
        
        # Location match modifier (0.1 bonus if matches filter)
        if result_row.get('metro_area') and 'locations' in filters:
            if result_row['metro_area'] in filters['locations']:
                modifiers['location_match'] = 0.1
        
        # Seniority match modifier (0.1 bonus if matches filter)
        if result_row.get('seniority') and 'seniority_levels' in filters:
            if result_row['seniority'] in filters['seniority_levels']:
                modifiers['seniority_match'] = 0.1
        
        # Education match modifier
        if 'education' in filters and result_row.get('education_school'):
            if any(edu in result_row['education_school'].lower() 
                   for edu in filters['education']):
                modifiers['education_match'] = 0.1
        
        return modifiers
    
    def _apply_modifiers(self, base_score: float, modifiers: Dict[str, float]) -> float:
        """
        Apply modifiers while keeping score in 0-1 range
        Uses a diminishing returns formula
        """
        if not modifiers:
            return base_score
        
        # Sum all modifiers
        modifier_sum = sum(modifiers.values())
        
        # Apply modifiers with diminishing returns
        # This ensures we stay in 0-1 range
        modified_score = base_score + (1 - base_score) * modifier_sum * 0.5
        
        # Ensure we stay in bounds
        return min(max(modified_score, 0.0), 1.0)
    
    def generate_score_explanation(self, score_result: Dict[str, Any]) -> str:
        """Generate human-readable explanation of the score"""
        explanation = []
        
        explanation.append(f"Final Score: {score_result['final_score']:.2%}")
        explanation.append("\nScore Components:")
        
        for comp_name, comp_info in score_result['explanation'].items():
            explanation.append(
                f"  - {comp_info['description']}: "
                f"{comp_info['score']:.2%} (weight: {comp_info['weight']:.0%})"
            )
        
        if score_result.get('modifiers'):
            explanation.append("\nScore Modifiers:")
            for mod_name, mod_value in score_result['modifiers'].items():
                if mod_value > 0:
                    explanation.append(f"  - {mod_name}: +{mod_value:.2%}")
        
        return '\n'.join(explanation)