#!/usr/bin/env python3
"""
Finance-Focused Yale Alumni Search
Specifically designed to handle finance queries like "Wall St. vc banker"
"""

import json
import psycopg2
import sys
import re
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class FinanceSearchResult:
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
    final_score: float
    explanation: str

class FinanceFocusedSearch:
    """
    Search engine specifically optimized for finance queries
    """
    
    def __init__(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        self.db_config = self.config['database']
        
        # Finance-specific mappings
        self.finance_terms = {
            'wall st': 'wall street',
            'wall st.': 'wall street',
            'vc': 'venture capital',
            'banker': 'investment banking',
            'investment': 'investment banking',
            'banking': 'investment banking',
            'goldman': 'goldman sachs',
            'morgan': 'morgan stanley',
            'jp': 'jpmorgan',
            'jpmorgan': 'jp morgan',
        }
        
        # Finance companies
        self.finance_companies = [
            'goldman sachs', 'morgan stanley', 'jp morgan', 'jpmorgan', 'jpmorgan chase',
            'citadel', 'blackstone', 'apollo', 'kkr', 'bridgewater', 'two sigma',
            'renaissance technologies', 'carlyle', 'bain capital', 'silver lake',
            'credit suisse', 'deutsche bank', 'barclays', 'ubs', 'wells fargo',
            'bank of america', 'citigroup', 'centerview', 'evercore', 'lazard'
        ]
        
        # Finance roles
        self.finance_roles = [
            'investment banker', 'investment banking', 'analyst', 'associate',
            'vice president', 'managing director', 'portfolio manager', 'trader',
            'research analyst', 'equity research', 'private equity', 'venture capital',
            'hedge fund', 'asset management', 'wealth management', 'financial advisor'
        ]
    
    def is_finance_query(self, query: str) -> bool:
        """Determine if this is a finance-related query"""
        query_lower = query.lower()
        finance_indicators = [
            'wall st', 'wall street', 'vc', 'banker', 'investment', 'banking',
            'finance', 'financial', 'goldman', 'morgan', 'private equity',
            'venture capital', 'hedge fund', 'asset management'
        ]
        return any(indicator in query_lower for indicator in finance_indicators)
    
    def expand_finance_query(self, query: str) -> List[str]:
        """Expand query terms for finance domain"""
        query_lower = query.lower()
        expanded_terms = []
        
        # Handle common finance phrases
        if 'wall st' in query_lower or 'wall street' in query_lower:
            expanded_terms.extend(['wall street', 'finance', 'financial', 'investment banking'])
        
        if 'vc' in query_lower:
            expanded_terms.extend(['venture capital', 'private equity', 'investment'])
        
        if 'banker' in query_lower:
            expanded_terms.extend(['investment banking', 'analyst', 'associate', 'finance'])
        
        # Add finance companies and roles
        for term in query_lower.split():
            if term in self.finance_terms:
                expanded_terms.append(self.finance_terms[term])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms
    
    def search(self, query: str, limit: int = 10) -> List[FinanceSearchResult]:
        """Search with finance domain focus"""
        print(f"🏦 Finance-Focused Yale Alumni Search")
        print("=" * 60)
        print(f"Query: '{query}'")
        
        is_finance = self.is_finance_query(query)
        print(f"Finance query detected: {is_finance}")
        
        if is_finance:
            expanded_terms = self.expand_finance_query(query)
            print(f"Expanded terms: {expanded_terms}")
            return self._execute_finance_search(query, expanded_terms, limit)
        else:
            # For non-finance queries, fall back to basic search
            return self._execute_basic_search(query, limit)
    
    def _execute_finance_search(self, original_query: str, expanded_terms: List[str], limit: int) -> List[FinanceSearchResult]:
        """Execute finance-optimized search"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        # Build finance-specific SQL with strict filtering
        sql = """
        WITH finance_candidates AS (
            SELECT 
                person_id, graduation_year, name, email, city, state_territory,
                current_position, current_company, industry, function, major,
                skills, linkedin_url, experience_history, education_history, location,
                -- Finance-specific scoring
                (
                    -- Investment banking terms (highest weight)
                    (CASE 
                        WHEN LOWER(current_position) LIKE '%investment%' AND LOWER(current_position) LIKE '%bank%' THEN 20
                        WHEN LOWER(current_position) LIKE '%investment%' THEN 15
                        WHEN LOWER(current_position) LIKE '%banking%' THEN 15
                        ELSE 0
                    END) +
                    -- Finance companies (very high weight)
                    (CASE 
                        WHEN LOWER(current_company) LIKE '%goldman%' THEN 18
                        WHEN LOWER(current_company) LIKE '%morgan%' THEN 18
                        WHEN LOWER(current_company) LIKE '%jpmorgan%' THEN 18
                        WHEN LOWER(current_company) LIKE '%citadel%' THEN 16
                        WHEN LOWER(current_company) LIKE '%apollo%' THEN 16
                        WHEN LOWER(current_company) LIKE '%blackstone%' THEN 16
                        WHEN LOWER(current_company) LIKE '%centerview%' THEN 14
                        ELSE 0
                    END) +
                    -- Finance roles (high weight)
                    (CASE 
                        WHEN LOWER(current_position) LIKE '%analyst%' THEN 10
                        WHEN LOWER(current_position) LIKE '%associate%' THEN 10
                        WHEN LOWER(current_position) LIKE '%vice president%' THEN 12
                        WHEN LOWER(current_position) LIKE '%managing director%' THEN 15
                        ELSE 0
                    END) +
                    -- Finance industry/function (medium weight)
                    (CASE 
                        WHEN LOWER(industry) LIKE '%finance%' THEN 8
                        WHEN LOWER(function) LIKE '%finance%' THEN 8
                        WHEN LOWER(industry) LIKE '%investment%' THEN 6
                        ELSE 0
                    END) +
                    -- Skills match (lower weight)
                    (CASE 
                        WHEN LOWER(skills) LIKE '%financial%' THEN 4
                        WHEN LOWER(skills) LIKE '%modeling%' THEN 3
                        ELSE 0
                    END)
                ) as finance_score
            FROM alumni_real
            WHERE 
                -- Must match at least one finance criterion
                (LOWER(current_position) LIKE '%investment%' OR
                 LOWER(current_position) LIKE '%banking%' OR
                 LOWER(current_position) LIKE '%analyst%' OR
                 LOWER(current_position) LIKE '%associate%' OR
                 LOWER(current_position) LIKE '%vice president%' OR
                 LOWER(current_position) LIKE '%managing director%' OR
                 LOWER(current_company) LIKE '%goldman%' OR
                 LOWER(current_company) LIKE '%morgan%' OR
                 LOWER(current_company) LIKE '%jpmorgan%' OR
                 LOWER(current_company) LIKE '%citadel%' OR
                 LOWER(current_company) LIKE '%apollo%' OR
                 LOWER(current_company) LIKE '%blackstone%' OR
                 LOWER(current_company) LIKE '%centerview%' OR
                 LOWER(industry) LIKE '%finance%' OR
                 LOWER(function) LIKE '%finance%')
        )
        SELECT * FROM finance_candidates
        WHERE finance_score > 0
        ORDER BY finance_score DESC, graduation_year DESC, name
        LIMIT %s
        """
        
        cur.execute(sql, [limit])
        
        results = []
        for row in cur.fetchall():
            result = FinanceSearchResult(
                person_id=row[0],
                graduation_year=row[1] or 0,
                name=row[2] or '',
                email=row[3] or '',
                city=row[4] or '',
                state_territory=row[5] or '',
                current_position=row[6] or '',
                current_company=row[7] or '',
                industry=row[8] or '',
                function=row[9] or '',
                major=row[10] or '',
                skills=row[11] or '',
                linkedin_url=row[12] or '',
                final_score=row[16] / 50.0,  # Normalize score
                explanation=self._create_explanation(row, expanded_terms)
            )
            results.append(result)
        
        conn.close()
        return results
    
    def _execute_basic_search(self, query: str, limit: int) -> List[FinanceSearchResult]:
        """Basic search for non-finance queries"""
        # Implementation for non-finance queries
        return []
    
    def _create_explanation(self, row, expanded_terms: List[str]) -> str:
        """Create explanation for the match"""
        explanations = []
        
        position = (row[6] or '').lower()
        company = (row[7] or '').lower()
        industry = (row[8] or '').lower()
        
        if 'investment' in position and 'bank' in position:
            explanations.append("Investment Banking role")
        elif 'investment' in position:
            explanations.append("Investment role")
        elif any(term in position for term in ['analyst', 'associate', 'vice president', 'managing director']):
            explanations.append("Finance title")
        
        if any(term in company for term in ['goldman', 'morgan', 'jpmorgan', 'citadel', 'apollo', 'blackstone']):
            explanations.append("Top finance firm")
        
        if 'finance' in industry:
            explanations.append("Finance industry")
        
        return "; ".join(explanations) if explanations else "Finance-related match"
    
    def print_results(self, results: List[FinanceSearchResult]):
        """Print search results"""
        if not results:
            print("No finance professionals found.")
            return
        
        print(f"\n🏆 Finance-Focused Results ({len(results)} found):")
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
            
            print(f"   🎯 Finance Score: {result.final_score:.3f}")
            print(f"   💡 Explanation: {result.explanation}")
            print(f"   🆔 ID: {result.person_id}")
            print()

def main():
    """Main interface"""
    if len(sys.argv) < 2:
        print("""
Finance-Focused Yale Alumni Search Engine

Usage:
  python finance_focused_search.py "search query"

Examples:
  python finance_focused_search.py "Wall St. vc banker"
  python finance_focused_search.py "investment banking Goldman"
  python finance_focused_search.py "private equity analyst"
        """)
        return
    
    search_engine = FinanceFocusedSearch()
    query = ' '.join(sys.argv[1:])
    
    results = search_engine.search(query, limit=10)
    search_engine.print_results(results)

if __name__ == "__main__":
    main()