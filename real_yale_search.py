#!/usr/bin/env python3
"""
Real Yale Alumni Search Engine
Search through 4,165 real Yale alumni with complete profiles
"""

import json
import psycopg2
import sys
import re
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass 
class RealYaleAlumni:
    person_id: int
    graduation_year: int
    name: str
    email: str
    country: str
    state_territory: str
    city: str
    graduate_school: str
    employer: str
    industry: str
    function: str
    major: str
    linkedin_url: str
    linkedin_profile_picture: str
    current_position: str
    current_company: str
    location: str
    education_history: str
    experience_history: str
    skills: str
    enrichment_date: str
    enrichment_status: str

class RealYaleSearch:
    """
    Search engine for real Yale alumni data
    """
    
    def __init__(self):
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        self.db_config = self.config['database']
    
    def search_alumni(self, query: str, limit: int = 10) -> List[RealYaleAlumni]:
        """
        Search real Yale alumni
        """
        print(f"🎓 Searching Real Yale Alumni Database")
        print(f"Query: '{query}'")
        print("=" * 60)
        
        # Tokenize query
        query_tokens = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
        print(f"Search terms: {query_tokens}")
        
        # Build comprehensive search
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        # Create search conditions for all relevant fields
        search_conditions = []
        params = []
        
        for token in query_tokens:
            if len(token) > 2:
                search_conditions.append("""
                    (LOWER(name) LIKE %s OR 
                     LOWER(email) LIKE %s OR
                     LOWER(current_position) LIKE %s OR 
                     LOWER(current_company) LIKE %s OR 
                     LOWER(employer) LIKE %s OR
                     LOWER(industry) LIKE %s OR
                     LOWER(function) LIKE %s OR
                     LOWER(major) LIKE %s OR
                     LOWER(skills) LIKE %s OR
                     LOWER(education_history) LIKE %s OR
                     LOWER(experience_history) LIKE %s OR
                     LOWER(city) LIKE %s OR
                     LOWER(state_territory) LIKE %s)
                """)
                like_pattern = f'%{token}%'
                params.extend([like_pattern] * 13)
        
        if not search_conditions:
            return []
        
        # Build comprehensive scoring SQL
        sql = f"""
        SELECT 
            person_id, graduation_year, name, email, country, state_territory, city,
            graduate_school, employer, industry, function, major, linkedin_url,
            linkedin_profile_picture, current_position, current_company, location,
            education_history, experience_history, skills, enrichment_date, enrichment_status,
            -- Scoring based on field matches
            (CASE WHEN LOWER(current_position) LIKE %s THEN 10 ELSE 0 END +
             CASE WHEN LOWER(current_company) LIKE %s THEN 8 ELSE 0 END +
             CASE WHEN LOWER(industry) LIKE %s THEN 6 ELSE 0 END +
             CASE WHEN LOWER(skills) LIKE %s THEN 5 ELSE 0 END +
             CASE WHEN LOWER(major) LIKE %s THEN 4 ELSE 0 END +
             CASE WHEN LOWER(name) LIKE %s THEN 3 ELSE 0 END +
             CASE WHEN LOWER(experience_history) LIKE %s THEN 2 ELSE 0 END +
             CASE WHEN LOWER(city) LIKE %s THEN 1 ELSE 0 END) as relevance_score
        FROM alumni_real
        WHERE {' AND '.join(search_conditions)}
        ORDER BY relevance_score DESC, graduation_year DESC, name
        LIMIT %s
        """
        
        # Add scoring parameters (using first token for relevance scoring)
        first_token = f'%{query_tokens[0]}%' if query_tokens else '%%'
        scoring_params = [first_token] * 8
        
        all_params = scoring_params + params + [limit]
        
        print(f"Executing search across {len(search_conditions)} search conditions...")
        cur.execute(sql, all_params)
        
        results = []
        for row in cur.fetchall():
            alumni = RealYaleAlumni(
                person_id=row[0],
                graduation_year=row[1] or 0,
                name=row[2] or '',
                email=row[3] or '',
                country=row[4] or '',
                state_territory=row[5] or '',
                city=row[6] or '',
                graduate_school=row[7] or '',
                employer=row[8] or '',
                industry=row[9] or '',
                function=row[10] or '',
                major=row[11] or '',
                linkedin_url=row[12] or '',
                linkedin_profile_picture=row[13] or '',
                current_position=row[14] or '',
                current_company=row[15] or '',
                location=row[16] or '',
                education_history=row[17] or '',
                experience_history=row[18] or '',
                skills=row[19] or '',
                enrichment_date=row[20] or '',
                enrichment_status=row[21] or ''
            )
            results.append(alumni)
        
        conn.close()
        print(f"Found {len(results)} matching alumni")
        return results
    
    def search_by_company(self, company: str, limit: int = 10) -> List[RealYaleAlumni]:
        """Search alumni by company"""
        return self.search_alumni(f"company:{company}", limit)
    
    def search_by_industry(self, industry: str, limit: int = 10) -> List[RealYaleAlumni]:
        """Search alumni by industry"""
        return self.search_alumni(f"industry:{industry}", limit)
    
    def search_by_graduation_year(self, year: int, limit: int = 10) -> List[RealYaleAlumni]:
        """Search alumni by graduation year"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        sql = """
        SELECT 
            person_id, graduation_year, name, email, country, state_territory, city,
            graduate_school, employer, industry, function, major, linkedin_url,
            linkedin_profile_picture, current_position, current_company, location,
            education_history, experience_history, skills, enrichment_date, enrichment_status
        FROM alumni_real
        WHERE graduation_year = %s
        ORDER BY name
        LIMIT %s
        """
        
        cur.execute(sql, [year, limit])
        
        results = []
        for row in cur.fetchall():
            alumni = RealYaleAlumni(
                person_id=row[0],
                graduation_year=row[1] or 0,
                name=row[2] or '',
                email=row[3] or '',
                country=row[4] or '',
                state_territory=row[5] or '',
                city=row[6] or '',
                graduate_school=row[7] or '',
                employer=row[8] or '',
                industry=row[9] or '',
                function=row[10] or '',
                major=row[11] or '',
                linkedin_url=row[12] or '',
                linkedin_profile_picture=row[13] or '',
                current_position=row[14] or '',
                current_company=row[15] or '',
                location=row[16] or '',
                education_history=row[17] or '',
                experience_history=row[18] or '',
                skills=row[19] or '',
                enrichment_date=row[20] or '',
                enrichment_status=row[21] or ''
            )
            results.append(alumni)
        
        conn.close()
        return results
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get database analytics"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        # Basic statistics
        cur.execute("SELECT COUNT(*) FROM alumni_real")
        total_alumni = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(DISTINCT current_company) FROM alumni_real WHERE current_company != ''")
        unique_companies = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(DISTINCT industry) FROM alumni_real WHERE industry != ''")
        unique_industries = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM alumni_real WHERE email != ''")
        alumni_with_email = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM alumni_real WHERE linkedin_url != ''")
        alumni_with_linkedin = cur.fetchone()[0]
        
        # Top companies
        cur.execute("""
            SELECT current_company, COUNT(*) as count
            FROM alumni_real 
            WHERE current_company != ''
            GROUP BY current_company 
            ORDER BY count DESC 
            LIMIT 10
        """)
        top_companies = cur.fetchall()
        
        # Top industries
        cur.execute("""
            SELECT industry, COUNT(*) as count
            FROM alumni_real 
            WHERE industry != ''
            GROUP BY industry 
            ORDER BY count DESC 
            LIMIT 10
        """)
        top_industries = cur.fetchall()
        
        # Graduation years
        cur.execute("""
            SELECT graduation_year, COUNT(*) as count
            FROM alumni_real 
            WHERE graduation_year IS NOT NULL
            GROUP BY graduation_year 
            ORDER BY graduation_year DESC 
            LIMIT 10
        """)
        graduation_years = cur.fetchall()
        
        conn.close()
        
        return {
            'total_alumni': total_alumni,
            'unique_companies': unique_companies,
            'unique_industries': unique_industries,
            'alumni_with_email': alumni_with_email,
            'alumni_with_linkedin': alumni_with_linkedin,
            'top_companies': top_companies,
            'top_industries': top_industries,
            'recent_graduation_years': graduation_years
        }
    
    def print_results(self, results: List[RealYaleAlumni], show_full: bool = False):
        """Print search results with full profiles"""
        if not results:
            print("No alumni found.")
            return
        
        print(f"\\n🎯 Real Yale Alumni Results ({len(results)} found):")
        print("=" * 100)
        
        for i, alumni in enumerate(results, 1):
            print(f"{i}. {alumni.name} (Class of {alumni.graduation_year})")
            print(f"   📧 Email: {alumni.email}")
            print(f"   💼 Current: {alumni.current_position} at {alumni.current_company}")
            print(f"   🏢 Industry: {alumni.industry}")
            print(f"   🎯 Function: {alumni.function}")
            print(f"   📍 Location: {alumni.city}, {alumni.state_territory}")
            print(f"   🎓 Major: {alumni.major}")
            
            if alumni.graduate_school:
                print(f"   🎓 Graduate School: {alumni.graduate_school}")
            
            if alumni.linkedin_url:
                print(f"   💼 LinkedIn: {alumni.linkedin_url}")
            
            # Show skills (truncated)
            if alumni.skills:
                skills_display = alumni.skills[:100] + "..." if len(alumni.skills) > 100 else alumni.skills
                print(f"   💻 Skills: {skills_display}")
            
            if show_full:
                # Show education history
                if alumni.education_history:
                    print(f"   🎓 Education: {alumni.education_history[:200]}...")
                
                # Show experience history
                if alumni.experience_history:
                    print(f"   💼 Experience: {alumni.experience_history[:200]}...")
            
            print(f"   🆔 ID: {alumni.person_id}")
            print()


def main():
    """Main interface for real Yale search"""
    if len(sys.argv) < 2:
        print("""
Real Yale Alumni Search Engine (4,165 Alumni)

Usage:
  python real_yale_search.py "search query"               # General search
  python real_yale_search.py "search query" --full        # Show full profiles
  python real_yale_search.py --company "Google"           # Company search
  python real_yale_search.py --industry "Technology"      # Industry search
  python real_yale_search.py --year 2020                  # Graduation year
  python real_yale_search.py --analytics                  # Database analytics

Examples:
  python real_yale_search.py "software engineer"
  python real_yale_search.py "data scientist python" --full
  python real_yale_search.py --company "Goldman Sachs"
  python real_yale_search.py --industry "Investment Banking"
  python real_yale_search.py --year 2022
        """)
        return
    
    search_engine = RealYaleSearch()
    show_full = '--full' in sys.argv
    
    # Remove flags from arguments
    args = [arg for arg in sys.argv if not arg.startswith('--')]
    
    try:
        if sys.argv[1] == '--analytics':
            analytics = search_engine.get_analytics()
            print("\\n📊 Real Yale Alumni Database Analytics")
            print("=" * 50)
            print(f"Total Alumni: {analytics['total_alumni']:,}")
            print(f"Unique Companies: {analytics['unique_companies']:,}")
            print(f"Unique Industries: {analytics['unique_industries']:,}")
            print(f"Alumni with Email: {analytics['alumni_with_email']:,}")
            print(f"Alumni with LinkedIn: {analytics['alumni_with_linkedin']:,}")
            
            print(f"\\n🏢 Top Companies:")
            for company, count in analytics['top_companies']:
                print(f"  {company}: {count} alumni")
            
            print(f"\\n🏭 Top Industries:")
            for industry, count in analytics['top_industries']:
                print(f"  {industry}: {count} alumni")
            
            print(f"\\n📅 Recent Graduation Years:")
            for year, count in analytics['recent_graduation_years']:
                print(f"  {year}: {count} alumni")
            
        elif sys.argv[1] == '--company' and len(sys.argv) > 2:
            results = search_engine.search_by_company(sys.argv[2])
            search_engine.print_results(results, show_full)
            
        elif sys.argv[1] == '--industry' and len(sys.argv) > 2:
            results = search_engine.search_by_industry(sys.argv[2])
            search_engine.print_results(results, show_full)
            
        elif sys.argv[1] == '--year' and len(sys.argv) > 2:
            year = int(sys.argv[2])
            results = search_engine.search_by_graduation_year(year)
            search_engine.print_results(results, show_full)
            
        else:
            # General search
            query = ' '.join(args[1:])
            results = search_engine.search_alumni(query, limit=10)
            search_engine.print_results(results, show_full)
            
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()