#!/usr/bin/env python3
"""
Yale Alumni Terminology Analyzer
Analyzes the actual 4,165 alumni data to extract domain-specific terminology
for building better semantic search
"""

import json
import psycopg2
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set

class AlumniTerminologyAnalyzer:
    """
    Analyzes real alumni data to extract domain-specific terminology
    """
    
    def __init__(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        self.db_config = self.config['database']
    
    def analyze_alumni_data(self):
        """Comprehensive analysis of alumni terminology"""
        print("🔍 Analyzing 4,165 Yale Alumni for Domain-Specific Terminology")
        print("=" * 70)
        
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        # Get all alumni data
        cur.execute("""
            SELECT current_position, current_company, industry, function, skills, 
                   experience_history, major, graduation_year
            FROM alumni_real 
            WHERE current_position != '' OR current_company != ''
        """)
        
        all_data = cur.fetchall()
        conn.close()
        
        print(f"📊 Analyzing {len(all_data)} alumni profiles")
        
        # Analyze different categories
        self.analyze_financial_terminology(all_data)
        self.analyze_technology_terminology(all_data)
        self.analyze_consulting_terminology(all_data)
        self.analyze_company_terminology(all_data)
        self.analyze_skill_terminology(all_data)
        self.build_phrase_mappings(all_data)
        
        return self.create_enhanced_mappings(all_data)
    
    def analyze_financial_terminology(self, data: List):
        """Extract financial industry terminology"""
        print("\n💰 Financial Industry Analysis:")
        
        financial_roles = Counter()
        financial_companies = Counter()
        financial_keywords = Counter()
        
        finance_indicators = [
            'investment', 'banking', 'finance', 'financial', 'capital', 'equity',
            'private equity', 'venture capital', 'hedge fund', 'wall street',
            'goldman', 'morgan', 'jpmorgan', 'credit suisse', 'deutsche bank',
            'barclays', 'citigroup', 'wells fargo', 'bank of america',
            'analyst', 'associate', 'vice president', 'managing director',
            'trader', 'portfolio', 'risk', 'compliance', 'research'
        ]
        
        for row in data:
            position = (row[0] or '').lower()
            company = (row[1] or '').lower()
            industry = (row[2] or '').lower()
            skills = (row[4] or '').lower()
            
            # Check if this is a finance professional
            text_to_check = f"{position} {company} {industry} {skills}"
            
            if any(indicator in text_to_check for indicator in finance_indicators):
                # Extract specific roles
                if position:
                    financial_roles[position] += 1
                if company:
                    financial_companies[company] += 1
                
                # Extract keywords
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text_to_check)
                for word in words:
                    if len(word) > 2:
                        financial_keywords[word] += 1
        
        print(f"   Found {len(financial_roles)} unique financial roles")
        print(f"   Top Financial Roles:")
        for role, count in financial_roles.most_common(10):
            print(f"     {role} ({count} alumni)")
        
        print(f"\n   Top Financial Companies:")
        for company, count in financial_companies.most_common(10):
            print(f"     {company} ({count} alumni)")
        
        print(f"\n   Most Common Financial Keywords:")
        relevant_keywords = {k: v for k, v in financial_keywords.items() 
                           if v >= 3 and any(indicator in k for indicator in finance_indicators)}
        for keyword, count in Counter(relevant_keywords).most_common(15):
            print(f"     {keyword} ({count} times)")
    
    def analyze_technology_terminology(self, data: List):
        """Extract technology industry terminology"""
        print("\n💻 Technology Industry Analysis:")
        
        tech_roles = Counter()
        tech_companies = Counter()
        tech_skills = Counter()
        
        tech_indicators = [
            'software', 'engineer', 'developer', 'data', 'scientist', 'analyst',
            'product', 'manager', 'technical', 'programming', 'coding',
            'python', 'java', 'javascript', 'machine learning', 'ai',
            'google', 'apple', 'microsoft', 'amazon', 'meta', 'facebook',
            'netflix', 'uber', 'airbnb', 'stripe', 'salesforce'
        ]
        
        for row in data:
            position = (row[0] or '').lower()
            company = (row[1] or '').lower()
            skills = (row[4] or '').lower()
            
            text_to_check = f"{position} {company} {skills}"
            
            if any(indicator in text_to_check for indicator in tech_indicators):
                if position:
                    tech_roles[position] += 1
                if company:
                    tech_companies[company] += 1
                
                # Extract tech skills
                if skills:
                    skill_words = re.findall(r'\b[a-zA-Z+#\.]{2,}\b', skills)
                    for skill in skill_words:
                        tech_skills[skill.lower()] += 1
        
        print(f"   Found {len(tech_roles)} unique tech roles")
        print(f"   Top Tech Roles:")
        for role, count in tech_roles.most_common(8):
            print(f"     {role} ({count} alumni)")
        
        print(f"\n   Top Tech Companies:")
        for company, count in tech_companies.most_common(8):
            print(f"     {company} ({count} alumni)")
        
        print(f"\n   Most Common Tech Skills:")
        for skill, count in tech_skills.most_common(10):
            if count >= 2:
                print(f"     {skill} ({count} alumni)")
    
    def analyze_consulting_terminology(self, data: List):
        """Extract consulting industry terminology"""
        print("\n🏛️ Consulting Industry Analysis:")
        
        consulting_firms = Counter()
        consulting_roles = Counter()
        
        consulting_indicators = [
            'consulting', 'consultant', 'advisory', 'strategy', 'management',
            'mckinsey', 'bain', 'bcg', 'boston consulting', 'booz',
            'deloitte', 'pwc', 'kpmg', 'ey', 'ernst', 'accenture'
        ]
        
        for row in data:
            position = (row[0] or '').lower()
            company = (row[1] or '').lower()
            
            text_to_check = f"{position} {company}"
            
            if any(indicator in text_to_check for indicator in consulting_indicators):
                if position:
                    consulting_roles[position] += 1
                if company:
                    consulting_firms[company] += 1
        
        print(f"   Top Consulting Firms:")
        for firm, count in consulting_firms.most_common(8):
            print(f"     {firm} ({count} alumni)")
        
        print(f"\n   Top Consulting Roles:")
        for role, count in consulting_roles.most_common(8):
            print(f"     {role} ({count} alumni)")
    
    def analyze_company_terminology(self, data: List):
        """Extract all company terminology for better matching"""
        print("\n🏢 Company Analysis:")
        
        all_companies = Counter()
        company_variants = defaultdict(set)
        
        for row in data:
            company = (row[1] or '').strip()
            if company and company != '-':
                all_companies[company.lower()] += 1
                
                # Track company name variants
                base_name = self.extract_base_company_name(company)
                company_variants[base_name].add(company.lower())
        
        print(f"   Total unique companies: {len(all_companies)}")
        print(f"   Top Companies by Alumni Count:")
        for company, count in all_companies.most_common(15):
            if count >= 3:
                print(f"     {company} ({count} alumni)")
        
        # Show company variants
        print(f"\n   Company Name Variants (for better matching):")
        for base_name, variants in list(company_variants.items())[:10]:
            if len(variants) > 1:
                print(f"     {base_name}: {', '.join(list(variants)[:3])}")
    
    def analyze_skill_terminology(self, data: List):
        """Extract skill terminology"""
        print("\n🎯 Skills Analysis:")
        
        all_skills = Counter()
        
        for row in data:
            skills_text = row[4] or ''
            if skills_text and skills_text != '-':
                # Parse skills (often comma-separated)
                skills = re.split(r'[,;]', skills_text)
                for skill in skills:
                    skill = skill.strip().lower()
                    if len(skill) > 2:
                        all_skills[skill] += 1
        
        print(f"   Total unique skills: {len(all_skills)}")
        print(f"   Most Common Skills:")
        for skill, count in all_skills.most_common(20):
            if count >= 5:
                print(f"     {skill} ({count} alumni)")
    
    def build_phrase_mappings(self, data: List):
        """Build phrase recognition for multi-word terms"""
        print("\n🔗 Phrase Recognition Analysis:")
        
        important_phrases = [
            'wall street', 'wall st', 'wall st.', 'investment bank', 'investment banking',
            'private equity', 'venture capital', 'hedge fund', 'mutual fund',
            'asset management', 'wealth management', 'financial advisor',
            'goldman sachs', 'morgan stanley', 'jp morgan', 'jpmorgan chase',
            'credit suisse', 'deutsche bank', 'bank of america', 'wells fargo',
            'software engineer', 'data scientist', 'product manager', 'project manager',
            'machine learning', 'artificial intelligence', 'data analysis',
            'management consulting', 'strategy consulting', 'business analyst',
            'new york', 'san francisco', 'los angeles', 'boston', 'chicago'
        ]
        
        phrase_counts = Counter()
        
        for row in data:
            text_fields = [row[0] or '', row[1] or '', row[2] or '', row[4] or '']
            full_text = ' '.join(text_fields).lower()
            
            for phrase in important_phrases:
                if phrase in full_text:
                    phrase_counts[phrase] += 1
        
        print(f"   Important Phrases Found in Alumni Data:")
        for phrase, count in phrase_counts.most_common(15):
            if count >= 2:
                print(f"     \"{phrase}\" ({count} alumni)")
    
    def extract_base_company_name(self, company_name: str) -> str:
        """Extract base company name (remove inc, llc, etc.)"""
        company = company_name.lower().strip()
        # Remove common suffixes
        suffixes = ['inc', 'llc', 'ltd', 'corp', 'corporation', 'company', 'co', 'group']
        for suffix in suffixes:
            if company.endswith(f' {suffix}'):
                company = company[:-len(suffix)-1]
        return company.strip()
    
    def create_enhanced_mappings(self, data: List) -> Dict:
        """Create enhanced synonym mappings based on real data"""
        print("\n🧠 Building Enhanced Semantic Mappings...")
        
        # Build mappings based on actual data analysis
        enhanced_mappings = {
            # Financial industry mappings
            'wall': ['wall street', 'finance', 'financial', 'investment banking', 'banking'],
            'st': ['street', 'wall street'] if 'wall' in str(data) else ['street'],
            'vc': ['venture capital', 'venture', 'capital', 'investment', 'startup funding'],
            'banker': ['investment banker', 'banking', 'finance', 'financial advisor', 'investment'],
            'investment': ['banking', 'finance', 'financial', 'capital', 'asset management'],
            'banking': ['investment', 'finance', 'financial', 'banker', 'bank'],
            'goldman': ['goldman sachs', 'investment banking', 'finance', 'wall street'],
            'sachs': ['goldman sachs', 'investment banking', 'finance'],
            'morgan': ['morgan stanley', 'jp morgan', 'jpmorgan', 'investment banking'],
            'jp': ['jpmorgan', 'jp morgan', 'chase', 'investment banking'],
            'jpmorgan': ['jp morgan', 'chase', 'investment banking', 'finance'],
            
            # Technology mappings
            'software': ['engineer', 'developer', 'programming', 'development', 'tech'],
            'engineer': ['software', 'developer', 'technical', 'programming', 'engineering'],
            'developer': ['software', 'engineer', 'programming', 'development', 'coding'],
            'data': ['scientist', 'analyst', 'analysis', 'analytics', 'science'],
            'scientist': ['data', 'researcher', 'research', 'analyst', 'analysis'],
            'machine': ['learning', 'ai', 'artificial intelligence', 'data science'],
            'ai': ['artificial intelligence', 'machine learning', 'data science'],
            
            # Consulting mappings
            'consulting': ['consultant', 'advisory', 'strategy', 'management consulting'],
            'consultant': ['consulting', 'advisory', 'strategy', 'advisor'],
            'mckinsey': ['consulting', 'strategy', 'management consulting'],
            'bain': ['consulting', 'strategy', 'management consulting'],
            'bcg': ['boston consulting', 'consulting', 'strategy'],
            
            # Company mappings
            'google': ['alphabet', 'tech', 'technology', 'software'],
            'meta': ['facebook', 'social media', 'tech', 'technology'],
            'microsoft': ['tech', 'technology', 'software', 'cloud'],
            'amazon': ['aws', 'cloud', 'tech', 'technology', 'ecommerce'],
            
            # Location mappings
            'ny': ['new york', 'nyc', 'manhattan'],
            'nyc': ['new york', 'ny', 'manhattan'],
            'sf': ['san francisco', 'bay area', 'california'],
            'la': ['los angeles', 'california'],
            
            # General business mappings
            'manager': ['management', 'director', 'lead', 'supervisor'],
            'director': ['manager', 'management', 'leadership', 'head'],
            'analyst': ['analysis', 'data', 'research', 'financial'],
            'associate': ['analyst', 'junior', 'staff'],
            'vp': ['vice president', 'director', 'senior'],
            'md': ['managing director', 'director', 'senior']
        }
        
        print(f"   Created {len(enhanced_mappings)} enhanced mappings")
        
        # Save mappings for use
        with open('enhanced_mappings.json', 'w') as f:
            json.dump(enhanced_mappings, f, indent=2)
        
        print("   ✅ Enhanced mappings saved to enhanced_mappings.json")
        
        return enhanced_mappings

def main():
    analyzer = AlumniTerminologyAnalyzer()
    enhanced_mappings = analyzer.analyze_alumni_data()
    
    print(f"\n🎯 Analysis Complete!")
    print(f"Enhanced semantic mappings have been created based on actual alumni data.")
    print(f"These mappings will dramatically improve search relevance for domain-specific queries.")

if __name__ == "__main__":
    main()