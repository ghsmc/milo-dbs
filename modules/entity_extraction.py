"""
Entity Extraction Module
Tokenizes job titles, extracts seniority levels, specializations, and normalizes entities
"""

import re
import spacy
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any
import pandas as pd
import logging
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedTitle:
    """Structured representation of an extracted job title"""
    original: str
    normalized: str
    seniority: Optional[str] = None
    role_type: Optional[str] = None
    specialization: Optional[str] = None
    department: Optional[str] = None
    industry_focus: Optional[str] = None
    tokens: List[str] = None


class TitleExtractor:
    """Extracts and normalizes job titles with seniority and specialization detection"""
    
    def __init__(self):
        # Load spaCy model - using small model for efficiency
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        # Seniority levels (ordered from junior to senior)
        self.seniority_levels = {
            'intern': 1,
            'junior': 2,
            'associate': 3,
            'analyst': 3,
            'senior analyst': 4,
            'manager': 5,
            'senior manager': 6,
            'director': 7,
            'senior director': 8,
            'vp': 9,
            'vice president': 9,
            'svp': 10,
            'senior vice president': 10,
            'evp': 11,
            'executive vice president': 11,
            'partner': 12,
            'managing director': 12,
            'md': 12,
            'principal': 12,
            'founder': 13,
            'ceo': 14,
            'cto': 14,
            'cfo': 14,
            'coo': 14,
            'president': 14,
        }
        
        # Role type mappings
        self.role_types = {
            'engineering': ['engineer', 'developer', 'programmer', 'architect', 'swe', 'dev'],
            'data': ['data scientist', 'data analyst', 'data engineer', 'ml engineer', 'machine learning'],
            'product': ['product manager', 'pm', 'product owner', 'product lead'],
            'design': ['designer', 'ux', 'ui', 'user experience', 'creative director'],
            'business': ['business analyst', 'strategy', 'operations', 'business development'],
            'finance': ['analyst', 'associate', 'investment', 'trading', 'portfolio', 'fund'],
            'consulting': ['consultant', 'advisor', 'advisory'],
            'sales': ['sales', 'account executive', 'business development', 'bd'],
            'marketing': ['marketing', 'growth', 'brand', 'communications'],
            'legal': ['lawyer', 'attorney', 'counsel', 'legal'],
            'hr': ['hr', 'human resources', 'recruiting', 'talent', 'people ops'],
        }
        
        # Specialization patterns
        self.specializations = {
            'backend': ['backend', 'back-end', 'server', 'api', 'infrastructure'],
            'frontend': ['frontend', 'front-end', 'ui', 'react', 'angular', 'vue'],
            'fullstack': ['fullstack', 'full-stack', 'full stack'],
            'mobile': ['mobile', 'ios', 'android', 'react native'],
            'devops': ['devops', 'sre', 'infrastructure', 'platform', 'cloud'],
            'security': ['security', 'cybersecurity', 'infosec'],
            'blockchain': ['blockchain', 'crypto', 'web3', 'defi'],
            'ai_ml': ['ai', 'ml', 'machine learning', 'deep learning', 'nlp'],
        }
        
        # Common abbreviation mappings
        self.abbreviations = {
            'swe': 'software engineer',
            'sde': 'software development engineer',
            'pm': 'product manager',
            'tpm': 'technical program manager',
            'em': 'engineering manager',
            'vp': 'vice president',
            'svp': 'senior vice president',
            'evp': 'executive vice president',
            'md': 'managing director',
            'ib': 'investment banking',
            'pe': 'private equity',
            'vc': 'venture capital',
            'hf': 'hedge fund',
            'bd': 'business development',
        }
        
        # Build frequency dictionaries
        self.title_frequency = Counter()
        self.token_frequency = Counter()
        
    def normalize_title(self, title: str) -> str:
        """Basic normalization of job title"""
        if not title:
            return ''
            
        normalized = title.lower().strip()
        
        # Expand common abbreviations
        for abbr, full in self.abbreviations.items():
            normalized = re.sub(r'\b' + abbr + r'\b', full, normalized)
            
        # Remove special characters but keep important ones
        normalized = re.sub(r'[^\w\s\-&/]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def extract_seniority(self, title: str) -> Tuple[Optional[str], int]:
        """Extract seniority level from title"""
        title_lower = title.lower()
        
        # Check for exact matches first
        for seniority, level in sorted(self.seniority_levels.items(), 
                                     key=lambda x: -len(x[0])):  # Longest match first
            if seniority in title_lower:
                return seniority, level
                
        # Check for patterns
        if re.search(r'\b(senior|sr\.?)\s+\w+', title_lower):
            return 'senior', 4
        elif re.search(r'\b(junior|jr\.?)\s+\w+', title_lower):
            return 'junior', 2
        elif re.search(r'\b(lead|principal|staff)\s+\w+', title_lower):
            return 'lead', 6
        elif re.search(r'\bhead\s+of\b', title_lower):
            return 'head', 8
            
        return None, 0
    
    def extract_role_type(self, title: str) -> Optional[str]:
        """Extract primary role type from title"""
        title_lower = title.lower()
        
        # Score each role type based on keyword matches
        role_scores = {}
        for role_type, keywords in self.role_types.items():
            score = sum(1 for keyword in keywords if keyword in title_lower)
            if score > 0:
                role_scores[role_type] = score
                
        # Return the role type with highest score
        if role_scores:
            return max(role_scores, key=role_scores.get)
            
        return None
    
    def extract_specialization(self, title: str) -> Optional[str]:
        """Extract technical specialization from title"""
        title_lower = title.lower()
        
        for spec, keywords in self.specializations.items():
            if any(keyword in title_lower for keyword in keywords):
                return spec
                
        return None
    
    def tokenize_title(self, title: str) -> List[str]:
        """Tokenize title into meaningful components"""
        normalized = self.normalize_title(title)
        
        if self.nlp:
            # Use spaCy tokenization
            doc = self.nlp(normalized)
            tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
        else:
            # Fallback to simple tokenization
            tokens = normalized.split()
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
            tokens = [t for t in tokens if t not in stop_words]
            
        return tokens
    
    def extract_title_entities(self, title: str) -> ExtractedTitle:
        """Extract all entities from a job title"""
        normalized = self.normalize_title(title)
        tokens = self.tokenize_title(title)
        
        seniority, seniority_level = self.extract_seniority(normalized)
        role_type = self.extract_role_type(normalized)
        specialization = self.extract_specialization(normalized)
        
        # Update frequency counters
        self.title_frequency[normalized] += 1
        self.token_frequency.update(tokens)
        
        # Detect department from common patterns
        department = None
        if any(term in normalized for term in ['engineering', 'technology', 'tech', 'it']):
            department = 'engineering'
        elif any(term in normalized for term in ['product', 'pm']):
            department = 'product'
        elif any(term in normalized for term in ['finance', 'accounting', 'treasury']):
            department = 'finance'
        elif any(term in normalized for term in ['marketing', 'growth', 'brand']):
            department = 'marketing'
        elif any(term in normalized for term in ['sales', 'business development']):
            department = 'sales'
            
        # Detect industry focus
        industry_focus = None
        if any(term in normalized for term in ['investment banking', 'ib', 'capital markets']):
            industry_focus = 'investment_banking'
        elif any(term in normalized for term in ['private equity', 'pe', 'buyout']):
            industry_focus = 'private_equity'
        elif any(term in normalized for term in ['venture capital', 'vc', 'venture']):
            industry_focus = 'venture_capital'
        elif any(term in normalized for term in ['hedge fund', 'hf', 'trading']):
            industry_focus = 'hedge_fund'
            
        return ExtractedTitle(
            original=title,
            normalized=normalized,
            seniority=seniority,
            role_type=role_type,
            specialization=specialization,
            department=department,
            industry_focus=industry_focus,
            tokens=tokens
        )
    
    def get_title_variations(self, title: str) -> List[str]:
        """Generate common variations of a title"""
        extracted = self.extract_title_entities(title)
        variations = [extracted.normalized]
        
        # Add abbreviated version
        if 'software engineer' in extracted.normalized:
            variations.append(extracted.normalized.replace('software engineer', 'swe'))
        if 'product manager' in extracted.normalized:
            variations.append(extracted.normalized.replace('product manager', 'pm'))
            
        # Add version without seniority
        if extracted.seniority:
            without_seniority = extracted.normalized.replace(extracted.seniority + ' ', '')
            variations.append(without_seniority)
            
        # Add common variations
        if 'developer' in extracted.normalized:
            variations.append(extracted.normalized.replace('developer', 'engineer'))
        if 'engineer' in extracted.normalized:
            variations.append(extracted.normalized.replace('engineer', 'developer'))
            
        return list(set(variations))


class LocationExtractor:
    """Advanced location extraction and normalization"""
    
    def __init__(self):
        # Major metro areas and their variations
        self.metro_areas = {
            'sf_bay_area': {
                'canonical': 'San Francisco Bay Area',
                'cities': ['san francisco', 'sf', 'oakland', 'san jose', 'berkeley', 
                          'palo alto', 'mountain view', 'menlo park', 'redwood city',
                          'sunnyvale', 'cupertino', 'fremont'],
                'variations': ['bay area', 'sf bay', 'silicon valley', 'south bay']
            },
            'nyc_metro': {
                'canonical': 'New York City Metro',
                'cities': ['new york', 'nyc', 'manhattan', 'brooklyn', 'queens', 
                          'bronx', 'staten island', 'jersey city', 'newark'],
                'variations': ['new york city', 'ny metro', 'tri-state area']
            },
            'la_metro': {
                'canonical': 'Los Angeles Metro',
                'cities': ['los angeles', 'la', 'santa monica', 'pasadena', 'long beach',
                          'glendale', 'burbank', 'beverly hills', 'culver city'],
                'variations': ['la metro', 'greater la', 'socal']
            },
            'chicago_metro': {
                'canonical': 'Chicago Metro',
                'cities': ['chicago', 'evanston', 'oak park', 'schaumburg', 'naperville'],
                'variations': ['chicagoland', 'chi-town']
            },
            'boston_metro': {
                'canonical': 'Boston Metro',
                'cities': ['boston', 'cambridge', 'somerville', 'brookline', 'newton'],
                'variations': ['greater boston', 'boston area']
            }
        }
        
        # State abbreviations
        self.state_abbrevs = {
            'ca': 'california', 'ny': 'new york', 'tx': 'texas', 'fl': 'florida',
            'il': 'illinois', 'pa': 'pennsylvania', 'oh': 'ohio', 'ga': 'georgia',
            'nc': 'north carolina', 'mi': 'michigan', 'nj': 'new jersey', 
            'va': 'virginia', 'wa': 'washington', 'ma': 'massachusetts',
            'az': 'arizona', 'in': 'indiana', 'tn': 'tennessee', 'mo': 'missouri',
            'md': 'maryland', 'wi': 'wisconsin', 'co': 'colorado', 'mn': 'minnesota',
            'sc': 'south carolina', 'al': 'alabama', 'la': 'louisiana', 'ky': 'kentucky',
            'or': 'oregon', 'ok': 'oklahoma', 'ct': 'connecticut', 'ut': 'utah',
            'nv': 'nevada', 'dc': 'district of columbia'
        }
        
        # International locations
        self.international = {
            'london': ['london', 'london uk', 'london england', 'city of london'],
            'hong_kong': ['hong kong', 'hk', 'hong kong sar'],
            'singapore': ['singapore', 'sg'],
            'tokyo': ['tokyo', 'tokyo japan'],
            'paris': ['paris', 'paris france'],
            'dubai': ['dubai', 'dubai uae', 'dxb'],
            'toronto': ['toronto', 'toronto canada', 'gta'],
            'sydney': ['sydney', 'sydney australia'],
            'mumbai': ['mumbai', 'bombay', 'mumbai india'],
            'shanghai': ['shanghai', 'shanghai china'],
        }
        
        # Build reverse lookup
        self.city_to_metro = {}
        for metro, data in self.metro_areas.items():
            for city in data['cities']:
                self.city_to_metro[city.lower()] = metro
                
    def extract_location_components(self, location: str) -> Dict[str, str]:
        """Extract city, state, country from location string"""
        if not location:
            return {}
            
        components = {}
        location_lower = location.lower().strip()
        
        # Remove common suffixes
        location_lower = re.sub(r',?\s*(usa|united states|u\.s\.|us)$', '', location_lower)
        
        # Check for metro area
        for metro, data in self.metro_areas.items():
            if any(city in location_lower for city in data['cities']):
                components['metro_area'] = data['canonical']
                components['city'] = self._extract_city(location_lower, data['cities'])
                components['state'] = self._get_state_for_metro(metro)
                components['country'] = 'United States'
                return components
                
        # Check for international
        for loc_key, variations in self.international.items():
            if any(var in location_lower for var in variations):
                components['city'] = loc_key.replace('_', ' ').title()
                components['country'] = self._get_country_for_international(loc_key)
                return components
                
        # Try to parse City, State format
        if ',' in location:
            parts = [p.strip() for p in location.split(',')]
            if len(parts) >= 2:
                components['city'] = parts[0]
                state_part = parts[1].lower()
                
                # Check if it's a state abbreviation
                if state_part in self.state_abbrevs:
                    components['state'] = self.state_abbrevs[state_part].title()
                    components['country'] = 'United States'
                else:
                    components['state'] = parts[1]
                    
        return components
    
    def _extract_city(self, location: str, city_list: List[str]) -> str:
        """Extract the most specific city from a location string"""
        for city in sorted(city_list, key=len, reverse=True):  # Longest match first
            if city in location:
                return city.title()
        return location.title()
    
    def _get_state_for_metro(self, metro: str) -> str:
        """Get state for a metro area"""
        metro_states = {
            'sf_bay_area': 'California',
            'nyc_metro': 'New York',
            'la_metro': 'California',
            'chicago_metro': 'Illinois',
            'boston_metro': 'Massachusetts'
        }
        return metro_states.get(metro, '')
    
    def _get_country_for_international(self, location: str) -> str:
        """Get country for international location"""
        location_countries = {
            'london': 'United Kingdom',
            'hong_kong': 'Hong Kong',
            'singapore': 'Singapore',
            'tokyo': 'Japan',
            'paris': 'France',
            'dubai': 'United Arab Emirates',
            'toronto': 'Canada',
            'sydney': 'Australia',
            'mumbai': 'India',
            'shanghai': 'China'
        }
        return location_countries.get(location, '')
    
    def normalize_location(self, location: str) -> str:
        """Normalize location to standard format"""
        components = self.extract_location_components(location)
        
        if 'metro_area' in components:
            return components['metro_area']
        elif 'city' in components:
            if 'state' in components:
                return f"{components['city']}, {components['state']}"
            elif 'country' in components and components['country'] != 'United States':
                return f"{components['city']}, {components['country']}"
            else:
                return components['city']
                
        return location.title() if location else ''


class EntityExtractionPipeline:
    """Main pipeline for entity extraction from alumni data"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.title_extractor = TitleExtractor()
        self.location_extractor = LocationExtractor()
        
    def build_frequency_dictionaries(self):
        """Build frequency dictionaries from database"""
        import psycopg2
        
        conn = psycopg2.connect(**self.db_config)
        
        # Extract all titles
        df_titles = pd.read_sql("""
            SELECT current_title, COUNT(*) as count
            FROM alumni
            WHERE current_title IS NOT NULL
            GROUP BY current_title
        """, conn)
        
        # Extract all locations
        df_locations = pd.read_sql("""
            SELECT location, COUNT(*) as count
            FROM alumni
            WHERE location IS NOT NULL
            GROUP BY location
        """, conn)
        
        conn.close()
        
        # Process titles
        for _, row in df_titles.iterrows():
            title = row['current_title']
            count = row['count']
            
            # Extract entities
            extracted = self.title_extractor.extract_title_entities(title)
            
            # Update frequency with actual count
            self.title_extractor.title_frequency[extracted.normalized] += count
            self.title_extractor.token_frequency.update({
                token: count for token in extracted.tokens
            })
            
        logger.info(f"Processed {len(df_titles)} unique titles")
        logger.info(f"Top 10 title tokens: {self.title_extractor.token_frequency.most_common(10)}")
        
        return {
            'title_frequency': dict(self.title_extractor.title_frequency),
            'token_frequency': dict(self.title_extractor.token_frequency),
            'unique_titles': len(df_titles),
            'unique_locations': len(df_locations)
        }
    
    def extract_and_store_entities(self):
        """Extract entities from all records and store in database"""
        import psycopg2
        from psycopg2.extras import execute_batch
        
        conn = psycopg2.connect(**self.db_config)
        
        # Create entity tables
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS title_entities (
                    person_id VARCHAR(50) REFERENCES alumni(person_id),
                    original_title VARCHAR(255),
                    normalized_title VARCHAR(255),
                    seniority VARCHAR(50),
                    seniority_level INTEGER,
                    role_type VARCHAR(50),
                    specialization VARCHAR(50),
                    department VARCHAR(50),
                    industry_focus VARCHAR(50),
                    PRIMARY KEY (person_id)
                )
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS location_entities (
                    person_id VARCHAR(50) REFERENCES alumni(person_id),
                    original_location VARCHAR(255),
                    normalized_location VARCHAR(255),
                    city VARCHAR(100),
                    state VARCHAR(100),
                    country VARCHAR(100),
                    metro_area VARCHAR(100),
                    PRIMARY KEY (person_id)
                )
            """)
            
            # Create indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_title_entities_seniority ON title_entities(seniority)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_title_entities_role_type ON title_entities(role_type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_location_entities_metro ON location_entities(metro_area)")
            
        conn.commit()
        
        # Process records in batches
        batch_size = 1000
        offset = 0
        
        while True:
            df = pd.read_sql(f"""
                SELECT person_id, current_title, location
                FROM alumni
                LIMIT {batch_size} OFFSET {offset}
            """, conn)
            
            if df.empty:
                break
                
            title_data = []
            location_data = []
            
            for _, row in df.iterrows():
                person_id = row['person_id']
                
                # Extract title entities
                if row['current_title']:
                    title_extracted = self.title_extractor.extract_title_entities(row['current_title'])
                    seniority, seniority_level = self.title_extractor.extract_seniority(title_extracted.normalized)
                    
                    title_data.append((
                        person_id,
                        row['current_title'],
                        title_extracted.normalized,
                        title_extracted.seniority,
                        seniority_level,
                        title_extracted.role_type,
                        title_extracted.specialization,
                        title_extracted.department,
                        title_extracted.industry_focus
                    ))
                
                # Extract location entities
                if row['location']:
                    location_components = self.location_extractor.extract_location_components(row['location'])
                    normalized_location = self.location_extractor.normalize_location(row['location'])
                    
                    location_data.append((
                        person_id,
                        row['location'],
                        normalized_location,
                        location_components.get('city'),
                        location_components.get('state'),
                        location_components.get('country'),
                        location_components.get('metro_area')
                    ))
            
            # Batch insert
            with conn.cursor() as cur:
                if title_data:
                    execute_batch(cur, """
                        INSERT INTO title_entities 
                        (person_id, original_title, normalized_title, seniority, seniority_level,
                         role_type, specialization, department, industry_focus)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (person_id) DO UPDATE
                        SET original_title = EXCLUDED.original_title,
                            normalized_title = EXCLUDED.normalized_title,
                            seniority = EXCLUDED.seniority,
                            seniority_level = EXCLUDED.seniority_level,
                            role_type = EXCLUDED.role_type,
                            specialization = EXCLUDED.specialization,
                            department = EXCLUDED.department,
                            industry_focus = EXCLUDED.industry_focus
                    """, title_data)
                
                if location_data:
                    execute_batch(cur, """
                        INSERT INTO location_entities
                        (person_id, original_location, normalized_location, city, state, country, metro_area)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (person_id) DO UPDATE
                        SET original_location = EXCLUDED.original_location,
                            normalized_location = EXCLUDED.normalized_location,
                            city = EXCLUDED.city,
                            state = EXCLUDED.state,
                            country = EXCLUDED.country,
                            metro_area = EXCLUDED.metro_area
                    """, location_data)
            
            conn.commit()
            offset += batch_size
            logger.info(f"Processed {offset} records")
        
        conn.close()
        logger.info("Entity extraction completed")
    
    def get_entity_statistics(self) -> Dict[str, Any]:
        """Get statistics about extracted entities"""
        import psycopg2
        
        conn = psycopg2.connect(**self.db_config)
        stats = {}
        
        with conn.cursor() as cur:
            # Seniority distribution
            cur.execute("""
                SELECT seniority, COUNT(*) as count
                FROM title_entities
                WHERE seniority IS NOT NULL
                GROUP BY seniority
                ORDER BY count DESC
            """)
            stats['seniority_distribution'] = cur.fetchall()
            
            # Role type distribution
            cur.execute("""
                SELECT role_type, COUNT(*) as count
                FROM title_entities
                WHERE role_type IS NOT NULL
                GROUP BY role_type
                ORDER BY count DESC
            """)
            stats['role_type_distribution'] = cur.fetchall()
            
            # Metro area distribution
            cur.execute("""
                SELECT metro_area, COUNT(*) as count
                FROM location_entities
                WHERE metro_area IS NOT NULL
                GROUP BY metro_area
                ORDER BY count DESC
            """)
            stats['metro_area_distribution'] = cur.fetchall()
            
            # Industry focus distribution
            cur.execute("""
                SELECT industry_focus, COUNT(*) as count
                FROM title_entities
                WHERE industry_focus IS NOT NULL
                GROUP BY industry_focus
                ORDER BY count DESC
            """)
            stats['industry_focus_distribution'] = cur.fetchall()
        
        conn.close()
        return stats


# Example usage
if __name__ == "__main__":
    db_config = {
        'host': 'localhost',
        'database': 'yale_alumni',
        'user': 'postgres',
        'password': 'password',
        'port': 5432
    }
    
    pipeline = EntityExtractionPipeline(db_config)
    
    # Build frequency dictionaries
    freq_stats = pipeline.build_frequency_dictionaries()
    print(json.dumps(freq_stats, indent=2))
    
    # Extract and store entities
    pipeline.extract_and_store_entities()
    
    # Get statistics
    stats = pipeline.get_entity_statistics()
    print(json.dumps(stats, indent=2))