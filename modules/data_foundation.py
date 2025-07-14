"""
Data Foundation Module
Handles data ingestion, deduplication, normalization, and database operations
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_batch
import logging
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import hashlib
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AlumniRecord:
    """Structured alumni data record"""
    person_id: str
    name: str
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    location: Optional[str] = None
    graduation_year: Optional[int] = None
    degree: Optional[str] = None
    major: Optional[str] = None
    experience: List[Dict[str, Any]] = field(default_factory=list)
    education: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    normalized_company: Optional[str] = None
    normalized_title: Optional[str] = None
    normalized_location: Optional[str] = None
    record_hash: Optional[str] = None
    
    def generate_hash(self) -> str:
        """Generate unique hash for deduplication"""
        hash_string = f"{self.name.lower()}_{self.current_company or ''}_{self.graduation_year or ''}"
        return hashlib.md5(hash_string.encode()).hexdigest()


class CompanyNormalizer:
    """Handles company name normalization and variant mapping"""
    
    def __init__(self):
        self.company_mappings = {
            # FAANG and variants
            'google': ['google', 'google inc', 'google llc', 'alphabet', 'alphabet inc'],
            'meta': ['meta', 'facebook', 'facebook inc', 'fb', 'meta platforms'],
            'amazon': ['amazon', 'amazon.com', 'amazon inc', 'amazon web services', 'aws'],
            'apple': ['apple', 'apple inc', 'apple computer'],
            'netflix': ['netflix', 'netflix inc'],
            'microsoft': ['microsoft', 'microsoft corp', 'microsoft corporation', 'msft'],
            
            # Investment Banks
            'goldman sachs': ['goldman sachs', 'goldman', 'gs', 'goldman sachs & co'],
            'morgan stanley': ['morgan stanley', 'ms', 'morgan stanley & co'],
            'jp morgan': ['jp morgan', 'jpmorgan', 'jpmorgan chase', 'jpm', 'jp morgan chase'],
            'bank of america': ['bank of america', 'bofa', 'boa', 'bank of america merrill lynch', 'baml'],
            'citigroup': ['citigroup', 'citi', 'citibank', 'citicorp'],
            
            # Consulting
            'mckinsey': ['mckinsey', 'mckinsey & company', 'mckinsey and company'],
            'bain': ['bain', 'bain & company', 'bain and company'],
            'bcg': ['bcg', 'boston consulting group', 'the boston consulting group'],
            
            # Private Equity
            'blackstone': ['blackstone', 'the blackstone group', 'blackstone group'],
            'kkr': ['kkr', 'kohlberg kravis roberts', 'kkr & co'],
            'carlyle': ['carlyle', 'the carlyle group', 'carlyle group'],
        }
        
        # Build reverse mapping for fast lookup
        self.variant_to_canonical = {}
        for canonical, variants in self.company_mappings.items():
            for variant in variants:
                self.variant_to_canonical[variant.lower()] = canonical
                
        # Regex patterns for common variations
        self.cleaning_patterns = [
            (r'\s+(inc|incorporated|corp|corporation|llc|ltd|limited|co\.?|company)\.?$', ''),
            (r'^the\s+', ''),
            (r'\s+&\s+', ' and '),
            (r'\s+', ' '),  # normalize whitespace
        ]
    
    def normalize(self, company_name: str) -> str:
        """Normalize company name to canonical form"""
        if not company_name:
            return ''
            
        normalized = company_name.lower().strip()
        
        # Apply regex cleaning patterns
        for pattern, replacement in self.cleaning_patterns:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        # Check for known variants
        if normalized in self.variant_to_canonical:
            return self.variant_to_canonical[normalized]
            
        # If no exact match, try fuzzy matching for known companies
        canonical_names = list(self.company_mappings.keys())
        match, score = process.extractOne(normalized, canonical_names, scorer=fuzz.ratio)
        
        if score >= 85:  # High confidence threshold
            return match
            
        return normalized.title()  # Return cleaned version if no match


class LocationNormalizer:
    """Handles location normalization"""
    
    def __init__(self):
        self.location_mappings = {
            'new york': ['new york', 'new york city', 'nyc', 'ny', 'new york, ny', 'manhattan'],
            'san francisco': ['san francisco', 'sf', 'san francisco, ca', 'san francisco bay area'],
            'los angeles': ['los angeles', 'la', 'los angeles, ca', 'l.a.'],
            'chicago': ['chicago', 'chicago, il', 'chi town'],
            'boston': ['boston', 'boston, ma', 'bos'],
            'washington dc': ['washington dc', 'washington, dc', 'dc', 'washington d.c.', 'district of columbia'],
            'london': ['london', 'london, uk', 'london, england', 'london, united kingdom'],
            'hong kong': ['hong kong', 'hk', 'hong kong sar'],
            'singapore': ['singapore', 'sg', 'singapore, singapore'],
        }
        
        self.variant_to_canonical = {}
        for canonical, variants in self.location_mappings.items():
            for variant in variants:
                self.variant_to_canonical[variant.lower()] = canonical
    
    def normalize(self, location: str) -> str:
        """Normalize location to canonical form"""
        if not location:
            return ''
            
        normalized = location.lower().strip()
        
        # Remove common suffixes
        normalized = re.sub(r',?\s*(usa|united states|u\.s\.|us)$', '', normalized, flags=re.IGNORECASE)
        
        # Check for known variants
        if normalized in self.variant_to_canonical:
            return self.variant_to_canonical[normalized]
            
        # Try to extract city from "City, State" format
        if ',' in normalized:
            city = normalized.split(',')[0].strip()
            if city in self.variant_to_canonical:
                return self.variant_to_canonical[city]
                
        return location.title()


class DataFoundation:
    """Main class for data foundation operations"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.company_normalizer = CompanyNormalizer()
        self.location_normalizer = LocationNormalizer()
        self.dedup_cache = set()
        
    def connect_db(self):
        """Create database connection"""
        return psycopg2.connect(**self.db_config)
    
    def create_tables(self):
        """Create PostgreSQL tables with proper indexes"""
        create_statements = [
            """
            CREATE TABLE IF NOT EXISTS alumni (
                person_id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                current_title VARCHAR(255),
                current_company VARCHAR(255),
                normalized_company VARCHAR(255),
                normalized_title VARCHAR(255),
                location VARCHAR(255),
                normalized_location VARCHAR(255),
                graduation_year INTEGER,
                degree VARCHAR(100),
                major VARCHAR(255),
                record_hash VARCHAR(32) UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS experience (
                id SERIAL PRIMARY KEY,
                person_id VARCHAR(50) REFERENCES alumni(person_id),
                company VARCHAR(255),
                normalized_company VARCHAR(255),
                title VARCHAR(255),
                normalized_title VARCHAR(255),
                location VARCHAR(255),
                start_date DATE,
                end_date DATE,
                description TEXT,
                is_current BOOLEAN DEFAULT FALSE
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS education (
                id SERIAL PRIMARY KEY,
                person_id VARCHAR(50) REFERENCES alumni(person_id),
                school VARCHAR(255),
                degree VARCHAR(100),
                major VARCHAR(255),
                graduation_year INTEGER,
                start_year INTEGER,
                end_year INTEGER
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS skills (
                id SERIAL PRIMARY KEY,
                person_id VARCHAR(50) REFERENCES alumni(person_id),
                skill VARCHAR(255)
            )
            """,
            
            # Indexes for search performance
            "CREATE INDEX IF NOT EXISTS idx_alumni_normalized_company ON alumni(normalized_company)",
            "CREATE INDEX IF NOT EXISTS idx_alumni_normalized_title ON alumni(normalized_title)",
            "CREATE INDEX IF NOT EXISTS idx_alumni_normalized_location ON alumni(normalized_location)",
            "CREATE INDEX IF NOT EXISTS idx_alumni_graduation_year ON alumni(graduation_year)",
            "CREATE INDEX IF NOT EXISTS idx_experience_normalized_company ON experience(normalized_company)",
            "CREATE INDEX IF NOT EXISTS idx_experience_normalized_title ON experience(normalized_title)",
            "CREATE INDEX IF NOT EXISTS idx_skills_skill ON skills(skill)",
            
            # Full text search indexes
            "CREATE INDEX IF NOT EXISTS idx_alumni_name_gin ON alumni USING gin(to_tsvector('english', name))",
            "CREATE INDEX IF NOT EXISTS idx_alumni_title_gin ON alumni USING gin(to_tsvector('english', current_title))",
            "CREATE INDEX IF NOT EXISTS idx_experience_desc_gin ON experience USING gin(to_tsvector('english', description))",
        ]
        
        with self.connect_db() as conn:
            with conn.cursor() as cur:
                for statement in create_statements:
                    cur.execute(statement)
            conn.commit()
            
        logger.info("Database tables created successfully")
    
    def parse_raw_record(self, raw_data: Dict[str, Any]) -> AlumniRecord:
        """Parse raw data into structured AlumniRecord"""
        record = AlumniRecord(
            person_id=raw_data.get('id', str(hash(raw_data.get('name', '')))),
            name=raw_data.get('name', ''),
            current_title=raw_data.get('current_title'),
            current_company=raw_data.get('current_company'),
            location=raw_data.get('location'),
            graduation_year=self._parse_year(raw_data.get('graduation_year')),
            degree=raw_data.get('degree'),
            major=raw_data.get('major'),
            experience=raw_data.get('experience', []),
            education=raw_data.get('education', []),
            skills=raw_data.get('skills', [])
        )
        
        # Normalize fields
        record.normalized_company = self.company_normalizer.normalize(record.current_company or '')
        record.normalized_title = self._normalize_title(record.current_title or '')
        record.normalized_location = self.location_normalizer.normalize(record.location or '')
        record.record_hash = record.generate_hash()
        
        return record
    
    def _parse_year(self, year_str: Any) -> Optional[int]:
        """Parse year from various formats"""
        if isinstance(year_str, int):
            return year_str
        if isinstance(year_str, str):
            # Extract 4-digit year
            match = re.search(r'\b(19|20)\d{2}\b', year_str)
            if match:
                return int(match.group())
        return None
    
    def _normalize_title(self, title: str) -> str:
        """Basic title normalization"""
        if not title:
            return ''
        
        # Remove common variations
        title = re.sub(r'\s+', ' ', title.lower().strip())
        title = re.sub(r'\s*[-–—]\s*', ' - ', title)  # Normalize dashes
        
        return title
    
    def deduplicate_records(self, records: List[AlumniRecord]) -> List[AlumniRecord]:
        """Deduplicate records using fuzzy matching"""
        deduped = []
        
        for record in records:
            if record.record_hash in self.dedup_cache:
                continue
                
            # Check for fuzzy duplicates
            is_duplicate = False
            for existing in deduped:
                # Fuzzy match on name + company combination
                name_similarity = fuzz.ratio(record.name.lower(), existing.name.lower())
                company_similarity = fuzz.ratio(
                    record.normalized_company or '', 
                    existing.normalized_company or ''
                )
                
                if name_similarity > 90 and company_similarity > 85:
                    is_duplicate = True
                    logger.debug(f"Duplicate found: {record.name} at {record.current_company}")
                    break
            
            if not is_duplicate:
                deduped.append(record)
                self.dedup_cache.add(record.record_hash)
        
        logger.info(f"Deduplication: {len(records)} -> {len(deduped)} records")
        return deduped
    
    def batch_insert_alumni(self, records: List[AlumniRecord], batch_size: int = 1000):
        """Batch insert alumni records with related data"""
        with self.connect_db() as conn:
            with conn.cursor() as cur:
                # Insert alumni records
                alumni_data = [
                    (r.person_id, r.name, r.current_title, r.current_company,
                     r.normalized_company, r.normalized_title, r.location,
                     r.normalized_location, r.graduation_year, r.degree,
                     r.major, r.record_hash)
                    for r in records
                ]
                
                execute_batch(cur, """
                    INSERT INTO alumni (person_id, name, current_title, current_company,
                                      normalized_company, normalized_title, location,
                                      normalized_location, graduation_year, degree,
                                      major, record_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (record_hash) DO UPDATE
                    SET updated_at = CURRENT_TIMESTAMP
                """, alumni_data, page_size=batch_size)
                
                # Insert experience records
                experience_data = []
                for record in records:
                    for exp in record.experience:
                        exp_company = exp.get('company', '')
                        experience_data.append((
                            record.person_id,
                            exp_company,
                            self.company_normalizer.normalize(exp_company),
                            exp.get('title', ''),
                            self._normalize_title(exp.get('title', '')),
                            exp.get('location', ''),
                            exp.get('start_date'),
                            exp.get('end_date'),
                            exp.get('description', ''),
                            exp.get('is_current', False)
                        ))
                
                if experience_data:
                    execute_batch(cur, """
                        INSERT INTO experience (person_id, company, normalized_company,
                                              title, normalized_title, location,
                                              start_date, end_date, description, is_current)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, experience_data, page_size=batch_size)
                
                # Insert skills
                skills_data = []
                for record in records:
                    for skill in record.skills:
                        skills_data.append((record.person_id, skill))
                
                if skills_data:
                    execute_batch(cur, """
                        INSERT INTO skills (person_id, skill)
                        VALUES (%s, %s)
                    """, skills_data, page_size=batch_size)
                
            conn.commit()
        
        logger.info(f"Inserted {len(records)} alumni records with related data")
    
    def run_data_quality_checks(self) -> Dict[str, Any]:
        """Run data quality checks and return statistics"""
        with self.connect_db() as conn:
            with conn.cursor() as cur:
                checks = {}
                
                # Total records
                cur.execute("SELECT COUNT(*) FROM alumni")
                checks['total_records'] = cur.fetchone()[0]
                
                # Missing current company
                cur.execute("SELECT COUNT(*) FROM alumni WHERE current_company IS NULL OR current_company = ''")
                checks['missing_company'] = cur.fetchone()[0]
                
                # Missing title
                cur.execute("SELECT COUNT(*) FROM alumni WHERE current_title IS NULL OR current_title = ''")
                checks['missing_title'] = cur.fetchone()[0]
                
                # Records with experience
                cur.execute("SELECT COUNT(DISTINCT person_id) FROM experience")
                checks['records_with_experience'] = cur.fetchone()[0]
                
                # Top companies
                cur.execute("""
                    SELECT normalized_company, COUNT(*) as count
                    FROM alumni
                    WHERE normalized_company IS NOT NULL AND normalized_company != ''
                    GROUP BY normalized_company
                    ORDER BY count DESC
                    LIMIT 10
                """)
                checks['top_companies'] = cur.fetchall()
                
                # Year distribution
                cur.execute("""
                    SELECT graduation_year, COUNT(*) as count
                    FROM alumni
                    WHERE graduation_year IS NOT NULL
                    GROUP BY graduation_year
                    ORDER BY graduation_year DESC
                    LIMIT 10
                """)
                checks['year_distribution'] = cur.fetchall()
        
        return checks
    
    def process_data_files(self, file_paths: List[str]):
        """Process multiple data files"""
        all_records = []
        
        for file_path in file_paths:
            logger.info(f"Processing file: {file_path}")
            
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                continue
            
            # Convert DataFrame to records
            raw_records = df.to_dict('records')
            
            # Parse records
            parsed_records = [self.parse_raw_record(r) for r in raw_records]
            all_records.extend(parsed_records)
        
        # Deduplicate all records
        unique_records = self.deduplicate_records(all_records)
        
        # Batch insert
        self.batch_insert_alumni(unique_records)
        
        # Run quality checks
        quality_report = self.run_data_quality_checks()
        logger.info(f"Data quality report: {json.dumps(quality_report, indent=2)}")
        
        return quality_report


# Example usage
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'yale_alumni',
        'user': 'postgres',
        'password': 'password',
        'port': 5432
    }
    
    # Initialize foundation
    foundation = DataFoundation(db_config)
    
    # Create tables
    foundation.create_tables()
    
    # Process data files
    # foundation.process_data_files(['path/to/yale_data.csv', 'path/to/more_data.xlsx'])