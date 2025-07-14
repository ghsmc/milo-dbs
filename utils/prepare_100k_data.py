#!/usr/bin/env python3
"""
100K Alumni Data Preparation and Cleaning Pipeline
=================================================

This script prepares 100K alumni records for AWS processing by:
1. Cleaning and normalizing raw data
2. Splitting into manageable batches 
3. Creating embeddings preparation tasks
4. Setting up data quality validation
"""

import pandas as pd
import numpy as np
import boto3
import json
import os
from datetime import datetime
from pathlib import Path
import hashlib
import re
from typing import List, Dict, Any

class AlumniDataPreparation:
    def __init__(self, input_file: str, batch_size: int = 1000):
        self.input_file = input_file
        self.batch_size = batch_size
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'yale-alumni-processing-2025'
        
        # Data quality thresholds
        self.quality_thresholds = {
            'name_completeness': 0.95,
            'title_completeness': 0.85, 
            'company_completeness': 0.80,
            'location_completeness': 0.75,
            'duplicate_threshold': 0.02
        }
        
        # Create output directories
        self.output_dir = Path('processed_100k')
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'batches').mkdir(exist_ok=True)
        (self.output_dir / 'embeddings').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)

    def load_and_analyze_data(self) -> pd.DataFrame:
        """Load 100K records and perform initial analysis"""
        print("📊 Loading and analyzing 100K alumni records...")
        
        # Determine file type and load appropriately
        if self.input_file.endswith('.xlsx'):
            df = pd.read_excel(self.input_file)
        elif self.input_file.endswith('.csv'):
            df = pd.read_csv(self.input_file)
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .csv")
        
        print(f"✓ Loaded {len(df)} records")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Generate data quality report
        quality_report = {
            'total_records': len(df),
            'columns': list(df.columns),
            'missing_data': df.isnull().sum().to_dict(),
            'completeness_rates': (1 - df.isnull().sum() / len(df)).to_dict(),
            'data_types': df.dtypes.to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'unique_values': {col: df[col].nunique() for col in df.columns}
        }
        
        # Save quality report
        with open(self.output_dir / 'reports' / 'initial_quality_report.json', 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        print(f"✓ Quality report saved: {quality_report['total_records']} records, {quality_report['duplicate_rows']} duplicates")
        
        return df

    def clean_and_normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize the alumni data"""
        print("🧹 Cleaning and normalizing data...")
        
        cleaned_df = df.copy()
        
        # 1. Standardize column names
        column_mapping = {
            'Full Name': 'name',
            'Name': 'name', 
            'Current Title': 'title',
            'Title': 'title',
            'Current Company': 'company',
            'Company': 'company',
            'Location': 'location',
            'City': 'location',
            'Email': 'email',
            'LinkedIn': 'linkedin_url',
            'Year': 'graduation_year',
            'Class Year': 'graduation_year',
            'School': 'school',
            'Degree': 'degree',
            'Major': 'major'
        }
        
        # Apply column mapping
        for old_name, new_name in column_mapping.items():
            if old_name in cleaned_df.columns:
                cleaned_df = cleaned_df.rename(columns={old_name: new_name})
        
        # 2. Clean names
        if 'name' in cleaned_df.columns:
            cleaned_df['name'] = cleaned_df['name'].astype(str).str.strip()
            cleaned_df['name'] = cleaned_df['name'].str.title()
            # Remove extra whitespace
            cleaned_df['name'] = cleaned_df['name'].str.replace(r'\s+', ' ', regex=True)
        
        # 3. Clean titles
        if 'title' in cleaned_df.columns:
            cleaned_df['title'] = cleaned_df['title'].astype(str).str.strip()
            # Standardize common title variations
            title_standardization = {
                r'Software Engineer.*': 'Software Engineer',
                r'Data Scientist.*': 'Data Scientist', 
                r'Product Manager.*': 'Product Manager',
                r'Investment Banking.*': 'Investment Banking Analyst',
                r'Consultant.*': 'Consultant',
                r'Vice President.*': 'Vice President',
                r'Director.*': 'Director',
                r'Senior.*': lambda x: f"Senior {x.split('Senior')[-1].strip()}"
            }
            
            for pattern, replacement in title_standardization.items():
                if callable(replacement):
                    mask = cleaned_df['title'].str.contains(pattern, case=False, na=False)
                    cleaned_df.loc[mask, 'title'] = cleaned_df.loc[mask, 'title'].apply(replacement)
                else:
                    cleaned_df['title'] = cleaned_df['title'].str.replace(pattern, replacement, regex=True, case=False)
        
        # 4. Clean companies
        if 'company' in cleaned_df.columns:
            cleaned_df['company'] = cleaned_df['company'].astype(str).str.strip()
            # Standardize company names
            company_standardization = {
                r'Goldman Sachs.*': 'Goldman Sachs',
                r'Morgan Stanley.*': 'Morgan Stanley', 
                r'McKinsey.*': 'McKinsey & Company',
                r'Boston Consulting.*': 'Boston Consulting Group',
                r'Deloitte.*': 'Deloitte',
                r'PwC.*': 'PwC',
                r'Ernst.*Young.*': 'EY',
                r'Google.*': 'Google',
                r'Microsoft.*': 'Microsoft',
                r'Amazon.*': 'Amazon'
            }
            
            for pattern, replacement in company_standardization.items():
                cleaned_df['company'] = cleaned_df['company'].str.replace(pattern, replacement, regex=True, case=False)
        
        # 5. Clean locations
        if 'location' in cleaned_df.columns:
            cleaned_df['location'] = cleaned_df['location'].astype(str).str.strip()
            # Extract city, state format
            location_pattern = r'([^,]+),\s*([A-Z]{2})'
            cleaned_df['city'] = cleaned_df['location'].str.extract(location_pattern)[0]
            cleaned_df['state'] = cleaned_df['location'].str.extract(location_pattern)[1]
        
        # 6. Add derived fields
        cleaned_df['record_id'] = range(1, len(cleaned_df) + 1)
        cleaned_df['created_at'] = datetime.utcnow().isoformat()
        cleaned_df['data_source'] = '100k_alumni_dataset'
        
        # Extract role type from title
        if 'title' in cleaned_df.columns:
            role_mapping = {
                r'engineer': 'engineering',
                r'scientist': 'data_science',
                r'analyst': 'finance',
                r'consultant': 'consulting',
                r'manager': 'management',
                r'director': 'management',
                r'vice president': 'executive',
                r'president': 'executive',
                r'ceo': 'executive',
                r'founder': 'entrepreneurship'
            }
            
            cleaned_df['role_type'] = 'other'
            for pattern, role in role_mapping.items():
                mask = cleaned_df['title'].str.contains(pattern, case=False, na=False)
                cleaned_df.loc[mask, 'role_type'] = role
        
        # Extract seniority level
        if 'title' in cleaned_df.columns:
            seniority_mapping = {
                r'senior|sr\.': 'senior',
                r'principal|lead': 'senior',
                r'junior|jr\.|entry': 'junior',
                r'associate': 'entry',
                r'director|vp|vice president': 'executive',
                r'analyst': 'entry'
            }
            
            cleaned_df['seniority'] = 'mid'
            for pattern, level in seniority_mapping.items():
                mask = cleaned_df['title'].str.contains(pattern, case=False, na=False)
                cleaned_df.loc[mask, 'seniority'] = level
        
        print(f"✓ Cleaned data: {len(cleaned_df)} records")
        return cleaned_df

    def deduplicate_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records using fuzzy matching"""
        print("🔍 Deduplicating records...")
        
        # Create composite key for exact duplicates
        if all(col in df.columns for col in ['name', 'company', 'title']):
            df['composite_key'] = (df['name'].astype(str) + '|' + 
                                 df['company'].astype(str) + '|' + 
                                 df['title'].astype(str)).str.lower()
            
            # Remove exact duplicates
            before_count = len(df)
            df = df.drop_duplicates(subset=['composite_key'], keep='first')
            exact_duplicates_removed = before_count - len(df)
            
            print(f"✓ Removed {exact_duplicates_removed} exact duplicates")
        
        # Fuzzy deduplication for similar records
        try:
            from fuzzywuzzy import fuzz
            
            duplicates_to_remove = []
            
            # Group by similar names for efficiency
            name_groups = df.groupby(df['name'].str[:3].str.upper() if 'name' in df.columns else df.index)
            
            for _, group in name_groups:
                if len(group) > 1:
                    for i, row1 in group.iterrows():
                        for j, row2 in group.iterrows():
                            if i >= j:  # Avoid duplicate comparisons
                                continue
                            
                            # Calculate similarity scores
                            name_sim = fuzz.ratio(str(row1.get('name', '')), str(row2.get('name', '')))
                            company_sim = fuzz.ratio(str(row1.get('company', '')), str(row2.get('company', '')))
                            
                            # If very similar, mark as duplicate
                            if name_sim > 85 and company_sim > 80:
                                duplicates_to_remove.append(j)
            
            # Remove fuzzy duplicates
            df = df.drop(index=duplicates_to_remove)
            fuzzy_duplicates_removed = len(duplicates_to_remove)
            
            print(f"✓ Removed {fuzzy_duplicates_removed} fuzzy duplicates")
            
        except ImportError:
            print("⚠️  fuzzywuzzy not available, skipping fuzzy deduplication")
        
        return df

    def create_embeddings_preparation(self, df: pd.DataFrame):
        """Prepare data for embedding generation on AWS"""
        print("🧠 Preparing embedding generation tasks...")
        
        # Create text fields for different embedding aspects
        embedding_tasks = []
        
        for idx, row in df.iterrows():
            # Multi-aspect text preparation
            aspects = {
                'profile_summary': self._create_profile_summary(row),
                'role_experience': self._create_role_experience_text(row),
                'skills_keywords': self._extract_skills_keywords(row),
                'career_trajectory': self._create_career_trajectory(row),
                'location_context': self._create_location_context(row)
            }
            
            task = {
                'record_id': row.get('record_id', idx),
                'name': row.get('name'),
                'aspects': aspects,
                'priority': self._calculate_embedding_priority(row),
                'batch_group': idx // self.batch_size
            }
            
            embedding_tasks.append(task)
        
        # Save embedding tasks by priority
        high_priority = [t for t in embedding_tasks if t['priority'] == 'high']
        medium_priority = [t for t in embedding_tasks if t['priority'] == 'medium'] 
        low_priority = [t for t in embedding_tasks if t['priority'] == 'low']
        
        for priority, tasks in [('high', high_priority), ('medium', medium_priority), ('low', low_priority)]:
            filename = self.output_dir / 'embeddings' / f'{priority}_priority_embeddings.json'
            with open(filename, 'w') as f:
                json.dump(tasks, f, indent=2)
            
            print(f"✓ Created {len(tasks)} {priority} priority embedding tasks")
        
        return embedding_tasks

    def _create_profile_summary(self, row: Dict) -> str:
        """Create comprehensive profile summary for embeddings"""
        parts = []
        
        if row.get('name'):
            parts.append(f"Name: {row['name']}")
        if row.get('title'):
            parts.append(f"Current Role: {row['title']}")
        if row.get('company'):
            parts.append(f"Company: {row['company']}")
        if row.get('location'):
            parts.append(f"Location: {row['location']}")
        if row.get('school'):
            parts.append(f"Education: {row['school']}")
        if row.get('degree'):
            parts.append(f"Degree: {row['degree']}")
        if row.get('major'):
            parts.append(f"Major: {row['major']}")
        
        return ". ".join(parts)

    def _create_role_experience_text(self, row: Dict) -> str:
        """Create role and experience focused text"""
        parts = []
        
        if row.get('title'):
            parts.append(row['title'])
        if row.get('role_type'):
            parts.append(f"in {row['role_type']}")
        if row.get('seniority'):
            parts.append(f"at {row['seniority']} level")
        if row.get('company'):
            parts.append(f"at {row['company']}")
        
        return " ".join(parts)

    def _extract_skills_keywords(self, row: Dict) -> str:
        """Extract and standardize skills keywords"""
        skills = []
        
        # Extract from title
        title = str(row.get('title', '')).lower()
        skill_keywords = {
            'software': ['python', 'java', 'javascript', 'programming', 'development'],
            'data': ['machine learning', 'analytics', 'sql', 'statistics', 'modeling'],
            'finance': ['financial modeling', 'valuation', 'investment', 'trading', 'risk'],
            'consulting': ['strategy', 'operations', 'transformation', 'advisory'],
            'product': ['product management', 'roadmap', 'user experience', 'agile'],
            'marketing': ['digital marketing', 'brand', 'campaigns', 'growth']
        }
        
        for category, keywords in skill_keywords.items():
            if any(keyword in title for keyword in keywords):
                skills.extend(keywords[:3])  # Add top 3 relevant skills
        
        return ", ".join(skills[:10])  # Limit to top 10 skills

    def _create_career_trajectory(self, row: Dict) -> str:
        """Create career trajectory text"""
        parts = []
        
        if row.get('graduation_year'):
            parts.append(f"Graduated {row['graduation_year']}")
        if row.get('seniority'):
            parts.append(f"Currently {row['seniority']} level")
        if row.get('role_type'):
            parts.append(f"in {row['role_type']} field")
        
        return ". ".join(parts)

    def _create_location_context(self, row: Dict) -> str:
        """Create location context for geographic embeddings"""
        parts = []
        
        if row.get('city'):
            parts.append(f"Based in {row['city']}")
        if row.get('state'):
            parts.append(f"{row['state']}")
        
        # Add location context
        location_context = {
            'San Francisco': 'tech hub',
            'New York': 'financial center', 
            'Boston': 'biotech and education',
            'Seattle': 'tech and aerospace',
            'Chicago': 'financial and consulting',
            'Los Angeles': 'entertainment and tech'
        }
        
        city = row.get('city', '')
        if city in location_context:
            parts.append(f"in {location_context[city]}")
        
        return " ".join(parts)

    def _calculate_embedding_priority(self, row: Dict) -> str:
        """Calculate embedding generation priority"""
        score = 0
        
        # Complete profiles get higher priority
        if row.get('name') and row.get('title') and row.get('company'):
            score += 3
        
        # Strategic roles get higher priority
        strategic_roles = ['ceo', 'founder', 'vp', 'director', 'partner', 'principal']
        title = str(row.get('title', '')).lower()
        if any(role in title for role in strategic_roles):
            score += 2
        
        # High-value companies get higher priority
        high_value_companies = ['google', 'microsoft', 'amazon', 'apple', 'goldman sachs', 'mckinsey']
        company = str(row.get('company', '')).lower()
        if any(comp in company for comp in high_value_companies):
            score += 2
        
        if score >= 5:
            return 'high'
        elif score >= 3:
            return 'medium'
        else:
            return 'low'

    def split_into_batches(self, df: pd.DataFrame) -> List[str]:
        """Split data into processing batches"""
        print(f"📦 Splitting {len(df)} records into batches of {self.batch_size}...")
        
        batch_files = []
        total_batches = (len(df) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(df), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch_df = df.iloc[i:i + self.batch_size].copy()
            
            # Add batch metadata
            batch_df['batch_number'] = batch_num
            batch_df['batch_size'] = len(batch_df)
            batch_df['total_batches'] = total_batches
            
            # Save batch file
            batch_filename = self.output_dir / 'batches' / f'batch_{batch_num:03d}.csv'
            batch_df.to_csv(batch_filename, index=False)
            batch_files.append(str(batch_filename))
            
            print(f"✓ Created batch {batch_num}/{total_batches}: {len(batch_df)} records")
        
        return batch_files

    def upload_to_s3(self, batch_files: List[str]):
        """Upload batch files to S3 for AWS processing"""
        print(f"☁️  Uploading {len(batch_files)} batches to S3...")
        
        for batch_file in batch_files:
            try:
                key = f"raw_batches/{Path(batch_file).name}"
                self.s3_client.upload_file(batch_file, self.bucket_name, key)
                print(f"✓ Uploaded {Path(batch_file).name}")
                
            except Exception as e:
                print(f"✗ Failed to upload {batch_file}: {e}")

    def generate_processing_manifest(self, df: pd.DataFrame, batch_files: List[str]):
        """Generate manifest for AWS processing orchestration"""
        print("📋 Generating processing manifest...")
        
        manifest = {
            'processing_job': {
                'job_id': f"yale_alumni_100k_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'created_at': datetime.utcnow().isoformat(),
                'total_records': len(df),
                'total_batches': len(batch_files),
                'batch_size': self.batch_size
            },
            'data_quality': {
                'completeness_check': True,
                'deduplication_check': True,
                'validation_rules': self.quality_thresholds
            },
            'processing_steps': [
                {
                    'step': 1,
                    'name': 'entity_extraction',
                    'description': 'Extract entities from alumni profiles',
                    'priority': 'high'
                },
                {
                    'step': 2, 
                    'name': 'embedding_generation',
                    'description': 'Generate multi-aspect embeddings',
                    'priority': 'high'
                },
                {
                    'step': 3,
                    'name': 'relationship_mapping', 
                    'description': 'Build relationship graphs',
                    'priority': 'medium'
                },
                {
                    'step': 4,
                    'name': 'search_indexing',
                    'description': 'Build search indexes',
                    'priority': 'medium'
                }
            ],
            'aws_resources': {
                'ec2_instances': 3,
                'expected_runtime_hours': 2,
                'estimated_cost': '$50-100'
            },
            'batch_files': [Path(f).name for f in batch_files]
        }
        
        manifest_file = self.output_dir / 'processing_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        print(f"✓ Processing manifest saved: {manifest_file}")
        return manifest

def main():
    """Main data preparation function"""
    print("🚀 Starting 100K Alumni Data Preparation Pipeline")
    print("=" * 60)
    
    # Initialize data preparation
    # Note: Replace with actual 100K data file path
    input_file = "alumni_100k_raw.xlsx"  # or .csv
    
    if not os.path.exists(input_file):
        print(f"⚠️  Input file not found: {input_file}")
        print("Creating sample 100K dataset for demonstration...")
        
        # Create sample data for testing
        sample_data = []
        for i in range(100000):
            record = {
                'name': f'Alumni {i+1}',
                'title': np.random.choice(['Software Engineer', 'Data Scientist', 'Investment Banking Analyst', 'Consultant', 'Product Manager']),
                'company': np.random.choice(['Google', 'Goldman Sachs', 'McKinsey', 'Microsoft', 'Meta']),
                'location': np.random.choice(['San Francisco, CA', 'New York, NY', 'Boston, MA', 'Seattle, WA']),
                'graduation_year': np.random.randint(2000, 2024),
                'school': 'Yale University',
                'degree': np.random.choice(['BA', 'BS', 'MBA', 'JD', 'MD'])
            }
            sample_data.append(record)
        
        sample_df = pd.DataFrame(sample_data)
        input_file = 'sample_alumni_100k.csv'
        sample_df.to_csv(input_file, index=False)
        print(f"✓ Created sample dataset: {input_file}")
    
    prep = AlumniDataPreparation(input_file, batch_size=1000)
    
    # Step 1: Load and analyze
    print("\n1. Loading and analyzing data...")
    df = prep.load_and_analyze_data()
    
    # Step 2: Clean and normalize
    print("\n2. Cleaning and normalizing...")
    cleaned_df = prep.clean_and_normalize_data(df)
    
    # Step 3: Deduplicate
    print("\n3. Deduplicating records...")
    deduped_df = prep.deduplicate_records(cleaned_df)
    
    # Step 4: Create embeddings preparation
    print("\n4. Preparing embedding tasks...")
    embedding_tasks = prep.create_embeddings_preparation(deduped_df)
    
    # Step 5: Split into batches
    print("\n5. Creating processing batches...")
    batch_files = prep.split_into_batches(deduped_df)
    
    # Step 6: Upload to S3 (optional)
    # print("\n6. Uploading to S3...")
    # prep.upload_to_s3(batch_files)
    
    # Step 7: Generate manifest
    print("\n6. Generating processing manifest...")
    manifest = prep.generate_processing_manifest(deduped_df, batch_files)
    
    print(f"\n✅ Data preparation completed!")
    print(f"📊 Final dataset: {len(deduped_df)} records")
    print(f"📦 Created {len(batch_files)} processing batches")
    print(f"🧠 Prepared {len(embedding_tasks)} embedding tasks")
    print(f"📋 Ready for AWS processing with manifest")

if __name__ == "__main__":
    main()