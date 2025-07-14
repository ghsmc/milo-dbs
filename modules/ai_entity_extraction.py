"""
AI-Enhanced Entity Extraction Module
Implements hybrid approach: rules first, AI fallback for low confidence cases
Addresses feedback #2: Use realtime AI for better field extraction
"""

import json
import logging
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch

# Try to import OpenAI for AI extraction
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Install with: pip install openai")

# Try to import transformers for local models
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers")

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of entity extraction with confidence score"""
    normalized_title: str
    seniority: Optional[str]
    role_type: Optional[str]
    specialization: Optional[str]
    department: Optional[str]
    industry_focus: Optional[str]
    confidence: float  # 0.0 to 1.0
    extraction_method: str  # 'rules' or 'ai'
    raw_extractions: Dict[str, Any] = None


@dataclass
class AIExtractionRequest:
    """Request for AI-based extraction"""
    text: str
    context: str  # 'title', 'company', 'description'
    person_id: str


class RuleBasedExtractor:
    """Enhanced rule-based extractor from original system"""
    
    def __init__(self):
        # Seniority levels (ordered by hierarchy)
        self.seniority_patterns = {
            'intern': ['intern', 'internship', 'co-op', 'coop'],
            'entry': ['associate', 'junior', 'entry', 'entry-level', 'trainee', 'graduate'],
            'mid': ['analyst', 'specialist', 'coordinator', 'representative'],
            'senior': ['senior', 'sr', 'lead', 'principal', 'staff'],
            'management': ['manager', 'mgr', 'supervisor', 'head', 'team lead', 'team leader'],
            'director': ['director', 'dir', 'vp', 'vice president', 'executive'],
            'c-level': ['ceo', 'cto', 'cfo', 'coo', 'chief', 'president', 'founder', 'co-founder']
        }
        
        # Role types
        self.role_patterns = {
            'engineering': ['engineer', 'developer', 'programmer', 'architect', 'devops', 'sre'],
            'product': ['product manager', 'product owner', 'pm', 'product analyst'],
            'design': ['designer', 'ux', 'ui', 'creative', 'art director'],
            'data': ['data scientist', 'data analyst', 'data engineer', 'ml engineer'],
            'sales': ['sales', 'account manager', 'business development', 'bd'],
            'marketing': ['marketing', 'brand', 'content', 'social media', 'growth'],
            'operations': ['operations', 'ops', 'program manager', 'project manager'],
            'finance': ['analyst', 'associate', 'banker', 'trader', 'risk'],
            'consulting': ['consultant', 'advisor', 'strategist'],
            'research': ['researcher', 'scientist', 'analyst', 'associate researcher']
        }
        
        # Specializations (more specific)
        self.specialization_patterns = {
            'backend': ['backend', 'back-end', 'server', 'api', 'microservices'],
            'frontend': ['frontend', 'front-end', 'ui', 'react', 'angular', 'vue'],
            'fullstack': ['fullstack', 'full-stack', 'full stack'],
            'mobile': ['mobile', 'ios', 'android', 'react native', 'flutter'],
            'machine_learning': ['machine learning', 'ml', 'ai', 'deep learning', 'nlp'],
            'data_science': ['data science', 'analytics', 'statistics', 'modeling'],
            'devops': ['devops', 'infrastructure', 'cloud', 'aws', 'kubernetes'],
            'security': ['security', 'cybersecurity', 'infosec', 'penetration'],
            'blockchain': ['blockchain', 'crypto', 'web3', 'defi', 'nft']
        }
        
        # Departments
        self.department_patterns = {
            'engineering': ['engineering', 'technology', 'tech', 'development'],
            'product': ['product', 'product management'],
            'design': ['design', 'creative', 'user experience'],
            'data': ['data', 'analytics', 'business intelligence'],
            'sales': ['sales', 'revenue', 'business development'],
            'marketing': ['marketing', 'brand', 'communications'],
            'operations': ['operations', 'ops', 'program management'],
            'finance': ['finance', 'accounting', 'treasury', 'fp&a'],
            'hr': ['human resources', 'hr', 'people', 'talent'],
            'legal': ['legal', 'compliance', 'regulatory']
        }
        
        # Industry focus
        self.industry_patterns = {
            'fintech': ['fintech', 'financial technology', 'payments', 'banking'],
            'healthcare': ['healthcare', 'medical', 'biotech', 'pharma'],
            'ecommerce': ['ecommerce', 'e-commerce', 'retail', 'marketplace'],
            'saas': ['saas', 'software as a service', 'b2b software'],
            'consumer': ['consumer', 'b2c', 'social media', 'entertainment'],
            'enterprise': ['enterprise', 'b2b', 'business software'],
            'gaming': ['gaming', 'games', 'entertainment', 'mobile games'],
            'edtech': ['edtech', 'education', 'learning', 'training'],
            'proptech': ['proptech', 'real estate', 'property'],
            'logistics': ['logistics', 'supply chain', 'transportation']
        }
    
    def extract_entities(self, title: str, company: str = None) -> ExtractionResult:
        """Extract entities using rule-based patterns"""
        title_lower = title.lower().strip()
        
        # Extract seniority
        seniority = self._extract_seniority(title_lower)
        
        # Extract role type
        role_type = self._extract_role_type(title_lower)
        
        # Extract specialization
        specialization = self._extract_specialization(title_lower)
        
        # Extract department
        department = self._extract_department(title_lower)
        
        # Extract industry focus (from company if available)
        industry_focus = None
        if company:
            industry_focus = self._extract_industry(company.lower())
        
        # Normalize title (clean up)
        normalized_title = self._normalize_title(title_lower)
        
        # Calculate confidence based on matches
        confidence = self._calculate_confidence(
            seniority, role_type, specialization, department, industry_focus
        )
        
        return ExtractionResult(
            normalized_title=normalized_title,
            seniority=seniority,
            role_type=role_type,
            specialization=specialization,
            department=department,
            industry_focus=industry_focus,
            confidence=confidence,
            extraction_method='rules'
        )
    
    def _extract_seniority(self, title: str) -> Optional[str]:
        """Extract seniority level from title"""
        for level, patterns in self.seniority_patterns.items():
            for pattern in patterns:
                if pattern in title:
                    return level
        return None
    
    def _extract_role_type(self, title: str) -> Optional[str]:
        """Extract role type from title"""
        for role, patterns in self.role_patterns.items():
            for pattern in patterns:
                if pattern in title:
                    return role
        return None
    
    def _extract_specialization(self, title: str) -> Optional[str]:
        """Extract specialization from title"""
        for spec, patterns in self.specialization_patterns.items():
            for pattern in patterns:
                if pattern in title:
                    return spec
        return None
    
    def _extract_department(self, title: str) -> Optional[str]:
        """Extract department from title"""
        for dept, patterns in self.department_patterns.items():
            for pattern in patterns:
                if pattern in title:
                    return dept
        return None
    
    def _extract_industry(self, company: str) -> Optional[str]:
        """Extract industry focus from company name"""
        for industry, patterns in self.industry_patterns.items():
            for pattern in patterns:
                if pattern in company:
                    return industry
        return None
    
    def _normalize_title(self, title: str) -> str:
        """Normalize and clean title"""
        # Remove common prefixes/suffixes
        title = re.sub(r'\b(the|a|an)\b', '', title)
        title = re.sub(r'\s+', ' ', title)  # Multiple spaces
        title = title.strip()
        
        # Expand common abbreviations
        abbreviations = {
            'sr': 'senior',
            'jr': 'junior',
            'mgr': 'manager',
            'dir': 'director',
            'vp': 'vice president',
            'swe': 'software engineer',
            'pm': 'product manager',
            'qa': 'quality assurance'
        }
        
        for abbrev, full in abbreviations.items():
            title = re.sub(rf'\b{abbrev}\b', full, title)
        
        return title
    
    def _calculate_confidence(self, seniority, role_type, specialization, 
                            department, industry_focus) -> float:
        """Calculate confidence score based on extracted entities"""
        score = 0.0
        total_weight = 0.0
        
        # Weight different extractions
        weights = {
            'seniority': 0.2,
            'role_type': 0.3,
            'specialization': 0.2,
            'department': 0.2,
            'industry_focus': 0.1
        }
        
        extractions = {
            'seniority': seniority,
            'role_type': role_type,
            'specialization': specialization,
            'department': department,
            'industry_focus': industry_focus
        }
        
        for field, value in extractions.items():
            weight = weights[field]
            total_weight += weight
            if value:
                score += weight
        
        # Normalize by total possible weight
        return score / total_weight if total_weight > 0 else 0.0


class AIExtractor:
    """AI-based entity extraction for fallback cases"""
    
    def __init__(self, use_openai: bool = True, openai_api_key: str = None):
        self.use_openai = use_openai and OPENAI_AVAILABLE
        self.use_local_model = TRANSFORMERS_AVAILABLE
        
        if self.use_openai and openai_api_key:
            openai.api_key = openai_api_key
        
        # Initialize local NER model if available
        self.ner_pipeline = None
        if self.use_local_model:
            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple"
                )
            except Exception as e:
                logger.warning(f"Failed to load local NER model: {e}")
                self.use_local_model = False
    
    def extract_entities(self, request: AIExtractionRequest) -> ExtractionResult:
        """Extract entities using AI"""
        if self.use_openai:
            return self._extract_with_openai(request)
        elif self.use_local_model:
            return self._extract_with_local_model(request)
        else:
            # Fallback: return empty result with low confidence
            return ExtractionResult(
                normalized_title=request.text,
                seniority=None,
                role_type=None,
                specialization=None,
                department=None,
                industry_focus=None,
                confidence=0.1,
                extraction_method='ai_fallback'
            )
    
    def _extract_with_openai(self, request: AIExtractionRequest) -> ExtractionResult:
        """Extract using OpenAI API"""
        try:
            prompt = self._build_extraction_prompt(request)
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured information from job titles and descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content
            return self._parse_ai_response(result_text, request.text)
            
        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
            return ExtractionResult(
                normalized_title=request.text,
                seniority=None,
                role_type=None,
                specialization=None,
                department=None,
                industry_focus=None,
                confidence=0.2,
                extraction_method='ai_error'
            )
    
    def _extract_with_local_model(self, request: AIExtractionRequest) -> ExtractionResult:
        """Extract using local transformer model"""
        try:
            # Use NER to identify entities
            entities = self.ner_pipeline(request.text)
            
            # Simple mapping - in production this would be more sophisticated
            role_type = None
            seniority = None
            
            # Look for person entities (might indicate roles)
            for entity in entities:
                if entity['entity_group'] == 'PER':
                    # This is a simple heuristic - real implementation would be more complex
                    text = entity['word'].lower()
                    if any(word in text for word in ['engineer', 'manager', 'analyst']):
                        role_type = 'engineering' if 'engineer' in text else 'management'
            
            return ExtractionResult(
                normalized_title=request.text,
                seniority=seniority,
                role_type=role_type,
                specialization=None,
                department=None,
                industry_focus=None,
                confidence=0.6,
                extraction_method='ai_local'
            )
            
        except Exception as e:
            logger.error(f"Local model extraction failed: {e}")
            return ExtractionResult(
                normalized_title=request.text,
                seniority=None,
                role_type=None,
                specialization=None,
                department=None,
                industry_focus=None,
                confidence=0.2,
                extraction_method='ai_local_error'
            )
    
    def _build_extraction_prompt(self, request: AIExtractionRequest) -> str:
        """Build prompt for AI extraction"""
        return f"""
Extract structured information from this job title: "{request.text}"

Please identify:
1. Seniority level (intern, entry, mid, senior, management, director, c-level)
2. Role type (engineering, product, design, data, sales, marketing, operations, finance, consulting, research)
3. Specialization (backend, frontend, fullstack, mobile, machine_learning, data_science, devops, security, etc.)
4. Department (engineering, product, design, data, sales, marketing, operations, finance, hr, legal)
5. Industry focus (if apparent from context)

Return as JSON:
{{"seniority": "...", "role_type": "...", "specialization": "...", "department": "...", "industry_focus": "..."}}

Use null for fields that cannot be determined.
"""
    
    def _parse_ai_response(self, response_text: str, original_title: str) -> ExtractionResult:
        """Parse AI response into structured result"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                return ExtractionResult(
                    normalized_title=original_title,
                    seniority=data.get('seniority'),
                    role_type=data.get('role_type'),
                    specialization=data.get('specialization'),
                    department=data.get('department'),
                    industry_focus=data.get('industry_focus'),
                    confidence=0.8,  # High confidence for AI extraction
                    extraction_method='ai_openai',
                    raw_extractions=data
                )
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            return ExtractionResult(
                normalized_title=original_title,
                seniority=None,
                role_type=None,
                specialization=None,
                department=None,
                industry_focus=None,
                confidence=0.3,
                extraction_method='ai_parse_error'
            )


class HybridEntityExtractor:
    """
    Hybrid extractor that combines rule-based and AI approaches
    Uses rules first, falls back to AI for low-confidence cases
    """
    
    def __init__(self, db_config: Dict[str, str], 
                 confidence_threshold: float = 0.7,
                 openai_api_key: str = None):
        self.db_config = db_config
        self.confidence_threshold = confidence_threshold
        
        # Initialize extractors
        self.rule_extractor = RuleBasedExtractor()
        self.ai_extractor = AIExtractor(openai_api_key=openai_api_key)
        
        # Statistics
        self.stats = {
            'total_extractions': 0,
            'rule_based_success': 0,
            'ai_fallback_used': 0,
            'low_confidence_results': 0
        }
    
    def extract_entities_hybrid(self, title: str, company: str = None, 
                              person_id: str = None) -> ExtractionResult:
        """
        Hybrid extraction: rules first, AI fallback for low confidence
        """
        self.stats['total_extractions'] += 1
        
        # Step 1: Try rule-based extraction
        rule_result = self.rule_extractor.extract_entities(title, company)
        
        # Step 2: Check if confidence is sufficient
        if rule_result.confidence >= self.confidence_threshold:
            self.stats['rule_based_success'] += 1
            return rule_result
        
        # Step 3: Use AI fallback for low confidence cases
        logger.info(f"Low confidence ({rule_result.confidence:.2f}) for '{title}', using AI fallback")
        
        ai_request = AIExtractionRequest(
            text=title,
            context='title',
            person_id=person_id or 'unknown'
        )
        
        ai_result = self.ai_extractor.extract_entities(ai_request)
        self.stats['ai_fallback_used'] += 1
        
        # Step 4: Merge results (prefer AI for new fields, keep rule-based for confirmed ones)
        merged_result = self._merge_extraction_results(rule_result, ai_result)
        
        if merged_result.confidence < 0.5:
            self.stats['low_confidence_results'] += 1
        
        return merged_result
    
    def _merge_extraction_results(self, rule_result: ExtractionResult, 
                                ai_result: ExtractionResult) -> ExtractionResult:
        """Merge rule-based and AI results intelligently"""
        
        # Use AI result as base, but keep high-confidence rule extractions
        merged = ExtractionResult(
            normalized_title=ai_result.normalized_title,
            seniority=ai_result.seniority or rule_result.seniority,
            role_type=ai_result.role_type or rule_result.role_type,
            specialization=ai_result.specialization or rule_result.specialization,
            department=ai_result.department or rule_result.department,
            industry_focus=ai_result.industry_focus or rule_result.industry_focus,
            confidence=max(rule_result.confidence, ai_result.confidence),
            extraction_method='hybrid',
            raw_extractions={
                'rule_result': rule_result.__dict__,
                'ai_result': ai_result.__dict__
            }
        )
        
        return merged
    
    def process_batch(self, profiles: List[Dict[str, Any]], 
                     batch_size: int = 100) -> List[ExtractionResult]:
        """Process a batch of profiles"""
        results = []
        
        for i, profile in enumerate(profiles):
            if i % batch_size == 0:
                logger.info(f"Processing batch {i//batch_size + 1}, profile {i+1}/{len(profiles)}")
            
            title = profile.get('current_title', '')
            company = profile.get('current_company', '')
            person_id = profile.get('person_id', '')
            
            if title:
                result = self.extract_entities_hybrid(title, company, person_id)
                results.append(result)
        
        return results
    
    def save_enhanced_extractions(self, results: List[ExtractionResult]):
        """Save enhanced extractions to database"""
        conn = psycopg2.connect(**self.db_config)
        
        try:
            with conn.cursor() as cur:
                # Create enhanced extractions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS enhanced_title_entities (
                        person_id VARCHAR(50) PRIMARY KEY,
                        normalized_title TEXT,
                        seniority VARCHAR(50),
                        role_type VARCHAR(50),
                        specialization VARCHAR(50),
                        department VARCHAR(50),
                        industry_focus VARCHAR(50),
                        confidence FLOAT,
                        extraction_method VARCHAR(20),
                        raw_extractions JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Prepare data for insertion
                data_to_insert = []
                for i, result in enumerate(results):
                    # We need person_id - for now, use index as placeholder
                    person_id = f"person_{i:06d}"  # This should come from actual data
                    
                    data_to_insert.append((
                        person_id,
                        result.normalized_title,
                        result.seniority,
                        result.role_type,
                        result.specialization,
                        result.department,
                        result.industry_focus,
                        result.confidence,
                        result.extraction_method,
                        json.dumps(result.raw_extractions) if result.raw_extractions else None
                    ))
                
                # Batch insert
                execute_batch(cur, """
                    INSERT INTO enhanced_title_entities 
                    (person_id, normalized_title, seniority, role_type, specialization, 
                     department, industry_focus, confidence, extraction_method, raw_extractions)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (person_id) DO UPDATE SET
                        normalized_title = EXCLUDED.normalized_title,
                        seniority = EXCLUDED.seniority,
                        role_type = EXCLUDED.role_type,
                        specialization = EXCLUDED.specialization,
                        department = EXCLUDED.department,
                        industry_focus = EXCLUDED.industry_focus,
                        confidence = EXCLUDED.confidence,
                        extraction_method = EXCLUDED.extraction_method,
                        raw_extractions = EXCLUDED.raw_extractions,
                        updated_at = CURRENT_TIMESTAMP
                """, data_to_insert)
                
                conn.commit()
                logger.info(f"Saved {len(results)} enhanced extractions to database")
                
        finally:
            conn.close()
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        if self.stats['total_extractions'] == 0:
            return self.stats
        
        return {
            **self.stats,
            'rule_success_rate': self.stats['rule_based_success'] / self.stats['total_extractions'],
            'ai_fallback_rate': self.stats['ai_fallback_used'] / self.stats['total_extractions'],
            'low_confidence_rate': self.stats['low_confidence_results'] / self.stats['total_extractions']
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the hybrid extractor
    db_config = {
        'host': 'localhost',
        'database': 'yale_alumni',
        'user': 'postgres',
        'password': 'password'
    }
    
    extractor = HybridEntityExtractor(db_config)
    
    # Test cases
    test_titles = [
        "Senior Software Engineer",
        "ML Engineer",
        "Investment Banking Analyst",
        "Product Manager - Consumer Growth",
        "Chief Technology Officer",
        "Junior Data Scientist",
        "Blockchain Developer",  # Might need AI fallback
        "Web3 Growth Hacker",   # Definitely needs AI fallback
    ]
    
    for title in test_titles:
        result = extractor.extract_entities_hybrid(title)
        print(f"\nTitle: {title}")
        print(f"Result: {result.seniority}, {result.role_type}, {result.specialization}")
        print(f"Confidence: {result.confidence:.2f}, Method: {result.extraction_method}")
    
    print(f"\nExtraction Stats: {extractor.get_extraction_stats()}")