# Yale Alumni Search Engine (MILO_DBS)

A sophisticated search engine for Yale alumni data with semantic understanding, query expansion, and domain-specific intelligence.

## 🎯 Overview

This system searches through 4,165 real Yale alumni profiles with advanced features:
- **Semantic Search**: Understands query meaning (e.g., "Wall St. vc banker" → finds investment bankers)
- **Query Expansion**: Automatically expands terms using co-occurrence analysis
- **Domain Intelligence**: Recognizes financial, technology, and consulting terminology
- **Phrase Recognition**: Handles multi-word phrases like "Goldman Sachs", "machine learning"

## 📁 Repository Structure

```
milo_dbs/
├── README.md                          # This file
├── CLAUDE.md                          # Instructions for Claude AI  
├── requirements.txt                   # Python dependencies
├── config.json                        # Database configuration
│
├── Data/
│   ├── FULLY_RECOVERED_YALE_DATABASE.xlsx  # 4,165 real Yale alumni
│   └── enhanced_mappings.json         # Semantic mappings from real data
│
├── Search Engine/
│   └── enhanced_yale_search.py        # MAIN: Enhanced semantic search
│
├── Core Pipeline/
│   └── main.py                        # Full ML pipeline (advanced features)
│
├── modules/                           # Advanced ML modules (12 modules)
│   ├── ai_entity_extraction.py       # AI-powered entity extraction
│   ├── cooccurrence_analysis.py      # Co-occurrence matrix building
│   ├── data_foundation.py            # Data ingestion and normalization
│   ├── enhanced_cooccurrence.py      # Enhanced co-occurrence with filtering
│   ├── enhanced_relationship_mapping.py  # Relationship scoring
│   ├── entity_extraction.py          # Basic entity extraction
│   ├── graph_query_expansion.py      # Graph-based query expansion
│   ├── multi_aspect_embeddings.py    # Multi-aspect semantic embeddings
│   ├── normalized_scoring.py         # Normalized scoring system
│   ├── query_expansion.py            # Query expansion service
│   ├── search_infrastructure.py      # Core search infrastructure
│   └── semantic_layer.py             # Semantic search layer
│
└── models/                            # Model storage directory (auto-created)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Database
```bash
# Create PostgreSQL database
createdb yale_alumni

# The system will auto-create config.json on first run
```

### 3. Run Enhanced Search (Recommended)
```bash
# Basic search
python enhanced_yale_search.py "Wall St. vc banker"

# Show query processing details
python enhanced_yale_search.py "software engineer machine learning" --show-processing

# More examples
python enhanced_yale_search.py "mckinsey consultant"
python enhanced_yale_search.py "data scientist startup"
```

## 🔍 Search Examples & Results

### Financial Search: "Wall St. vc banker"
```
✅ Managing Director at Apollo Global Management
✅ Managing Director at JPMorgan  
✅ Investment Banking Analyst at Goldman Sachs
✅ Investment Banking Associate at Morgan Stanley
```

### Technology Search: "software engineer machine learning"
```
✅ Software Engineer at Google
✅ Software Engineering Manager at Apple
✅ Senior Software Engineer at Google DeepMind
✅ Software Engineer at Microsoft
```

### Consulting Search: "mckinsey consultant"
```
✅ Senior Associate Consultant at Bain & Company
✅ Senior Consultant at Deloitte
✅ Consultant at Boston Consulting Group (BCG)
```

## 📊 Key Features

### 1. **Semantic Understanding**
- Recognizes "Wall St." as "Wall Street" 
- Maps "vc" to "venture capital"
- Understands "banker" means "investment banking"

### 2. **Query Expansion**
- Uses co-occurrence analysis from real alumni data
- Expands queries with related terms
- Example: "software" → "engineer", "developer", "programming"

### 3. **Domain Classification**
- Automatically detects query domain (finance/tech/consulting)
- Applies domain-specific scoring boosts
- Uses industry-specific terminology

### 4. **Enhanced Scoring**
- Position match: Highest weight (15 points)
- Company match: Very high weight (12 points)  
- Industry match: High weight (8 points)
- Skills match: Medium weight (6 points)
- Recency bonus: Recent graduates get boost

## 🛠️ Advanced Usage


### Build Full ML Pipeline (Optional)
```bash
# Requires stable TensorFlow/PyTorch environment
python main.py --setup
python main.py --ingest
python main.py --build-semantic
```

## 📝 Data Schema

Alumni records contain 22 fields:
- **Basic Info**: name, email, graduation_year
- **Professional**: current_position, current_company, industry, function
- **Location**: city, state_territory, country
- **Education**: major, graduate_school, education_history
- **Experience**: experience_history, skills
- **Social**: linkedin_url, linkedin_profile_picture
- **Metadata**: enrichment_date, enrichment_status

## 🔧 Configuration

The system auto-generates `config.json` with:
```json
{
  "database": {
    "dbname": "yale_alumni",
    "user": "your_username",
    "password": "",
    "host": "localhost",
    "port": 5432
  }
}
```

## 🤝 Team Integration

1. **For Developers**: Use `enhanced_yale_search.py` as the main entry point
2. **For Data Scientists**: Explore modules/ for ML components
3. **For Product**: See search examples above for capabilities

## 📈 Performance

- **Search Speed**: ~100ms per query
- **Database Size**: 4,165 alumni profiles
- **Accuracy**: Domain-specific understanding ensures relevant results
- **Scalability**: Optimized for up to 100K+ records

## 🐛 Troubleshooting

1. **Database Connection Error**: 
   - Ensure PostgreSQL is running
   - Check config.json credentials

2. **No Results Found**:
   - Try broader search terms
   - Use --show-processing to see query expansion

3. **ML Dependencies Issues**:
   - Use enhanced_yale_search.py (no heavy ML deps)
   - Install only: psycopg2-binary, no TensorFlow needed

## 📧 Support

For questions about the Yale Alumni Search Engine, refer to CLAUDE.md for detailed technical documentation.