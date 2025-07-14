# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Production Setup (Current)
```bash
# Install dependencies
pip install -r requirements.txt

# Setup database (first time only)
createdb yale_alumni
python main.py --setup

# Generate sample data (for testing)
python utils/create_sample_data.py

# Ingest data (direct method - bypasses ML timeout issues)
python -c "
import pandas as pd
import json
import psycopg2
from psycopg2.extras import execute_values

with open('config.json', 'r') as f:
    config = json.load(f)

df = pd.read_excel('OCS_YALE_PEOPLE_5K.xlsx')
conn = psycopg2.connect(**config['database'])
cur = conn.cursor()

cur.execute('TRUNCATE TABLE title_entities, location_entities, skills, experience, alumni RESTART IDENTITY CASCADE')

alumni_data = [(row['person_id'], row['name'], row['current_title'], row['current_company'], 
               row['current_company'], row['location'], row['location'], row['degree'], 
               row['major'], row['education_school'], int(row['graduation_year'])) for _, row in df.iterrows()]

execute_values(cur, '''INSERT INTO alumni (person_id, name, current_title, current_company, 
                      normalized_company, location, normalized_location, degree, major, 
                      education_school, graduation_year) VALUES %s''', alumni_data)

conn.commit()
print(f'Inserted {cur.rowcount} records')
conn.close()
"

# Search (production interface)
python search.py "software engineer"              # General search
python search.py --company "Google"               # Company search  
python search.py --role "Product Manager"         # Role search
python search.py --analytics                      # Database analytics
```

### ML Pipeline (Advanced - if needed)
```bash
# Build ML indexes (optional - main search works without this)
python build_indexes.py

# Full ML pipeline (may timeout due to library issues)
python main.py --build-semantic
python main.py --build-multi-aspect
python main.py --build-relationships
```

### Development
```bash
# Format code
black modules/ main.py

# Lint code
flake8 modules/ main.py

# Run tests (not yet implemented)
pytest

# Test new features
python test_implementations.py    # High-priority features
python quick_test_medium.py      # Medium-priority features
```

## Architecture Overview

A comprehensive Yale Alumni Search Engine with two implementation paths:

### Simple Pipeline (Recommended)
Lightweight implementation without heavy ML dependencies:
- `simple_setup.py` → Database setup
- `simple_ingest.py` → Data ingestion  
- `unified_search.py` → All search methods (keyword, TF-IDF, filtered)
- Fast setup (~15 seconds), low memory usage (<100MB)

### Advanced ML Pipeline
Full-featured implementation via `main.py`:
- **Core modules** (`modules/`):
  - `data_foundation.py` - Ingestion, deduplication
  - `entity_extraction.py` - Entity extraction
  - `search_infrastructure.py` - SQL generation, caching
  - `semantic_layer.py` - Embeddings search
- **Enhanced features**:
  - `multi_aspect_embeddings.py` - Separate embeddings by profile aspect
  - `normalized_scoring.py` - 0-1 score normalization
  - `ai_entity_extraction.py` - Hybrid rule+AI extraction
  - `graph_query_expansion.py` - Dijkstra-based expansion
  - `enhanced_relationship_mapping.py` - Temporal weighting

### Configuration
- **Database**: PostgreSQL on localhost:5432 (user: georgemccain)
- **Cache**: Redis on localhost:6379 (optional)
- **Config**: Auto-generated `config.json`
- **Models**: `all-MiniLM-L6-v2` for embeddings

### Current File Structure
```
milo_dbs/
├── main.py                 # Full ML pipeline (complex, may timeout)
├── search.py              # Production search interface (recommended)
├── build_indexes.py       # ML index builder (optional)
├── config.json           # Database configuration
├── requirements.txt      # Python dependencies
├── CLAUDE.md            # This documentation
├── modules/             # ML pipeline modules
│   ├── data_foundation.py
│   ├── semantic_layer.py
│   ├── multi_aspect_embeddings.py
│   └── ... (other ML modules)
├── utils/               # Utility scripts
│   ├── create_sample_data.py
│   ├── prepare_100k_data.py
│   └── aws_setup.py
└── data files:
    ├── OCS_YALE_PEOPLE_5K.xlsx  # Yale alumni data (5,000 records)
    └── 361_GPT_COMPANIES.csv    # Company reference data
```

## Key Implementation Details

### Search Pipeline Flow
1. **Query Processing**: Expansion using co-occurrence graph
2. **Search Execution**: Parallel keyword + semantic search
3. **Score Normalization**: 0-1 range with component weights
4. **Result Caching**: Redis for sub-100ms responses

### Enhanced Features
- **Multi-Aspect Embeddings**: Separate vectors for role/skills/education
- **AI Entity Extraction**: Hybrid rule-based + AI fallback
- **Graph Query Expansion**: Dijkstra's algorithm for optimal paths
- **Temporal Relationship Scoring**: Recency decay + duration penalties

### Common Issues & Solutions
1. **PostgreSQL not running**: Start with `pg_ctl start` or `brew services start postgresql`
2. **ML library timeouts**: Use simple pipeline (`unified_search.py`)
3. **Memory issues**: Reduce batch size in `config.json`
4. **No data**: Run `create_sample_data.py` for test data

### Performance Characteristics
- **Simple Pipeline**: ~50-100ms search, <100MB memory
- **ML Pipeline**: ~200-500ms search, ~2GB memory for 5K records
- **Database**: Handles 100K+ records efficiently
- **Caching**: Redis reduces repeat queries to <10ms