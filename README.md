# Reliable-llm-project
Building A Trustable Large Language Model With Paper Retrieval Capabilities
# Reliable Language Model with Paper Retrieval

This project implements a trustable large language model system with paper retrieval capabilities, developed as part of the SEN515 Advanced Data Mining course at Istanbul Aydin University.

## Project Overview

The project consists of three main phases:
1. Initial data collection and analysis (1,701 papers)
2. Refined analysis with feature engineering (229 papers)
3. RAG system implementation for paper retrieval

### Key Features
- Multi-source academic paper processing (Springer, IEEE, arXiv)
- TF-IDF based feature engineering
- PCA and t-SNE dimensionality reduction
- K-means clustering for document organization
- RAG system for paper retrieval and citation

## Project Structure

- `docs/`: Project documentation and figures
- `src/`: Source code for all project phases
- `data/`: Data storage (raw and processed)

## Installation

```bash
git clone https://github.com/your-username/reliable-llm-project.git
cd reliable-llm-project
pip install -r requirements.txt
```

## Usage

### Phase 1: Initial Analysis
```python
python src/phase1/data_collection.py
python src/phase1/preprocessing.py
python src/phase1/clustering.py
```

### Phase 2: Refined Analysis
```python
python src/phase2/data_processing.py
python src/phase2/feature_engineering.py
python src/phase2/clustering_analysis.py
```

### RAG System
```python
python src/rag_system/query_engine.py
```

## Results

- Successfully clustered academic papers into 10 distinct research domains
- Achieved meaningful subject-based clustering with clear thematic separation
- Implemented RAG system for accurate paper retrieval and citation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Fatemeh Shahrokhshahi
- Email: ftm.shahrokhshahi@gmail.com
- University Email: fatemehshahrokhshahi@stu.aydin.edu.tr
