import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AcademicPaperRAG:
    def __init__(self, json_path):
        self.papers = self._load_papers(json_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.document_vectors = self._create_vectors()
        
    def _load_papers(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_vectors(self):
        documents = [f"{paper['title']} {paper['abstract']}" for paper in self.papers]
        return self.vectorizer.fit_transform(documents)
    
    def search(self, query, top_k=5):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [{
            'title': self.papers[idx]['title'],
            'abstract': self.papers[idx]['abstract'][:300] + '...',  # Truncate long abstracts
            'subjects': self.papers[idx]['subjects'],
            'similarity_score': similarities[idx],
            'doi': self.papers[idx]['identifier']
        } for idx in top_indices]

# Initialize and use RAG system
file_path = r"C:\Users\Lenovo\Desktop\Data Analysis\Datasets from Reputable Publishers\Publication-Dataset\publication dataset\only springers\Third step\manipulated_springer_dataset.json"
rag = AcademicPaperRAG(file_path)

# Example searches
queries = [
    "artificial intelligence ethics",
    "climate change impact",
    "machine learning applications"
]

for query in queries:
    print(f"\nSearch Query: {query}")
    results = rag.search(query)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Score: {result['similarity_score']:.3f})")
        print(f"Title: {result['title']}")
        print(f"DOI: {result['doi']}")
        print("-" * 50)