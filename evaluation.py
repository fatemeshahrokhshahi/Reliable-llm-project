import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from collections import Counter
import os

def evaluate_paper_clusters(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        clusters_data = json.load(f)
    
    all_papers = []
    cluster_labels = []
    
    for cluster_id, cluster_info in clusters_data.items():
        if cluster_id == 'statistics':
            continue
        if isinstance(cluster_info, dict) and 'papers' in cluster_info:
            papers = cluster_info['papers']
            cluster_num = int(cluster_id.split('_')[1])
            all_papers.extend(papers)
            cluster_labels.extend([cluster_num] * len(papers))
    
    cluster_labels = np.array(cluster_labels)
    
    abstracts = [paper.get('abstract_snippet', '') for paper in all_papers]
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(abstracts).toarray()
    
    metrics = {
        'silhouette': float(silhouette_score(X, cluster_labels)),
        'davies_bouldin': float(davies_bouldin_score(X, cluster_labels)),
        'calinski_harabasz': float(calinski_harabasz_score(X, cluster_labels))
    }
    
    subject_consistency = {}
    for cluster_id in np.unique(cluster_labels):
        cluster_papers = [p for i, p in enumerate(all_papers) if cluster_labels[i] == cluster_id]
        all_subjects = [s for p in cluster_papers for s in p.get('subjects', [])]
        subject_counts = Counter(all_subjects)
        top_subjects = subject_counts.most_common(5)
        subject_consistency[f'cluster_{cluster_id}'] = top_subjects
    
    # Convert numpy int64 to regular int for JSON serialization
    cluster_sizes = Counter(cluster_labels)
    cluster_sizes_dict = {str(k): int(v) for k, v in cluster_sizes.items()}
    
    results = {
        'evaluation_metrics': metrics,
        'subject_consistency': subject_consistency,
        'cluster_sizes': cluster_sizes_dict
    }
    
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'clustering_evaluation.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    return metrics, subject_consistency

input_path = r"C:\Users\Lenovo\Desktop\Data Analysis\Datasets from Reputable Publishers\Publication-Dataset\publication dataset\only springers\modeling 2\modeling\cluster_analysis.json"
output_path = r"C:\Users\Lenovo\Desktop\Data Analysis\Datasets from Reputable Publishers\Publication-Dataset\publication dataset\only springers\modeling 2\evaluation of the model"

metrics, subject_analysis = evaluate_paper_clusters(input_path, output_path)
print("Results saved to:", output_path)
print("\nMetrics:", metrics)