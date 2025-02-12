import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

def load_json_data(file_path):
    """Load data from JSON file with improved error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} papers")
        print("\nColumns in the dataset:", df.columns.tolist())
        
        # Add basic data validation
        required_columns = ['title', 'abstract', 'subjects', 'source', 'identifier']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        return df
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def create_document_vectors(df):
    """Create TF-IDF vectors with improved text preprocessing"""
    # Improved text combination with proper weighting
    combined_texts = (
        df['title'].fillna('') + ' ' + 
        df['title'].fillna('') + ' ' +  # Title repeated for more weight
        df['abstract'].fillna('')
    )
    
    # Enhanced TF-IDF configuration
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,  # Minimum document frequency
        max_df=0.95,  # Maximum document frequency
        strip_accents='unicode',
        norm='l2'
    )
    
    feature_matrix = vectorizer.fit_transform(combined_texts)
    return vectorizer, feature_matrix

def find_optimal_clusters(feature_matrix, max_clusters=10):
    """Find optimal number of clusters using elbow method and silhouette score"""
    inertias = []
    silhouette_scores = []
    k_values = range(2, max_clusters + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_matrix)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(feature_matrix, labels))
    
    # Plot elbow curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    
    plt.tight_layout()
    return k_values[np.argmax(silhouette_scores)]

def perform_clustering(feature_matrix, n_clusters=None):
    """Perform K-means clustering with optimal number of clusters"""
    if n_clusters is None:
        n_clusters = find_optimal_clusters(feature_matrix)
        print(f"\nOptimal number of clusters determined: {n_clusters}")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(feature_matrix)
    return kmeans, labels

def analyze_clusters(kmeans, labels, vectorizer, feature_matrix, df):
    """Enhanced cluster analysis with additional metrics"""
    feature_names = vectorizer.get_feature_names_out()
    cluster_analysis = {}
    
    # Calculate global metrics
    global_stats = {
        "total_papers": len(df),
        "avg_cluster_size": len(df) / kmeans.n_clusters,
        "cluster_size_std": np.std([sum(labels == i) for i in range(kmeans.n_clusters)])
    }
    
    for i in range(kmeans.n_clusters):
        cluster_papers = df[labels == i]
        print(f"\nCluster {i} ({len(cluster_papers)} papers):")
        
        # Enhanced term analysis
        centroid = kmeans.cluster_centers_[i]
        top_indices = centroid.argsort()[-20:][::-1]  # Increased to top 20 terms
        top_terms = [feature_names[j] for j in top_indices]
        term_weights = centroid[top_indices]
        
        # Advanced subject analysis
        all_subjects = [subj for paper_subjects in cluster_papers['subjects'] for subj in paper_subjects]
        subject_counts = Counter(all_subjects)
        top_subjects = dict(sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:15])
        
        # Calculate cluster cohesion
        cluster_vectors = feature_matrix[labels == i]
        centroid_vector = kmeans.cluster_centers_[i]
        distances = np.sqrt(np.sum((cluster_vectors.toarray() - centroid_vector) ** 2, axis=1))
        cohesion = {
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances)),
            "max_distance": float(np.max(distances))
        }
        
        # Enhanced paper details
        paper_details = []
        for _, paper in cluster_papers.iterrows():
            paper_info = {
                'title': paper['title'],
                'identifier': paper['identifier'],
                'source': paper['source'],
                'subjects': paper['subjects'],
                'abstract_snippet': paper['abstract'][:300] + '...' if len(paper['abstract']) > 300 else paper['abstract'],
                'distance_to_centroid': float(distances[len(paper_details)])
            }
            paper_details.append(paper_info)
        
        # Store comprehensive cluster information
        cluster_analysis[f"cluster_{i}"] = {
            "size": len(cluster_papers),
            "percentage": len(cluster_papers) / len(df) * 100,
            "top_terms": list(zip(top_terms, term_weights.tolist())),
            "top_subjects": top_subjects,
            "papers": paper_details,
            "cohesion_metrics": cohesion,
            "statistics": {
                "sources_distribution": dict(Counter(cluster_papers['source'])),
                "avg_abstract_length": int(cluster_papers['abstract'].str.len().mean()),
                "total_unique_subjects": len(set(all_subjects)),
                "subject_diversity": len(set(all_subjects)) / len(all_subjects) if all_subjects else 0
            }
        }
        
        # Print enhanced cluster summary
        print(f"\nCluster {i} Summary:")
        print(f"Size: {len(cluster_papers)} papers ({cluster_analysis[f'cluster_{i}']['percentage']:.1f}%)")
        print(f"Cohesion (mean distance to centroid): {cohesion['mean_distance']:.3f}")
        print("\nTop Terms (with weights):")
        for term, weight in list(zip(top_terms, term_weights))[:10]:
            print(f"- {term}: {weight:.3f}")
        
    # Add global statistics to the analysis
    cluster_analysis["global_statistics"] = global_stats
    return cluster_analysis

def save_detailed_results(df, labels, cluster_analysis, output_dir):
    """Save comprehensive results with improved organization"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save clustering results CSV without individual distances
    df['cluster'] = labels
    
    # Create a mapping of document indices to distances for each cluster
    distances_map = {}
    for cluster_id, cluster_info in cluster_analysis.items():
        if cluster_id != "global_statistics":
            for paper in cluster_info["papers"]:
                identifier = paper["identifier"]
                distances_map[identifier] = paper["distance_to_centroid"]
    
    # Add distances to dataframe where available
    df['distance_to_centroid'] = df['identifier'].map(distances_map)
    
    results_path = output_dir / "clustering_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Save detailed cluster analysis
    analysis_path = output_dir / "cluster_analysis.json"
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_analysis, f, indent=2, ensure_ascii=False)
    
    # Generate enhanced cluster summaries
    summary_path = output_dir / "cluster_summaries.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        # Write global statistics
        f.write("GLOBAL CLUSTERING STATISTICS\n")
        f.write("=" * 80 + "\n")
        for key, value in cluster_analysis["global_statistics"].items():
            f.write(f"{key}: {value}\n")
        
        # Write individual cluster information
        for cluster_id, analysis in cluster_analysis.items():
            if cluster_id == "global_statistics":
                continue
                
            f.write(f"\n{'='*80}\n")
            f.write(f"# {cluster_id.upper()}\n")
            f.write(f"Number of papers: {analysis['size']} ")
            f.write(f"({analysis['percentage']:.1f}% of total)\n\n")
            
            f.write("## Cohesion Metrics\n")
            for metric, value in analysis['cohesion_metrics'].items():
                f.write(f"{metric}: {value:.3f}\n")
            
            f.write("\n## Top Terms (with weights)\n")
            for term, weight in analysis['top_terms'][:15]:
                f.write(f"- {term}: {weight:.3f}\n")
            
            f.write("\n## Top Subjects\n")
            for subject, count in analysis['top_subjects'].items():
                f.write(f"- {subject}: {count} papers\n")
            
            f.write("\n## Statistics\n")
            stats = analysis['statistics']
            f.write(f"Average abstract length: {stats['avg_abstract_length']} characters\n")
            f.write(f"Total unique subjects: {stats['total_unique_subjects']}\n")
            f.write(f"Subject diversity index: {stats['subject_diversity']:.3f}\n")
            f.write("\nSource distribution:\n")
            for source, count in stats['sources_distribution'].items():
                f.write(f"- {source}: {count} papers\n")
            
            f.write("\n## Representative Papers (closest to centroid)\n")
            sorted_papers = sorted(analysis['papers'], key=lambda x: x['distance_to_centroid'])
            for paper in sorted_papers[:5]:
                f.write(f"\nTitle: {paper['title']}\n")
                f.write(f"Distance to centroid: {paper['distance_to_centroid']:.3f}\n")
                f.write(f"Source: {paper['source']}\n")
                f.write(f"Subjects: {', '.join(paper['subjects'])}\n")
                f.write(f"Abstract: {paper['abstract_snippet']}\n")

def main():
    try:
        # File paths
        input_path = Path(r"C:\Users\Lenovo\Desktop\Data Analysis\Datasets from Reputable Publishers\Publication-Dataset\publication dataset\only springers\Third step\manipulated_springer_dataset.json")
        output_dir = Path(r"C:\Users\Lenovo\Desktop\Data Analysis\Datasets from Reputable Publishers\Publication-Dataset\publication dataset\only springers\modeling 2")
        
        # Load and preprocess data
        print("Loading data...")
        df = load_json_data(input_path)
        
        # Create document vectors
        print("\nCreating document vectors...")
        vectorizer, feature_matrix = create_document_vectors(df)
        print(f"Created feature matrix with shape: {feature_matrix.shape}")
        
        # Perform clustering with optimal number of clusters
        print("\nDetermining optimal number of clusters...")
        kmeans, labels = perform_clustering(feature_matrix)
        
        # Analyze results
        print("\nAnalyzing clusters...")
        cluster_analysis = analyze_clusters(kmeans, labels, vectorizer, feature_matrix, df)
        
        # Save all results
        save_detailed_results(df, labels, cluster_analysis, output_dir)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()