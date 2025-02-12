import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File paths
input_path = r"C:\Users\Lenovo\Desktop\Data Analysis\Datasets from Reputable Publishers\Publication-Dataset\publication dataset\manipulated data\manipulated_dataset.json"
output_dir = r"C:\Users\Lenovo\Desktop\Data Analysis\Datasets from Reputable Publishers\Publication-Dataset\publication dataset\Visualization\visualization 2"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Convert TF-IDF dictionaries to a matrix
def prepare_tfidf_matrix(data, tfidf_key='title_tfidf'):
    # Collect all unique terms
    all_terms = set()
    for doc in data:
        if doc.get(tfidf_key):
            all_terms.update(json.loads(doc[tfidf_key]).keys())
    
    # Create the matrix
    term_list = sorted(list(all_terms))
    matrix = []
    for doc in data:
        if doc.get(tfidf_key):
            tfidf_dict = json.loads(doc[tfidf_key])
            row = [tfidf_dict.get(term, 0.0) for term in term_list]
            matrix.append(row)
    
    return np.array(matrix), term_list

# Perform PCA and create visualizations
def analyze_and_visualize(data):
    # Prepare the TF-IDF matrix
    X, terms = prepare_tfidf_matrix(data)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create DataFrame for plotting
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    
    # 1. Scree plot
    plt.figure(figsize=(10, 6))
    explained_variance = pca.explained_variance_ratio_ * 100
    plt.bar(range(1, 4), explained_variance)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance (%)')
    plt.title('Scree Plot: Explained Variance by Principal Components')
    plt.savefig(os.path.join(output_dir, 'scree_plot.png'))
    plt.close()
    
    # 2. 2D scatter plot (PC1 vs PC2)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', alpha=0.6)
    plt.title('PCA: First Two Principal Components')
    plt.xlabel(f'PC1 ({explained_variance[0]:.1f}% variance explained)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.1f}% variance explained)')
    plt.savefig(os.path.join(output_dir, '2d_scatter.png'))
    plt.close()
    
    # 3. Loading plot
    # Get the top 10 terms for each component
    n_top_terms = 10
    loading_scores = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2', 'PC3'],
        index=terms
    )
    
    plt.figure(figsize=(15, 10))
    for i, pc in enumerate(['PC1', 'PC2', 'PC3']):
        plt.subplot(1, 3, i+1)
        component_loadings = loading_scores[pc].sort_values(ascending=False)
        top_terms = component_loadings.head(n_top_terms)
        
        sns.barplot(x=top_terms.values, y=top_terms.index)
        plt.title(f'Top Terms in {pc}')
        plt.xlabel('Loading Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loading_plot.png'))
    plt.close()
    
    # 4. 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.6)
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.1f}%)')
    ax.set_zlabel(f'PC3 ({explained_variance[2]:.1f}%)')
    plt.title('3D PCA Visualization')
    plt.savefig(os.path.join(output_dir, '3d_scatter.png'))
    plt.close()

    # Save component information
    component_info = {
        'explained_variance': explained_variance.tolist(),
        'loading_scores': loading_scores.to_dict(),
        'top_terms': {
            f'PC{i+1}': loading_scores[f'PC{i+1}']
                .sort_values(ascending=False)
                .head(n_top_terms).to_dict()
            for i in range(3)
        }
    }
    
    with open(os.path.join(output_dir, 'pca_components_info.json'), 'w') as f:
        json.dump(component_info, f, indent=2)

def main():
    try:
        # Load data
        print("Loading dataset...")
        data = load_data(input_path)
        print(f"Loaded {len(data)} documents")
        
        # Perform analysis and create visualizations
        print("Performing PCA and creating visualizations...")
        analyze_and_visualize(data)
        
        print("Analysis complete! Visualizations saved to:", output_dir)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()