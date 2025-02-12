import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import json
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
import os

class ResearchSearchSystem:
    def __init__(self):
        self.search_weights = {
            "subject": 3.0,
            "title": 2.0,
            "abstract": 1.0
        }
        self.subject_aliases = {
            'atmospheric science': 'atmospheric sciences',
            'environment, general': 'environment',
            'environmental sciences': 'environment',
            'biotechnology': 'environmental engineering/biotechnology'
        }
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    def standardize_subjects(self, subjects):
        """Standardize subject names and remove duplicates."""
        if not isinstance(subjects, list):
            return []
        subjects = [str(s).lower() for s in subjects if s is not None]
        subjects = [self.subject_aliases.get(s, s) for s in subjects]
        return list(dict.fromkeys(subjects))
    
    def calculate_subject_weights(self, df):
        """Calculate weights for subjects based on frequency."""
        all_subjects = []
        for subjects in df['subjects']:
            if isinstance(subjects, list):
                all_subjects.extend(subjects)
        
        subject_counts = Counter(all_subjects)
        total_count = sum(subject_counts.values())
        
        if total_count == 0:
            return {}
            
        return {subject: count/total_count 
                for subject, count in subject_counts.items()}
    
    def calculate_tfidf_scores(self, df):
        """Calculate TF-IDF scores for titles, abstracts, and subjects."""
        # Calculate TF-IDF for titles
        titles = df['title'].fillna('')
        title_tfidf = self.tfidf_vectorizer.fit_transform(titles)
        title_features = self.tfidf_vectorizer.get_feature_names_out()
        
        # Calculate TF-IDF for abstracts
        abstracts = df['abstract'].fillna('')
        abstract_tfidf = self.tfidf_vectorizer.fit_transform(abstracts)
        abstract_features = self.tfidf_vectorizer.get_feature_names_out()
        
        # Calculate TF-IDF for subjects
        subject_texts = [' '.join(map(str, subjects)) if isinstance(subjects, list) else '' 
                        for subjects in df['subjects']]
        subject_tfidf = self.tfidf_vectorizer.fit_transform(subject_texts)
        subject_features = self.tfidf_vectorizer.get_feature_names_out()
        
        # Create score dictionaries
        title_scores = []
        abstract_scores = []
        subject_scores = []
        
        for i in range(len(df)):
            title_dict = {
                title_features[j]: float(title_tfidf[i,j])
                for j in title_tfidf[i].nonzero()[1]
            }
            title_scores.append(title_dict)
            
            abstract_dict = {
                abstract_features[j]: float(abstract_tfidf[i,j])
                for j in abstract_tfidf[i].nonzero()[1]
            }
            abstract_scores.append(abstract_dict)
            
            subject_dict = {
                subject_features[j]: float(subject_tfidf[i,j])
                for j in subject_tfidf[i].nonzero()[1]
            }
            subject_scores.append(subject_dict)
        
        return title_scores, abstract_scores, subject_scores

    def prepare_matrix_for_pca(self, df):
        """Convert stored TF-IDF scores into a matrix format for PCA."""
        # Collect all unique terms
        all_terms = set()
        for idx, row in df.iterrows():
            title_terms = json.loads(row['title_tfidf']).keys()
            abstract_terms = json.loads(row['abstract_tfidf']).keys()
            subject_terms = json.loads(row['subject_tfidf']).keys()
            all_terms.update(title_terms, abstract_terms, subject_terms)
        
        # Create matrix
        matrix = np.zeros((len(df), len(all_terms)))
        terms_list = sorted(list(all_terms))
        term_to_idx = {term: idx for idx, term in enumerate(terms_list)}
        
        # Fill matrix with weighted scores
        for i, row in df.iterrows():
            title_scores = json.loads(row['title_tfidf'])
            abstract_scores = json.loads(row['abstract_tfidf'])
            subject_scores = json.loads(row['subject_tfidf'])
            
            for term, score in title_scores.items():
                matrix[i, term_to_idx[term]] += score * self.search_weights['title']
            for term, score in abstract_scores.items():
                matrix[i, term_to_idx[term]] += score * self.search_weights['abstract']
            for term, score in subject_scores.items():
                matrix[i, term_to_idx[term]] += score * self.search_weights['subject']
        
        return matrix, terms_list

    def perform_pca_analysis(self, df, n_components=2):
        """Perform PCA analysis with robust scaling."""
        print("Preparing matrix for PCA...")
        X, terms = self.prepare_matrix_for_pca(df)
        
        print("Performing robust scaling...")
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        print("Performing PCA...")
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # Get feature importance
        feature_importance = np.abs(pca.components_)
        top_terms_idx = feature_importance[0].argsort()[-10:][::-1]
        top_terms = [terms[idx] for idx in top_terms_idx]
        
        return {
            'pca_coordinates': X_pca,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance_ratio': cumulative_variance_ratio,
            'top_terms': top_terms,
            'pca_object': pca,
            'scaler': scaler,
            'terms': terms
        }

    def visualize_pca_results(self, df, pca_results, output_path):
        """Create separate, clear visualizations for PCA results."""
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Created output directory: {output_path}")

        # 1. Document Distribution Plot
        print("Generating document distribution plot...")
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            pca_results['pca_coordinates'][:, 0],
            pca_results['pca_coordinates'][:, 1],
            alpha=0.6,
            s=100,
            c=range(len(df)),
            cmap='viridis'
        )
        plt.colorbar(scatter, label='Document Index')
        plt.xlabel(f'First Principal Component (Explains {pca_results["explained_variance_ratio"][0]*100:.1f}% of variance)')
        plt.ylabel(f'Second Principal Component (Explains {pca_results["explained_variance_ratio"][1]*100:.1f}% of variance)')
        plt.title('Document Distribution in PC Space')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_path, 'document_distribution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved document distribution plot to: {plot_path}")
        plt.close()
        
        # 2. Explained Variance Plot
        print("Generating explained variance plot...")
        plt.figure(figsize=(10, 6))
        components = range(1, len(pca_results['explained_variance_ratio']) + 1)
        plt.plot(components, 
                pca_results['cumulative_variance_ratio'], 
                'bo-', 
                linewidth=2, 
                markersize=8)
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Cumulative Explained Variance Ratio', fontsize=12)
        plt.title('Explained Variance Ratio', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_path, 'explained_variance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved explained variance plot to: {plot_path}")
        plt.close()
        
        # 3. Top Terms Plot
        print("Generating top terms plot...")
        plt.figure(figsize=(12, 8))
        top_terms = pca_results['top_terms']
        importance = np.abs(pca_results['pca_object'].components_[0, :])
        top_importance = importance[importance.argsort()[-10:][::-1]]
        
        plt.barh(range(len(top_terms)), 
                top_importance,
                height=0.6)
        plt.yticks(range(len(top_terms)), top_terms, fontsize=10)
        plt.xlabel('Contribution to First PC', fontsize=12)
        plt.title('Top Terms Contributing to First Principal Component', 
                 fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_path, 'top_terms.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved top terms plot to: {plot_path}")
        plt.close()

    def analyze_principal_components(self, df, pca_results, n_features=10):
        """Analyze and display detailed information about principal components."""
        terms_list = pca_results['terms']
        
        # 1. Get feature contributions for PC1 and PC2
        pc1_contributions = pd.DataFrame({
            'term': terms_list,
            'contribution': np.abs(pca_results['pca_object'].components_[0])
        }).sort_values('contribution', ascending=False)
        
        pc2_contributions = pd.DataFrame({
            'term': terms_list,
            'contribution': np.abs(pca_results['pca_object'].components_[1])
        }).sort_values('contribution', ascending=False)
        
        # Print top contributing features
        print("\nTop Features Contributing to PC1:")
        print(pc1_contributions.head(n_features).to_string())
        print("\nTop Features Contributing to PC2:")
        print(pc2_contributions.head(n_features).to_string())
        
        # Print example documents for each extreme
        print("\nExample Documents at Extremes of Principal Components:")
        
        for component, pc_name in [(0, "PC1"), (1, "PC2")]:
            print(f"\n{pc_name} Analysis:")
            
            # Get highest and lowest scoring documents
            top_docs = df.iloc[pca_results['pca_coordinates'][:, component].argsort()[-3:]]
            bottom_docs = df.iloc[pca_results['pca_coordinates'][:, component].argsort()[:3]]
            
            print(f"\nTop 3 documents for {pc_name}:")
            for _, doc in top_docs.iterrows():
                print(f"Title: {doc['title']}")
                print(f"Subjects: {', '.join(doc['subjects']) if isinstance(doc['subjects'], list) else ''}\n")
            
            print(f"\nBottom 3 documents for {pc_name}:")
            for _, doc in bottom_docs.iterrows():
                print(f"Title: {doc['title']}")
                print(f"Subjects: {', '.join(doc['subjects']) if isinstance(doc['subjects'], list) else ''}\n")
                
        return pc1_contributions, pc2_contributions

    def process_data_from_df(self, df):
        """Process DataFrame and calculate TF-IDF scores."""
        print("Available columns:", df.columns.tolist())
        
        required_columns = ['title', 'abstract', 'subjects']
        for col in required_columns:
            if col not in df.columns:
                print(f"Adding missing column: {col}")
                df[col] = ''
        
        df_processed = df.copy()
        df_processed['subjects'] = df_processed['subjects'].apply(self.standardize_subjects)
        
        title_tfidf, abstract_tfidf, subject_tfidf = self.calculate_tfidf_scores(df_processed)
        subject_weights = self.calculate_subject_weights(df_processed)
        
        df_processed['title_tfidf'] = [json.dumps(score) for score in title_tfidf]
        df_processed['abstract_tfidf'] = [json.dumps(score) for score in abstract_tfidf]
        df_processed['subject_tfidf'] = [json.dumps(score) for score in subject_tfidf]
        df_processed['subject_weights'] = df_processed['subjects'].apply(
            lambda x: json.dumps({subject: subject_weights.get(subject, 0) for subject in x})
        )
        
        return df_processed

    def load_and_process_json(self, json_path):
        """Load and process JSON data."""
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
            
            df = pd.json_normalize(data)
            print("Initial columns:", df.columns.tolist())
            print("Number of rows:", len(df))
            
            return self.process_data_from_df(df)
            
        except Exception as e:
            print(f"Error processing JSON data: {str(e)}")
            raise

if __name__ == "__main__":
    search_system = ResearchSearchSystem()
    
    # File paths
    input_path = r"C:\Users\Lenovo\Desktop\Data Analysis\Datasets from Reputable Publishers\Publication-Dataset\publication dataset\manipulated data\manipulated_dataset.json"
    output_path = r"C:\Users\Lenovo\Desktop\Data Analysis\Datasets from Reputable Publishers\Publication-Dataset\publication dataset\Visualization"
    
    try:
        print("Loading JSON file...")
        processed_df = search_system.load_and_process_json(input_path)
        
        print("\nPerforming PCA analysis...")
        pca_results = search_system.perform_pca_analysis(processed_df)
        
        print("\nAnalyzing principal components...")
        pc1_contributions, pc2_contributions = search_system.analyze_principal_components(
            processed_df, pca_results
        )
        
        print("\nGenerating visualizations...")
        # Pass output_path to visualize_pca_results
        search_system.visualize_pca_results(processed_df, pca_results, output_path)
        
        print(f"\nVisualizations have been saved to: {output_path}")
        
        # Print variance explained
        print("\nVariance Explained by Principal Components:")
        print(f"PC1: {pca_results['explained_variance_ratio'][0]*100:.2f}%")
        print(f"PC2: {pca_results['explained_variance_ratio'][1]*100:.2f}%")
        print(f"Total: {sum(pca_results['explained_variance_ratio'])*100:.2f}%")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()