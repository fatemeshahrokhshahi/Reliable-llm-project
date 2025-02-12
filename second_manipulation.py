import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import json
import re

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
        # Convert to lowercase for comparison
        subjects = [str(s).lower() for s in subjects if s is not None]
        # Apply aliases
        subjects = [self.subject_aliases.get(s, s) for s in subjects]
        # Remove duplicates while preserving order
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
            # Get non-zero TF-IDF scores for titles
            title_dict = {
                title_features[j]: float(title_tfidf[i,j])
                for j in title_tfidf[i].nonzero()[1]
            }
            title_scores.append(title_dict)
            
            # Get non-zero TF-IDF scores for abstracts
            abstract_dict = {
                abstract_features[j]: float(abstract_tfidf[i,j])
                for j in abstract_tfidf[i].nonzero()[1]
            }
            abstract_scores.append(abstract_dict)
            
            # Get non-zero TF-IDF scores for subjects
            subject_dict = {
                subject_features[j]: float(subject_tfidf[i,j])
                for j in subject_tfidf[i].nonzero()[1]
            }
            subject_scores.append(subject_dict)
        
        return title_scores, abstract_scores, subject_scores

    def load_and_process_json(self, json_path):
        """Load JSON file with proper handling of nested structures."""
        try:
            # Read the JSON file
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Extract the data array if it's nested under 'data' key
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
            
            # Convert to DataFrame
            df = pd.json_normalize(data)
            
            print("Initial columns:", df.columns.tolist())
            print("Number of rows:", len(df))
            
            return self.process_data_from_df(df)
            
        except Exception as e:
            print(f"Error processing JSON data: {str(e)}")
            raise

    def process_data_from_df(self, df):
        """Process DataFrame with proper column handling."""
        # Print current columns for debugging
        print("Available columns:", df.columns.tolist())
        
        # Ensure required columns exist
        required_columns = ['title', 'abstract', 'subjects']
        for col in required_columns:
            if col not in df.columns:
                print(f"Adding missing column: {col}")
                df[col] = ''
        
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Clean and standardize subjects
        df_processed['subjects'] = df_processed['subjects'].apply(self.standardize_subjects)
        
        # Calculate TF-IDF scores
        title_tfidf, abstract_tfidf, subject_tfidf = self.calculate_tfidf_scores(df_processed)
        
        # Calculate subject weights
        subject_weights = self.calculate_subject_weights(df_processed)
        
        # Store everything as JSON strings to avoid DataFrame issues
        df_processed['title_tfidf'] = [json.dumps(score) for score in title_tfidf]
        df_processed['abstract_tfidf'] = [json.dumps(score) for score in abstract_tfidf]
        df_processed['subject_tfidf'] = [json.dumps(score) for score in subject_tfidf]
        df_processed['subject_weights'] = df_processed['subjects'].apply(
            lambda x: json.dumps({subject: subject_weights.get(subject, 0) for subject in x})
        )
        
        return df_processed

    def search(self, df, query, field=None):
        """Search the dataset using the query."""
        scores = []
        for idx, row in df.iterrows():
            score = 0
            
            try:
                if field is None or field == 'subject':
                    # Direct subject match
                    if any(query.lower() in subject.lower() for subject in row['subjects']):
                        score += self.search_weights['subject']
                    # TF-IDF subject match
                    subject_tfidf = json.loads(row['subject_tfidf'])
                    if query in subject_tfidf:
                        score += subject_tfidf[query] * self.search_weights['subject']
                
                if field is None or field == 'title':
                    # Direct title match
                    if query.lower() in str(row['title']).lower():
                        score += self.search_weights['title']
                    # TF-IDF title match
                    title_tfidf = json.loads(row['title_tfidf'])
                    if query in title_tfidf:
                        score += title_tfidf[query] * self.search_weights['title']
                
                if field is None or field == 'abstract':
                    # Direct abstract match
                    if query.lower() in str(row['abstract']).lower():
                        score += self.search_weights['abstract']
                    # TF-IDF abstract match
                    abstract_tfidf = json.loads(row['abstract_tfidf'])
                    if query in abstract_tfidf:
                        score += abstract_tfidf[query] * self.search_weights['abstract']
            
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                score = 0
            
            scores.append(score)
        
        # Create a copy of the DataFrame with search scores
        results_df = df.copy()
        results_df['search_score'] = scores
        return results_df.sort_values('search_score', ascending=False)

if __name__ == "__main__":
    search_system = ResearchSearchSystem()
    
    input_path = r"C:\Users\Lenovo\Desktop\Data Analysis\Datasets from Reputable Publishers\Publication-Dataset\publication dataset\only springers\merged_dataset.json"
    output_csv = r"C:\Users\Lenovo\Desktop\Data Analysis\Datasets from Reputable Publishers\Publication-Dataset\publication dataset\only springers\manipulated_springer_dataset.csv"
    output_json = r"C:\Users\Lenovo\Desktop\Data Analysis\Datasets from Reputable Publishers\Publication-Dataset\publication dataset\only springers\manipulated_springer_dataset.json"
    
    try:
        print("Loading JSON file...")
        with open(input_path, 'r', encoding='utf-8') as file:
            json_text = file.read()
            print("First 500 characters of JSON:", json_text[:500])
        
        print("\nProcessing data...")
        processed_df = search_system.load_and_process_json(input_path)
        
        print("\nSaving processed data...")
        processed_df.to_csv(output_csv, index=False)
        processed_df.to_json(output_json, orient='records', indent=2)
        
        print("Data processing complete!")
        print(f"Saved CSV to: {output_csv}")
        print(f"Saved JSON to: {output_json}")
        
        # Example search
        print("\nTesting search functionality...")
        results = search_system.search(processed_df, "artificial intelligence")
        print("\nTop 5 results:")
        print(results[['title', 'subjects', 'search_score']].head().to_string())
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()