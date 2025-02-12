import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv

# File paths
input_path = r"C:\Users\Lenovo\Desktop\Data Analysis\Datasets from Reputable Publishers\Publication-Dataset\merged_dataset.json"
output_path = r"C:\Users\Lenovo\Desktop\Data Analysis\Datasets from Reputable Publishers\Publication-Dataset\title_tfidf_scores.csv"

# Load and examine the JSON data
with open(input_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Print the structure of the data
print("Keys in the JSON file:", data.keys())
print("\nMetadata:", json.dumps(data.get('metadata', {}), indent=2))

# Get the actual data records (they might be in the root level)
records = data.get('data', [])  # Try 'data' key first
if not records:
    # If no 'data' key, the records might be directly in a list
    records = [item for item in data if isinstance(item, dict) and 'title' in item]

# Print sample of records
print("\nFirst few records:", json.dumps(records[:2], indent=2))

# Create DataFrame
df = pd.DataFrame(records)

print("\nDataFrame columns:", df.columns.tolist())
print("Number of records:", len(df))

# Now proceed with TF-IDF calculation if we have the data
if 'title' in df.columns and len(df) > 0:
    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        lowercase=True,
        strip_accents='unicode',
        max_features=1000
    )

    # Calculate TF-IDF
    df['title'] = df['title'].fillna('')  # Handle any missing titles
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['title'])

    # Get feature names (words)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Convert to DataFrame for better visualization
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=df['title']
    )

    # Create results list
    results = []
    for title in df['title']:
        # Get non-zero TF-IDF scores for this title
        title_vector = tfidf_df.loc[title]
        word_scores = {word: score for word, score in zip(feature_names, title_vector) if score > 0}
        
        # Sort by score in descending order
        sorted_scores = dict(sorted(word_scores.items(), key=lambda x: x[1], reverse=True))
        
        results.append({
            'title': title,
            'word_scores': sorted_scores
        })

    # Save results to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Word', 'TF-IDF Score'])
        
        for result in results:
            title = result['title']
            for word, score in result['word_scores'].items():
                writer.writerow([title, word, score])

    # Save overall word importance
    overall_importance = tfidf_df.sum().sort_values(ascending=False)
    overall_importance.to_csv(output_path.replace('.csv', '_overall_importance.csv'))

    print(f"\nProcessed {len(df)} titles")
    print(f"Extracted {len(feature_names)} unique terms")
    print(f"Results saved to: {output_path}")
    print(f"Overall word importance saved to: {output_path.replace('.csv', '_overall_importance.csv')}")
else:
    print("\nError: Could not find 'title' column in the data")