import json
import csv
import os
from datetime import datetime

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

def standardize_springer_record(record):
    abstract = record.get('abstract', '')
    if isinstance(abstract, dict):
        abstract = abstract.get('p', '')
    if isinstance(abstract, list):
        abstract = ' '.join(abstract)
    
    return {
        'source': 'Springer',
        'identifier': record.get('doi', record.get('identifier', '')),
        'title': record.get('title', ''),
        'abstract': abstract,
        'subjects': record.get('subjects', []) + [disc.get('term', '') for disc in record.get('disciplines', [])]
    }

def validate_record(record):
    required_fields = ['title', 'identifier']
    return all(record.get(field) for field in required_fields)

def merge_datasets(springer_path, output_json_path, output_csv_path):
    springer_data = read_json_file(springer_path)
    if not springer_data:
        print("No valid data to process")
        return
        
    merged_data = []
    for record in springer_data if isinstance(springer_data, list) else [springer_data]:
        standardized = standardize_springer_record(record)
        if standardized and validate_record(standardized):
            merged_data.append(standardized)

    metadata = {
        "total_records": len(merged_data),
        "sources": {
            "springer": len([r for r in merged_data if r['source'] == 'Springer']),
            "ieee": len([r for r in merged_data if r['source'] == 'IEEE']),
            "arxiv": len([r for r in merged_data if r['source'] == 'arXiv'])
        },
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fields": list(merged_data[0].keys()) if merged_data else []
    }

    final_dataset = {
        "metadata": metadata,
        "data": merged_data
    }

    # Save JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, indent=2, ensure_ascii=False)
        print(f"Successfully created JSON dataset at {output_json_path}")
    except Exception as e:
        print(f"Error saving JSON dataset: {str(e)}")

    # Save CSV
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=merged_data[0].keys())
            writer.writeheader()
            writer.writerows(merged_data)
        print(f"Successfully created CSV dataset at {output_csv_path}")
    except Exception as e:
        print(f"Error saving CSV dataset: {str(e)}")

    print(f"Total records: {metadata['total_records']}")
    print("Records per source:")
    for source, count in metadata['sources'].items():
        print(f"- {source}: {count}")

def main():
    base_path = r'C:\Users\Lenovo\Desktop\Data Analysis\Datasets from Reputable Publishers\Publication-Dataset\publication dataset\only springers'
    springer_path = os.path.join(base_path, 'merged_springer_data_20250123_121848.json')
    output_json_path = os.path.join(base_path, 'merged_dataset.json')
    output_csv_path = os.path.join(base_path, 'merged_dataset.csv')
    
    merge_datasets(springer_path, output_json_path, output_csv_path)

if __name__ == "__main__":
    main()