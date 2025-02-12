import json
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
    # Get abstract content, handling both string and list cases
    abstract = record.get('abstract', {}).get('p', '')
    if isinstance(abstract, list):
        abstract = ' '.join(abstract)  # Combine multiple paragraphs
    
    return {
        'source': 'Springer',
        'identifier': record.get('doi', record.get('identifier', '')),
        'title': record.get('title', ''),
        'abstract': abstract,
        'subjects': record.get('subjects', []) + [disc.get('term', '') for disc in record.get('disciplines', [])]
    }

def standardize_ieee_record(record):
    return {
        'source': 'IEEE',
        'identifier': str(record.get('Id', '')),
        'title': record.get('title', ''),
        'abstract': record.get('abstract', ''),
        'subjects': []
    }

def standardize_arxiv_record(record):
    # Split subjects by space and create a list
    categories = record.get('categories', '')
    subject_list = categories.split() if isinstance(categories, str) else [categories]
    
    return {
        'source': 'arXiv',
        'identifier': record.get('doi', record.get('id', '')),
        'title': record.get('title', ''),
        'abstract': record.get('abstract', ''),  # Keep original LaTeX formatting
        'subjects': subject_list
    }

def merge_datasets(springer_path, ieee_path, arxiv_path, output_path):
    springer_data = read_json_file(springer_path)
    ieee_data = read_json_file(ieee_path)
    arxiv_data = read_json_file(arxiv_path)

    merged_data = []

    if springer_data:
        if isinstance(springer_data, dict):
            springer_data = [springer_data]
        for record in springer_data:
            merged_data.append(standardize_springer_record(record))

    if ieee_data:
        if isinstance(ieee_data, dict):
            ieee_data = [ieee_data]
        for record in ieee_data:
            merged_data.append(standardize_ieee_record(record))

    if arxiv_data:
        if isinstance(arxiv_data, dict):
            arxiv_data = [arxiv_data]
        for record in arxiv_data:
            merged_data.append(standardize_arxiv_record(record))

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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, indent=2, ensure_ascii=False)
        print(f"Successfully created merged dataset at {output_path}")
        print(f"Total records: {metadata['total_records']}")
        print("Records per source:")
        for source, count in metadata['sources'].items():
            print(f"- {source}: {count}")
    except Exception as e:
        print(f"Error saving merged dataset: {str(e)}")

def main():
    base_path = r'C:\Users\Lenovo\Desktop\Data Analysis\Datasets from Reputable Publishers'
    springer_path = os.path.join(base_path, 'Nature Springer', 'springer_data_20250101_234834_raw.json')
    ieee_path = os.path.join(base_path, 'IEEE', 'csvjson.json')
    arxiv_path = os.path.join(base_path, 'ArXiv', 'arxiv_sample_1000.json')
    output_path = os.path.join(base_path, 'Publication-Dataset', 'merged_dataset.json')

    merge_datasets(springer_path, ieee_path, arxiv_path, output_path)

if __name__ == "__main__":
    main()