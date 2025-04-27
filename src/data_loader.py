"""
Module for loading and processing legal document data from JSON files.
"""
import os
import json
import codecs
from typing import List, Dict, Any

def load_json_data(directory_path: str) -> List[Dict[str, Any]]:
    """
    Load all JSON files from a directory and return a list of document fragments.
    
    Args:
        directory_path: Path to the directory containing JSON files
        
    Returns:
        List of document fragments as dictionaries
    """
    documents = []
    
    # Check if directory exists
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            try:
                # Try different encodings
                encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1']
                data = None
                
                for encoding in encodings:
                    try:
                        with codecs.open(file_path, 'r', encoding=encoding) as file:
                            content = file.read()
                            data = json.loads(content)
                            print(f"Successfully loaded file {filename} with encoding {encoding}")
                            break
                    except UnicodeDecodeError:
                        continue
                
                if data is None:
                    print(f"Error: Could not decode file {filename} with any of the attempted encodings")
                    continue
                    
                print(f"Loaded file: {filename}")
                
                # If the loaded data is a list, extend documents with its items
                if isinstance(data, list):
                    print(f"  Found {len(data)} documents in list format")
                    for i, doc in enumerate(data[:3]):  # Print first 3 docs for debugging
                        print(f"  Document {i} keys: {doc.keys()}")
                        if 'content' in doc:
                            content_preview = doc['content'][:50].replace('\n', ' ')
                            print(f"  Content preview: {content_preview}...")
                    documents.extend(data)
                # If it's a single document, append it to the list
                else:
                    print(f"  Found a single document")
                    print(f"  Document keys: {data.keys()}")
                    if 'content' in data:
                        content_preview = data['content'][:50].replace('\n', ' ')
                        print(f"  Content preview: {content_preview}...")
                    documents.append(data)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON format in file {filename}")
            except Exception as e:
                print(f"Error loading file {filename}: {str(e)}")
    
    # Convert documents to the standard format expected by the system
    standardized_docs = []
    for doc in documents:
        # Check if document has the required fields
        if 'content' in doc:
            # Create a standardized document
            std_doc = {
                'content': doc['content'],
                'article_id': '',
                'law_name': '',
                'article_number': '',
                'category': '',
                'source': ''
            }
            
            # Extract metadata if available
            if 'metadata' in doc:
                metadata = doc['metadata']
                if 'article' in metadata:
                    std_doc['article_number'] = metadata['article']
                    std_doc['article_id'] = f"{metadata.get('code', 'unknown')}_{metadata['article']}"
                if 'code' in metadata:
                    std_doc['law_name'] = metadata['code']
                if 'chapter' in metadata:
                    std_doc['category'] = metadata['chapter']
            
            standardized_docs.append(std_doc)
    
    print(f"Standardized {len(standardized_docs)} documents")
    
    return standardized_docs
