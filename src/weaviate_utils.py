"""
Module for integrating with Weaviate vector database.
"""
import weaviate
from weaviate.auth import AuthApiKey
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import os
import pickle
import hashlib

def connect_weaviate(weaviate_url: str, weaviate_api_key: Optional[str] = None) -> weaviate.Client:
    """
    Establish connection to Weaviate.
    
    Args:
        weaviate_url: URL of the Weaviate instance
        weaviate_api_key: API key for authentication (optional)
        
    Returns:
        Weaviate client instance
    """
    auth_config = AuthApiKey(api_key=weaviate_api_key) if weaviate_api_key else None
    
    try:
        client = weaviate.Client(
            url=weaviate_url,
            auth_client_secret=auth_config
        )
        
        # Verify connection
        if not client.is_ready():
            raise Exception("Weaviate is not ready")
            
        return client
    except Exception as e:
        raise Exception(f"Failed to connect to Weaviate: {str(e)}")

def create_weaviate_schema(client: weaviate.Client, collection_name: str) -> None:
    """
    Create schema for legal articles collection in Weaviate.
    
    Args:
        client: Weaviate client instance
        collection_name: Name of the collection to create
    """
    # Check if collection already exists
    try:
        schema = client.schema.get()
        classes = [c["class"] for c in schema["classes"]] if "classes" in schema else []
        
        if collection_name in classes:
            print(f"Collection '{collection_name}' already exists.")
            return
    except Exception as e:
        print(f"Error checking schema: {str(e)}")
    
    # Define default properties for legal articles
    properties = [
        {
            "name": "content",
            "description": "The full text content of the article",
            "dataType": ["text"]
        },
        {
            "name": "article_id",
            "description": "Unique identifier for the article",
            "dataType": ["string"]
        },
        {
            "name": "law_name",
            "description": "Name of the law or code",
            "dataType": ["string"]
        },
        {
            "name": "article_number",
            "description": "Article number within the law",
            "dataType": ["string"]
        },
        {
            "name": "category",
            "description": "Category or section of the law",
            "dataType": ["string"]
        },
        {
            "name": "source",
            "description": "Source of the article",
            "dataType": ["string"]
        }
    ]
    
    # Define the schema for legal articles
    schema = {
        "class": collection_name,
        "description": "Legal articles from various codes and laws",
        "vectorizer": "none",  # We'll provide our own vectors
        "properties": properties
    }
    
    try:
        client.schema.create_class(schema)
        print(f"Successfully created '{collection_name}' collection in Weaviate.")
    except Exception as e:
        # Intenta con el método alternativo si el primero falla
        try:
            client.schema.create({"classes": [schema]})
            print(f"Successfully created '{collection_name}' collection in Weaviate.")
        except Exception as e2:
            raise Exception(f"Failed to create schema: {str(e)} / {str(e2)}")

def generate_embeddings(
    documents: List[Dict[str, Any]],
    embedding_model: str,
    cache_dir: str,
    use_cache: bool = True
) -> List[Tuple[Dict[str, Any], List[float]]]:
    """
    Generate embeddings for documents with caching support.
    
    Args:
        documents: List of document dictionaries
        embedding_model: Name of the sentence-transformers model to use
        cache_dir: Directory to store cached embeddings
        use_cache: Whether to use cached embeddings if available
        
    Returns:
        List of tuples (document, embedding)
    """
    # Create cache directory if it doesn't exist
    if use_cache and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Generate a cache filename based on the documents and model
    cache_id = hashlib.md5(f"{embedding_model}_{len(documents)}".encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"embeddings_{cache_id}.pkl")
    
    # Check if cache exists and is valid
    if use_cache and os.path.exists(cache_file):
        try:
            print(f"Loading embeddings from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
    
    # Load the embedding model
    print(f"Generating embeddings using model: {embedding_model}")
    model = SentenceTransformer(embedding_model)
    
    # Generate embeddings
    result = []
    for i, doc in enumerate(documents):
        # Extract document content
        content = doc.get("content", "")
        if not content:
            print(f"Warning: Document at index {i} has no content, skipping.")
            continue
            
        # Generate embedding
        embedding = model.encode(content).tolist()
        result.append((doc, embedding))
        
        if i % 100 == 0 and i > 0:
            print(f"Generated embeddings for {i} documents...")
    
    # Save to cache if enabled
    if use_cache:
        try:
            print(f"Saving embeddings to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"Error saving cache: {str(e)}")
    
    return result

def store_embeddings_weaviate(
    client: weaviate.Client, 
    collection_name: str, 
    documents: List[Dict[str, Any]], 
    embedding_model: str,
    use_cache: bool = False
) -> None:
    """
    Generate embeddings for documents and store them in Weaviate.
    
    Args:
        client: Weaviate client instance
        collection_name: Name of the collection to store embeddings
        documents: List of document dictionaries
        embedding_model: Name of the sentence-transformers model to use
        use_cache: Whether to use cached embeddings if available
    """
    # Generate or load embeddings
    doc_embeddings = generate_embeddings(
        documents, 
        embedding_model=embedding_model,
        cache_dir="cache",
        use_cache=use_cache
    )
    
    # Create a batch process
    with client.batch as batch:
        batch.batch_size = 100
        
        for i, (doc, embedding) in enumerate(doc_embeddings):
            # Extract document content and metadata
            content = doc.get("content", "")
            if not content:
                continue
                
            # Prepare properties
            properties = {
                "content": content,
                "article_id": doc.get("article_id", f"article_{i}"),
                "law_name": doc.get("law_name", ""),
                "article_number": doc.get("article_number", ""),
                "category": doc.get("category", ""),
                "source": doc.get("source", "")
            }
            
            # Add the object with its embedding
            batch.add_data_object(
                data_object=properties,
                class_name=collection_name,
                vector=embedding
            )
            
            if i % 100 == 0 and i > 0:
                print(f"Processed {i} documents...")
                
    print(f"Successfully stored {len(doc_embeddings)} documents with embeddings in Weaviate.")

def search_weaviate(
    client: weaviate.Client,
    collection_name: str,
    query: str,
    embedding_model: str,
    top_n: int = 5,
    use_cache: bool = False
) -> List[Dict[str, Any]]:
    """
    Perform semantic search in Weaviate.
    
    Args:
        client: Weaviate client instance
        collection_name: Name of the collection to search
        query: Query text
        embedding_model: Name of the sentence-transformers model to use
        top_n: Number of results to return
        use_cache: Whether to use cached model if available
        
    Returns:
        List of matching documents with their metadata and similarity scores
    """
    # Generate query embedding with optional caching for the model
    cache_dir = "cache"
    cache_id = hashlib.md5(embedding_model.encode()).hexdigest()
    model_cache_file = os.path.join(cache_dir, f"model_{cache_id}.pkl")
    
    # Try to load cached model
    model = None
    if use_cache and os.path.exists(model_cache_file):
        try:
            print(f"Loading model from cache: {model_cache_file}")
            with open(model_cache_file, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            print(f"Error loading cached model: {str(e)}")
    
    # Load model if not loaded from cache
    if model is None:
        print(f"Loading embedding model: {embedding_model}")
        model = SentenceTransformer(embedding_model)
        
        # Save model to cache if enabled
        if use_cache:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            try:
                print(f"Saving model to cache: {model_cache_file}")
                with open(model_cache_file, 'wb') as f:
                    pickle.dump(model, f)
            except Exception as e:
                print(f"Error saving model cache: {str(e)}")
    
    # Generate embedding for the query
    query_embedding = model.encode(query)
    
    # Perform the vector search
    result = client.query.get(
        collection_name, 
        ["content", "article_id", "law_name", "article_number", "category", "source"]
    ).with_near_vector({
        "vector": query_embedding,
        "certainty": 0.7
    }).with_limit(top_n).do()
    
    # Process results
    results = []
    if "data" in result and "Get" in result["data"]:
        for item in result["data"]["Get"].get(collection_name, []):
            # Extract similarity score
            if "_additional" in item and "certainty" in item["_additional"]:
                score = item["_additional"]["certainty"]
            else:
                score = 0.0
                
            # Create result object
            result_obj = {
                "content": item.get("content", ""),
                "article_id": item.get("article_id", ""),
                "law_name": item.get("law_name", ""),
                "article_number": item.get("article_number", ""),
                "category": item.get("category", ""),
                "source": item.get("source", ""),
                "score": score
            }
            results.append(result_obj)
    
    return results
