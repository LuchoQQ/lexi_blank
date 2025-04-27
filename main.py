"""
Script principal para el sistema de recuperación de documentos legales.
Configuración inicial básica que verifica conexiones y genera embeddings si no están en caché.
"""
import os
import argparse
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Tuple

# Importar módulos del sistema
from src.config_loader import load_config
from src.data_loader import load_json_data
from src.weaviate_utils import connect_weaviate, create_weaviate_schema, store_embeddings_weaviate
from src.neo4j_utils import connect_neo4j, create_neo4j_nodes, create_law_relationship, check_data_exists

# Cargar variables de entorno
load_dotenv()

# Rutas por defecto
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

def check_connections(config: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Verifica las conexiones a las bases de datos configuradas.
    
    Args:
        config: Diccionario de configuración
        
    Returns:
        Tupla con los clientes de conexión (weaviate_client, neo4j_driver)
    """
    weaviate_client = None
    neo4j_driver = None
    
    # Verificar conexión a Weaviate
    if config.get("weaviate", {}).get("enabled", False):
        try:
            weaviate_url = config["weaviate"].get("url")
            weaviate_api_key = config["weaviate"].get("api_key")
            print(f"Conectando a Weaviate en {weaviate_url}...")
            weaviate_client = connect_weaviate(weaviate_url, weaviate_api_key)
            print("✓ Conexión a Weaviate exitosa")
        except Exception as e:
            print(f"✗ Error al conectar con Weaviate: {str(e)}")
    
    # Verificar conexión a Neo4j
    if config.get("neo4j", {}).get("enabled", False):
        try:
            neo4j_uri = config["neo4j"].get("uri")
            neo4j_username = config["neo4j"].get("username")
            neo4j_password = config["neo4j"].get("password")
            print(f"Conectando a Neo4j en {neo4j_uri}...")
            neo4j_driver = connect_neo4j(neo4j_uri, neo4j_username, neo4j_password)
            print("✓ Conexión a Neo4j exitosa")
        except Exception as e:
            print(f"✗ Error al conectar con Neo4j: {str(e)}")
    
    return weaviate_client, neo4j_driver

def setup_weaviate(weaviate_client, config: Dict[str, Any], documents: List[Dict[str, Any]]) -> None:
    """
    Configura Weaviate y carga documentos si es necesario.
    Genera embeddings solo si no existen en caché.
    
    Args:
        weaviate_client: Cliente de Weaviate
        config: Diccionario de configuración
        documents: Lista de documentos
    """
    if not weaviate_client:
        return
        
    collection_name = config["weaviate"].get("collection_name", "ArticulosLegales")
    embedding_model = config["weaviate"].get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
    use_cache = config["weaviate"].get("use_cache", True)
    
    # Crear esquema si no existe
    try:
        print(f"Verificando esquema para colección '{collection_name}'...")
        create_weaviate_schema(weaviate_client, collection_name)
    except Exception as e:
        print(f"Error al crear esquema: {str(e)}")
        return
    
    # Almacenar embeddings (se generarán solo si no están en caché)
    try:
        print("Almacenando documentos con embeddings en Weaviate...")
        print(f"Usando caché: {'Sí' if use_cache else 'No'}")
        store_embeddings_weaviate(
            weaviate_client,
            collection_name,
            documents,
            embedding_model=embedding_model,
            use_cache=use_cache
        )
        print("✓ Documentos almacenados correctamente")
    except Exception as e:
        print(f"Error al almacenar documentos: {str(e)}")

def main():
    """Función principal del programa."""
    print("\n=== Sistema de Recuperación de Documentos Legales ===")
    print("Iniciando configuración básica...\n")
    
    # Cargar configuración desde la ruta por defecto
    config_path = DEFAULT_CONFIG_PATH
    print(f"Cargando configuración desde {config_path}...")
    config = load_config(config_path)
    if not config:
        print("Error: No se pudo cargar la configuración.")
        return
    
    # Verificar conexiones
    weaviate_client, neo4j_driver = check_connections(config)
    
    # Cargar datos desde la ruta por defecto
    data_path = DEFAULT_DATA_PATH
    documents = []
    try:
        print(f"Cargando documentos desde {data_path}...")
        documents = load_json_data(data_path)
        print(f"✓ Cargados {len(documents)} documentos")
    except Exception as e:
        print(f"Error al cargar documentos: {str(e)}")
    
    # Configurar Weaviate si está habilitado
    if config.get("weaviate", {}).get("enabled", False) and weaviate_client and documents:
        setup_weaviate(weaviate_client, config, documents)
    
    print("\n=== Configuración completada ===")
    print("El sistema está listo para su uso.")

if __name__ == "__main__":
    main()