"""
Script para configurar y ejecutar todo el sistema de recuperación de documentos legales.
"""
import os
import subprocess
import sys
import time
import argparse

# Agregar el directorio raíz al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_command(command, cwd=None):
    """Ejecuta un comando y muestra su salida en tiempo real."""
    print(f"Ejecutando: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=cwd
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def check_dependencies():
    """Verifica si las dependencias están instaladas."""
    print("Verificando dependencias...")
    
    try:
        import weaviate
        print("✓ weaviate-client está instalado")
    except ImportError:
        print("✗ weaviate-client no está instalado")
        return False
    
    try:
        import neo4j
        print("✓ neo4j está instalado")
    except ImportError:
        print("✗ neo4j no está instalado")
        return False
    
    try:
        import rank_bm25
        print("✓ rank_bm25 está instalado")
    except ImportError:
        print("✗ rank_bm25 no está instalado")
        return False
    
    try:
        import yaml
        print("✓ pyyaml está instalado")
    except ImportError:
        print("✗ pyyaml no está instalado")
        return False
    
    try:
        import numpy
        print("✓ numpy está instalado")
    except ImportError:
        print("✗ numpy no está instalado")
        return False
    
    try:
        import sentence_transformers
        print("✓ sentence-transformers está instalado")
    except ImportError:
        print("✗ sentence-transformers no está instalado")
        return False
    
    return True

def setup_docker_services():
    """Configura los servicios Docker (Weaviate y Neo4j)."""
    print("\n=== Configurando servicios Docker ===")
    
    # Ejecutar setup_weaviate.py
    print("\nConfigurando Weaviate...")
    run_command(f"{sys.executable} {os.path.join(os.path.dirname(__file__), 'setup_weaviate.py')}")
    
    # Ejecutar setup_neo4j.py
    print("\nConfigurando Neo4j...")
    run_command(f"{sys.executable} {os.path.join(os.path.dirname(__file__), 'setup_neo4j.py')}")

def create_default_config():
    """Crea un archivo de configuración por defecto si no existe."""
    from src.config_loader import save_config
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    
    # Si el archivo ya existe, no sobrescribirlo
    if os.path.exists(config_path):
        print(f"El archivo de configuración ya existe en {config_path}")
        return
    
    # Configuración por defecto
    default_config = {
        "weaviate": {
            "enabled": True,
            "url": "http://localhost:8080",
            "api_key": None,
            "collection_name": "ArticulosLegales",
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "use_cache": True
        },
        "neo4j": {
            "enabled": True,
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password"
        },
        "bm25": {
            "enabled": True
        },
        "retrieval": {
            "top_n": 5,
            "weights": [0.5, 0.3, 0.2],  # vectorial, grafo, léxico
            "save_results": True,
            "results_dir": "results"
        }
    }
    
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Guardar configuración
        save_config(default_config, config_path)
        print(f"Archivo de configuración creado en {config_path}")
    except Exception as e:
        print(f"Error al crear archivo de configuración: {str(e)}")

def run_search_example():
    """Ejecuta una búsqueda de ejemplo."""
    print("\n=== Ejecutando búsqueda de ejemplo ===")
    
    # Ejecutar main.py con una consulta de ejemplo
    query = "estafa defraudación incumplimiento contractual daños y perjuicios"
    run_command(f"{sys.executable} {os.path.join(os.path.dirname(__file__), '..', 'main.py')} --query \"{query}\"")

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Configurar y ejecutar el sistema de recuperación de documentos legales")
    parser.add_argument("--skip-docker", action="store_true", help="Omitir la configuración de servicios Docker")
    parser.add_argument("--skip-search", action="store_true", help="Omitir la ejecución de la búsqueda de ejemplo")
    args = parser.parse_args()
    
    print("=== Configuración del Sistema de Recuperación de Documentos Legales ===")
    
    # Verificar dependencias
    if not check_dependencies():
        print("\nAlgunas dependencias no están instaladas.")
        print("Por favor, ejecute primero el script install_dependencies.py")
        return
    
    # Crear archivo de configuración por defecto si no existe
    create_default_config()
    
    # Configurar servicios Docker (Weaviate y Neo4j)
    if not args.skip_docker:
        setup_docker_services()
    else:
        print("\nOmitiendo la configuración de servicios Docker.")
    
    # Ejecutar búsqueda de ejemplo
    if not args.skip_search:
        run_search_example()
    else:
        print("\nOmitiendo la ejecución de la búsqueda de ejemplo.")
    
    print("\n=== Configuración completada ===")
    print("Ahora puede ejecutar búsquedas con: python main.py --query \"su consulta aquí\"")

if __name__ == "__main__":
    main()