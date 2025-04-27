# Sistema de Recuperación de Documentos Legales

Este sistema permite recuperar artículos legales utilizando múltiples métodos de búsqueda:
1. Búsqueda vectorial (semántica) con Weaviate
2. Búsqueda en grafo con Neo4j
3. Búsqueda por palabras clave con BM25

## Requisitos

- Python 3.8+
- Weaviate (opcional, para búsqueda vectorial)
- Neo4j (opcional, para búsqueda en grafo)

## Instalación

Instale las dependencias necesarias:

```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

### Directorios Principales
- `src/`: Contiene los módulos principales del sistema
  - `data_loader.py`: Carga de datos desde archivos JSON
  - `weaviate_utils.py`: Integración con Weaviate para búsqueda vectorial
  - `neo4j_utils.py`: Integración con Neo4j para búsqueda en grafo
  - `bm25_utils.py`: Implementación de búsqueda BM25 por palabras clave
  - `config_loader.py`: Carga de configuración desde archivos YAML
  - `legal_retriever.py`: Módulo principal de recuperación
- `setup/`: Contiene scripts de configuración y preparación
  - `setup_weaviate.py`: Configuración de Weaviate con Docker
  - `setup_neo4j.py`: Configuración de Neo4j con Docker
  - `setup_system.py`: Script principal para configurar todo el sistema
  - `install_dependencies.py`: Instalación de dependencias
- `data/`: Contiene los archivos de datos en formato JSON
- `main.py`: Script principal para ejecutar el sistema
- `config.yaml`: Archivo de configuración

## Configuración

Edite el archivo `config.yaml` para configurar las conexiones a Weaviate y Neo4j, así como las estrategias de recuperación.

```yaml
# Ejemplo de configuración
weaviate:
  enabled: true
  url: "http://localhost:8080"
  api_key: ""  # Agregue su clave API si es necesario
  collection_name: "ArticulosLegales"

neo4j:
  enabled: true
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "password"  # Cambie esto en producción

bm25:
  enabled: true

retrieval:
  top_n: 5
  combine_method: "weighted"
  weights:
    weaviate: 0.4
    neo4j: 0.3
    bm25: 0.3
```

## Formato de Datos

Los documentos legales deben estar en formato JSON con la siguiente estructura:

```json
[
  {
    "article_id": "cc_9",
    "law_name": "Código Civil y Comercial de la Nación",
    "article_number": "9",
    "content": "Principio de buena fe. Los derechos deben ser ejercidos de buena fe.",
    "category": "Principios generales",
    "source": "Infoleg"
  },
  {
    "article_id": "cc_10",
    "law_name": "Código Civil y Comercial de la Nación",
    "article_number": "10",
    "content": "Abuso del derecho. El ejercicio regular de un derecho propio o el cumplimiento de una obligación legal no puede constituir como ilícito ningún acto.",
    "category": "Principios generales",
    "source": "Infoleg"
  }
]
```

## Uso

### Modo Interactivo

```bash
python main.py --config config.yaml --data ./data
```

### Búsqueda Directa

```bash
python main.py --config config.yaml --data ./data --query "incumplimiento contractual daños y perjuicios"
```

## Ejemplo de Uso en Código

```python
from src.config_loader import load_config
from src.data_loader import load_json_data
from src.weaviate_utils import connect_weaviate, create_weaviate_schema, store_embeddings_weaviate
from src.neo4j_utils import connect_neo4j, create_neo4j_nodes, create_law_relationship
from src.bm25_utils import create_bm25_index
from src.legal_retriever import retrieve_legal_articles

# Cargar configuración
config = load_config("config.yaml")

# Cargar documentos
documents = load_json_data("./data")

# Inicializar Weaviate
weaviate_client = connect_weaviate(config["weaviate"]["url"], config["weaviate"]["api_key"])
create_weaviate_schema(weaviate_client, config["weaviate"]["collection_name"])
store_embeddings_weaviate(weaviate_client, config["weaviate"]["collection_name"], documents)

# Inicializar Neo4j
neo4j_driver = connect_neo4j(config["neo4j"]["uri"], config["neo4j"]["username"], config["neo4j"]["password"])
create_neo4j_nodes(neo4j_driver, documents)

# Inicializar BM25
bm25_index = create_bm25_index(documents)

# Realizar búsqueda
query = "incumplimiento contractual daños y perjuicios"
results = retrieve_legal_articles(query, config, weaviate_client, neo4j_driver, bm25_index)

# Mostrar resultados
for result in results:
    print(f"{result['law_name']} - Art. {result['article_number']}")
    print(f"Contenido: {result['content'][:200]}...")
    print(f"Score: {result.get('combined_score', 0):.2f}")
    print("-" * 80)
