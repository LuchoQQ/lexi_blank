# Guía de Uso - Sistema de Recuperación de Documentos Legales

## Introducción

Este sistema permite la búsqueda y recuperación de documentos legales utilizando técnicas avanzadas de procesamiento de lenguaje natural y bases de datos gráficas y vectoriales. El sistema implementa un enfoque de búsqueda federada que combina:

1. **Búsqueda vectorial** con Weaviate
2. **Búsqueda basada en grafos** con Neo4j
3. **Búsqueda léxica** con BM25

## Requisitos Previos

- Python 3.7 o superior
- Docker y Docker Compose
- Dependencias especificadas en `install_dependencies.py`

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/sistema-recuperacion-legal.git
cd sistema-recuperacion-legal
```

### 2. Instalar dependencias

```bash
python setup/install_dependencies.py
```

### 3. Configurar el sistema

```bash
python setup/setup_system.py
```

Este comando:
- Verifica la instalación de dependencias
- Configura Weaviate y Neo4j con Docker
- Crea un archivo de configuración predeterminado si no existe

## Uso Básico

### Realizar una búsqueda

```bash
python main.py --query "estafa defraudación incumplimiento contractual"
```

### Configurar el sistema sin ejecutar búsqueda de ejemplo

```bash
python main.py --setup
```

### Opciones adicionales

```bash
python main.py --help
```

## Flujo de Procesamiento

El sistema implementa un flujo de procesamiento optimizado:

### 1. Expansión de Consulta Multi-perspectiva

La consulta del usuario se procesa para:
- Clasificar en categorías legales (penal, civil, comercial, etc.)
- Extraer entidades legales clave (acciones, sujetos, objetos, lugares, tiempos)
- Generar sub-consultas especializadas para cada categoría relevante

### 2. Búsqueda Multi-modal Federada

Las consultas expandidas se envían en paralelo a:
- **Weaviate**: Para búsqueda vectorial semántica
- **Neo4j**: Para búsqueda basada en relaciones entre artículos y leyes
- **BM25**: Para búsqueda léxica de coincidencia de términos

### 3. Fusión Inteligente de Resultados

Los resultados de las diferentes fuentes se combinan mediante:
- Ponderación configurable de cada fuente de búsqueda
- Eliminación de duplicados
- Normalización de puntuaciones
- Ordenamiento por relevancia

## Configuración Avanzada

El sistema se configura mediante el archivo `config.yaml`:

### Configuración de Weaviate

```yaml
weaviate:
  enabled: true
  url: "http://localhost:8080"
  api_key: null
  collection_name: "ArticulosLegales"
  embedding_model: "paraphrase-multilingual-MiniLM-L12-v2"
  use_cache: true
```

### Configuración de Neo4j

```yaml
neo4j:
  enabled: true
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "password"
```

### Configuración de BM25

```yaml
bm25:
  enabled: true
```

### Configuración de Recuperación

```yaml
retrieval:
  top_n: 5
  weights: [0.5, 0.3, 0.2]  # vectorial, grafo, léxico
  save_results: true
  results_dir: "results"
  fusion_strategy: "weighted_max"
  min_score_threshold: 0.3
```

## Estructura del Proyecto

```
sistema-recuperacion-legal/
├── main.py                 # Punto de entrada principal
├── config.yaml             # Configuración del sistema
├── data/                   # Directorio para archivos de datos
├── cache/                  # Directorio para caché de embeddings
├── results/                # Resultados de búsquedas guardados
├── setup/                  # Scripts de configuración
│   ├── install_dependencies.py
│   ├── setup_system.py
│   ├── setup_neo4j.py
│   └── setup_weaviate.py
└── src/                    # Código fuente
    ├── config_loader.py
    ├── data_loader.py
    ├── neo4j_utils.py
    └── weaviate_utils.py
```

## Técnicas Implementadas

### 1. Expansión de Consulta Multi-perspectiva

- **Clasificación temática**: Identifica las categorías legales más relevantes para la consulta
- **Extracción de entidades**: Reconoce acciones, sujetos, objetos y contexto temporal/espacial
- **Generación de subconsultas**: Crea variantes especializadas para mejorar la cobertura

### 2. Búsqueda Federada

- **Búsqueda vectorial**: Captura la semántica y el significado contextual
- **Búsqueda por grafo**: Explora relaciones y conexiones entre leyes y artículos
- **Búsqueda léxica**: Encuentra coincidencias basadas en términos específicos

### 3. Procesamiento Paralelo

- Ejecución concurrente de búsquedas para minimizar latencia
- Aprovechamiento eficiente de recursos computacionales
- Escalabilidad para manejar grandes volúmenes de consultas

## Extensiones Futuras

- Integración de modelos de IA específicos para el dominio legal
- Implementación de análisis de precedentes jurídicos
- Extracción de argumentos y razonamiento legal
- Interfaz de usuario web o API REST