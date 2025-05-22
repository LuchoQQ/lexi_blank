"""
Script principal para el sistema de recuperación de documentos legales.
Integra las capacidades de búsqueda vectorial, por grafo y léxica para
proporcionar resultados de alta relevancia y precisión.
"""
import os
import argparse
import time
import concurrent.futures
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import numpy as np
from collections import defaultdict
import json

# Importar módulos del sistema
from src.config_loader import load_config
from src.data_loader import load_json_data
from src.weaviate_utils import connect_weaviate, create_weaviate_schema, store_embeddings_weaviate, search_weaviate
from src.neo4j_utils import connect_neo4j, create_neo4j_nodes, create_law_relationship, check_data_exists, search_neo4j, create_thematic_relationships, create_cross_law_relationships, create_topic_relationships_from_tags, create_semantic_content_relationships, detect_query_topics, clear_neo4j_data, create_legal_domain_relationships, search_by_legal_domain
from src.legal_domains import LEGAL_DOMAINS, detect_domains_in_query, extract_labor_entities

# Cargar variables de entorno
load_dotenv()

# Rutas por defecto
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache")

# Asegurar que existan los directorios necesarios
for directory in [DEFAULT_CACHE_PATH]:
    os.makedirs(directory, exist_ok=True)

# Categorías legales principales para clasificación
LEGAL_CATEGORIES = {
    "PENAL": ["delito", "crimen", "pena", "sentencia", "prisión", "multa", "cárcel", "homicidio", "robo", "estafa", "defraudación", "lesiones"],
    "CIVIL": ["contrato", "obligación", "derecho real", "propiedad", "posesión", "daño", "perjuicio", "indemnización", "responsabilidad civil", "herencia", "testamento", "matrimonio", "divorcio"],
    "COMERCIAL": ["sociedad", "empresa", "accionista", "comerciante", "compraventa", "mercantil", "cheque", "pagaré", "quiebra", "concurso", "patente", "marca"],
    "ADMINISTRATIVO": ["administración pública", "acto administrativo", "procedimiento administrativo", "recurso administrativo", "contrato administrativo", "función pública", "servicio público"],
    "LABORAL": ["trabajador", "empleador", "contrato de trabajo", "salario", "despido", "indemnización laboral", "sindicato", "huelga", "convenio colectivo"],
    "CONSTITUCIONAL": ["derecho fundamental", "garantía constitucional", "amparo", "habeas corpus", "habeas data", "inconstitucionalidad", "acción de tutela"],
    "PROCESAL": ["demanda", "contestación", "prueba", "audiencia", "sentencia", "recurso", "apelación", "casación", "medida cautelar", "embargo", "ejecución"]
}

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
            weaviate_url = config["weaviate"].get("url", "http://localhost:8080")
            weaviate_api_key = config["weaviate"].get("api_key")
            print(f"Conectando a Weaviate en {weaviate_url}...")
            weaviate_client = connect_weaviate(weaviate_url, weaviate_api_key)
            print("✓ Conexión a Weaviate exitosa")
        except Exception as e:
            print(f"✗ Error al conectar con Weaviate: {str(e)}")
    
    # Verificar conexión a Neo4j
    if config.get("neo4j", {}).get("enabled", False):
        try:
            neo4j_uri = config["neo4j"].get("uri", "bolt://localhost:7687")
            neo4j_username = config["neo4j"].get("username", "neo4j")
            neo4j_password = config["neo4j"].get("password", "password")
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

def setup_neo4j_data(neo4j_driver, config: Dict[str, Any], documents: List[Dict[str, Any]]) -> None:
    """
    Configura Neo4j y carga documentos si es necesario.
    Crea automáticamente relaciones basadas en contenido y tags.
    
    Args:
        neo4j_driver: Driver de Neo4j
        config: Diccionario de configuración
        documents: Lista de documentos
    """
    if not neo4j_driver:
        return
        
    # Verificar si ya existen datos en Neo4j
    data_exists = check_data_exists(neo4j_driver)
    if data_exists:
        print("Ya existen datos en Neo4j, omitiendo carga...")
        return
    
    # Crear nodos de artículos
    print("Creando nodos de artículos en Neo4j...")
    article_ids = create_neo4j_nodes(neo4j_driver, documents)
    print(f"✓ Creados {len(article_ids)} nodos de artículos")
    
    # Agrupar artículos por ley
    law_articles = defaultdict(list)
    for doc in documents:
        # Extraer law_name del documento o de metadata si está disponible
        law_name = doc.get("law_name")
        if not law_name and "metadata" in doc:
            law_name = doc["metadata"].get("code")
            
        article_id = doc.get("article_id")
        if not article_id and "metadata" in doc:
            # Crear article_id a partir de código y número de artículo si está disponible
            code = doc["metadata"].get("code")
            article_num = doc["metadata"].get("article")
            if code and article_num:
                article_id = f"{code}_{article_num}"
                
        if law_name and article_id:
            law_articles[law_name].append(article_id)
    
    # Crear nodos de leyes y relaciones
    print("Creando nodos de leyes y relaciones...")
    for law_name, article_ids in law_articles.items():
        create_law_relationship(neo4j_driver, law_name, article_ids)
    
    # Crear relaciones temáticas basadas en contenido y estructura
    try:
        print("Creando relaciones basadas en contenido...")
        create_semantic_content_relationships(neo4j_driver)
        
        print("Creando relaciones entre códigos y leyes...")
        create_cross_law_relationships(neo4j_driver, documents)
        
        print("Creando relaciones basadas en tags...")
        create_topic_relationships_from_tags(neo4j_driver, documents)
        
        # Crear relaciones de dominios legales específicos
        print("Creando relaciones de dominios legales específicos...")
        create_legal_domain_relationships(neo4j_driver, documents)
        
        print("✓ Relaciones creadas correctamente")
    except Exception as e:
        print(f"Error al crear relaciones: {str(e)}")

def classify_query(query: str) -> Dict[str, float]:
    """
    Clasifica la consulta en categorías legales calculando un puntaje para cada categoría.
    
    Args:
        query: Consulta del usuario
        
    Returns:
        Diccionario con categorías y sus puntajes
    """
    categories_scores = {}
    
    # Convertir a minúsculas para comparación
    query_lower = query.lower()
    
    # Calcular puntaje para cada categoría
    for category, keywords in LEGAL_CATEGORIES.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in query_lower:
                # Incrementar puntaje basado en la especificidad de la palabra clave
                keyword_len = len(keyword.split())
                score += keyword_len * 0.1
        
        if score > 0:
            categories_scores[category] = score
    
    # Si no se encuentra ninguna categoría, asignar un puntaje bajo a todas
    if not categories_scores:
        for category in LEGAL_CATEGORIES:
            categories_scores[category] = 0.1
    
    # Normalizar puntajes
    total_score = sum(categories_scores.values())
    if total_score > 0:
        for category in categories_scores:
            categories_scores[category] /= total_score
    
    return categories_scores

def extract_legal_entities(query: str) -> Dict[str, List[str]]:
    """
    Extrae entidades legales clave de la consulta.
    
    Args:
        query: Consulta del usuario
        
    Returns:
        Diccionario con tipos de entidades y sus valores
    """
    # En una implementación completa, aquí se usaría NER o modelos específicos
    # para extraer entidades. Esta es una implementación simplificada.
    
    # Palabras clave para diferentes tipos de entidades
    entity_keywords = {
        "ACCION": ["apropiación", "falsificación", "estafa", "robo", "hurto", "daño", "lesión", "homicidio", 
                   "defraudación", "incumplimiento", "fraude", "violación", "abuso"],
        "SUJETO": ["persona", "individuo", "empresa", "sociedad", "menor", "cónyuge", "trabajador", 
                   "empleador", "funcionario", "acreedor", "deudor", "propietario", "inquilino"],
        "OBJETO": ["bien", "propiedad", "inmueble", "vehículo", "documento", "contrato", "dinero", 
                  "información", "datos", "derecho", "obra", "marca", "patente"],
        "LUGAR": ["domicilio", "establecimiento", "local", "vivienda", "lugar público", "territorio"],
        "TIEMPO": ["plazo", "término", "período", "prescripción", "caducidad"]
    }
    
    entities = {entity_type: [] for entity_type in entity_keywords}
    words = query.lower().split()
    
    # Buscar palabras clave en la consulta
    for i, word in enumerate(words):
        for entity_type, keywords in entity_keywords.items():
            for keyword in keywords:
                # Buscar coincidencias exactas o parciales
                if keyword.lower() in word or word in keyword.lower():
                    # Tratar de extraer un contexto (n-grama) alrededor de la palabra clave
                    start = max(0, i - 2)
                    end = min(len(words), i + 3)
                    context = " ".join(words[start:end])
                    if context not in entities[entity_type]:
                        entities[entity_type].append(context)
    
    return entities

def generate_expanded_queries(query: str, category_scores: Dict[str, float], entities: Dict[str, List[str]]) -> List[str]:
    """
    Genera consultas expandidas basadas en la categorización y entidades extraídas.
    
    Args:
        query: Consulta original del usuario
        category_scores: Puntajes de categorías
        entities: Entidades extraídas
        
    Returns:
        Lista de consultas expandidas
    """
    expanded_queries = [query]  # Incluir la consulta original
    
    # Seleccionar las categorías más relevantes (top 2)
    top_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    
    # Para cada categoría relevante, generar consultas específicas
    for category, score in top_categories:
        if score < 0.1:  # Ignorar categorías con puntaje muy bajo
            continue
            
        # Añadir términos específicos de la categoría
        category_keywords = LEGAL_CATEGORIES.get(category, [])
        if category_keywords:
            # Seleccionar algunas palabras clave relevantes (no todas para evitar sobreexpansión)
            selected_keywords = category_keywords[:3]
            category_query = f"{query} {' '.join(selected_keywords)}"
            expanded_queries.append(category_query)
    
    # Generar consultas basadas en entidades extraídas
    for entity_type, entity_values in entities.items():
        if not entity_values:
            continue
            
        # Usar hasta 2 entidades de cada tipo para evitar consultas demasiado largas
        for value in entity_values[:2]:
            entity_query = f"{query} {value}"
            if entity_query not in expanded_queries:
                expanded_queries.append(entity_query)
    
    # Eliminar duplicados mientras se mantiene el orden
    return list(dict.fromkeys(expanded_queries))

def search_with_weaviate(weaviate_client, config: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """
    Realiza una búsqueda vectorial en Weaviate.
    
    Args:
        weaviate_client: Cliente de Weaviate
        config: Diccionario de configuración
        query: Consulta expandida
        
    Returns:
        Lista de resultados relevantes
    """
    if not weaviate_client:
        return []
        
    collection_name = config["weaviate"].get("collection_name", "ArticulosLegales")
    embedding_model = config["weaviate"].get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
    use_cache = config["weaviate"].get("use_cache", True)
    top_n = config["retrieval"].get("top_n", 10)
    
    try:
        return search_weaviate(
            weaviate_client,
            collection_name,
            query,
            embedding_model=embedding_model,
            top_n=top_n,
            use_cache=use_cache
        )
    except Exception as e:
        print(f"Error en búsqueda vectorial: {str(e)}")
        return []

def search_with_neo4j(neo4j_driver, config: Dict[str, Any], query: str, entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Realiza una búsqueda basada en grafo en Neo4j.
    Usa detección automática de temas basada en la estructura del grafo.
    
    Args:
        neo4j_driver: Driver de Neo4j
        config: Diccionario de configuración
        query: Consulta expandida
        entities: Entidades extraídas de la consulta
        
    Returns:
        Lista de resultados relevantes
    """
    if not neo4j_driver:
        return []
        
    print(f"Realizando búsqueda en grafo para: '{query}'")
    
    # Configuración de búsqueda
    limit = config.get("neo4j", {}).get("limit", 10)
    
    # 1. Primero, intentar búsqueda por dominios legales específicos
    domain_results = search_by_legal_domain(neo4j_driver, query, limit)
    
    # Si encontramos resultados por dominio, usarlos
    if domain_results:
        print(f"Encontrados {len(domain_results)} resultados por dominios legales específicos")
        return domain_results
    
    # 2. Si no hay resultados por dominio, realizar búsqueda tradicional
    # Detectar temas relevantes en la consulta
    topics = detect_query_topics(query, neo4j_driver)
    print(f"Temas detectados: {topics}")
    
    # Preparar parámetros de búsqueda
    search_params = {
        "keywords": query.split(),
        "topics": topics
    }
    
    # Añadir entidades si están disponibles
    if entities:
        for entity_type, values in entities.items():
            if values:
                search_params[entity_type] = values
    
    # Realizar búsqueda en Neo4j
    results = search_neo4j(neo4j_driver, search_params, limit)
    
    # Formatear resultados
    formatted_results = []
    for result in results:
        formatted_result = {
            "article_id": result.get("article_id", ""),
            "content": result.get("content", ""),
            "law_name": result.get("law_name", ""),
            "article_number": result.get("article_number", ""),
            "category": result.get("category", ""),
            "source": result.get("source", ""),
            "score": result.get("score", 0.0)
        }
        formatted_results.append(formatted_result)
    
    print(f"Encontrados {len(formatted_results)} resultados en Neo4j")
    return formatted_results

def search_with_bm25(config: Dict[str, Any], query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Realiza una búsqueda léxica usando BM25.
    
    Args:
        config: Diccionario de configuración
        query: Consulta expandida
        documents: Lista de documentos
        
    Returns:
        Lista de resultados relevantes
    """
    # Importar localmente para evitar dependencia si el módulo no está disponible
    try:
        from rank_bm25 import BM25Okapi
        import re
    except ImportError:
        print("Error: No se pudo importar rank_bm25")
        return []
    
    top_n = config["retrieval"].get("top_n", 10)
    
    # Preprocesar documentos
    processed_docs = []
    for doc in documents:
        content = doc.get("content", "")
        if not content:
            continue
            
        # Preprocesar: convertir a minúsculas y tokenizar
        tokens = re.findall(r'\w+', content.lower())
        processed_docs.append(tokens)
    
    if not processed_docs:
        return []
    
    # Inicializar BM25
    bm25 = BM25Okapi(processed_docs)
    
    # Tokenizar consulta
    query_tokens = re.findall(r'\w+', query.lower())
    
    # Obtener puntuaciones
    try:
        scores = bm25.get_scores(query_tokens)
        
        # Ordenar documentos por puntuación y seleccionar los top_n
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        # Construir resultados
        results = []
        for i in top_indices:
            if scores[i] > 0:  # Solo incluir resultados con puntuación positiva
                doc = documents[i]
                results.append({
                    "content": doc.get("content", ""),
                    "article_id": doc.get("article_id", ""),
                    "law_name": doc.get("law_name", ""),
                    "article_number": doc.get("article_number", ""),
                    "category": doc.get("category", ""),
                    "source": doc.get("source", ""),
                    "score": float(scores[i])
                })
                
        return results
    except Exception as e:
        print(f"Error en búsqueda BM25: {str(e)}")
        return []

def merge_search_results(results_list: List[List[Dict[str, Any]]], weights: List[float] = None) -> List[Dict[str, Any]]:
    """
    Fusiona los resultados de diferentes métodos de búsqueda con un enfoque ponderado.
    
    Args:
        results_list: Lista de listas de resultados de diferentes métodos
        weights: Pesos para cada método de búsqueda (opcional)
        
    Returns:
        Lista fusionada de resultados
    """
    if not results_list:
        return []
        
    # Si no se proporcionan pesos, asumir pesos iguales
    if not weights:
        weights = [1.0] * len(results_list)
    elif len(weights) != len(results_list):
        weights = [1.0] * len(results_list)
    
    # Normalizar pesos
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1.0 / len(weights)] * len(weights)
    
    # Combinar todos los resultados en un diccionario con puntuaciones ponderadas
    merged = {}
    for i, results in enumerate(results_list):
        weight = weights[i]
        for result in results:
            # Usar article_id como clave de fusión
            key = result.get("article_id", "")
            if not key:
                # Generar un identificador único basado en el contenido
                content = result.get("content", "")
                if content:
                    key = f"content_{hash(content)}"
                else:
                    # Si no hay article_id ni contenido, generar un ID único
                    key = f"item_{i}_{results.index(result)}"
            
            if key in merged:
                # Actualizar puntuación con el máximo ponderado
                merged[key]["score"] = max(merged[key]["score"], result.get("score", 0) * weight)
            else:
                # Añadir nuevo resultado con puntuación ponderada
                result_copy = result.copy()
                result_copy["score"] = result.get("score", 0) * weight
                merged[key] = result_copy
    
    # Convertir el diccionario a lista y ordenar por puntuación
    results = list(merged.values())
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return results


def search_query(query: str, config: Dict[str, Any], weaviate_client=None, neo4j_driver=None, documents=None) -> List[Dict[str, Any]]:
    """
    Versión mejorada con especialización legal
    """
    print(f"\nProcesando consulta: '{query}'")
    start_time = time.time()
    
    # 1. Clasificar y extraer información de la consulta
    category_scores = classify_query(query)
    entities = extract_legal_entities(query)
    labor_domains = detect_domains_in_query(query)
    labor_entities = extract_labor_entities(query)
    
    # 2. Realizar búsquedas en diferentes fuentes
    all_results = []
    weights = []
    
    # Búsqueda vectorial en Weaviate
    if weaviate_client and config.get("weaviate", {}).get("enabled", False):
        weaviate_results = search_with_weaviate(weaviate_client, config, query)
        if weaviate_results:
            all_results.append(weaviate_results)
            weights.append(0.4)  # 40% peso para búsqueda vectorial
    
    # Búsqueda en grafo Neo4j
    if neo4j_driver and config.get("neo4j", {}).get("enabled", False):
        neo4j_results = search_with_neo4j(neo4j_driver, config, query, entities)
        if neo4j_results:
            all_results.append(neo4j_results)
            weights.append(0.4)  # 40% peso para búsqueda en grafo
    
    # Búsqueda léxica BM25
    if documents and config.get("bm25", {}).get("enabled", False):
        bm25_results = search_with_bm25(config, query, documents)
        if bm25_results:
            all_results.append(bm25_results)
            weights.append(0.2)  # 20% peso para búsqueda léxica
    
    # 3. Fusionar resultados
    if not all_results:
        print("No se encontraron resultados en ninguna fuente de búsqueda")
        return []
        
    base_results = merge_search_results(all_results, weights)
    
    # 4. NUEVA: Aplicar especialización legal
    print("Aplicando especialización legal...")
    
    # Importar el módulo especializado
    from src.specialized_retriever import enhance_search_with_specialization
    
    specialized_results = enhance_search_with_specialization(
        query=query,
        base_results=base_results,
        detected_domains=labor_domains,
        category_scores=category_scores,
        neo4j_driver=neo4j_driver
    )
    
    # 5. Continuar con enriquecimiento y verificación de artículos críticos
    results = enrich_labor_context(specialized_results, query, neo4j_driver, labor_domains, labor_entities)
    
    # 6. Tu lógica existente para artículos críticos faltantes...
    # (Esta parte se podría simplificar ya que la especialización debería manejar esto)
    
    # 7. Limitar resultados finales
    top_n = config.get("retrieval", {}).get("top_n", 15)
    final_results = results[:top_n]
    
    end_time = time.time()
    print(f"Búsqueda completada en {end_time - start_time:.2f} segundos, {len(final_results)} resultados.")
    
    return final_results

# También podrías crear una versión específica para casos laborales:

def search_labor_case(query: str, config: Dict[str, Any], neo4j_driver=None) -> List[Dict[str, Any]]:
    """
    Búsqueda especializada para casos laborales con mayor precisión
    """
    from src.specialized_retriever import LaborLawRetriever
    from src.legal_domains import detect_domains_in_query
    
    print(f"Búsqueda especializada laboral: '{query}'")
    
    # Búsqueda directa por dominios laborales
    labor_domains = detect_domains_in_query(query)
    
    if neo4j_driver and labor_domains:
        domain_results = []
        for domain in labor_domains:
            results = search_by_legal_domain(neo4j_driver, query, 20)  # Más resultados para filtrar
            domain_results.extend(results)
        
        # Aplicar especialización
        retriever = LaborLawRetriever()
        specialized_results = retriever.retrieve(query, domain_results)
        
        print(f"Encontrados {len(specialized_results)} resultados especializados")
        return specialized_results
    
    return []

def enrich_labor_context(results_list, query_text, neo4j_driver, labor_domains, labor_entities):
    """
    Enriquece los resultados de búsqueda con contexto específico laboral,
    asegurando que se incluyan artículos relevantes para el caso específico.
    
    Args:
        results_list: Lista de resultados de búsqueda
        query_text: Texto de la consulta del usuario
        neo4j_driver: Driver de conexión a Neo4j
        labor_domains: Dominios laborales identificados
        labor_entities: Entidades laborales extraídas
        
    Returns:
        Lista enriquecida de resultados
    """
    if not neo4j_driver:
        return results_list
        
    print("Enriqueciendo resultados con contexto laboral específico...")
    
    # Artículos ya incluidos en los resultados (para evitar duplicados)
    included_articles = set()
    for result in results_list:
        article_id = f"{result.get('law_name', '')}-{result.get('article_number', '')}"
        included_articles.add(article_id)
    
    additional_results = []
    
    # 1. Buscar artículos complementarios basados en dominios laborales
    if labor_domains:
        try:
            # Buscar artículos que contengan procedimientos específicos
            procedural_patterns = [
                "procedimiento", "plazo", "requisito", "formulario", "solicitud", 
                "presentar", "corresponde", "deberá", "días", "hábiles"
            ]
            
            with neo4j_driver.session() as session:
                for pattern in procedural_patterns[:3]:  # Limitar a 3 patrones para eficiencia
                    query = """
                    MATCH (d:LegalDomain)
                    WHERE d.name IN $domains
                    MATCH (d)<-[:RELATED_TO]-(a:Article)
                    WHERE toLower(a.content) CONTAINS $pattern
                    RETURN a.article_id as article_id, 
                           a.content as content,
                           a.law_name as law_name,
                           a.article_number as article_number,
                           a.category as category,
                           a.source as source
                    LIMIT 5
                    """
                    
                    result = session.run(query, domains=labor_domains, pattern=pattern)
                    
                    for record in result:
                        article_id = f"{record['law_name']}-{record['article_number']}"
                        
                        # Evitar duplicados
                        if article_id in included_articles:
                            continue
                            
                        # Crear resultado y añadirlo
                        proc_article = {
                            "article_id": record["article_id"],
                            "content": record["content"],
                            "law_name": record["law_name"],
                            "article_number": record["article_number"],
                            "category": record["category"],
                            "source": record["source"],
                            "score": 50.0,  # Score alto para asegurar inclusión
                            "domain": "Procedimiento"
                        }
                        
                        additional_results.append(proc_article)
                        included_articles.add(article_id)
                        print(f"Añadido artículo de procedimiento: {article_id}")
        except Exception as e:
            print(f"Error al buscar artículos de procedimiento: {str(e)}")
    
    # 2. Si hay entidades legales específicas (leyes, organismos), buscar artículos relacionados
    if labor_entities and 'laws' in labor_entities and labor_entities['laws']:
        try:
            with neo4j_driver.session() as session:
                for law in labor_entities['laws'][:2]:  # Limitar a 2 leyes
                    query = """
                    MATCH (l:Law)
                    WHERE toLower(l.name) CONTAINS toLower($law)
                    MATCH (l)-[:CONTAINS]->(a:Article)
                    RETURN a.article_id as article_id, 
                           a.content as content,
                           a.law_name as law_name,
                           a.article_number as article_number,
                           a.category as category,
                           a.source as source
                    ORDER BY a.article_number ASC
                    LIMIT 5
                    """
                    
                    result = session.run(query, law=law)
                    
                    for record in result:
                        article_id = f"{record['law_name']}-{record['article_number']}"
                        
                        # Evitar duplicados
                        if article_id in included_articles:
                            continue
                            
                        # Crear resultado y añadirlo
                        law_article = {
                            "article_id": record["article_id"],
                            "content": record["content"],
                            "law_name": record["law_name"],
                            "article_number": record["article_number"],
                            "category": record["category"],
                            "source": record["source"],
                            "score": 40.0,  # Score alto para asegurar inclusión
                            "domain": "Normativa_Específica"
                        }
                        
                        additional_results.append(law_article)
                        included_articles.add(article_id)
                        print(f"Añadido artículo de ley específica: {article_id}")
        except Exception as e:
            print(f"Error al buscar artículos de leyes específicas: {str(e)}")
    
    # Si no encontramos artículos adicionales, devolver los originales
    if not additional_results:
        return results_list
    
    # Combinar resultados originales con los adicionales
    combined_results = results_list + additional_results
    
    # Reordenar por puntuación
    combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    print(f"Añadidos {len(additional_results)} artículos complementarios para enriquecer el contexto laboral")
    
    return combined_results

def enrich_labor_context(results_list, query_text, neo4j_driver, labor_domains, labor_entities):
    """
    Enriquece los resultados de búsqueda con contexto específico laboral,
    asegurando que se incluyan artículos relevantes para el caso específico.
    
    Args:
        results_list: Lista de resultados de búsqueda
        query_text: Texto de la consulta del usuario
        neo4j_driver: Driver de conexión a Neo4j
        labor_domains: Dominios laborales identificados
        labor_entities: Entidades laborales extraídas
        
    Returns:
        Lista enriquecida de resultados
    """
    if not neo4j_driver:
        return results_list
        
    print("Enriqueciendo resultados con contexto laboral específico...")
    
    # Artículos ya incluidos en los resultados (para evitar duplicados)
    included_articles = set()
    for result in results_list:
        article_id = f"{result.get('law_name', '')}-{result.get('article_number', '')}"
        included_articles.add(article_id)
    
    additional_results = []
    
    # 1. Buscar artículos complementarios basados en dominios laborales
    if labor_domains:
        try:
            # Buscar artículos que contengan procedimientos específicos
            procedural_patterns = [
                "procedimiento", "plazo", "requisito", "formulario", "solicitud", 
                "presentar", "corresponde", "deberá", "días", "hábiles"
            ]
            
            with neo4j_driver.session() as session:
                for pattern in procedural_patterns[:3]:  # Limitar a 3 patrones para eficiencia
                    query = """
                    MATCH (d:LegalDomain)
                    WHERE d.name IN $domains
                    MATCH (d)<-[:RELATED_TO]-(a:Article)
                    WHERE toLower(a.content) CONTAINS $pattern
                    RETURN a.article_id as article_id, 
                           a.content as content,
                           a.law_name as law_name,
                           a.article_number as article_number,
                           a.category as category,
                           a.source as source
                    LIMIT 5
                    """
                    
                    result = session.run(query, domains=labor_domains, pattern=pattern)
                    
                    for record in result:
                        article_id = f"{record['law_name']}-{record['article_number']}"
                        
                        # Evitar duplicados
                        if article_id in included_articles:
                            continue
                            
                        # Crear resultado y añadirlo
                        proc_article = {
                            "article_id": record["article_id"],
                            "content": record["content"],
                            "law_name": record["law_name"],
                            "article_number": record["article_number"],
                            "category": record["category"],
                            "source": record["source"],
                            "score": 50.0,  # Score alto para asegurar inclusión
                            "domain": "Procedimiento"
                        }
                        
                        additional_results.append(proc_article)
                        included_articles.add(article_id)
                        print(f"Añadido artículo de procedimiento: {article_id}")
        except Exception as e:
            print(f"Error al buscar artículos de procedimiento: {str(e)}")
    
    # 2. Si hay entidades legales específicas (leyes, organismos), buscar artículos relacionados
    if labor_entities and 'laws' in labor_entities and labor_entities['laws']:
        try:
            with neo4j_driver.session() as session:
                for law in labor_entities['laws'][:2]:  # Limitar a 2 leyes
                    query = """
                    MATCH (l:Law)
                    WHERE toLower(l.name) CONTAINS toLower($law)
                    MATCH (l)-[:CONTAINS]->(a:Article)
                    RETURN a.article_id as article_id, 
                           a.content as content,
                           a.law_name as law_name,
                           a.article_number as article_number,
                           a.category as category,
                           a.source as source
                    ORDER BY a.article_number ASC
                    LIMIT 5
                    """
                    
                    result = session.run(query, law=law)
                    
                    for record in result:
                        article_id = f"{record['law_name']}-{record['article_number']}"
                        
                        # Evitar duplicados
                        if article_id in included_articles:
                            continue
                            
                        # Crear resultado y añadirlo
                        law_article = {
                            "article_id": record["article_id"],
                            "content": record["content"],
                            "law_name": record["law_name"],
                            "article_number": record["article_number"],
                            "category": record["category"],
                            "source": record["source"],
                            "score": 40.0,  # Score alto para asegurar inclusión
                            "domain": "Normativa_Específica"
                        }
                        
                        additional_results.append(law_article)
                        included_articles.add(article_id)
                        print(f"Añadido artículo de ley específica: {article_id}")
        except Exception as e:
            print(f"Error al buscar artículos de leyes específicas: {str(e)}")
    
    # Si no encontramos artículos adicionales, devolver los originales
    if not additional_results:
        return results_list
    
    # Combinar resultados originales con los adicionales
    combined_results = results_list + additional_results
    
    # Reordenar por puntuación
    combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    print(f"Añadidos {len(additional_results)} artículos complementarios para enriquecer el contexto laboral")
    
    return combined_results

def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    Formatea los resultados de búsqueda para su presentación.
    
    Args:
        results: Lista de resultados de búsqueda
        
    Returns:
        Texto formateado con los resultados
    """
    if not results:
        return "No se encontraron resultados para la consulta."
    
    formatted = "\n=== RESULTADOS DE BÚSQUEDA ===\n\n"
    
    for i, result in enumerate(results, 1):
        formatted += f"RESULTADO #{i} (Relevancia: {result.get('score', 0):.2f})\n"
        formatted += f"Ley/Código: {result.get('law_name', 'N/A')}\n"
        formatted += f"Artículo: {result.get('article_number', 'N/A')}\n"
        formatted += f"Categoría: {result.get('category', 'N/A')}\n"
        formatted += "-" * 50 + "\n"
        formatted += f"{result.get('content', 'Sin contenido')}\n"
        formatted += "=" * 80 + "\n\n"
    
    return formatted

def setup_system(config_path: str = DEFAULT_CONFIG_PATH, data_path: str = DEFAULT_DATA_PATH):
    """
    Configura todo el sistema: verifica conexiones y carga datos iniciales.
    
    Args:
        config_path: Ruta al archivo de configuración
        data_path: Ruta al directorio de datos
    """
    print("\n=== Configuración del Sistema de Recuperación de Documentos Legales ===")
    
    # Cargar configuración
    print(f"Cargando configuración desde {config_path}...")
    config = load_config(config_path)
    if not config:
        print("Error: No se pudo cargar la configuración.")
        return
    
    # Verificar conexiones
    weaviate_client, neo4j_driver = check_connections(config)
    
    # Cargar datos
    documents = []
    try:
        print(f"Cargando documentos desde {data_path}...")
        documents = load_json_data(data_path)
        print(f"✓ Cargados {len(documents)} documentos")
    except Exception as e:
        print(f"Error al cargar documentos: {str(e)}")
        return
    
    # Configurar Weaviate
    if config.get("weaviate", {}).get("enabled", False) and weaviate_client and documents:
        setup_weaviate(weaviate_client, config, documents)
    
    # Configurar Neo4j
    if config.get("neo4j", {}).get("enabled", False) and neo4j_driver and documents:
        setup_neo4j_data(neo4j_driver, config, documents)
    
    # Cerrar conexiones
    if neo4j_driver:
        neo4j_driver.close()
    
    print("\n=== Configuración completada ===")
    print("El sistema está listo para su uso.")

def main():
    """Función principal del programa."""
    parser = argparse.ArgumentParser(description="Sistema de Recuperación de Documentos Legales")
    parser.add_argument("--setup", action="store_true", help="Configurar el sistema antes de ejecutarlo")
    parser.add_argument("--query", type=str, help="Consulta para buscar documentos legales")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Ruta al archivo de configuración")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Ruta al directorio de datos")
    parser.add_argument("--clear-neo4j", action="store_true", help="Eliminar todos los datos de Neo4j")
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    if not config:
        print("Error: No se pudo cargar la configuración.")
        return
    
    # Verificar conexiones
    weaviate_client, neo4j_driver = check_connections(config)
    
    # Si se solicita limpiar Neo4j
    if args.clear_neo4j:
        if neo4j_driver:
            confirm = input("¿Está seguro de que desea eliminar todos los datos de Neo4j? Esta acción es irreversible. (s/n): ")
            if confirm.lower() == 's':
                clear_neo4j_data(neo4j_driver)
            else:
                print("Operación cancelada.")
        else:
            print("No se pudo conectar a Neo4j para limpiar los datos.")
        
        if neo4j_driver:
            neo4j_driver.close()
        return
    
    # Si se solicita configurar el sistema
    if args.setup:
        setup_system(args.config, args.data)
        return
    
    # Si no hay consulta, mostrar ayuda
    if not args.query:
        print("\n=== Sistema de Recuperación de Documentos Legales ===")
        print("Utilice --query para realizar una búsqueda o --setup para configurar el sistema")
        print("Ejemplo: python main.py --query \"estafa defraudación incumplimiento contractual\"")
        parser.print_help()
        
        if neo4j_driver:
            neo4j_driver.close()
        return
    
    # Cargar documentos para búsqueda léxica si está habilitada
    documents = None
    if config.get("bm25", {}).get("enabled", False):
        try:
            print(f"Cargando documentos desde {args.data} para búsqueda léxica...")
            documents = load_json_data(args.data)
            print(f"✓ Cargados {len(documents)} documentos")
        except Exception as e:
            print(f"Error al cargar documentos: {str(e)}")
    
    # Realizar búsqueda
    results = search_query(args.query, config, weaviate_client, neo4j_driver, documents)
    
    # Formatear y mostrar resultados
    formatted_results = format_search_results(results)
    print(formatted_results)
    
    # Guardar resultados en un archivo si está configurado
    if config.get("retrieval", {}).get("save_results", False):
        results_dir = config.get("retrieval", {}).get("results_dir", "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_file = os.path.join(results_dir, f"results_{timestamp}.txt")
        
        with open(results_file, "w", encoding="utf-8") as f:
            f.write(f"Consulta: {args.query}\n\n")
            f.write(formatted_results)
            
        print(f"Resultados guardados en {results_file}")
        
    # Cerrar conexiones
    if neo4j_driver:
        neo4j_driver.close()
    # Weaviate no necesita cierre explícito
    
if __name__ == "__main__":
    main()