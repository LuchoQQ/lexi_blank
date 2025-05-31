"""
Script principal mejorado - Sistema de recuperación semántica neutral
Sin keywords hardcodeadas ni sesgos hacia áreas legales específicas.
Análisis puramente semántico y lingüístico.
"""
import os
import argparse
import time
import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import defaultdict

# Importar módulos del sistema
from src.config_loader import load_config
from src.data_loader import load_json_data
from src.weaviate_utils import connect_weaviate, create_weaviate_schema, store_embeddings_weaviate, search_weaviate
from src.neo4j_utils import (
    connect_neo4j, create_neo4j_nodes, create_law_relationship, check_data_exists, 
    search_neo4j, create_thematic_relationships, create_cross_law_relationships, 
    create_topic_relationships_from_tags, create_semantic_content_relationships, 
    detect_query_topics, clear_neo4j_data
)

# Rutas por defecto
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache")

# Asegurar que existan los directorios necesarios
for directory in [DEFAULT_CACHE_PATH]:
    os.makedirs(directory, exist_ok=True)

def neutral_entity_extraction(query: str) -> Dict[str, Any]:
    """
    Extracción neutral de entidades usando análisis lingüístico puro.
    Sin keywords hardcodeadas ni sesgos hacia áreas específicas.
    """
    query_lower = query.lower()
    entities = {
        'tiempo': [],
        'negaciones': [],
        'query_tokens': query_lower.split(),
        'confidence': {}
    }
    
    # Intentar usar spaCy si está disponible para análisis lingüístico neutro
    try:
        import spacy
        nlp = spacy.load("es_core_news_sm")
        doc = nlp(query)
        
        # Extraer entidades nombradas de forma neutral
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME", "QUANTITY"]:
                entities['tiempo'].append(ent.text)
            # No categorizamos personas u organizaciones como "condiciones"
            # Mantenemos el análisis neutral
    except (ImportError, OSError):
        print("spaCy no disponible, usando análisis básico")
    
    # Patrones de tiempo (neutrales, sin interpretación semántica)
    time_patterns = [
        r'(\d+)\s*(años?|meses?|días?|semanas?)',
        r'durante\s+(\d+\s*\w+)',
        r'luego\s+de\s+(\d+\s*\w+)',
        r'después\s+de\s+(\d+\s*\w+)',
        r'por\s+(\d+\s*\w+)',
        r'hace\s+(\d+\s*\w+)'
    ]
    
    for pattern in time_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            if isinstance(match, tuple):
                entities['tiempo'].append(' '.join(match))
            else:
                entities['tiempo'].append(match)
    
    # Detección de negaciones (neutral, sin interpretación específica)
    negation_patterns = [
        r'sin\s+(\w+(?:\s+\w+)?)',
        r'no\s+(\w+(?:\s+\w+)?)',
        r'nunca\s+(\w+)',
        r'falta\s+de\s+(\w+(?:\s+\w+)?)',
        r'ausencia\s+de\s+(\w+)',
        r'carencia\s+de\s+(\w+)'
    ]
    
    for pattern in negation_patterns:
        matches = re.findall(pattern, query_lower)
        entities['negaciones'].extend(matches)
    
    return entities

def semantic_similarity_search(query: str, articles: List[Dict], top_k: int = 10) -> List[Dict]:
    """
    Búsqueda por similitud semántica pura usando embeddings.
    Sin sesgos ni boosts específicos por área legal.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import pickle
        import hashlib
        
        # Usar cache para el modelo
        cache_file = os.path.join(DEFAULT_CACHE_PATH, "semantic_model.pkl")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    model = pickle.load(f)
            else:
                model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                with open(cache_file, 'wb') as f:
                    pickle.dump(model, f)
        except:
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Generar embedding de la consulta
        query_embedding = model.encode(query)
        
        # Cache para embeddings de artículos
        articles_hash = hashlib.md5(str(len(articles)).encode()).hexdigest()
        articles_cache_file = os.path.join(DEFAULT_CACHE_PATH, f"articles_embeddings_{articles_hash}.pkl")
        
        if os.path.exists(articles_cache_file):
            with open(articles_cache_file, 'rb') as f:
                article_embeddings = pickle.load(f)
        else:
            # Generar embeddings de artículos
            article_embeddings = []
            for i, article in enumerate(articles):
                content = article.get('content', '')
                if content:
                    embedding = model.encode(content)
                    article_embeddings.append((i, embedding, article))
                    
                if i % 100 == 0:
                    print(f"Generando embeddings: {i}/{len(articles)}")
            
            # Guardar en cache
            with open(articles_cache_file, 'wb') as f:
                pickle.dump(article_embeddings, f)
        
        # Calcular similitudes
        similarities = []
        for i, article_embedding, article in article_embeddings:
            similarity = np.dot(query_embedding, article_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(article_embedding)
            )
            similarities.append((i, similarity, article))
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Retornar top_k resultados
        results = []
        for i, sim, article in similarities[:top_k]:
            results.append({
                **article,
                'score': float(sim * 10),  # Escalar para comparar con otros métodos
                'method': 'semantic_similarity'
            })
        
        return results
        
    except ImportError:
        print("sentence-transformers no disponible, omitiendo búsqueda semántica")
        return []
    except Exception as e:
        print(f"Error en búsqueda semántica: {str(e)}")
        return []

def neutral_neo4j_search(neo4j_driver, query: str, entities: Dict, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Búsqueda neutral en Neo4j usando términos de la consulta sin categorización predefinida.
    """
    if not neo4j_driver:
        return []
    
    print(f"Búsqueda Neo4j neutral para: '{query}'")
    
    # Usar tokens de la consulta directamente sin categorización
    search_terms = entities.get('query_tokens', [])
    
    # Añadir términos de tiempo detectados
    search_terms.extend(entities.get('tiempo', []))
    
    # Añadir términos de negaciones para contexto
    search_terms.extend(entities.get('negaciones', []))
    
    # Filtrar términos muy cortos o comunes
    search_terms = [term for term in search_terms if len(term) > 2 and term not in ['que', 'por', 'con', 'sin', 'para', 'una', 'los', 'las', 'del', 'como']]
    
    # Remover duplicados manteniendo orden
    search_terms = list(dict.fromkeys(search_terms))
    
    if not search_terms:
        return []
    
    try:
        with neo4j_driver.session() as session:
            # Consulta simplificada y neutral
            cypher_query = """
            MATCH (a:Article)
            WHERE """ + " OR ".join([f"toLower(a.content) CONTAINS toLower(${i})" for i in range(len(search_terms))]) + """
            
            WITH a,
                 // Contar coincidencias de términos (neutral)
                 size([term IN $terms WHERE toLower(a.content) CONTAINS toLower(term)]) as term_matches,
                 
                 // Factor de longitud neutral (evitar artículos extremadamente largos o cortos)
                 CASE 
                    WHEN size(a.content) > 3000 THEN 0.8
                    WHEN size(a.content) < 100 THEN 0.7
                    ELSE 1.0
                 END as length_factor
            
            // Calcular score neutral basado solo en coincidencias de términos
            WITH a, 
                 (toFloat(term_matches) / size($terms)) * length_factor as relevance_score
            
            WHERE relevance_score > 0.1
            
            RETURN a.article_id as article_id,
                   a.content as content,
                   a.law_name as law_name,
                   a.article_number as article_number,
                   a.category as category,
                   a.source as source,
                   relevance_score as score
            ORDER BY relevance_score DESC
            LIMIT $limit
            """
            
            params = {str(i): term for i, term in enumerate(search_terms)}
            params.update({
                'terms': search_terms,
                'limit': limit
            })
            
            result = session.run(cypher_query, params)
            results = []
            
            for record in result:
                article = {
                    "article_id": record["article_id"],
                    "content": record["content"],
                    "law_name": record["law_name"],
                    "article_number": record["article_number"],
                    "category": record["category"],
                    "source": record["source"],
                    "score": float(record["score"]) * 5,  # Escalar para comparar con otros métodos
                    "method": "neutral_neo4j"
                }
                results.append(article)
            
            print(f"Neo4j encontró {len(results)} resultados")
            return results
            
    except Exception as e:
        print(f"Error en búsqueda Neo4j neutral: {str(e)}")
        return []

def neutral_bm25_search(query: str, documents: List[Dict], entities: Dict, top_k: int = 10) -> List[Dict]:
    """
    Búsqueda BM25 neutral usando expansión mínima de consulta.
    """
    try:
        from rank_bm25 import BM25Okapi
        import re
        
        # Expandir consulta mínimamente con términos detectados
        expanded_query_terms = query.lower().split()
        
        # Añadir solo términos temporales y de negación (neutrales)
        expanded_query_terms.extend(entities.get('tiempo', []))
        expanded_query_terms.extend(entities.get('negaciones', []))
        
        # Preprocesar documentos
        processed_docs = []
        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue
            # Tokenizar y limpiar
            tokens = re.findall(r'\w+', content.lower())
            processed_docs.append(tokens)
        
        if not processed_docs:
            return []
        
        # Inicializar BM25
        bm25 = BM25Okapi(processed_docs)
        
        # Tokenizar consulta expandida
        query_tokens = []
        for term in expanded_query_terms:
            query_tokens.extend(re.findall(r'\w+', term.lower()))
        
        # Obtener puntuaciones
        scores = bm25.get_scores(query_tokens)
        
        # Ordenar y seleccionar top_k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i in top_indices:
            if scores[i] > 0:
                doc = documents[i]
                results.append({
                    "content": doc.get("content", ""),
                    "article_id": doc.get("article_id", ""),
                    "law_name": doc.get("law_name", ""),
                    "article_number": doc.get("article_number", ""),
                    "category": doc.get("category", ""),
                    "source": doc.get("source", ""),
                    "score": float(scores[i]),
                    "method": "neutral_bm25"
                })
        
        return results
        
    except ImportError:
        print("rank_bm25 no disponible")
        return []
    except Exception as e:
        print(f"Error en BM25 neutral: {str(e)}")
        return []

def neutral_scoring(results: List[Dict], entities: Dict, query: str) -> List[Dict]:
    """
    Scoring neutral basado únicamente en similitud de términos y factores técnicos.
    Sin sesgos hacia áreas legales específicas.
    """
    scored_results = []
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    for result in results:
        score = result.get('score', 0.0)
        content_lower = result.get('content', '').lower()
        method = result.get('method', '')
        
        # Factor neutral por método (basado en capacidad técnica, no en dominio)
        method_multipliers = {
            'semantic_similarity': 2.0,  # Era 1.3, aumentar a 2.0
            'weaviate': 1.5,             # Era 1.1, aumentar a 1.5
            'neutral_neo4j': 0.8,        # Era 1.0, reducir a 0.8
            'neutral_bm25': 0.6          # Era 0.9, reducir a 0.6
        }
        score *= method_multipliers.get(method, 1.0)
        
        # Boost neutral por coincidencias exactas de palabras
        content_words = set(content_lower.split())
        exact_matches = len(query_words.intersection(content_words))
        # Booth por coincidencias exactas SOLO si son semánticamente relevantes
        if exact_matches > 0:
            # Verificar que las coincidencias no sean solo stopwords comunes
            meaningful_matches = [w for w in query_words.intersection(content_words) 
                                 if w not in ['de', 'la', 'el', 'que', 'por', 'con', 'sin', 'para']]
            if meaningful_matches:
                score *= (1.0 + len(meaningful_matches) * 0.1)  # Reducido de 0.15 a 0.1
        
        # Factor neutral de longitud (evitar extremos)
        content_length = len(content_lower)
        if content_length > 3000:  # Artículos muy largos pueden ser menos específicos
            score *= 0.9
        elif content_length < 100:  # Artículos muy cortos pueden ser incompletos
            score *= 0.8
        
        # Boost mínimo por términos de tiempo (neutral, no específico a labor)
        tiempo_detectado = entities.get('tiempo', [])
        if tiempo_detectado:
            for tiempo in tiempo_detectado:
                if any(t_word in content_lower for t_word in tiempo.split()):
                    score *= 1.1  # Boost mínimo y neutral
        
        # Boost mínimo por contexto de negaciones (neutral)
        negaciones = entities.get('negaciones', [])
        if negaciones:
            for negacion in negaciones:
                # Boost neutral si el artículo menciona el término negado
                if negacion in content_lower:
                    score *= 1.05  # Boost muy pequeño y neutral
        
        scored_results.append({
            **result,
            'score': score,
            'neutral_factors': {
                'exact_matches': exact_matches,
                'tiempo_detected': len(tiempo_detectado) > 0,
                'negaciones_detected': len(negaciones) > 0
            }
        })
    
    return sorted(scored_results, key=lambda x: x['score'], reverse=True)

def search_query_neutral(query: str, config: Dict[str, Any], weaviate_client=None, neo4j_driver=None, documents=None) -> List[Dict[str, Any]]:
    """
    Búsqueda completamente neutral sin sesgos hacia áreas legales específicas.
    Análisis puramente semántico y técnico.
    """
    print(f"\n🔍 Procesando consulta neutral: '{query}'")
    start_time = time.time()
    
    # 1. Extracción neutral de entidades
    print("📊 Analizando entidades de forma neutral...")
    entities = neutral_entity_extraction(query)
    
    print(f"✅ Análisis neutral completado:")
    print(f"   - Términos temporales: {entities.get('tiempo', [])}")
    print(f"   - Negaciones detectadas: {entities.get('negaciones', [])}")
    print(f"   - Tokens de consulta: {len(entities.get('query_tokens', []))}")
    
    # 2. Búsquedas multi-modales neutrales
    all_results = []
    
    # A. Búsqueda vectorial en Weaviate (neutral)
    if weaviate_client and config.get("weaviate", {}).get("enabled", False):
        try:
            print("🔮 Ejecutando búsqueda vectorial neutral (Weaviate)...")
            collection_name = config["weaviate"].get("collection_name", "ArticulosLegales")
            embedding_model = config["weaviate"].get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
            use_cache = config["weaviate"].get("use_cache", True)
            top_n = config["retrieval"].get("top_n", 15)
            
            weaviate_results = search_weaviate(
                weaviate_client, collection_name, query,
                embedding_model=embedding_model, top_n=top_n, use_cache=use_cache
            )
            
            if weaviate_results:
                for result in weaviate_results:
                    result['method'] = 'weaviate'
                all_results.extend(weaviate_results)
                print(f"   ✅ Weaviate: {len(weaviate_results)} resultados")
        except Exception as e:
            print(f"   ❌ Error en Weaviate: {str(e)}")
    
    # B. Búsqueda neutral en Neo4j
    if neo4j_driver and config.get("neo4j", {}).get("enabled", False):
        print("🕸️  Ejecutando búsqueda neutral en grafo (Neo4j)...")
        neo4j_results = neutral_neo4j_search(neo4j_driver, query, entities, limit=15)
        all_results.extend(neo4j_results)
    
    # C. Búsqueda semántica directa (neutral)
    if documents:
        print("🧠 Ejecutando búsqueda semántica neutral...")
        semantic_results = semantic_similarity_search(query, documents, top_k=10)
        all_results.extend(semantic_results)
        if semantic_results:
            print(f"   ✅ Semántica: {len(semantic_results)} resultados")
    
    # D. Búsqueda BM25 neutral
    if documents and config.get("bm25", {}).get("enabled", False):
        print("📝 Ejecutando búsqueda BM25 neutral...")
        bm25_results = neutral_bm25_search(query, documents, entities, top_k=8)
        all_results.extend(bm25_results)
        if bm25_results:
            print(f"   ✅ BM25: {len(bm25_results)} resultados")
    
    print(f"📈 Total de resultados antes de fusión: {len(all_results)}")
    
    # 3. Eliminar duplicados manteniendo el mejor score
    print("🔄 Eliminando duplicados...")
    unique_results = {}
    for result in all_results:
        # Usar article_id como clave principal, content como fallback
        key = result.get('article_id') or result.get('content', '')[:100]
        
        if key not in unique_results or result.get('score', 0) > unique_results[key].get('score', 0):
            unique_results[key] = result
    
    unique_list = list(unique_results.values())
    print(f"📊 Resultados únicos: {len(unique_list)}")
    
    # 4. Scoring neutral
    print("🎯 Aplicando scoring neutral...")
    scored_results = neutral_scoring(unique_list, entities, query)
    
    # 5. Aplicar filtro de relevancia semántica
    print("🔍 Aplicando filtro de relevancia semántica...")
    filtered_results = final_rerank(scored_results, query)
    print(f"📊 Resultados después del filtro: {len(filtered_results)}")
    
    # 6. Limitar resultados finales
    top_n = config.get("retrieval", {}).get("top_n", 20)
    final_results = filtered_results[:top_n]
    
    end_time = time.time()
    print(f"⚡ Búsqueda completada en {end_time - start_time:.2f} segundos")
    print(f"🎉 Resultados finales: {len(final_results)}")
    
    return final_results

# === Funciones de soporte (mantienen funcionalidad original) ===

def check_connections(config: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
    """Verifica las conexiones a las bases de datos configuradas."""
    weaviate_client = None
    neo4j_driver = None
    
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
    """Configura Weaviate y carga documentos si es necesario."""
    if not weaviate_client:
        return
        
    collection_name = config["weaviate"].get("collection_name", "ArticulosLegales")
    embedding_model = config["weaviate"].get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
    use_cache = config["weaviate"].get("use_cache", True)
    
    try:
        print(f"Verificando esquema para colección '{collection_name}'...")
        create_weaviate_schema(weaviate_client, collection_name)
    except Exception as e:
        print(f"Error al crear esquema: {str(e)}")
        return
    
    try:
        print("Almacenando documentos con embeddings en Weaviate...")
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
    """Configura Neo4j y carga documentos si es necesario."""
    if not neo4j_driver:
        return
        
    data_exists = check_data_exists(neo4j_driver)
    if data_exists:
        print("Ya existen datos en Neo4j, omitiendo carga...")
        return
    
    print("Creando nodos de artículos en Neo4j...")
    article_ids = create_neo4j_nodes(neo4j_driver, documents)
    print(f"✓ Creados {len(article_ids)} nodos de artículos")
    
    law_articles = defaultdict(list)
    for doc in documents:
        law_name = doc.get("law_name")
        if not law_name and "metadata" in doc:
            law_name = doc["metadata"].get("code")
            
        article_id = doc.get("article_id")
        if not article_id and "metadata" in doc:
            code = doc["metadata"].get("code")
            article_num = doc["metadata"].get("article")
            if code and article_num:
                article_id = f"{code}_{article_num}"
                
        if law_name and article_id:
            law_articles[law_name].append(article_id)
    
    print("Creando nodos de leyes y relaciones...")
    for law_name, article_ids in law_articles.items():
        create_law_relationship(neo4j_driver, law_name, article_ids)
    
    try:
        print("Creando relaciones basadas en contenido...")
        create_semantic_content_relationships(neo4j_driver)
        print("Creando relaciones entre códigos y leyes...")
        create_cross_law_relationships(neo4j_driver, documents)
        print("Creando relaciones basadas en tags...")
        create_topic_relationships_from_tags(neo4j_driver, documents)
        print("✓ Relaciones creadas correctamente")
    except Exception as e:
        print(f"Error al crear relaciones: {str(e)}")

def format_search_results(results: List[Dict[str, Any]]) -> str:
    """Formatea los resultados de búsqueda neutral para presentación."""
    if not results:
        return "No se encontraron resultados para la consulta."
    
    formatted = "\n=== RESULTADOS DE BÚSQUEDA NEUTRAL (SIN SESGOS) ===\n\n"
    
    for i, result in enumerate(results, 1):
        formatted += f"RESULTADO #{i} (Relevancia: {result.get('score', 0):.2f})\n"
        formatted += f"Ley/Código: {result.get('law_name', 'N/A')}\n"
        formatted += f"Artículo: {result.get('article_number', 'N/A')}\n"
        formatted += f"Categoría: {result.get('category', 'N/A')}\n"
        formatted += f"Método: {result.get('method', 'N/A')}\n"
        
        # Mostrar factores neutrales aplicados si están disponibles
        if 'neutral_factors' in result:
            factors = result['neutral_factors']
            if factors.get('exact_matches', 0) > 0:
                formatted += f"Coincidencias exactas: {factors['exact_matches']}\n"
            if factors.get('tiempo_detected'):
                formatted += f"Referencias temporales detectadas\n"
            if factors.get('negaciones_detected'):
                formatted += f"Contexto de negación detectado\n"
        
        formatted += "-" * 50 + "\n"
        formatted += f"{result.get('content', 'Sin contenido')}\n"
        formatted += "=" * 80 + "\n\n"
    
    return formatted

def setup_system(config_path: str = DEFAULT_CONFIG_PATH, data_path: str = DEFAULT_DATA_PATH):
    """Configura todo el sistema: verifica conexiones y carga datos iniciales."""
    print("\n=== Configuración del Sistema de Recuperación Neutral ===")
    
    config = load_config(config_path)
    if not config:
        print("Error: No se pudo cargar la configuración.")
        return
    
    weaviate_client, neo4j_driver = check_connections(config)
    
    documents = []
    try:
        print(f"Cargando documentos desde {data_path}...")
        documents = load_json_data(data_path)
        print(f"✓ Cargados {len(documents)} documentos")
    except Exception as e:
        print(f"Error al cargar documentos: {str(e)}")
        return
    
    if config.get("weaviate", {}).get("enabled", False) and weaviate_client and documents:
        setup_weaviate(weaviate_client, config, documents)
    
    if config.get("neo4j", {}).get("enabled", False) and neo4j_driver and documents:
        setup_neo4j_data(neo4j_driver, config, documents)
    
    if neo4j_driver:
        neo4j_driver.close()
    
    print("\n=== Configuración completada ===")
    print("El sistema neutral está listo para su uso.")

def main():
    """Función principal del programa neutral."""
    parser = argparse.ArgumentParser(description="Sistema de Recuperación Legal - Versión Semántica Neutral")
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
    
    # Limpiar Neo4j si se solicita
    if args.clear_neo4j:
        if neo4j_driver:
            confirm = input("¿Está seguro de que desea eliminar todos los datos de Neo4j? (s/n): ")
            if confirm.lower() == 's':
                from src.neo4j_utils import clear_neo4j_data
                clear_neo4j_data(neo4j_driver)
            else:
                print("Operación cancelada.")
        else:
            print("No se pudo conectar a Neo4j para limpiar los datos.")
        
        if neo4j_driver:
            neo4j_driver.close()
        return
    
    # Configurar sistema si se solicita
    if args.setup:
        setup_system(args.config, args.data)
        return
    
    # Mostrar ayuda si no hay consulta
    if not args.query:
        print("\n=== Sistema de Recuperación Legal - Versión Semántica Neutral ===")
        print("🚀 Sin sesgos hacia áreas específicas - Análisis puramente técnico")
        print("\nUso:")
        print("  python main.py --query \"tu consulta aquí\"")
        print("  python main.py --setup  # Para configurar el sistema")
        print("\nEjemplos de consultas que funcionan de forma neutral:")
        print('  • "fui despedida sin indemnización por embarazo"')
        print('  • "me desvincularon después de 5 años sin previo aviso"')
        print('  • "no me pagaron al terminar el contrato"')
        print('  • "derecho a vacaciones pagas"')
        print('  • "procedimiento para divorcio"')
        parser.print_help()
        
        if neo4j_driver:
            neo4j_driver.close()
        return
    
    # Cargar documentos
    documents = None
    try:
        print(f"📁 Cargando documentos desde {args.data}...")
        documents = load_json_data(args.data)
        print(f"✅ Cargados {len(documents)} documentos")
    except Exception as e:
        print(f"❌ Error al cargar documentos: {str(e)}")
    
    # Realizar búsqueda neutral
    print(f"\n{'='*60}")
    print("🔍 INICIANDO BÚSQUEDA SEMÁNTICA NEUTRAL")
    print(f"{'='*60}")
    
    results = search_query_neutral(args.query, config, weaviate_client, neo4j_driver, documents)
    
    # Formatear y mostrar resultados
    formatted_results = format_search_results(results)
    print(formatted_results)
    
    # Guardar resultados
    if config.get("retrieval", {}).get("save_results", False):
        results_dir = config.get("retrieval", {}).get("results_dir", "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_file = os.path.join(results_dir, f"results_neutral_{timestamp}.txt")
        
        with open(results_file, "w", encoding="utf-8") as f:
            f.write(f"Consulta: {args.query}\n")
            f.write(f"Sistema: Búsqueda Semántica Neutral (Sin Sesgos)\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(formatted_results)
        
        print(f"💾 Resultados guardados en {results_file}")
    
    # Cerrar conexiones
    if neo4j_driver:
        neo4j_driver.close()

def final_rerank(results: List[Dict], query: str) -> List[Dict]:
    """Re-ranking final basado en relevancia semántica real"""
    # Filtrar resultados irrelevantes
    filtered = [r for r in results if semantic_relevance_filter(r, query)]
    
    # Si se filtraron demasiados, usar threshold más bajo
    if len(filtered) < len(results) * 0.3:
        filtered = [r for r in results if semantic_relevance_filter(r, query, 0.2)]
    
    return filtered

def semantic_relevance_filter(result: Dict, query: str, threshold: float = 0.3) -> bool:
    """Filtrar resultados que no son semánticamente relevantes"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        query_emb = model.encode(query)
        content_emb = model.encode(result.get('content', '')[:500])  # Primeros 500 chars
        
        similarity = np.dot(query_emb, content_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(content_emb)
        )
        
        return similarity > threshold
    except:
        return True  # Si falla, no filtrar

if __name__ == "__main__":
    # Parsear argumentos de la línea de comandos
    parser = argparse.ArgumentParser(description="Sistema de búsqueda legal con análisis semántico neutral")
    parser.add_argument("--config", type=str, default="config.yaml", help="Ruta al archivo de configuración")
    parser.add_argument("--query", type=str, help="Consulta legal para buscar")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo")
    args = parser.parse_args()
    
    config = load_config(args.config)
    if config.get("weaviate", {}).get("enabled", False):
        try:
            weaviate_url = config["weaviate"].get("url", "http://localhost:8080")
            weaviate_api_key = config["weaviate"].get("api_key")
            print(f"Conectando a Weaviate en {weaviate_url}...")
            weaviate_client = connect_weaviate(weaviate_url, weaviate_api_key)
            print("✓ Conexión a Weaviate exitosa")
        except Exception as e:
            print(f"✗ Error al conectar con Weaviate: {str(e)}")
    
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

    main()