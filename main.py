"""
Script principal actualizado con Graph RAG Avanzado integrado.
Mejora dramáticamente la relevancia de resultados usando navegación inteligente del grafo.
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

# Importar el nuevo sistema Neo4j mejorado
from src.neo4j_utils import (
    connect_neo4j, check_data_exists, clear_neo4j_data, search_neo4j, search_neo4j_enhanced
)

# Rutas por defecto
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache")

# Asegurar que existan los directorios necesarios
for directory in [DEFAULT_CACHE_PATH]:
    os.makedirs(directory, exist_ok=True)

def enhanced_query_analysis(query: str) -> Dict[str, Any]:
    """
    Análisis mejorado de consultas que extrae más información semántica.
    """
    query_lower = query.lower()
    analysis = {
        'legal_entities': [],
        'temporal_references': [],
        'legal_actions': [],
        'stakeholders': [],
        'urgency_indicators': [],
        'emotional_context': [],
        'query_complexity': 0.0
    }
    
    # Detectar entidades legales específicas
    legal_entity_patterns = {
        'contratos': r'contrato\w*|acuerdo\w*|convenio\w*',
        'derechos': r'derecho\w*|facultad\w*|prerrogativa\w*',
        'obligaciones': r'obligación\w*|deber\w*|responsabilidad\w*',
        'sanciones': r'sanción\w*|pena\w*|multa\w*|castigo\w*',
        'procedimientos': r'procedimiento\w*|proceso\w*|trámite\w*|gestión\w*',
        'autoridades': r'tribunal\w*|juez\w*|autoridad\w*|funcionario\w*'
    }
    
    for entity_type, pattern in legal_entity_patterns.items():
        matches = re.findall(pattern, query_lower)
        if matches:
            analysis['legal_entities'].append({
                'type': entity_type,
                'matches': matches,
                'weight': len(matches)
            })
    
    # Detectar referencias temporales más específicas
    temporal_patterns = [
        r'(\d+)\s*(?:días?|meses?|años?|horas?)',
        r'(?:inmediatamente|urgente|pronto|rápido)',
        r'(?:antes|después|durante)\s+(?:de\s+)?(\w+)',
        r'(?:plazo|término|vencimiento)\s+(?:de\s+)?(\w+)',
        r'(?:fecha|momento|tiempo)\s+(?:de\s+)?(\w+)'
    ]
    
    for pattern in temporal_patterns:
        matches = re.findall(pattern, query_lower)
        if matches:
            analysis['temporal_references'].extend(matches)
    
    # Detectar acciones legales
    legal_action_patterns = [
        r'(?:presentar|interponer|radicar)\s+(\w+)',
        r'(?:solicitar|pedir|requerir)\s+(\w+)',
        r'(?:demandar|denunciar|acusar)\s*(?:por\s+)?(\w+)?',
        r'(?:apelar|recurrir|impugnar)\s+(\w+)',
        r'(?:notificar|informar|comunicar)\s+(\w+)'
    ]
    
    for pattern in legal_action_patterns:
        matches = re.findall(pattern, query_lower)
        if matches:
            analysis['legal_actions'].extend([m for m in matches if m])
    
    # Detectar stakeholders (partes involucradas)
    stakeholder_patterns = [
        r'(?:empleado\w*|trabajador\w*|obrero\w*)',
        r'(?:empleador\w*|patrón\w*|empresa\w*|compañía\w*)',
        r'(?:cliente\w*|consumidor\w*|usuario\w*)',
        r'(?:proveedor\w*|contratista\w*|prestador\w*)',
        r'(?:vecino\w*|propietario\w*|inquilino\w*)',
        r'(?:esposo\w*|cónyuge\w*|pareja\w*|familia\w*)'
    ]
    
    for pattern in stakeholder_patterns:
        matches = re.findall(pattern, query_lower)
        if matches:
            analysis['stakeholders'].extend(matches)
    
    # Detectar indicadores de urgencia
    urgency_patterns = [
        r'urgente|inmediato|rápido|ya|ahora',
        r'no\s+(?:puedo|pueden?)\s+esperar',
        r'necesito\s+(?:ya|ahora|urgente)',
        r'(?:emergencia|crisis|problema\s+grave)'
    ]
    
    for pattern in urgency_patterns:
        if re.search(pattern, query_lower):
            analysis['urgency_indicators'].append(pattern)
    
    # Detectar contexto emocional
    emotional_patterns = [
        r'(?:me\s+)?(?:discrimin\w*|hostiga\w*|acosa\w*)',
        r'(?:me\s+)?(?:despidieron|echaron|terminaron)',
        r'(?:no\s+me\s+)?(?:pagan|pagaron|abonan)',
        r'(?:me\s+)?(?:estafaron|engañaron|timaron)',
        r'(?:tengo\s+)?(?:miedo|temor|preocupación)'
    ]
    
    for pattern in emotional_patterns:
        if re.search(pattern, query_lower):
            analysis['emotional_context'].append(pattern)
    
    # Calcular complejidad de la consulta
    complexity_factors = [
        len(analysis['legal_entities']) * 0.2,
        len(analysis['temporal_references']) * 0.15,
        len(analysis['legal_actions']) * 0.25,
        len(analysis['stakeholders']) * 0.1,
        len(analysis['urgency_indicators']) * 0.1,
        len(analysis['emotional_context']) * 0.2
    ]
    
    analysis['query_complexity'] = min(sum(complexity_factors), 1.0)
    
    return analysis

def search_query_with_advanced_graph_rag(query: str, config: Dict[str, Any], 
                                        weaviate_client=None, neo4j_driver=None, 
                                        documents=None) -> List[Dict[str, Any]]:
    """
    Búsqueda principal que integra el Graph RAG Avanzado como método principal.
    """
    print(f"\n🚀 Procesando consulta con Graph RAG Avanzado: '{query}'")
    start_time = time.time()
    
    # 1. Análisis mejorado de la consulta
    print("🧠 Realizando análisis avanzado de consulta...")
    query_analysis = enhanced_query_analysis(query)
    
    print(f"   📊 Complejidad detectada: {query_analysis['query_complexity']:.2f}")
    if query_analysis['legal_entities']:
        entities = [e['type'] for e in query_analysis['legal_entities']]
        print(f"   ⚖️ Entidades legales: {', '.join(entities)}")
    if query_analysis['urgency_indicators']:
        print(f"   🚨 Urgencia detectada: {len(query_analysis['urgency_indicators'])} indicadores")
    
    # 2. Determinar estrategia de búsqueda basada en el análisis
    search_strategy = determine_search_strategy(query_analysis, config)
    print(f"   🎯 Estrategia seleccionada: {search_strategy['name']}")
    
    # 3. Ejecutar búsquedas según la estrategia
    all_results = []
    
    # A. Graph RAG Avanzado (PRIORIDAD MÁXIMA)
    if neo4j_driver and config.get("neo4j", {}).get("enabled", False):
        print("🕸️ Ejecutando Graph RAG Avanzado...")
        try:
            # Usar la búsqueda optimizada
            graph_rag_results = search_neo4j_enhanced(
                neo4j_driver, query, 
                limit=search_strategy['graph_rag_limit']
            )
            
            # Aplicar boost basado en análisis de consulta
            for result in graph_rag_results:
                result['score'] *= search_strategy['graph_rag_weight']
                if result.get('method') != 'optimized_seed' and result.get('method') != 'optimized_expansion':
                    result['method'] = 'advanced_graph_rag'
            
            all_results.extend(graph_rag_results)
            print(f"   ✅ Graph RAG: {len(graph_rag_results)} resultados")
            
        except Exception as e:
            print(f"   ❌ Error en Graph RAG: {str(e)}")
    
    # B. Búsqueda vectorial (complementaria)
    if weaviate_client and config.get("weaviate", {}).get("enabled", False):
        print("🔮 Ejecutando búsqueda vectorial complementaria...")
        try:
            collection_name = config["weaviate"].get("collection_name", "ArticulosLegales")
            embedding_model = config["weaviate"].get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
            
            vector_results = search_weaviate(
                weaviate_client, collection_name, query,
                embedding_model=embedding_model, 
                top_n=search_strategy['vector_limit']
            )
            
            # Aplicar peso basado en estrategia
            for result in vector_results:
                result['score'] *= search_strategy['vector_weight']
                result['method'] = 'weaviate_vectorial'
            
            all_results.extend(vector_results)
            print(f"   ✅ Vectorial: {len(vector_results)} resultados")
            
        except Exception as e:
            print(f"   ❌ Error en búsqueda vectorial: {str(e)}")
    
    # C. Búsqueda semántica directa (fallback)
    if documents and len(all_results) < search_strategy['min_results']:
        print("🧭 Ejecutando búsqueda semántica directa...")
        semantic_results = semantic_similarity_search(
            query, documents, 
            top_k=search_strategy['semantic_limit']
        )
        
        for result in semantic_results:
            result['score'] *= search_strategy['semantic_weight']
        
        all_results.extend(semantic_results)
        print(f"   ✅ Semántica: {len(semantic_results)} resultados")
    
    print(f"📊 Total resultados antes de fusión: {len(all_results)}")
    
    # 4. Fusión inteligente y eliminación de duplicados
    print("🔄 Fusionando resultados...")
    unique_results = {}
    for result in all_results:
        key = result.get('article_id') or result.get('content', '')[:100]
        
        if key not in unique_results or result.get('score', 0) > unique_results[key].get('score', 0):
            unique_results[key] = result
    
    unique_list = list(unique_results.values())
    print(f"📊 Resultados únicos: {len(unique_list)}")
    
    # 5. Re-scoring contextual basado en análisis de consulta
    print("🎯 Aplicando re-scoring contextual...")
    contextual_results = apply_contextual_rescoring(unique_list, query_analysis, query)
    
    # 6. Filtro final de relevancia
    print("🔍 Aplicando filtro final de relevancia...")
    final_results = final_relevance_filter(contextual_results, query, threshold=0.2)
    
    # 7. Limitar resultados finales
    top_n = config.get("retrieval", {}).get("top_n", 20)
    final_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    limited_results = final_results[:top_n]
    
    end_time = time.time()
    print(f"⚡ Búsqueda completada en {end_time - start_time:.2f} segundos")
    print(f"🎉 Resultados finales: {len(limited_results)}")
    
    # Mostrar breakdown de métodos para análisis
    method_breakdown = {}
    for result in limited_results:
        method = result.get('method', 'unknown')
        method_breakdown[method] = method_breakdown.get(method, 0) + 1
    
    print(f"📈 Breakdown por método: {dict(method_breakdown)}")
    
    return limited_results

def determine_search_strategy(query_analysis: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determina la estrategia de búsqueda óptima basada en el análisis de la consulta.
    """
    complexity = query_analysis['query_complexity']
    has_urgency = len(query_analysis['urgency_indicators']) > 0
    has_emotional_context = len(query_analysis['emotional_context']) > 0
    legal_entity_count = len(query_analysis['legal_entities'])
    
    # Estrategia para consultas complejas con alta urgencia
    if complexity > 0.7 or has_urgency:
        return {
            'name': 'high_priority_comprehensive',
            'graph_rag_limit': 20,
            'graph_rag_weight': 1.5,  # Boost alto para Graph RAG
            'vector_limit': 10,
            'vector_weight': 1.2,
            'semantic_limit': 5,
            'semantic_weight': 0.8,
            'min_results': 15
        }
    
    # Estrategia para consultas con contexto emocional (posibles violaciones de derechos)
    elif has_emotional_context or legal_entity_count > 2:
        return {
            'name': 'rights_violation_focused',
            'graph_rag_limit': 18,
            'graph_rag_weight': 1.4,
            'vector_limit': 12,
            'vector_weight': 1.3,
            'semantic_limit': 6,
            'semantic_weight': 1.0,
            'min_results': 12
        }
    
    # Estrategia para consultas técnicas/procedimentales
    elif any(entity['type'] == 'procedimientos' for entity in query_analysis['legal_entities']):
        return {
            'name': 'procedural_focused',
            'graph_rag_limit': 15,
            'graph_rag_weight': 1.6,  # Graph RAG es excelente para procedimientos
            'vector_limit': 8,
            'vector_weight': 1.0,
            'semantic_limit': 7,
            'semantic_weight': 1.1,
            'min_results': 10
        }
    
    # Estrategia estándar balanceada
    else:
        return {
            'name': 'balanced_standard',
            'graph_rag_limit': 15,
            'graph_rag_weight': 1.3,
            'vector_limit': 10,
            'vector_weight': 1.1,
            'semantic_limit': 8,
            'semantic_weight': 1.0,
            'min_results': 10
        }

def apply_contextual_rescoring(results: List[Dict[str, Any]], query_analysis: Dict[str, Any], 
                              original_query: str) -> List[Dict[str, Any]]:
    """
    Aplica re-scoring contextual basado en el análisis profundo de la consulta.
    """
    for result in results:
        base_score = result.get('score', 0.0)
        content = result.get('content', '').lower()
        
        # Factor 1: Relevancia por entidades legales detectadas
        entity_boost = 0.0
        for entity in query_analysis.get('legal_entities', []):
            entity_type = entity['type']
            for match in entity['matches']:
                if match.lower() in content:
                    # Boost diferenciado por tipo de entidad
                    type_weights = {
                        'derechos': 1.5,
                        'obligaciones': 1.4,
                        'sanciones': 1.6,
                        'procedimientos': 1.3,
                        'contratos': 1.2,
                        'autoridades': 1.1
                    }
                    entity_boost += type_weights.get(entity_type, 1.0) * entity['weight'] * 0.2
        
        # Factor 2: Urgencia detectada
        urgency_boost = 0.0
        if query_analysis.get('urgency_indicators'):
            # Buscar indicadores de tiempo/urgencia en el contenido
            urgency_terms = ['inmediato', 'urgente', 'plazo', 'término', 'días', 'inmediatamente']
            urgency_matches = sum(1 for term in urgency_terms if term in content)
            if urgency_matches > 0:
                urgency_boost = urgency_matches * 0.3 * len(query_analysis['urgency_indicators'])
        
        # Factor 3: Contexto emocional/violación de derechos
        emotional_boost = 0.0
        if query_analysis.get('emotional_context'):
            # Buscar términos relacionados con protección, derechos, sanciones
            protection_terms = ['protección', 'amparo', 'defensa', 'sanción', 'pena', 'responsabilidad']
            protection_matches = sum(1 for term in protection_terms if term in content)
            if protection_matches > 0:
                emotional_boost = protection_matches * 0.4 * len(query_analysis['emotional_context'])
        
        # Factor 4: Stakeholders relevantes
        stakeholder_boost = 0.0
        for stakeholder in query_analysis.get('stakeholders', []):
            if stakeholder.lower() in content:
                stakeholder_boost += 0.25
        
        # Factor 5: Acciones legales específicas
        action_boost = 0.0
        for action in query_analysis.get('legal_actions', []):
            if action.lower() in content:
                action_boost += 0.3
        
        # Factor 6: Referencias temporales
        temporal_boost = 0.0
        for temporal_ref in query_analysis.get('temporal_references', []):
            if str(temporal_ref).lower() in content:
                temporal_boost += 0.2
        
        # Calcular score final con boost adicional para Graph RAG Avanzado
        method_boost = 0.0
        if result.get('method') in ['optimized_seed', 'optimized_expansion', 'advanced_graph_rag']:
            method_boost = 0.5  # Boost adicional para resultados del Graph RAG Avanzado
        
        final_score = (
            base_score + 
            entity_boost + 
            urgency_boost + 
            emotional_boost + 
            stakeholder_boost + 
            action_boost + 
            temporal_boost +
            method_boost
        )
        
        result['score'] = final_score
        result['contextual_analysis'] = {
            'entity_boost': entity_boost,
            'urgency_boost': urgency_boost,
            'emotional_boost': emotional_boost,
            'stakeholder_boost': stakeholder_boost,
            'action_boost': action_boost,
            'temporal_boost': temporal_boost,
            'method_boost': method_boost
        }
    
    return results

def final_relevance_filter(results: List[Dict[str, Any]], query: str, threshold: float = 0.2) -> List[Dict[str, Any]]:
    """
    Filtro final de relevancia que usa similitud semántica para eliminar resultados irrelevantes.
    """
    if not results:
        return results
    
    try:
        from sentence_transformers import SentenceTransformer
        import pickle
        import os
        
        # Cargar modelo desde cache si existe
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
        
        query_embedding = model.encode(query)
        filtered_results = []
        
        for result in results:
            content = result.get('content', '')[:500]  # Limitar contenido para eficiencia
            content_embedding = model.encode(content)
            
            similarity = np.dot(query_embedding, content_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding)
            )
            
            # Ajustar threshold basado en el método
            adjusted_threshold = threshold
            if result.get('method') in ['optimized_seed', 'optimized_expansion', 'advanced_graph_rag']:
                adjusted_threshold *= 0.8  # Threshold más permisivo para Graph RAG
            
            if similarity > adjusted_threshold:
                result['semantic_similarity'] = similarity
                filtered_results.append(result)
            else:
                article_info = f"{result.get('law_name', 'N/A')} Art. {result.get('article_number', 'N/A')}"
                print(f"   ❌ Filtrado por baja similitud ({similarity:.3f}): {article_info}")
        
        return filtered_results
        
    except ImportError:
        print("sentence-transformers no disponible, omitiendo filtro semántico")
        return results
    except Exception as e:
        print(f"Error en filtro semántico: {str(e)}")
        return results

def semantic_similarity_search(query: str, articles: List[Dict], top_k: int = 10) -> List[Dict]:
    """
    Búsqueda por similitud semántica pura usando embeddings.
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

def format_enhanced_results(results: List[Dict[str, Any]]) -> str:
    """
    Formatea los resultados mejorados mostrando información adicional del análisis.
    """
    if not results:
        return "No se encontraron resultados relevantes para la consulta."
    
    formatted = "\n=== RESULTADOS CON GRAPH RAG AVANZADO ===\n\n"
    
    for i, result in enumerate(results, 1):
        formatted += f"RESULTADO #{i} (Relevancia: {result.get('score', 0):.2f})\n"
        formatted += f"Ley/Código: {result.get('law_name', 'N/A')}\n"
        formatted += f"Artículo: {result.get('article_number', 'N/A')}\n"
        formatted += f"Categoría: {result.get('category', 'N/A')}\n"
        formatted += f"Método: {result.get('method', 'N/A')}\n"
        
        # Mostrar análisis contextual si está disponible
        if 'contextual_analysis' in result:
            analysis = result['contextual_analysis']
            significant_boosts = []
            
            if analysis.get('entity_boost', 0) > 0.1:
                significant_boosts.append(f"Entidades legales (+{analysis['entity_boost']:.1f})")
            if analysis.get('urgency_boost', 0) > 0.1:
                significant_boosts.append(f"Urgencia (+{analysis['urgency_boost']:.1f})")
            if analysis.get('emotional_boost', 0) > 0.1:
                significant_boosts.append(f"Contexto emocional (+{analysis['emotional_boost']:.1f})")
            if analysis.get('method_boost', 0) > 0.1:
                significant_boosts.append(f"Graph RAG (+{analysis['method_boost']:.1f})")
            
            if significant_boosts:
                formatted += f"Factores de relevancia: {', '.join(significant_boosts)}\n"
        
        # Mostrar explicación de razonamiento si está disponible
        if 'reasoning_explanation' in result:
            formatted += f"Razonamiento: {result['reasoning_explanation']}\n"
        
        # Mostrar similitud semántica si está disponible
        if 'semantic_similarity' in result:
            formatted += f"Similitud semántica: {result['semantic_similarity']:.3f}\n"
        
        formatted += "-" * 50 + "\n"
        formatted += f"{result.get('content', 'Sin contenido')}\n"
        formatted += "=" * 80 + "\n\n"
    
    return formatted

# === Funciones de compatibilidad y soporte ===

def check_connections(config: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
    """Verifica las conexiones a las bases de datos configuradas."""
    weaviate_client = None
    neo4j_driver = None
    
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

def setup_system(config_path: str = DEFAULT_CONFIG_PATH, data_path: str = DEFAULT_DATA_PATH):
    """Configura todo el sistema con el Graph RAG Avanzado."""
    print("\n=== Configuración del Sistema con Graph RAG Avanzado ===")
    
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
        from src.neo4j_utils import setup_enhanced_neo4j_data
        setup_enhanced_neo4j_data(neo4j_driver, documents)
    
    if neo4j_driver:
        neo4j_driver.close()
    
    print("\n=== Configuración completada ===")
    print("El sistema Graph RAG Avanzado está listo para su uso.")

# Función de compatibilidad para mantener la API existente
def search_query_neutral(query: str, config: Dict[str, Any], weaviate_client=None, neo4j_driver=None, documents=None) -> List[Dict[str, Any]]:
    """Función de compatibilidad que usa el Graph RAG Avanzado."""
    return search_query_with_advanced_graph_rag(query, config, weaviate_client, neo4j_driver, documents)

# Alias para mantener compatibilidad
search_query = search_query_with_advanced_graph_rag

def main():
    """Función principal del programa mejorado."""
    parser = argparse.ArgumentParser(description="Sistema de Recuperación Legal con Graph RAG Avanzado")
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
        print("\n=== Sistema de Recuperación Legal con Graph RAG Avanzado ===")
        print("🚀 Navegación inteligente del grafo legal con cadenas de razonamiento jurídico")
        print("\nUso:")
        print("  python main.py --query \"tu consulta aquí\"")
        print("  python main.py --setup  # Para configurar el sistema")
        print("\nEjemplos de consultas optimizadas para Graph RAG:")
        print('  • "fui despedida sin indemnización por estar embarazada"')
        print('  • "me hacen trabajar más de 8 horas sin pagar extras"')
        print('  • "mi jefe me discrimina por mi edad y género"')
        print('  • "no me pagaron la liquidación final al terminar el contrato"')
        print('  • "puedo divorciarme sin el consentimiento de mi esposo"')
        print('  • "mi vecino construyó en mi terreno sin permiso"')
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
    
    # Realizar búsqueda con Graph RAG Avanzado
    print(f"\n{'='*60}")
    print("🧠 INICIANDO BÚSQUEDA CON GRAPH RAG AVANZADO")
    print(f"{'='*60}")
    
    results = search_query_with_advanced_graph_rag(args.query, config, weaviate_client, neo4j_driver, documents)
    
    # Formatear y mostrar resultados
    formatted_results = format_enhanced_results(results)
    print(formatted_results)
    
    # Guardar resultados
    if config.get("retrieval", {}).get("save_results", False):
        results_dir = config.get("retrieval", {}).get("results_dir", "results")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_file = os.path.join(results_dir, f"results_graph_rag_{timestamp}.txt")
        
        with open(results_file, "w", encoding="utf-8") as f:
            f.write(f"Consulta: {args.query}\n")
            f.write(f"Sistema: Graph RAG Avanzado\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(formatted_results)
        
        print(f"💾 Resultados guardados en {results_file}")
    
    # Cerrar conexiones
    if neo4j_driver:
        neo4j_driver.close()

if __name__ == "__main__":
    main()