"""
advanced_graph_rag_optimized.py

Versión optimizada del Graph RAG Avanzado que resuelve problemas de timeout
y mejora la performance para grandes volúmenes de datos (3500+ documentos).
"""
import re
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, Counter
import numpy as np
from contextlib import contextmanager

class OptimizedAdvancedGraphRAG:
    """
    Versión optimizada del Graph RAG Avanzado con:
    1. Consultas Cypher más eficientes
    2. Timeouts manejados
    3. Fallbacks rápidos
    4. Caché de consultas frecuentes
    """
    
    def __init__(self, neo4j_driver, max_query_time: int = 15):
        self.driver = neo4j_driver
        self.max_query_time = max_query_time  # Timeout en segundos
        self.legal_connectors = self._build_legal_connectors()
        self.semantic_patterns = self._build_semantic_patterns()
        self.query_cache = {}  # Cache simple para consultas frecuentes
        
    @contextmanager
    def get_session(self):
        """Context manager para sesiones Neo4j con timeout."""
        session = None
        try:
            session = self.driver.session()
            yield session
        finally:
            if session:
                session.close()
    
    def _build_legal_connectors(self) -> Dict[str, List[str]]:
        """Conectores legales simplificados para mejor performance."""
        return {
            'causal': ['cuando', 'si', 'en caso de', 'debido a'],
            'temporal': ['después', 'antes', 'durante', 'plazo', 'días'],
            'conditional': ['salvo', 'excepto', 'sin perjuicio'],
            'procedural': ['procedimiento', 'proceso', 'trámite'],
            'opposition': ['contra', 'prohibido', 'vedado']
        }
    
    def _build_semantic_patterns(self) -> Dict[str, str]:
        """Patrones semánticos optimizados."""
        return {
            'derechos': r'derecho\w*',
            'obligaciones': r'obligación\w*|deber\w*',
            'prohibiciones': r'prohib\w*|ved\w*',
            'sanciones': r'sanción\w*|pena\w*|multa\w*',
            'procedimientos': r'procedimiento\w*|proceso\w*',
            'plazos': r'plazo\w*|término\w*|días\w*'
        }
    
    def analyze_query_semantics_fast(self, query: str) -> Dict[str, Any]:
        """Análisis semántico rápido y eficiente."""
        query_lower = query.lower()
        analysis = {
            'legal_intentions': [],
            'key_terms': [],
            'urgency_level': 0,
            'confidence_score': 0.0
        }
        
        # Detectar intenciones legales principales
        for intention, pattern in self.semantic_patterns.items():
            if re.search(pattern, query_lower):
                analysis['legal_intentions'].append(intention)
        
        # Extraer términos clave (palabras > 3 caracteres, excluyendo stopwords)
        stopwords = {'que', 'por', 'con', 'sin', 'para', 'una', 'los', 'las', 'del', 'como', 'más'}
        key_terms = [word for word in query_lower.split() 
                    if len(word) > 3 and word not in stopwords]
        analysis['key_terms'] = key_terms[:10]  # Limitar a 10 términos clave
        
        # Detectar urgencia
        urgency_indicators = ['urgente', 'inmediato', 'rápido', 'ya', 'ahora']
        analysis['urgency_level'] = sum(1 for indicator in urgency_indicators if indicator in query_lower)
        
        # Calcular confianza
        analysis['confidence_score'] = min(
            (len(analysis['legal_intentions']) * 0.3 + 
             len(analysis['key_terms']) * 0.1 + 
             analysis['urgency_level'] * 0.2), 1.0
        )
        
        return analysis
    
    def search_with_optimized_graph_rag(self, query: str, limit: int = 15) -> List[Dict[str, Any]]:
        """
        Búsqueda Graph RAG optimizada con timeouts y fallbacks.
        """
        print(f"\n🧠 Graph RAG Optimizado para: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        start_time = time.time()
        
        # 1. Análisis semántico rápido
        semantic_analysis = self.analyze_query_semantics_fast(query)
        print(f"   📊 Análisis rápido (confianza: {semantic_analysis['confidence_score']:.2f})")
        
        # 2. Búsqueda de artículos semilla optimizada
        try:
            seed_articles = self._find_seed_articles_fast(query, semantic_analysis)
            print(f"   🌱 Artículos semilla: {len(seed_articles)}")
        except Exception as e:
            print(f"   ❌ Error en semilla: {str(e)}")
            return self._fallback_search(query, limit)
        
        if not seed_articles:
            return self._fallback_search(query, limit)
        
        # 3. Expansión de grafo optimizada (solo 1 nivel)
        try:
            expanded_articles = self._expand_graph_optimized(seed_articles, semantic_analysis)
            print(f"   🔗 Expansión del grafo: {len(expanded_articles)}")
        except Exception as e:
            print(f"   ❌ Error en expansión: {str(e)}")
            expanded_articles = []
        
        # 4. Combinar y puntuar resultados
        all_results = seed_articles + expanded_articles
        
        # 5. Eliminar duplicados y re-puntuar
        unique_results = self._deduplicate_and_score(all_results, semantic_analysis, query)
        
        # 6. Limitar resultados
        final_results = sorted(unique_results, key=lambda x: x.get('score', 0), reverse=True)[:limit]
        
        elapsed_time = time.time() - start_time
        print(f"   ⚡ Completado en {elapsed_time:.2f}s - {len(final_results)} resultados")
        
        return final_results
    
    def _find_seed_articles_fast(self, query: str, semantic_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Búsqueda de artículos semilla optimizada."""
        seed_articles = []
        key_terms = semantic_analysis['key_terms'][:5]  # Solo top 5 términos
        
        if not key_terms:
            return []
        
        with self.get_session() as session:
            # Consulta optimizada con LIMIT y índices
            seed_query = """
            MATCH (a:Article)
            WHERE """ + " OR ".join([f"toLower(a.content) CONTAINS $term{i}" for i in range(len(key_terms))]) + """
            
            WITH a,
                 size([term IN $all_terms WHERE toLower(a.content) CONTAINS term]) as term_matches,
                 
                 // Boost rápido por metadatos
                 CASE 
                    WHEN a.has_penalties = true AND 'sanciones' IN $legal_intentions THEN 2.0
                    WHEN a.tag_count > 3 THEN 1.5
                    ELSE 0.0 
                 END as metadata_boost
            
            WITH a, (toFloat(term_matches) / size($all_terms)) * 10.0 + metadata_boost as score
            
            WHERE score > 2.0
            
            RETURN a.article_id as article_id,
                   a.content as content,
                   a.law_name as law_name,
                   a.article_number as article_number,
                   a.category as category,
                   a.source as source,
                   score
            
            ORDER BY score DESC
            LIMIT 15
            """
            
            params = {
                **{f'term{i}': term for i, term in enumerate(key_terms)},
                'all_terms': key_terms,
                'legal_intentions': semantic_analysis['legal_intentions']
            }
            
            try:
                # Ejecutar con timeout
                result = session.run(seed_query, params, timeout=self.max_query_time)
                
                for record in result:
                    seed_articles.append({
                        'article_id': record['article_id'],
                        'content': record['content'],
                        'law_name': record['law_name'],
                        'article_number': record['article_number'],
                        'category': record['category'],
                        'source': record['source'],
                        'score': float(record['score']),
                        'method': 'optimized_seed'
                    })
                    
            except Exception as e:
                print(f"Error en búsqueda de semilla: {str(e)}")
                # Fallback a búsqueda más simple
                return self._simple_text_search(query, 10)
        
        return seed_articles
    
    def _expand_graph_optimized(self, seed_articles: List[Dict[str, Any]], 
                               semantic_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expansión optimizada del grafo (solo 1 salto)."""
        expanded_articles = []
        seed_ids = [art['article_id'] for art in seed_articles[:5]]  # Solo top 5 semillas
        
        if not seed_ids:
            return []
        
        with self.get_session() as session:
            # Consulta de expansión simple y rápida
            expansion_query = """
MATCH (seed:Article)-[r]-(related:Article)
WHERE seed.article_id IN $seed_ids
AND NOT related.article_id IN $seed_ids

WITH related, r, 
     CASE type(r)
        WHEN 'REFERENCES' THEN 3.0
        WHEN 'SHARES_TAG' THEN 2.0
        WHEN 'SAME_SECTION' THEN 1.5
        WHEN 'SIMILAR_PENALTY' THEN 2.5
        ELSE 1.0 
     END as relation_weight

WITH related, max(relation_weight) as best_relation_weight

WHERE best_relation_weight > 1.2

RETURN DISTINCT related.article_id as article_id,
       related.content as content,
       related.law_name as law_name,
       related.article_number as article_number,
       related.category as category,
       related.source as source,
       best_relation_weight as score

ORDER BY score DESC
LIMIT 20
"""
            
            params = {
                'seed_ids': seed_ids
            }
            
            try:
                result = session.run(expansion_query, params, timeout=self.max_query_time)
                
                for record in result:
                    expanded_articles.append({
                        'article_id': record['article_id'],
                        'content': record['content'],
                        'law_name': record['law_name'],
                        'article_number': record['article_number'],
                        'category': record['category'],
                        'source': record['source'],
                        'score': float(record['score']),
                        'method': 'optimized_expansion'
                    })
                    
            except Exception as e:
                print(f"Error en expansión del grafo: {str(e)}")
        
        return expanded_articles
    
    def _simple_text_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Búsqueda de texto simple como fallback."""
        results = []
        query_words = [word.lower() for word in query.split() if len(word) > 3][:5]
        
        if not query_words:
            return []
        
        with self.get_session() as session:
            simple_query = """
            MATCH (a:Article)
            WHERE """ + " OR ".join([f"toLower(a.content) CONTAINS $word{i}" for i in range(len(query_words))]) + """
            
            WITH a, size([word IN $words WHERE toLower(a.content) CONTAINS word]) as matches
            
            WHERE matches > 0
            
            RETURN a.article_id as article_id,
                   a.content as content,
                   a.law_name as law_name,
                   a.article_number as article_number,
                   a.category as category,
                   a.source as source,
                   toFloat(matches) as score
            
            ORDER BY score DESC
            LIMIT $limit
            """
            
            params = {
                **{f'word{i}': word for i, word in enumerate(query_words)},
                'words': query_words,
                'limit': limit
            }
            
            try:
                result = session.run(simple_query, params, timeout=5)  # Timeout corto
                
                for record in result:
                    results.append({
                        'article_id': record['article_id'],
                        'content': record['content'],
                        'law_name': record['law_name'],
                        'article_number': record['article_number'],
                        'category': record['category'],
                        'source': record['source'],
                        'score': float(record['score']),
                        'method': 'simple_fallback'
                    })
                    
            except Exception as e:
                print(f"Error en búsqueda simple: {str(e)}")
        
        return results
    
    def _fallback_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Búsqueda de fallback cuando Graph RAG falla."""
        print("   🔄 Usando búsqueda de fallback...")
        return self._simple_text_search(query, limit)
    
    def _deduplicate_and_score(self, results: List[Dict[str, Any]], 
                              semantic_analysis: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """Elimina duplicados y re-puntúa resultados."""
        unique_results = {}
        query_lower = query.lower()
        
        for result in results:
            article_id = result.get('article_id', '')
            
            if article_id not in unique_results or result.get('score', 0) > unique_results[article_id].get('score', 0):
                # Re-scoring rápido
                content = result.get('content', '').lower()
                base_score = result.get('score', 0.0)
                
                # Boost por términos clave en contenido
                key_term_boost = 0.0
                for term in semantic_analysis.get('key_terms', []):
                    if term in content:
                        key_term_boost += 0.5
                
                # Boost por intenciones legales
                intention_boost = 0.0
                for intention in semantic_analysis.get('legal_intentions', []):
                    if intention.replace('es', '') in content:  # Buscar raíz de la palabra
                        intention_boost += 1.0
                
                # Score final
                final_score = base_score + key_term_boost + intention_boost
                result['score'] = final_score
                
                unique_results[article_id] = result
        
        return list(unique_results.values())


def optimized_advanced_neo4j_search(neo4j_driver, query: str, limit: int = 15) -> List[Dict[str, Any]]:
    """
    Función wrapper optimizada para integrar con el sistema existente.
    """
    if not neo4j_driver:
        print("⚠️ Neo4j driver no disponible")
        return []
    
    try:
        optimized_rag = OptimizedAdvancedGraphRAG(neo4j_driver, max_query_time=10)
        results = optimized_rag.search_with_optimized_graph_rag(query, limit)
        
        print(f"🎯 Graph RAG Optimizado: {len(results)} resultados")
        
        # Mostrar métodos utilizados
        methods = {}
        for result in results[:5]:
            method = result.get('method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
        
        if methods:
            methods_str = ", ".join([f"{method}: {count}" for method, count in methods.items()])
            print(f"   📊 Métodos: {methods_str}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error en Graph RAG Optimizado: {str(e)}")
        return []


# Función de compatibilidad para reemplazar en enhanced_neo4j_utils.py
def search_neo4j_enhanced(driver, query: str, limit: int = 15) -> List[Dict[str, Any]]:
    """
    Búsqueda mejorada que usa el Graph RAG Optimizado.
    """
    print(f"🔍 Búsqueda Neo4j optimizada para: '{query[:30]}{'...' if len(query) > 30 else ''}'")
    
    # Usar Graph RAG Optimizado como método principal
    optimized_results = optimized_advanced_neo4j_search(driver, query, limit)
    
    # Si no hay suficientes resultados, complementar con búsqueda tradicional
    if len(optimized_results) < limit // 2:
        print("   🔄 Complementando con búsqueda tradicional...")
        try:
            traditional_results = search_neo4j_traditional_fast(driver, query, limit - len(optimized_results))
            
            # Combinar resultados evitando duplicados
            existing_ids = {result['article_id'] for result in optimized_results}
            for result in traditional_results:
                if result['article_id'] not in existing_ids:
                    optimized_results.append(result)
        except Exception as e:
            print(f"   ❌ Error en búsqueda tradicional: {str(e)}")
    
    return optimized_results[:limit]


def search_neo4j_traditional_fast(driver, query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Búsqueda tradicional rápida y optimizada.
    """
    results = []
    query_words = [word.lower() for word in query.split() if len(word) > 3][:5]
    
    if not query_words:
        return results
    
    with driver.session() as session:
        try:
            cypher_query = """
            MATCH (a:Article)
            WHERE """ + " OR ".join([f"toLower(a.content) CONTAINS $word{i}" for i in range(len(query_words))]) + """
            
            WITH a, size([word IN $words WHERE toLower(a.content) CONTAINS word]) as matches
            
            WHERE matches > 0
            
            RETURN a.article_id as article_id,
                   a.content as content,
                   a.law_name as law_name,
                   a.article_number as article_number,
                   a.category as category,
                   a.source as source,
                   toFloat(matches) as score
            
            ORDER BY score DESC
            LIMIT $limit
            """
            
            params = {
                **{f'word{i}': word for i, word in enumerate(query_words)},
                'words': query_words,
                'limit': limit
            }
            
            result = session.run(cypher_query, params, timeout=5)
            for record in result:
                results.append({
                    "article_id": record["article_id"],
                    "content": record["content"],
                    "law_name": record["law_name"],
                    "article_number": record["article_number"],
                    "category": record["category"],
                    "source": record["source"],
                    "score": float(record["score"]) * 2,  # Escalar para comparar
                    "method": "neo4j_traditional_fast"
                })
        except Exception as e:
            print(f"Error en búsqueda tradicional rápida: {str(e)}")
    
    return results