"""
advanced_graph_navigator.py

Navegador inteligente del grafo legal que implementa búsqueda contextual profunda
y navegación semántica avanzada para encontrar más artículos relevantes.
"""

import re
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict

class AdvancedGraphNavigator:
    """
    Navegador inteligente del grafo legal que encuentra conexiones profundas
    entre artículos usando análisis semántico y navegación contextual.
    """
    
    def __init__(self, driver):
        self.driver = driver
        self.legal_context_patterns = self._build_legal_context_patterns()
        self.procedural_chains = self._build_procedural_chains()
        
    def _build_legal_context_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Patrones de contexto legal para navegación inteligente."""
        return {
            'embarazo_despido': {
                'core_concepts': ['embarazo', 'maternidad', 'despido', 'discriminación'],
                'related_concepts': ['indemnización', 'estabilidad', 'licencia', 'protección'],
                'procedural_concepts': ['notificación', 'presunción', 'prueba', 'procedimiento'],
                'penalty_concepts': ['sanción', 'multa', 'reparación', 'daños']
            },
            'contrato_trabajo': {
                'core_concepts': ['contrato', 'trabajo', 'empleador', 'trabajador'],
                'related_concepts': ['salario', 'jornada', 'vacaciones', 'licencia'],
                'procedural_concepts': ['rescisión', 'terminación', 'preaviso', 'indemnización'],
                'penalty_concepts': ['multa', 'sanción', 'responsabilidad']
            },
            'derechos_laborales': {
                'core_concepts': ['derecho', 'obligación', 'trabajador', 'empleador'],
                'related_concepts': ['igualdad', 'no_discriminación', 'protección', 'seguridad'],
                'procedural_concepts': ['denuncia', 'reclamo', 'procedimiento', 'recursos'],
                'penalty_concepts': ['sanción', 'inhabilitación', 'multa']
            }
        }
    
    def _build_procedural_chains(self) -> Dict[str, List[str]]:
        """Cadenas procedimentales típicas en derecho laboral."""
        return {
            'despido_proceso': [
                'causa_despido', 'notificacion_despido', 'indemnizacion_calculo', 
                'procedimiento_reclamo', 'plazos_prescripcion'
            ],
            'embarazo_proteccion': [
                'notificacion_embarazo', 'estabilidad_laboral', 'licencia_maternidad',
                'prohibicion_despido', 'presuncion_discriminacion'
            ],
            'indemnizacion_proceso': [
                'calculo_indemnizacion', 'conceptos_incluidos', 'forma_pago',
                'plazos_pago', 'intereses_mora'
            ]
        }
    
    def detect_query_context(self, query: str) -> Dict[str, Any]:
        """Detecta el contexto legal de la consulta para navegación dirigida."""
        query_lower = query.lower()
        detected_contexts = []
        confidence_scores = {}
        
        for context_name, patterns in self.legal_context_patterns.items():
            score = 0
            matched_concepts = []
            
            # Buscar conceptos centrales (peso alto)
            for concept in patterns['core_concepts']:
                if concept in query_lower:
                    score += 3
                    matched_concepts.append(('core', concept))
            
            # Buscar conceptos relacionados (peso medio)
            for concept in patterns['related_concepts']:
                if concept in query_lower:
                    score += 2
                    matched_concepts.append(('related', concept))
            
            # Buscar conceptos procedimentales (peso medio)
            for concept in patterns['procedural_concepts']:
                if concept in query_lower:
                    score += 2
                    matched_concepts.append(('procedural', concept))
            
            if score > 0:
                detected_contexts.append(context_name)
                confidence_scores[context_name] = {
                    'score': score,
                    'matched_concepts': matched_concepts
                }
        
        return {
            'primary_context': detected_contexts[0] if detected_contexts else 'general',
            'all_contexts': detected_contexts,
            'confidence_scores': confidence_scores
        }
    
    def intelligent_graph_expansion(self, seed_articles: List[Dict[str, Any]], 
                                   query: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        Navegación inteligente del grafo legal usando análisis contextual.
        """
        if not seed_articles:
            return []
        
        print(f"🧠 Iniciando navegación inteligente con {len(seed_articles)} artículos semilla")
        
        # Detectar contexto de la consulta
        context_info = self.detect_query_context(query)
        primary_context = context_info['primary_context']
        
        print(f"   🎯 Contexto detectado: {primary_context}")
        
        # Extraer IDs de artículos semilla
        seed_ids = [art['article_id'] for art in seed_articles]
        
        expanded_articles = list(seed_articles)  # Empezar con artículos semilla
        all_found_ids = set(seed_ids)
        
        with self.driver.session() as session:
            # Nivel 1: Navegación directa por relaciones fuertes
            level1_articles = self._expand_by_strong_relations(session, seed_ids, primary_context)
            for article in level1_articles:
                if article['article_id'] not in all_found_ids:
                    article['expansion_level'] = 1
                    article['expansion_method'] = 'strong_relations'
                    expanded_articles.append(article)
                    all_found_ids.add(article['article_id'])
            
            print(f"   📊 Nivel 1: +{len(level1_articles)} artículos por relaciones fuertes")
            
            # Nivel 2: Navegación por contexto semántico
            level1_ids = [art['article_id'] for art in level1_articles]
            all_current_ids = seed_ids + level1_ids
            
            level2_articles = self._expand_by_semantic_context(session, all_current_ids, primary_context, query)
            for article in level2_articles:
                if article['article_id'] not in all_found_ids:
                    article['expansion_level'] = 2
                    article['expansion_method'] = 'semantic_context'
                    expanded_articles.append(article)
                    all_found_ids.add(article['article_id'])
            
            print(f"   📊 Nivel 2: +{len(level2_articles)} artículos por contexto semántico")
            
            # Nivel 3: Navegación por cadenas procedimentales
            if max_depth >= 3:
                level3_articles = self._expand_by_procedural_chains(session, all_found_ids, primary_context)
                for article in level3_articles:
                    if article['article_id'] not in all_found_ids:
                        article['expansion_level'] = 3
                        article['expansion_method'] = 'procedural_chains'
                        expanded_articles.append(article)
                        all_found_ids.add(article['article_id'])
                
                print(f"   📊 Nivel 3: +{len(level3_articles)} artículos por cadenas procedimentales")
        
        print(f"🎯 Navegación completada: {len(expanded_articles)} artículos totales")
        return expanded_articles
    
    def _expand_by_strong_relations(self, session, seed_ids: List[str], context: str) -> List[Dict[str, Any]]:
        """Expandir por relaciones fuertes: referencias, tags compartidos, penalties similares."""
        query = """
        MATCH (seed:Article) 
        WHERE seed.article_id IN $seed_ids
        
        // Relaciones directas fuertes
        OPTIONAL MATCH (seed)-[r1:REFERENCES]-(ref_article:Article)
        OPTIONAL MATCH (seed)-[r2:SHARES_TAG]-(tag_article:Article)
        OPTIONAL MATCH (seed)-[r3:SIMILAR_PENALTY]-(penalty_article:Article)
        OPTIONAL MATCH (seed)-[r4:SAME_SECTION]-(section_article:Article)
        
        WITH DISTINCT COALESCE(ref_article, tag_article, penalty_article, section_article) as related_article,
             CASE 
                WHEN ref_article IS NOT NULL THEN 4.0
                WHEN tag_article IS NOT NULL THEN 3.0  
                WHEN penalty_article IS NOT NULL THEN 3.5
                WHEN section_article IS NOT NULL THEN 2.5
                ELSE 0.0
             END as relation_strength
        
        WHERE related_article IS NOT NULL 
        AND NOT related_article.article_id IN $seed_ids
        AND relation_strength > 2.0
        
        RETURN DISTINCT related_article.article_id as article_id,
               related_article.content as content,
               related_article.law_name as law_name,
               related_article.article_number as article_number,
               related_article.category as category,
               related_article.source as source,
               relation_strength as score
        
        ORDER BY relation_strength DESC
        LIMIT 10
        """
        
        try:
            result = session.run(query, seed_ids=seed_ids, timeout=10)
            articles = []
            for record in result:
                articles.append({
                    'article_id': record['article_id'],
                    'content': record['content'],
                    'law_name': record['law_name'],
                    'article_number': record['article_number'],
                    'category': record['category'],
                    'source': record['source'],
                    'score': float(record['score'])
                })
            return articles
        except Exception as e:
            print(f"   ❌ Error en expansión por relaciones fuertes: {str(e)}")
            return []
    
    def _expand_by_semantic_context(self, session, current_ids: List[str], context: str, query: str) -> List[Dict[str, Any]]:
        """Expandir por contexto semántico específico."""
        # Obtener patrones del contexto
        context_patterns = self.legal_context_patterns.get(context, {})
        
        # Construir términos de búsqueda contextual
        search_terms = []
        search_terms.extend(context_patterns.get('core_concepts', []))
        search_terms.extend(context_patterns.get('related_concepts', []))
        
        # Agregar términos específicos de la consulta
        query_words = [word for word in query.lower().split() if len(word) > 3]
        search_terms.extend(query_words[:5])
        
        # Eliminar duplicados y términos muy comunes
        search_terms = list(set(search_terms))
        common_words = ['para', 'con', 'por', 'sin', 'trabajo', 'ley']
        search_terms = [term for term in search_terms if term not in common_words][:8]
        
        if not search_terms:
            return []
        
        # Construir consulta dinámica
        where_conditions = [f"toLower(a.content) CONTAINS '{term}'" for term in search_terms]
        where_clause = " OR ".join(where_conditions)
        
        query_cypher = f"""
        MATCH (a:Article)
        WHERE ({where_clause})
        AND NOT a.article_id IN $current_ids
        
        WITH a,
             size([term IN $search_terms WHERE toLower(a.content) CONTAINS term]) as term_matches,
             
             // Boost por metadatos específicos del contexto
             CASE 
                WHEN a.has_penalties = true AND $context CONTAINS 'despido' THEN 2.0
                WHEN a.tag_count > 2 THEN 1.5
                WHEN toLower(a.category) CONTAINS 'trabajo' OR toLower(a.category) CONTAINS 'laboral' THEN 1.3
                ELSE 0.0 
             END as context_boost
        
        WITH a, (toFloat(term_matches) / size($search_terms)) * 8.0 + context_boost as semantic_score
        
        WHERE semantic_score > 2.0
        
        RETURN a.article_id as article_id,
               a.content as content,
               a.law_name as law_name,
               a.article_number as article_number,
               a.category as category,
               a.source as source,
               semantic_score as score
        
        ORDER BY semantic_score DESC
        LIMIT 12
        """
        
        try:
            result = session.run(query_cypher, {
                'current_ids': current_ids,
                'search_terms': search_terms,
                'context': context
            }, timeout=10)
            
            articles = []
            for record in result:
                articles.append({
                    'article_id': record['article_id'],
                    'content': record['content'],
                    'law_name': record['law_name'],
                    'article_number': record['article_number'],
                    'category': record['category'],
                    'source': record['source'],
                    'score': float(record['score'])
                })
            return articles
        except Exception as e:
            print(f"   ❌ Error en expansión semántica: {str(e)}")
            return []
    
    def _expand_by_procedural_chains(self, session, current_ids: Set[str], context: str) -> List[Dict[str, Any]]:
        """Expandir siguiendo cadenas procedimentales típicas."""
        # Obtener cadena procedimental para el contexto
        if context == 'embarazo_despido':
            chain_terms = self.procedural_chains.get('embarazo_proteccion', [])
            chain_terms.extend(self.procedural_chains.get('despido_proceso', []))
        elif context == 'contrato_trabajo':
            chain_terms = self.procedural_chains.get('despido_proceso', [])
            chain_terms.extend(self.procedural_chains.get('indemnizacion_proceso', []))
        else:
            chain_terms = ['procedimiento', 'tramite', 'solicitud', 'recurso', 'plazo']
        
        # Convertir términos compuestos en búsquedas
        search_patterns = []
        for term in chain_terms:
            if '_' in term:
                # Términos compuestos como 'notificacion_embarazo'
                words = term.split('_')
                search_patterns.append(' AND '.join([f"toLower(a.content) CONTAINS '{word}'" for word in words]))
            else:
                search_patterns.append(f"toLower(a.content) CONTAINS '{term}'")
        
        if not search_patterns:
            return []
        
        where_clause = " OR ".join([f"({pattern})" for pattern in search_patterns[:6]])
        
        query_cypher = f"""
        MATCH (a:Article)
        WHERE ({where_clause})
        AND NOT a.article_id IN $current_ids
        
        WITH a,
             // Contar coincidencias procedimentales
             size([term IN $chain_terms WHERE toLower(a.content) CONTAINS term]) as procedural_matches,
             
             // Boost por tipo de contenido procedimental
             CASE 
                WHEN toLower(a.content) CONTAINS 'procedimiento' THEN 1.5
                WHEN toLower(a.content) CONTAINS 'plazo' THEN 1.3
                WHEN toLower(a.content) CONTAINS 'notificación' THEN 1.2
                ELSE 0.0 
             END as procedural_boost
        
        WITH a, toFloat(procedural_matches) * 2.0 + procedural_boost as procedural_score
        
        WHERE procedural_score > 1.0
        
        RETURN a.article_id as article_id,
               a.content as content,
               a.law_name as law_name,
               a.article_number as article_number,
               a.category as category,
               a.source as source,
               procedural_score as score
        
        ORDER BY procedural_score DESC
        LIMIT 8
        """
        
        try:
            result = session.run(query_cypher, {
                'current_ids': list(current_ids),
                'chain_terms': chain_terms
            }, timeout=10)
            
            articles = []
            for record in result:
                articles.append({
                    'article_id': record['article_id'],
                    'content': record['content'],
                    'law_name': record['law_name'],
                    'article_number': record['article_number'],
                    'category': record['category'],
                    'source': record['source'],
                    'score': float(record['score'])
                })
            return articles
        except Exception as e:
            print(f"   ❌ Error en expansión procedimental: {str(e)}")
            return []


class IntelligentFilter:
    """
    Filtro inteligente que considera contexto legal para decisiones de relevancia.
    """
    
    def __init__(self):
        self.legal_context_weights = {
            'embarazo_despido': {
                'core_terms': {'embarazo': 3.0, 'despido': 3.0, 'discriminación': 2.8},
                'related_terms': {'indemnización': 2.5, 'estabilidad': 2.3, 'protección': 2.0},
                'procedural_terms': {'presunción': 2.2, 'notificación': 2.0, 'procedimiento': 1.8}
            },
            'contrato_trabajo': {
                'core_terms': {'contrato': 3.0, 'trabajo': 2.8, 'empleador': 2.5},
                'related_terms': {'salario': 2.3, 'jornada': 2.0, 'licencia': 2.0},
                'procedural_terms': {'rescisión': 2.5, 'preaviso': 2.2, 'terminación': 2.0}
            }
        }
        
        self.law_hierarchy_weights = {
            'Ley de Contrato de Trabajo': 3.0,
            'Codigo Civil y Comercial': 2.5,
            'Codigo Penal': 2.0,
            'Ley de Empleo': 2.8
        }
    
    def contextual_filtering(self, articles: List[Dict[str, Any]], query: str, 
                           context: str = 'general', base_threshold: float = 0.15) -> List[Dict[str, Any]]:
        """
        Filtrado inteligente que considera contexto legal y jerarquía normativa.
        """
        query_lower = query.lower()
        
        # Obtener pesos para el contexto detectado
        context_weights = self.legal_context_weights.get(context, {})
        
        filtered_articles = []
        
        for article in articles:
            # Similitud base
            base_similarity = article.get('semantic_similarity', article.get('score', 0) / 10.0)
            
            content_lower = article.get('content', '').lower()
            law_name = article.get('law_name', '')
            
            # 1. Boost por contexto específico
            context_boost = 1.0
            
            # Aplicar pesos por términos contextuales
            for term_type, terms in context_weights.items():
                for term, weight in terms.items():
                    if term in content_lower:
                        context_boost = max(context_boost, weight * 0.3)
            
            # 2. Boost por jerarquía legal
            law_boost = self.law_hierarchy_weights.get(law_name, 1.0) * 0.2
            
            # 3. Boost por método de obtención
            method_boost = 1.0
            method = article.get('method', '')
            expansion_level = article.get('expansion_level', 0)
            
            if method in ['optimized_seed', 'strong_relations']:
                method_boost = 1.4
            elif method == 'semantic_context' or expansion_level == 2:
                method_boost = 1.2
            elif method == 'procedural_chains' or expansion_level == 3:
                method_boost = 1.1
            
            # 4. Boost por densidad de términos relevantes
            query_terms = [word for word in query_lower.split() if len(word) > 3]
            term_density = sum(1 for term in query_terms if term in content_lower) / max(len(query_terms), 1)
            density_boost = 1.0 + (term_density * 0.5)
            
            # Calcular similitud ajustada
            adjusted_similarity = base_similarity * context_boost * density_boost * method_boost + (law_boost * 0.1)
            
            # Ajustar umbral basado en el contexto
            if context in ['embarazo_despido', 'contrato_trabajo']:
                # Ser más permisivo con contextos específicos
                adjusted_threshold = base_threshold * 0.7
            else:
                adjusted_threshold = base_threshold
            
            # Boost adicional para artículos con expansión inteligente
            if expansion_level > 0:
                adjusted_threshold *= 0.8
            
            # Aplicar filtro
            if adjusted_similarity > adjusted_threshold:
                article['adjusted_similarity'] = adjusted_similarity
                article['context_boost'] = context_boost
                article['law_boost'] = law_boost
                article['method_boost'] = method_boost
                article['density_boost'] = density_boost
                filtered_articles.append(article)
            else:
                article_info = f"{article.get('law_name', 'N/A')} Art. {article.get('article_number', 'N/A')}"
                print(f"   ❌ Filtrado contextual ({adjusted_similarity:.3f}): {article_info}")
        
        # Ordenar por similitud ajustada
        filtered_articles.sort(key=lambda x: x['adjusted_similarity'], reverse=True)
        
        print(f"🔍 Filtro inteligente: {len(filtered_articles)}/{len(articles)} artículos pasaron el filtro")
        
        return filtered_articles
    
    def rank_by_legal_importance(self, articles: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Re-ranking final considerando importancia legal y relevancia contextual.
        """
        query_lower = query.lower()
        
        for article in articles:
            content_lower = article.get('content', '').lower()
            
            # Factores de importancia legal
            importance_score = 0.0
            
            # 1. Artículos que definen derechos fundamentales
            if any(term in content_lower for term in ['derecho', 'obligación', 'prohib']):
                importance_score += 1.0
            
            # 2. Artículos con procedimientos específicos
            if any(term in content_lower for term in ['procedimiento', 'plazo', 'notificación']):
                importance_score += 0.8
            
            # 3. Artículos con sanciones/penalties
            if any(term in content_lower for term in ['sanción', 'pena', 'multa', 'responsabilidad']):
                importance_score += 0.6
            
            # 4. Artículos con definiciones
            if any(term in content_lower for term in ['se entiende', 'definición', 'concepto']):
                importance_score += 0.5
            
            # Combinar con similitud ajustada
            final_score = article.get('adjusted_similarity', 0) + (importance_score * 0.3)
            article['final_legal_score'] = final_score
        
        # Ordenar por score legal final
        articles.sort(key=lambda x: x.get('final_legal_score', 0), reverse=True)
        
        return articles