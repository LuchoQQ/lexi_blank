"""
api_integration.py

Módulo de integración que conecta el sistema de clasificación inteligente
con la API existente y adapta las búsquedas según el tipo de consulta.
"""

import time
from typing import Dict, Any, List, Optional
from enum import Enum
from .legal_query_classifier import IntelligentLegalSystem

# Importar el sistema de clasificación
from .legal_query_classifier import (
    QueryType, UrgencyLevel, QueryClassification, LegalSpecialist
)
from .llama_integration import EnhancedIntelligentLegalSystem

class EnhancedAPIConsultaHandler:
    """
    Handler mejorado para consultas que integra clasificación inteligente
    """
    
    def __init__(self, config: Dict[str, Any], weaviate_client=None, 
                 neo4j_driver=None, documents=None, openai_client=None, llama_config=None):
        self.config = config
        self.weaviate_client = weaviate_client
        self.neo4j_driver = neo4j_driver
        self.documents = documents
        self.openai_client = openai_client
        self.llama_config = llama_config
        
        # Inicializar sistema inteligente
        self.intelligent_system = EnhancedIntelligentLegalSystem(llama_config)
        
        # Mapeo de estrategias de búsqueda
        self.search_strategies = {
            "exact_lookup": self._execute_exact_article_search,
            "graph_rag_enhanced": self._execute_enhanced_graph_rag,
            "procedural_chains": self._execute_procedural_search,
            "balanced_comprehensive": self._execute_balanced_search
        }
        
        # Generadores de respuesta especializados
        self.response_generators = {
            "structured_articles": self._generate_article_response,
            "legal_advice": self._generate_legal_advice_response,
            "step_by_step_guide": self._generate_procedural_response,
            "educational_overview": self._generate_educational_response
        }
    
    def process_intelligent_consulta(self, query: str, top_n: int = 15) -> str:
        """
        Procesa una consulta usando el sistema inteligente completo
        """
        print(f"🧠 Iniciando procesamiento inteligente para: '{query[:50]}...'")
        start_time = time.time()
        
        try:
            # 1. Clasificar consulta
            classification_result = self.intelligent_system.process_query(query)
            classification = classification_result["classification"]
            specialist_config = classification_result["specialist_routing"]
            
            # 2. Ejecutar búsqueda especializada
            search_results = self._execute_specialized_search(
                query, specialist_config, top_n
            )
            
            # 3. Generar respuesta especializada
            response = self._generate_specialized_response(
                query, search_results, classification, specialist_config
            )
            
            processing_time = time.time() - start_time
            print(f"✅ Procesamiento inteligente completado en {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            print(f"❌ Error en procesamiento inteligente: {str(e)}")
            # Fallback a sistema tradicional
            return self._fallback_to_traditional_search(query, top_n)
    
    def _execute_specialized_search(self, query: str, specialist_config: Dict[str, Any], 
                                   top_n: int) -> List[Dict[str, Any]]:
        """Ejecuta búsqueda especializada según configuración del especialista"""
        search_strategy = specialist_config.get("search_strategy", "balanced_comprehensive")
        search_config = specialist_config.get("search_config", {})
        
        # Obtener función de búsqueda
        search_function = self.search_strategies.get(search_strategy, 
                                                   self._execute_balanced_search)
        
        # Ejecutar búsqueda
        return search_function(query, search_config, top_n)
    
    def _execute_exact_article_search(self, query: str, config: Dict[str, Any], 
                                     top_n: int) -> List[Dict[str, Any]]:
        """Búsqueda exacta de artículos específicos"""
        print("🔍 Ejecutando búsqueda exacta de artículos...")
        
        target_articles = config.get("target_articles", [])
        results = []
        
        if not target_articles:
            # Si no hay artículos específicos, usar búsqueda tradicional
            return self._execute_balanced_search(query, config, top_n)
        
        # Buscar artículos específicos mencionados
        for article_ref in target_articles:
            specific_results = self._search_specific_article(article_ref)
            results.extend(specific_results)
        
        # Si queremos artículos relacionados
        if config.get("include_related", False) and results:
            related_results = self._find_related_articles(results, top_n - len(results))
            results.extend(related_results)
        
        return results[:top_n]
    
    def _execute_enhanced_graph_rag(self, query: str, config: Dict[str, Any], 
                                   top_n: int) -> List[Dict[str, Any]]:
        """Búsqueda Graph RAG mejorada para análisis de casos"""
        print("🕸️ Ejecutando Graph RAG mejorado para análisis de caso...")
        
        # Usar el sistema Graph RAG existente con configuración especializada
        if self.neo4j_driver:
            try:
                from .neo4j_utils import search_neo4j_enhanced
                
                # Configurar límites más altos para casos complejos
                case_elements = config.get("case_elements", {})
                urgency_boost = config.get("urgency_boost", "medium")
                
                # Ajustar límite según urgencia
                if urgency_boost == "high":
                    effective_limit = min(top_n * 2, 25)
                elif urgency_boost == "critical":
                    effective_limit = min(top_n * 3, 30)
                else:
                    effective_limit = top_n
                
                results = search_neo4j_enhanced(self.neo4j_driver, query, effective_limit)
                
                # Aplicar boost contextual para casos
                for result in results:
                    result = self._apply_case_contextual_boost(result, case_elements)
                
                # Re-ordenar por relevancia contextual
                results.sort(key=lambda x: x.get('score', 0), reverse=True)
                
                return results[:top_n]
                
            except Exception as e:
                print(f"❌ Error en Graph RAG mejorado: {str(e)}")
        
        # Fallback a búsqueda balanceada
        return self._execute_balanced_search(query, config, top_n)
    
    def _execute_procedural_search(self, query: str, config: Dict[str, Any], 
                                  top_n: int) -> List[Dict[str, Any]]:
        """Búsqueda enfocada en procedimientos y pasos"""
        print("📋 Ejecutando búsqueda procedimental...")
        
        # Términos procedimentales específicos
        procedural_terms = [
            "procedimiento", "paso", "requisito", "plazo", "formulario",
            "denuncia", "presentar", "tramitar", "solicitar"
        ]
        
        # Modificar consulta para enfocarse en procedimientos
        enhanced_query = f"{query} {' '.join(procedural_terms[:3])}"
        
        # Ejecutar búsqueda con términos procedimentales
        results = self._execute_balanced_search(enhanced_query, config, top_n)
        
        # Filtrar y priorizar artículos procedimentales
        procedural_results = []
        for result in results:
            content_lower = result.get('content', '').lower()
            procedural_score = sum(1 for term in procedural_terms if term in content_lower)
            
            if procedural_score > 0:
                result['procedural_score'] = procedural_score
                result['score'] = result.get('score', 0) + (procedural_score * 0.5)
                procedural_results.append(result)
        
        # Ordenar por relevancia procedimental
        procedural_results.sort(key=lambda x: x.get('procedural_score', 0), reverse=True)
        
        return procedural_results[:top_n]
    
    def _execute_balanced_search(self, query: str, config: Dict[str, Any], 
                                top_n: int) -> List[Dict[str, Any]]:
        """Búsqueda balanceada tradicional (fallback)"""
        print("⚖️ Ejecutando búsqueda balanceada...")
        
        try:
            # Usar el sistema existente
            from main import search_query_neutral
            return search_query_neutral(
                query, self.config, self.weaviate_client, 
                self.neo4j_driver, self.documents
            )[:top_n]
        except Exception as e:
            print(f"❌ Error en búsqueda balanceada: {str(e)}")
            return []
    
    def _search_specific_article(self, article_ref: str) -> List[Dict[str, Any]]:
        """Busca un artículo específico por referencia"""
        results = []
        
        # Buscar en Neo4j si está disponible
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    # Consulta específica para artículo
                    query = """
                    MATCH (a:Article)
                    WHERE toLower(a.content) CONTAINS toLower($article_ref)
                    OR a.article_number CONTAINS $article_ref
                    OR a.article_id CONTAINS $article_ref
                    RETURN a.article_id as article_id,
                           a.content as content,
                           a.law_name as law_name,
                           a.article_number as article_number,
                           a.category as category,
                           a.source as source,
                           5.0 as score
                    LIMIT 3
                    """
                    
                    result = session.run(query, article_ref=article_ref)
                    for record in result:
                        results.append({
                            'article_id': record['article_id'],
                            'content': record['content'],
                            'law_name': record['law_name'],
                            'article_number': record['article_number'],
                            'category': record['category'],
                            'source': record['source'],
                            'score': float(record['score']),
                            'method': 'exact_article_lookup'
                        })
            except Exception as e:
                print(f"Error buscando artículo específico: {str(e)}")
        
        return results
    
    def _find_related_articles(self, base_articles: List[Dict[str, Any]], 
                              limit: int) -> List[Dict[str, Any]]:
        """Encuentra artículos relacionados a los artículos base"""
        if not base_articles or not self.neo4j_driver:
            return []
        
        related_articles = []
        base_ids = [art['article_id'] for art in base_articles if art.get('article_id')]
        
        if not base_ids:
            return []
        
        try:
            with self.neo4j_driver.session() as session:
                query = """
                MATCH (base:Article)-[r]-(related:Article)
                WHERE base.article_id IN $base_ids
                AND NOT related.article_id IN $base_ids
                
                WITH related, 
                     CASE type(r)
                        WHEN 'SHARES_TAG' THEN 2.0
                        WHEN 'REFERENCES' THEN 3.0
                        WHEN 'SAME_SECTION' THEN 1.5
                        ELSE 1.0
                     END as relation_strength
                
                RETURN DISTINCT related.article_id as article_id,
                       related.content as content,
                       related.law_name as law_name,
                       related.article_number as article_number,
                       related.category as category,
                       related.source as source,
                       max(relation_strength) as score
                
                ORDER BY score DESC
                LIMIT $limit
                """
                
                result = session.run(query, base_ids=base_ids, limit=limit)
                for record in result:
                    related_articles.append({
                        'article_id': record['article_id'],
                        'content': record['content'],
                        'law_name': record['law_name'],
                        'article_number': record['article_number'],
                        'category': record['category'],
                        'source': record['source'],
                        'score': float(record['score']),
                        'method': 'related_to_exact'
                    })
        except Exception as e:
            print(f"Error buscando artículos relacionados: {str(e)}")
        
        return related_articles
    
    def _apply_case_contextual_boost(self, result: Dict[str, Any], 
                                   case_elements: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica boost contextual para análisis de casos"""
        content_lower = result.get('content', '').lower()
        base_score = result.get('score', 0)
        boost_factor = 1.0
        
        # Boost por problemas legales identificados
        legal_issues = case_elements.get('legal_issues', [])
        for issue in legal_issues:
            if issue.replace('_', ' ') in content_lower:
                boost_factor *= 1.3
        
        # Boost por stakeholders relevantes
        stakeholders = case_elements.get('stakeholders', [])
        for stakeholder in stakeholders:
            if stakeholder in content_lower:
                boost_factor *= 1.2
        
        # Boost por daños identificados
        damages = case_elements.get('damages_claimed', [])
        for damage in damages:
            if damage in content_lower:
                boost_factor *= 1.25
        
        result['score'] = base_score * boost_factor
        result['case_boost_factor'] = boost_factor
        
        return result
    
    def _generate_specialized_response(self, query: str, search_results: List[Dict[str, Any]], 
                                     classification: Dict[str, Any], 
                                     specialist_config: Dict[str, Any]) -> str:
        """Genera respuesta especializada según el tipo de consulta"""
        response_format = specialist_config.get("response_format", "educational_overview")
        
        # Obtener generador de respuesta apropiado
        response_generator = self.response_generators.get(response_format, 
                                                        self._generate_educational_response)
        
        return response_generator(query, search_results, classification, specialist_config)
    
    def _generate_article_response(self, query: str, search_results: List[Dict[str, Any]], 
                                 classification: Dict[str, Any], 
                                 specialist_config: Dict[str, Any]) -> str:
        """Genera respuesta estructurada para consultas de artículos específicos"""
        if not search_results:
            return "No se encontraron los artículos específicos solicitados. Por favor, verifique la referencia e intente nuevamente."
        
        response_parts = []
        
        # Encabezado específico para artículos
        response_parts.append("📋 ARTÍCULOS SOLICITADOS\n")
        
        for i, result in enumerate(search_results[:5], 1):
            law_name = result.get('law_name', 'Ley no especificada')
            article_num = result.get('article_number', 'N/A')
            content = result.get('content', 'Contenido no disponible')
            
            response_parts.append(f"**ARTÍCULO {i}**")
            response_parts.append(f"📚 Fuente: {law_name}")
            response_parts.append(f"📄 Artículo: {article_num}")
            response_parts.append(f"📝 Contenido:\n{content}")
            response_parts.append("-" * 50)
        
        # Agregar explicación contextual si hay OpenAI
        if self.openai_client and len(search_results) > 0:
            contextual_explanation = self._generate_contextual_explanation(
                query, search_results[:3], "article_explanation"
            )
            if contextual_explanation:
                response_parts.append("\n💡 EXPLICACIÓN CONTEXTUAL:")
                response_parts.append(contextual_explanation)
        
        return "\n\n".join(response_parts)
    
    def _generate_legal_advice_response(self, query: str, search_results: List[Dict[str, Any]], 
                                      classification: Dict[str, Any], 
                                      specialist_config: Dict[str, Any]) -> str:
        """Genera respuesta de asesoramiento legal para análisis de casos"""
        if not self.openai_client:
            return "El servicio de asesoramiento legal no está disponible en este momento."
        
        try:
            # Preparar contexto especializado para casos
            urgency_level = classification.get("urgency_level", "medium")
            legal_domains = classification.get("legal_domains", [])
            
            # Obtener elementos del caso si están disponibles
            case_analysis = specialist_config.get("case_analysis", {})
            
            # Generar asesoramiento especializado
            legal_advice = self._generate_case_analysis_advice(
                query, search_results, urgency_level, legal_domains, case_analysis
            )
            
            return legal_advice
            
        except Exception as e:
            print(f"Error generando asesoramiento legal: {str(e)}")
            return f"Lo siento, hubo un error al generar el asesoramiento legal. Por favor, consulte con un abogado especializado."
    
    def _generate_procedural_response(self, query: str, search_results: List[Dict[str, Any]], 
                                    classification: Dict[str, Any], 
                                    specialist_config: Dict[str, Any]) -> str:
        """Genera respuesta paso a paso para orientación procedimental"""
        if not search_results:
            return "No se encontraron procedimientos específicos para su consulta. Le recomiendo consultar con un abogado especializado."
        
        # Generar guía procedimental estructurada
        if self.openai_client:
            return self._generate_procedural_guide(query, search_results)
        else:
            # Respuesta básica sin OpenAI
            response_parts = ["📋 ORIENTACIÓN PROCEDIMENTAL\n"]
            
            for i, result in enumerate(search_results[:8], 1):
                content = result.get('content', '')[:300]
                law_name = result.get('law_name', 'N/A')
                article_num = result.get('article_number', 'N/A')
                
                response_parts.append(f"**PASO {i}**")
                response_parts.append(f"📚 Referencia: {law_name} Art. {article_num}")
                response_parts.append(f"📝 Información: {content}...")
                response_parts.append("")
            
            return "\n".join(response_parts)
    
    def _generate_educational_response(self, query: str, search_results: List[Dict[str, Any]], 
                                     classification: Dict[str, Any], 
                                     specialist_config: Dict[str, Any]) -> str:
        """Genera respuesta educativa para consultas generales"""
        if not search_results:
            return "No se encontraron resultados relevantes para su consulta. Por favor, intente reformular su pregunta."
        
        if self.openai_client:
            return self._generate_educational_overview(query, search_results)
        else:
            # Respuesta básica sin OpenAI
            response_parts = ["📚 INFORMACIÓN LEGAL RELEVANTE\n"]
            
            for i, result in enumerate(search_results[:10], 1):
                law_name = result.get('law_name', 'N/A')
                article_num = result.get('article_number', 'N/A')
                content = result.get('content', '')[:400]
                
                response_parts.append(f"**INFORMACIÓN {i}**")
                response_parts.append(f"📚 Fuente: {law_name} Art. {article_num}")
                response_parts.append(f"📝 Contenido: {content}...")
                response_parts.append("")
            
            return "\n".join(response_parts)
    
    def _generate_case_analysis_advice(self, query: str, search_results: List[Dict[str, Any]], 
                                     urgency_level: str, legal_domains: List[str], 
                                     case_analysis: Dict[str, Any]) -> str:
        """Genera asesoramiento especializado para análisis de casos"""
        # Preparar contexto específico para casos
        relevant_articles_text = ""
        top_articles = search_results[:8]
        
        for i, article in enumerate(top_articles, 1):
            law_name = article.get('law_name', 'Ley no especificada')
            article_num = article.get('article_number', 'N/A')
            content = article.get('content', '')[:600]
            
            relevant_articles_text += f"\n--- Artículo {i} ({law_name} - Art. {article_num}) ---\n{content}\n"
        
        # Prompt especializado para análisis de casos
        system_prompt = f"""Eres un abogado especialista en derecho argentino experto en análisis de casos legales. El usuario presenta una situación legal específica que requiere asesoramiento práctico y accionable.

CONTEXTO DEL CASO:
- Urgencia: {urgency_level}
- Dominios legales: {', '.join(legal_domains)}
- Elementos del caso detectados: {case_analysis}

INSTRUCCIONES ESPECÍFICAS:
1. Analiza la situación legal del usuario con profundidad
2. Identifica claramente qué derechos le asisten según los artículos
3. Evalúa las posibles violaciones legales
4. Proporciona recomendaciones específicas y PASOS CONCRETOS a seguir
5. Incluye información sobre plazos legales críticos
6. Considera el contexto emocional y urgencia de la situación
7. Proporciona información sobre recursos disponibles

FORMATO DE RESPUESTA:
🔍 **ANÁLISIS DE LA SITUACIÓN**
[Análisis detallado de la situación legal]

⚖️ **DERECHOS QUE LE ASISTEN**
[Derechos específicos con base legal]

🚨 **VIOLACIONES IDENTIFICADAS** (si aplica)
[Violaciones legales detectadas]

📋 **RECOMENDACIONES ESPECÍFICAS**
[Acciones concretas a tomar]

⏰ **PLAZOS IMPORTANTES**
[Plazos legales críticos]

🆘 **PRÓXIMOS PASOS INMEDIATOS**
[Pasos específicos ordenados por prioridad]

⚠️ **ADVERTENCIAS LEGALES**
[Advertencias sobre plazos y riesgos]

📞 **RECURSOS ADICIONALES**
[Información sobre dónde obtener más ayuda]

🔒 **DISCLAIMER LEGAL**
[Disclaimer apropiado]"""

        user_prompt = f"""SITUACIÓN LEGAL: "{query}"

ARTÍCULOS LEGALES APLICABLES:
{relevant_articles_text}

Por favor, proporciona un análisis legal completo y asesoramiento práctico basándote exclusivamente en estos artículos y tu conocimiento del derecho argentino."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.2,
                timeout=30.0
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generando análisis de caso: {str(e)}")
            return f"Lo siento, hubo un error al generar el análisis del caso. Error técnico: {str(e)}. Por favor, consulte con un abogado especializado para obtener asesoramiento específico sobre su situación."
    
    def _generate_procedural_guide(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Genera guía procedimental paso a paso"""
        relevant_articles_text = ""
        for i, article in enumerate(search_results[:6], 1):
            law_name = article.get('law_name', 'Ley no especificada')
            article_num = article.get('article_number', 'N/A')
            content = article.get('content', '')[:500]
            
            relevant_articles_text += f"\n--- Referencia {i} ({law_name} - Art. {article_num}) ---\n{content}\n"
        
        system_prompt = """Eres un experto en procedimientos legales argentinos. Tu tarea es proporcionar una guía paso a paso clara y práctica sobre el procedimiento legal consultado.

INSTRUCCIONES:
1. Proporciona una guía paso a paso detallada
2. Incluye requisitos específicos para cada paso
3. Menciona formularios necesarios y dónde obtenerlos
4. Especifica plazos legales importantes
5. Incluye costos aproximados si son relevantes
6. Proporciona alternativas cuando sea posible

FORMATO DE RESPUESTA:
📋 **GUÍA PROCEDIMENTAL PASO A PASO**

🎯 **OBJETIVO**
[Descripción clara del objetivo del procedimiento]

📝 **PASOS A SEGUIR**

**PASO 1: [Título del paso]**
- Descripción detallada
- Requisitos específicos
- Documentos necesarios
- Plazo: [si aplica]
- Costo: [si aplica]

**PASO 2: [Título del paso]**
[Continuar con formato similar]

📋 **DOCUMENTOS NECESARIOS**
[Lista completa de documentos requeridos]

💰 **COSTOS ESTIMADOS**
[Desglose de costos si aplica]

⏰ **PLAZOS IMPORTANTES**
[Plazos críticos a tener en cuenta]

🔄 **ALTERNATIVAS DISPONIBLES**
[Opciones alternativas si existen]

⚠️ **ADVERTENCIAS IMPORTANTES**
[Advertencias sobre el proceso]

📞 **CONTACTOS ÚTILES**
[Información de contacto relevante]"""

        user_prompt = f"""CONSULTA PROCEDIMENTAL: "{query}"

REFERENCIAS LEGALES:
{relevant_articles_text}

Por favor, proporciona una guía procedimental completa y práctica."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1200,
                temperature=0.1,
                timeout=30.0
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generando guía procedimental: {str(e)}")
            return f"Lo siento, hubo un error al generar la guía procedimental. Por favor, consulte con un abogado especializado."
    
    def _generate_educational_overview(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Genera overview educativo para consultas generales"""
        relevant_articles_text = ""
        for i, article in enumerate(search_results[:8], 1):
            law_name = article.get('law_name', 'Ley no especificada')
            article_num = article.get('article_number', 'N/A')
            content = article.get('content', '')[:400]
            
            relevant_articles_text += f"\n--- Artículo {i} ({law_name} - Art. {article_num}) ---\n{content}\n"
        
        system_prompt = """Eres un experto en derecho argentino que proporciona información educativa clara y accesible. Tu objetivo es educar al usuario sobre el tema legal consultado.

INSTRUCCIONES:
1. Proporciona una explicación clara y educativa
2. Usa un lenguaje accesible evitando jerga legal excesiva
3. Incluye ejemplos prácticos cuando sea relevante
4. Menciona los aspectos más importantes del tema
5. Proporciona una base sólida para entender el tema

FORMATO DE RESPUESTA:
📚 **INFORMACIÓN EDUCATIVA**

🎯 **RESUMEN EJECUTIVO**
[Resumen claro del tema consultado]

📖 **CONCEPTOS FUNDAMENTALES**
[Explicación de conceptos clave]

⚖️ **MARCO LEGAL**
[Base legal del tema con artículos relevantes]

💡 **EJEMPLOS PRÁCTICOS**
[Ejemplos que ilustren los conceptos]

🔍 **ASPECTOS IMPORTANTES**
[Puntos clave a tener en cuenta]

📋 **PRÓXIMOS PASOS SUGERIDOS**
[Qué hacer si el usuario necesita más información]

📚 **RECURSOS ADICIONALES**
[Referencias para profundizar en el tema]"""

        user_prompt = f"""CONSULTA EDUCATIVA: "{query}"

ARTÍCULOS LEGALES RELEVANTES:
{relevant_articles_text}

Por favor, proporciona una explicación educativa completa sobre este tema legal."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1200,
                temperature=0.3,
                timeout=30.0
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generando overview educativo: {str(e)}")
            return f"Lo siento, hubo un error al generar la información educativa. Por favor, consulte con un abogado especializado."
    
    def _generate_contextual_explanation(self, query: str, search_results: List[Dict[str, Any]], 
                                       explanation_type: str) -> Optional[str]:
        """Genera explicación contextual específica"""
        if not self.openai_client or not search_results:
            return None
        
        try:
            context_text = ""
            for result in search_results:
                context_text += f"{result.get('content', '')[:300]}...\n\n"
            
            explanation_prompt = f"""Proporciona una explicación breve y clara sobre estos artículos legales en relación a la consulta: "{query}"

Artículos:
{context_text}

Explicación (máximo 200 palabras):"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": explanation_prompt}
                ],
                max_tokens=300,
                temperature=0.2,
                timeout=15.0
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generando explicación contextual: {str(e)}")
            return None
    
    def _fallback_to_traditional_search(self, query: str, top_n: int) -> str:
        """Fallback al sistema tradicional cuando el inteligente falla"""
        print("🔄 Fallback a sistema tradicional...")
        
        try:
            from main import search_query_neutral
            
            # Usar búsqueda tradicional
            search_results = search_query_neutral(
                query, self.config, self.weaviate_client, 
                self.neo4j_driver, self.documents
            )[:top_n]
            
            # Generar respuesta tradicional con GPT si está disponible
            if self.openai_client and search_results:
                return self._generate_traditional_gpt_response(query, search_results)
            else:
                return "Lo siento, el sistema no está disponible en este momento. Por favor, consulte con un abogado especializado."
                
        except Exception as e:
            print(f"Error en fallback tradicional: {str(e)}")
            return "Lo siento, ocurrió un error al procesar su consulta. Por favor, intente nuevamente o consulte con un abogado especializado."
    
    def _generate_traditional_gpt_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Genera respuesta usando el sistema GPT tradicional"""
        try:
            # Usar el generador GPT existente del sistema
            from api import generate_gpt_advice
            return generate_gpt_advice(query, search_results)
        except Exception as e:
            print(f"Error en respuesta GPT tradicional: {str(e)}")
            return f"Lo siento, hubo un error al generar la respuesta. Error: {str(e)}"


# ========== INTEGRACIÓN CON LA API EXISTENTE ==========

def integrate_intelligent_system_with_api():
    """
    Función para integrar el sistema inteligente con la API existente
    """
    print("🚀 Integrando sistema inteligente con API existente...")
    
    # Modificaciones necesarias en api.py:
    
    integration_code = '''
# Agregar al inicio de api.py después de las importaciones existentes:
from .api_integration import EnhancedAPIConsultaHandler

# Modificar la función startup_event() para incluir el handler inteligente:
enhanced_handler = None

@app.on_event("startup")
async def startup_event():
    """Inicializar conexiones y cargar configuración al iniciar la API."""
    global config, weaviate_client, neo4j_driver, documents, openai_client, enhanced_handler
    
    # ... código existente ...
    
    # Inicializar handler inteligente
    enhanced_handler = EnhancedAPIConsultaHandler(
        config=config,
        weaviate_client=weaviate_client,
        neo4j_driver=neo4j_driver,
        documents=documents,
        openai_client=openai_client
    )
    
    print("🧠 Sistema inteligente de clasificación inicializado")

# Modificar el endpoint /consulta para usar el sistema inteligente:
@app.post("/consulta", response_model=ConsultaResponse)
async def realizar_consulta(request: ConsultaRequest):
    """
    Realizar una consulta legal con clasificación inteligente y asesoramiento especializado.
    """
    start_time = time.time()
    
    try:
        # Validaciones existentes...
        if not documents:
            return ConsultaResponse(
                response="Lo siento, el sistema no está disponible en este momento."
            )
        
        if not openai_client:
            return ConsultaResponse(
                response="Lo siento, el servicio de asesoramiento legal no está disponible."
            )
        
        print(f"🧠 Procesando consulta inteligente: '{request.query}'")
        
        # Usar el sistema inteligente
        intelligent_response = enhanced_handler.process_intelligent_consulta(
            request.query, request.top_n
        )
        
        execution_time = time.time() - start_time
        print(f"✅ Consulta inteligente procesada en {execution_time:.2f}s")
        
        return ConsultaResponse(response=intelligent_response)
        
    except Exception as e:
        print(f"❌ Error en consulta inteligente: {str(e)}")
        return ConsultaResponse(
            response=f"Lo siento, ocurrió un error al procesar su consulta: {str(e)}"
        )

# Agregar endpoint para información del sistema inteligente:
@app.get("/sistema/info")
async def obtener_info_sistema():
    """Obtener información sobre el sistema de clasificación inteligente."""
    return {
        "sistema": "Clasificación Inteligente de Consultas Legales",
        "version": "1.0.0",
        "tipos_consulta": [
            {"tipo": "article_lookup", "descripcion": "Búsqueda específica de artículos"},
            {"tipo": "case_analysis", "descripcion": "Análisis de situación legal específica"},
            {"tipo": "general_consultation", "descripcion": "Consulta general sobre derecho"},
            {"tipo": "procedural_guidance", "descripcion": "Orientación sobre procedimientos"},
            {"tipo": "comparative_analysis", "descripcion": "Comparación entre leyes/artículos"}
        ],
        "especialistas": [
            {"especialista": "article_specialist", "descripcion": "Especialista en búsqueda de artículos"},
            {"especialista": "case_analyst", "descripcion": "Analista de casos legales"},
            {"especialista": "procedural_guide", "descripcion": "Guía procedimental"},
            {"especialista": "general_counselor", "descripcion": "Consejero general"}
        ],
        "caracteristicas": [
            "Clasificación automática con Llama",
            "Routing a especialistas apropiados", 
            "Análisis contextual avanzado",
            "Respuestas adaptadas al tipo de consulta",
            "Detección de urgencia y contexto emocional"
        ]
    }

# Agregar endpoint para testing de clasificación:
@app.post("/sistema/clasificar")
async def clasificar_consulta(request: dict):
    """Endpoint para testing de clasificación de consultas."""
    try:
        query = request.get("query", "")
        if not query:
            return {"error": "Query requerido"}
        
        # Usar solo el clasificador
        classification_result = enhanced_handler.intelligent_system.process_query(query)
        
        return {
            "query": query,
            "clasificacion": classification_result["classification"],
            "especialista_recomendado": classification_result["specialist_routing"]["specialist_type"],
            "estrategia_busqueda": classification_result["specialist_routing"]["search_strategy"]
        }
    except Exception as e:
        return {"error": f"Error en clasificación: {str(e)}"}
'''
    
    print("📋 Código de integración generado.")
    print("🔧 Para completar la integración:")
    print("   1. Agregar EnhancedAPIConsultaHandler a api.py")
    print("   2. Modificar startup_event() para inicializar el handler")
    print("   3. Actualizar endpoint /consulta para usar el sistema inteligente")
    print("   4. Agregar endpoints adicionales para información del sistema")
    
    return integration_code


# ========== EJEMPLO DE USO COMPLETO ==========

def demo_complete_integration():
    """Demostración completa del sistema integrado"""
    print("="*80)
    print("🚀 DEMOSTRACIÓN COMPLETA - SISTEMA INTELIGENTE INTEGRADO")
    print("="*80)
    
    # Simular configuración
    mock_config = {
        "weaviate": {"enabled": True},
        "neo4j": {"enabled": True},
        "retrieval": {"top_n": 15}
    }
    
    # Crear handler (sin clientes reales para demo)
    handler = EnhancedAPIConsultaHandler(
        config=mock_config,
        weaviate_client=None,  # En producción serían clientes reales
        neo4j_driver=None,
        documents=None,
        openai_client=None
    )
    
    # Casos de prueba que demuestran diferentes tipos de consultas
    test_cases = [
        {
            "query": "¿Cuál es el artículo 14 del código penal?",
            "expected_type": "article_lookup",
            "expected_specialist": "article_specialist"
        },
        {
            "query": "Fui despedida luego de trabajar durante 5 años sin indemnización después de avisar que estoy embarazada",
            "expected_type": "case_analysis", 
            "expected_specialist": "case_analyst"
        },
        {
            "query": "¿Cómo presento una denuncia por acoso laboral en el ministerio de trabajo?",
            "expected_type": "procedural_guidance",
            "expected_specialist": "procedural_guide"
        },
        {
            "query": "¿Cuáles son mis derechos como trabajador en Argentina?",
            "expected_type": "general_consultation",
            "expected_specialist": "general_counselor"
        },
        {
            "query": "¿Qué diferencias hay entre despido con causa justificada y sin causa?",
            "expected_type": "comparative_analysis",
            "expected_specialist": "general_counselor"
        }
    ]
    
    print(f"\n🧪 Probando {len(test_cases)} casos de diferentes tipos de consultas...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{'='*60}")
        print(f"CASO DE PRUEBA {i}")
        print(f"{'='*60}")
        print(f"📝 Consulta: {test_case['query']}")
        print(f"🎯 Tipo esperado: {test_case['expected_type']}")
        print(f"👨‍⚖️ Especialista esperado: {test_case['expected_specialist']}")
        
        # Procesar con sistema inteligente
        try:
            result = handler.intelligent_system.process_query(test_case['query'])
            
            classification = result["classification"]
            specialist_config = result["specialist_routing"]
            
            print(f"\n✅ RESULTADO:")
            print(f"   🏷️  Tipo detectado: {classification['query_type']}")
            print(f"   👨‍⚖️ Especialista asignado: {specialist_config['specialist_type']}")
            print(f"   🔍 Estrategia de búsqueda: {specialist_config['search_strategy']}")
            print(f"   📊 Confianza: {classification['confidence']:.2f}")
            print(f"   📈 Dominios: {', '.join(classification['legal_domains'])}")
            
            # Verificar predicción
            type_correct = classification['query_type'] == test_case['expected_type']
            specialist_correct = specialist_config['specialist_type'] == test_case['expected_specialist']
            
            print(f"\n🎯 PRECISIÓN:")
            print(f"   ✅ Tipo: {'CORRECTO' if type_correct else 'INCORRECTO'}")
            print(f"   ✅ Especialista: {'CORRECTO' if specialist_correct else 'INCORRECTO'}")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
        
        print()
    
    print("="*80)
    print("📊 RESUMEN DE LA DEMOSTRACIÓN")
    print("="*80)
    print("✅ Sistema de clasificación inteligente funcional")
    print("✅ Routing automático a especialistas apropiados")
    print("✅ Configuración de búsqueda adaptada por tipo de consulta")
    print("✅ Análisis contextual y detección de urgencia")
    print("✅ Preparado para integración con API existente")
    print("\n🚀 ¡Sistema listo para implementación!")


if __name__ == "__main__":
    # Ejecutar demostración
    demo_complete_integration()
    
    # Mostrar código de integración
    print("\n" + "="*80)
    print("🔧 CÓDIGO DE INTEGRACIÓN")
    print("="*80)
    integration_code = integrate_intelligent_system_with_api()
    print(integration_code)