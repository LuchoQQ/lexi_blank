"""
Sistema de Clasificación Inteligente de Consultas Legales
Utiliza Llama para clasificar y enrutar consultas a especialistas apropiados.
"""

import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

class QueryType(Enum):
    """Tipos de consultas legales identificadas"""
    ARTICLE_LOOKUP = "article_lookup"           # Búsqueda específica de artículos
    CASE_ANALYSIS = "case_analysis"             # Análisis de situación legal específica
    GENERAL_CONSULTATION = "general_consultation" # Consulta general sobre derecho
    PROCEDURAL_GUIDANCE = "procedural_guidance"  # Orientación sobre procedimientos
    COMPARATIVE_ANALYSIS = "comparative_analysis" # Comparación entre leyes/artículos
    UNDEFINED = "undefined"                      # No se puede clasificar claramente

class UrgencyLevel(Enum):
    """Niveles de urgencia de la consulta"""
    LOW = "low"           # Consulta académica o general
    MEDIUM = "medium"     # Situación que requiere atención
    HIGH = "high"         # Situación urgente (despidos, discriminación activa)
    CRITICAL = "critical" # Emergencia legal

@dataclass
class QueryClassification:
    """Resultado de la clasificación de consulta"""
    query_type: QueryType
    urgency_level: UrgencyLevel
    confidence: float
    legal_domains: List[str]
    key_entities: List[str]
    emotional_indicators: List[str]
    specific_articles: List[str]
    reasoning: str
    recommended_specialist: str

class LlamaQueryClassifier:
    """
    Clasificador de consultas usando Llama para razonamiento avanzado
    """
    
    def __init__(self, llama_model_path: Optional[str] = None):
        self.llama_model_path = llama_model_path
        self.classification_prompt_template = self._build_classification_prompt()
        self.legal_domain_patterns = self._build_legal_domain_patterns()
        
    def _build_classification_prompt(self) -> str:
        """Construye el prompt para Llama para clasificación de consultas"""
        return """Eres un experto jurista argentino especializado en clasificar consultas legales. Tu tarea es analizar la consulta del usuario y clasificarla según estos criterios:

**TIPOS DE CONSULTA:**
1. **ARTICLE_LOOKUP**: Búsqueda específica de artículos o normas
   - Ejemplos: "¿Cuál es el artículo 14 del código penal?", "Muéstrame el artículo 75 de la LCT"
   
2. **CASE_ANALYSIS**: Análisis de situación legal específica del usuario
   - Ejemplos: "Fui despedido sin indemnización", "Mi jefe me discrimina por embarazo"
   
3. **GENERAL_CONSULTATION**: Consulta general sobre temas legales
   - Ejemplos: "¿Cuáles son mis derechos laborales?", "¿Cómo funciona el divorcio?"
   
4. **PROCEDURAL_GUIDANCE**: Orientación sobre procedimientos legales
   - Ejemplos: "¿Cómo presento una denuncia?", "¿Qué pasos seguir para reclamar?"
   
5. **COMPARATIVE_ANALYSIS**: Comparación entre leyes o situaciones
   - Ejemplos: "Diferencias entre despido con/sin causa", "Comparar códigos"

**NIVELES DE URGENCIA:**
- **CRITICAL**: Emergencias (violencia, amenazas inmediatas)
- **HIGH**: Situaciones urgentes (despidos recientes, discriminación activa)
- **MEDIUM**: Requiere atención pronta (reclamos pendientes, consultas específicas)
- **LOW**: Consultas académicas o generales

**DOMINIOS LEGALES:**
- Laboral, Civil, Penal, Comercial, Familia, Administrativo, Constitucional

**INSTRUCCIONES:**
1. Analiza cuidadosamente la consulta
2. Identifica el tipo principal de consulta
3. Evalúa el nivel de urgencia
4. Detecta dominios legales relevantes
5. Extrae entidades clave (artículos específicos, leyes, conceptos)
6. Identifica indicadores emocionales
7. Proporciona tu razonamiento

**FORMATO DE RESPUESTA (JSON):**
```json
{
    "query_type": "TIPO_DE_CONSULTA",
    "urgency_level": "NIVEL_URGENCIA", 
    "confidence": 0.95,
    "legal_domains": ["dominio1", "dominio2"],
    "key_entities": ["entidad1", "entidad2"],
    "emotional_indicators": ["indicador1", "indicador2"],
    "specific_articles": ["Art. X Código Y"],
    "reasoning": "Explicación detallada de tu análisis",
    "recommended_specialist": "tipo_especialista_recomendado"
}
```

**CONSULTA A ANALIZAR:**
{query}

**ANÁLISIS:**"""

    def _build_legal_domain_patterns(self) -> Dict[str, List[str]]:
        """Patrones para detectar dominios legales"""
        return {
            "laboral": [
                "trabajo", "empleado", "empleador", "jefe", "empresa", "patrón",
                "despido", "indemnización", "salario", "sueldo", "jornada",
                "vacaciones", "licencia", "art", "obra social", "sindicato",
                "convenio colectivo", "contrato trabajo", "lct"
            ],
            "civil": [
                "contrato", "daños", "perjuicios", "responsabilidad civil",
                "propiedad", "vecino", "terreno", "construcción", "herencia",
                "sucesión", "matrimonio", "divorcio", "código civil"
            ],
            "penal": [
                "delito", "robo", "hurto", "estafa", "amenaza", "lesiones",
                "código penal", "denuncia penal", "fiscalía", "querella"
            ],
            "familia": [
                "divorcio", "separación", "custodia", "alimentos", "régimen visitas",
                "violencia familiar", "adopción", "filiación"
            ],
            "comercial": [
                "sociedad", "empresa", "comercio", "quiebra", "concurso",
                "código comercial", "factura", "cheque"
            ],
            "administrativo": [
                "administración pública", "trámite", "municipio", "provincia",
                "estado", "funcionario público"
            ]
        }
    
    def classify_query_with_llama(self, query: str) -> QueryClassification:
        """
        Clasifica la consulta usando Llama (simulación para demo)
        En implementación real, aquí iría la llamada a Llama
        """
        try:
            # SIMULACIÓN: En producción aquí iría la llamada real a Llama
            llama_response = self._simulate_llama_response(query)
            
            # Parsear respuesta de Llama
            classification_data = self._parse_llama_response(llama_response)
            
            # Crear objeto de clasificación
            return QueryClassification(**classification_data)
            
        except Exception as e:
            print(f"Error en clasificación con Llama: {str(e)}")
            # Fallback a clasificación basada en reglas
            return self._fallback_rule_based_classification(query)
    
    def _simulate_llama_response(self, query: str) -> str:
        """
        Simulación de respuesta de Llama para demostración
        En producción, aquí iría la llamada real al modelo
        """
        query_lower = query.lower()
        
        # Detectar tipo de consulta
        if re.search(r'artículo?\s+\d+|art\.?\s*\d+', query_lower):
            query_type = "ARTICLE_LOOKUP"
            urgency = "LOW"
            specialist = "article_specialist"
        elif any(indicator in query_lower for indicator in ['fui', 'me', 'mi jefe', 'despidieron', 'discriminan']):
            query_type = "CASE_ANALYSIS" 
            urgency = "HIGH" if any(urgent in query_lower for urgent in ['despido', 'discrimina', 'embarazo']) else "MEDIUM"
            specialist = "case_analyst"
        elif any(proc in query_lower for proc in ['cómo', 'qué pasos', 'procedimiento', 'denuncia']):
            query_type = "PROCEDURAL_GUIDANCE"
            urgency = "MEDIUM"
            specialist = "procedural_guide"
        else:
            query_type = "GENERAL_CONSULTATION"
            urgency = "LOW"
            specialist = "general_counselor"
        
        # Detectar dominios
        domains = []
        for domain, patterns in self.legal_domain_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                domains.append(domain)
        
        # Simular respuesta JSON de Llama
        response = {
            "query_type": query_type,
            "urgency_level": urgency,
            "confidence": 0.85,
            "legal_domains": domains[:2],  # Top 2 dominios
            "key_entities": self._extract_entities(query),
            "emotional_indicators": self._detect_emotional_indicators(query),
            "specific_articles": self._extract_articles(query),
            "reasoning": f"Consulta clasificada como {query_type} basado en patrones detectados y contexto.",
            "recommended_specialist": specialist
        }
        
        return json.dumps(response, ensure_ascii=False, indent=2)
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extrae entidades clave de la consulta"""
        entities = []
        query_lower = query.lower()
        
        # Entidades laborales
        labor_entities = ['empleado', 'empleador', 'jefe', 'empresa', 'trabajo', 'despido', 'indemnización']
        entities.extend([entity for entity in labor_entities if entity in query_lower])
        
        # Entidades civiles
        civil_entities = ['contrato', 'propiedad', 'daños', 'responsabilidad']
        entities.extend([entity for entity in civil_entities if entity in query_lower])
        
        return list(set(entities))[:5]  # Máximo 5 entidades
    
    def _detect_emotional_indicators(self, query: str) -> List[str]:
        """Detecta indicadores emocionales en la consulta"""
        indicators = []
        query_lower = query.lower()
        
        emotional_patterns = {
            'frustracion': ['no entiendo', 'confundido', 'perdido'],
            'urgencia': ['urgente', 'inmediato', 'ya', 'rápido'],
            'injusticia': ['injusto', 'abuso', 'maltrato'],
            'miedo': ['miedo', 'temor', 'preocupado'],
            'victimizacion': ['me hicieron', 'me obligaron', 'no me dejan']
        }
        
        for emotion, patterns in emotional_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                indicators.append(emotion)
        
        return indicators
    
    def _extract_articles(self, query: str) -> List[str]:
        """Extrae referencias específicas a artículos"""
        articles = []
        
        # Patrones para artículos específicos
        patterns = [
            r'artículo\s+(\d+)',
            r'art\.?\s*(\d+)',
            r'artículo\s+(\d+)\s+del?\s+(.+)',
            r'art\.?\s*(\d+)\s+(.+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                if len(match.groups()) == 1:
                    articles.append(f"Art. {match.group(1)}")
                else:
                    articles.append(f"Art. {match.group(1)} {match.group(2)}")
        
        return articles
    
    def _parse_llama_response(self, llama_response: str) -> Dict[str, Any]:
        """Parsea la respuesta JSON de Llama"""
        try:
            # Extraer JSON de la respuesta
            json_start = llama_response.find('{')
            json_end = llama_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No se encontró JSON válido en la respuesta")
            
            json_str = llama_response[json_start:json_end]
            response_data = json.loads(json_str)
            
            # Convertir enums
            response_data['query_type'] = QueryType(response_data['query_type'].lower())
            response_data['urgency_level'] = UrgencyLevel(response_data['urgency_level'].lower())
            
            return response_data
            
        except Exception as e:
            print(f"Error parseando respuesta de Llama: {str(e)}")
            raise
    
    def _fallback_rule_based_classification(self, query: str) -> QueryClassification:
        """Clasificación de fallback basada en reglas cuando Llama falla"""
        query_lower = query.lower()
        
        # Determinar tipo de consulta
        if re.search(r'artículo?\s+\d+|art\.?\s*\d+', query_lower):
            query_type = QueryType.ARTICLE_LOOKUP
            urgency = UrgencyLevel.LOW
        elif any(indicator in query_lower for indicator in ['fui', 'me', 'mi jefe', 'despidieron']):
            query_type = QueryType.CASE_ANALYSIS
            urgency = UrgencyLevel.HIGH
        else:
            query_type = QueryType.GENERAL_CONSULTATION
            urgency = UrgencyLevel.MEDIUM
        
        return QueryClassification(
            query_type=query_type,
            urgency_level=urgency,
            confidence=0.6,  # Menor confianza para fallback
            legal_domains=["laboral"],  # Dominio por defecto
            key_entities=self._extract_entities(query),
            emotional_indicators=self._detect_emotional_indicators(query),
            specific_articles=self._extract_articles(query),
            reasoning="Clasificación basada en reglas de fallback",
            recommended_specialist="general_counselor"
        )

class SpecialistRouter:
    """
    Router que dirige las consultas al especialista apropiado
    """
    
    def __init__(self):
        self.specialists = {
            "article_specialist": ArticleSpecialist(),
            "case_analyst": CaseAnalyst(),
            "procedural_guide": ProceduralGuide(),
            "general_counselor": GeneralCounselor()
        }
    
    def route_query(self, classification: QueryClassification, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enruta la consulta al especialista apropiado"""
        specialist_name = classification.recommended_specialist
        specialist = self.specialists.get(specialist_name)
        
        if not specialist:
            specialist = self.specialists["general_counselor"]
        
        return specialist.handle_query(query, classification, context)

class LegalSpecialist(ABC):
    """Clase base para especialistas legales"""
    
    @abstractmethod
    def handle_query(self, query: str, classification: QueryClassification, context: Dict[str, Any]) -> Dict[str, Any]:
        pass

class ArticleSpecialist(LegalSpecialist):
    """Especialista en búsqueda específica de artículos"""
    
    def handle_query(self, query: str, classification: QueryClassification, context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"🔍 ArticleSpecialist manejando: {query}")
        
        # Extraer artículos específicos solicitados
        requested_articles = classification.specific_articles
        
        if not requested_articles:
            # Intentar extraer artículos de la consulta
            requested_articles = self._extract_article_requests(query)
        
        # Configurar búsqueda específica
        search_config = {
            "search_type": "exact_article_lookup",
            "target_articles": requested_articles,
            "include_related": True,
            "max_results": 5
        }
        
        return {
            "specialist_type": "article_specialist",
            "search_strategy": "exact_lookup",
            "search_config": search_config,
            "response_format": "structured_articles",
            "additional_context": {
                "explanation_level": "detailed",
                "include_examples": True,
                "cite_sources": True
            }
        }
    
    def _extract_article_requests(self, query: str) -> List[str]:
        """Extrae solicitudes específicas de artículos"""
        articles = []
        patterns = [
            r'artículo\s+(\d+)(?:\s+del?\s+(.+?))?',
            r'art\.?\s*(\d+)(?:\s+(.+?))?'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                article_num = match.group(1)
                law_context = match.group(2) if len(match.groups()) > 1 else None
                
                if law_context:
                    articles.append(f"Art. {article_num} {law_context.strip()}")
                else:
                    articles.append(f"Art. {article_num}")
        
        return articles

class CaseAnalyst(LegalSpecialist):
    """Especialista en análisis de casos legales específicos"""
    
    def handle_query(self, query: str, classification: QueryClassification, context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"⚖️ CaseAnalyst manejando: {query}")
        
        # Análisis profundo del caso
        case_analysis = self._analyze_case_elements(query, classification)
        
        # Configurar búsqueda contextual avanzada
        search_config = {
            "search_type": "contextual_case_analysis",
            "case_elements": case_analysis,
            "urgency_boost": classification.urgency_level.value,
            "domain_focus": classification.legal_domains,
            "max_results": 15
        }
        
        return {
            "specialist_type": "case_analyst",
            "search_strategy": "graph_rag_enhanced",
            "search_config": search_config,
            "response_format": "legal_advice",
            "case_analysis": case_analysis,
            "additional_context": {
                "provide_actionable_advice": True,
                "include_precedents": True,
                "urgency_level": classification.urgency_level.value,
                "emotional_support": len(classification.emotional_indicators) > 0
            }
        }
    
    def _analyze_case_elements(self, query: str, classification: QueryClassification) -> Dict[str, Any]:
        """Analiza los elementos del caso legal"""
        return {
            "stakeholders": self._identify_stakeholders(query),
            "legal_issues": self._identify_legal_issues(query),
            "timeline_indicators": self._extract_timeline(query),
            "damages_claimed": self._identify_damages(query),
            "emotional_context": classification.emotional_indicators,
            "urgency_factors": self._assess_urgency_factors(query)
        }
    
    def _identify_stakeholders(self, query: str) -> List[str]:
        """Identifica las partes involucradas"""
        stakeholders = []
        query_lower = query.lower()
        
        stakeholder_patterns = {
            "empleado": ["yo", "me", "mi", "trabajador", "empleado"],
            "empleador": ["jefe", "empresa", "empleador", "patrón", "supervisor"],
            "familia": ["esposo", "esposa", "hijo", "padre", "madre"],
            "terceros": ["vecino", "cliente", "proveedor"]
        }
        
        for role, patterns in stakeholder_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                stakeholders.append(role)
        
        return stakeholders
    
    def _identify_legal_issues(self, query: str) -> List[str]:
        """Identifica los problemas legales principales"""
        issues = []
        query_lower = query.lower()
        
        issue_patterns = {
            "despido_improcedente": ["despido", "echaron", "terminaron contrato"],
            "discriminacion": ["discrimina", "trato diferencial", "embarazo"],
            "falta_pago": ["no pagan", "no pagaron", "adeudan"],
            "acoso_laboral": ["acoso", "maltrato", "humillación"],
            "incumplimiento_contrato": ["no cumplen", "violaron contrato"]
        }
        
        for issue, patterns in issue_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                issues.append(issue)
        
        return issues
    
    def _extract_timeline(self, query: str) -> List[str]:
        """Extrae indicadores temporales"""
        timeline = []
        
        time_patterns = [
            r'(\d+)\s*años?',
            r'(\d+)\s*meses?',
            r'(\d+)\s*días?',
            r'hace\s+(\d+)',
            r'durante\s+(\d+)',
            r'después\s+de\s+(\d+)'
        ]
        
        for pattern in time_patterns:
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                timeline.append(match.group(0))
        
        return timeline
    
    def _identify_damages(self, query: str) -> List[str]:
        """Identifica daños o perjuicios reclamados"""
        damages = []
        query_lower = query.lower()
        
        damage_patterns = [
            "indemnización", "compensación", "pago", "dinero",
            "daños", "perjuicios", "pérdidas"
        ]
        
        for pattern in damage_patterns:
            if pattern in query_lower:
                damages.append(pattern)
        
        return damages
    
    def _assess_urgency_factors(self, query: str) -> List[str]:
        """Evalúa factores de urgencia"""
        urgency_factors = []
        query_lower = query.lower()
        
        urgency_patterns = {
            "temporal": ["urgente", "inmediato", "ya", "ahora"],
            "economico": ["sin dinero", "no tengo", "necesito"],
            "legal": ["plazo", "vence", "término"],
            "personal": ["embarazada", "enfermo", "familia"]
        }
        
        for factor_type, patterns in urgency_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                urgency_factors.append(factor_type)
        
        return urgency_factors

class ProceduralGuide(LegalSpecialist):
    """Especialista en orientación procedimental"""
    
    def handle_query(self, query: str, classification: QueryClassification, context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"📋 ProceduralGuide manejando: {query}")
        
        # Identificar tipo de procedimiento solicitado
        procedure_type = self._identify_procedure_type(query)
        
        search_config = {
            "search_type": "procedural_guidance",
            "procedure_focus": procedure_type,
            "step_by_step": True,
            "include_requirements": True,
            "max_results": 10
        }
        
        return {
            "specialist_type": "procedural_guide",
            "search_strategy": "procedural_chains",
            "search_config": search_config,
            "response_format": "step_by_step_guide",
            "additional_context": {
                "include_forms": True,
                "include_timeframes": True,
                "include_costs": True,
                "provide_alternatives": True
            }
        }
    
    def _identify_procedure_type(self, query: str) -> str:
        """Identifica el tipo de procedimiento solicitado"""
        query_lower = query.lower()
        
        procedures = {
            "denuncia_laboral": ["denuncia", "inspección trabajo", "ministerio trabajo"],
            "demanda_civil": ["demanda", "tribunal", "juicio"],
            "reclamo_administrativo": ["reclamo", "carta documento", "intimación"],
            "consulta_gratuita": ["asesoramiento", "consulta gratuita", "abogado"]
        }
        
        for procedure, patterns in procedures.items():
            if any(pattern in query_lower for pattern in patterns):
                return procedure
        
        return "procedimiento_general"

class GeneralCounselor(LegalSpecialist):
    """Consejero general para consultas no específicas"""
    
    def handle_query(self, query: str, classification: QueryClassification, context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"🎓 GeneralCounselor manejando: {query}")
        
        search_config = {
            "search_type": "general_consultation",
            "broad_search": True,
            "educational_focus": True,
            "max_results": 12
        }
        
        return {
            "specialist_type": "general_counselor",
            "search_strategy": "balanced_comprehensive",
            "search_config": search_config,
            "response_format": "educational_overview",
            "additional_context": {
                "provide_examples": True,
                "explain_concepts": True,
                "suggest_next_steps": True
            }
        }

class IntelligentLegalSystem:
    """
    Sistema principal que integra clasificación y routing
    """
    
    def __init__(self, llama_config: Optional[Dict[str, Any]] = None):
        # If llama_config is provided, extract the model path, otherwise use None
        llama_model_path = None
        if llama_config and isinstance(llama_config, dict):
            # Extract model path from config if available
            llama_model_path = llama_config.get('local_model')
            
        self.classifier = LlamaQueryClassifier(llama_model_path)
        self.router = SpecialistRouter()
        self.llama_config = llama_config
    
    def process_query_with_real_llama(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Método para clasificar una consulta usando Llama pero sin procesar la respuesta completa"""
        if context is None:
            context = {}
        
        print(f"\n🦙 SISTEMA INTELIGENTE - Clasificando solo: '{query}'\n")
        
        # Clasificar consulta con Llama
        start_time = time.time()
        try:
            classification = self.classifier.classify_query_with_llama(query)
            classification_time = time.time() - start_time
            print(f"✅ Clasificación completada en {classification_time:.2f}s")
            
            # Convertir a un diccionario con la estructura esperada
            classification_dict = {
                "query_type": classification.query_type.value,
                "urgency_level": classification.urgency_level.value,
                "confidence": classification.confidence,
                "legal_domains": classification.legal_domains,
                "key_entities": classification.key_entities,
                "emotional_indicators": classification.emotional_indicators,
                "specific_articles": classification.specific_articles,
                "reasoning": classification.reasoning,
                "recommended_specialist": classification.recommended_specialist,
                "classification_method": "llama"
            }
            
            return {
                "classification": classification_dict,
                "specialist_routing": {
                    "specialist_type": classification.recommended_specialist,
                    "routing_reason": classification.reasoning
                },
                "llama_available": True
            }
        except Exception as e:
            print(f"❌ Error en clasificación: {str(e)}")
            # Retornar una clasificación simple por defecto con la estructura correcta
            default_classification = {
                "query_type": "general_consultation",
                "urgency_level": "medium",
                "confidence": 0.5,
                "legal_domains": ["general"],
                "key_entities": [],
                "emotional_indicators": [],
                "specific_articles": [],
                "reasoning": "Clasificación por defecto debido a error.",
                "recommended_specialist": "general_counselor",
                "classification_method": "fallback"
            }
            
            return {
                "classification": default_classification,
                "specialist_routing": {
                    "specialist_type": "general_counselor",
                    "routing_reason": "Fallback debido a error en clasificación."
                },
                "llama_available": False
            }
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Procesa una consulta completa desde clasificación hasta respuesta"""
        if context is None:
            context = {}
        
        print(f"\n🧠 SISTEMA INTELIGENTE - Procesando: '{query}'\n")
        
        # 1. Clasificar consulta con Llama
        start_time = time.time()
        classification = self.classifier.classify_query_with_llama(query)
        classification_time = time.time() - start_time
        
        print(f"📊 CLASIFICACIÓN COMPLETADA ({classification_time:.2f}s):")
        print(f"   🏷️  Tipo: {classification.query_type.value}")
        print(f"   🚨 Urgencia: {classification.urgency_level.value}")
        print(f"   📈 Confianza: {classification.confidence:.2f}")
        print(f"   ⚖️  Dominios: {', '.join(classification.legal_domains)}")
        print(f"   🎯 Especialista: {classification.recommended_specialist}")
        
        if classification.emotional_indicators:
            print(f"   💭 Emocional: {', '.join(classification.emotional_indicators)}")
        
        # 2. Enrutar a especialista
        specialist_response = self.router.route_query(classification, query, context)
        
        # 3. Preparar respuesta completa
        complete_response = {
            "classification": {
                "query_type": classification.query_type.value,
                "urgency_level": classification.urgency_level.value,
                "confidence": classification.confidence,
                "legal_domains": classification.legal_domains,
                "reasoning": classification.reasoning
            },
            "specialist_routing": specialist_response,
            "processing_time": classification_time,
            "timestamp": time.time()
        }
        
        print(f"\n🎯 ROUTING COMPLETADO:")
        print(f"   🤖 Especialista: {specialist_response['specialist_type']}")
        print(f"   🔍 Estrategia: {specialist_response['search_strategy']}")
        print(f"   📋 Formato: {specialist_response['response_format']}")
        
        return complete_response

# ========== EJEMPLO DE USO ==========

def demo_intelligent_system():
    """Demostración del sistema inteligente"""
    system = IntelligentLegalSystem()
    
    # Casos de prueba
    test_queries = [
        "¿Cuál es el artículo 14 del código penal?",
        "Fui despedido luego de trabajar durante 5 años en relación de dependencia, sin anticipación o previo aviso y sin indemnización luego de avisar que estoy embarazada",
        "¿Cómo presento una denuncia por acoso laboral?",
        "¿Cuáles son mis derechos como trabajador?",
        "Mi vecino construyó una pared en mi terreno sin permiso",
        "¿Qué diferencias hay entre despido con causa y sin causa?"
    ]
    
    print("="*80)
    print("🚀 DEMOSTRACIÓN DEL SISTEMA INTELIGENTE DE CLASIFICACIÓN")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"CONSULTA {i}")
        print(f"{'='*60}")
        
        result = system.process_query(query)
        
        print(f"\n✅ RESULTADO COMPLETO:")
        print(f"   Tipo detectado: {result['classification']['query_type']}")
        print(f"   Especialista asignado: {result['specialist_routing']['specialist_type']}")
        print(f"   Estrategia de búsqueda: {result['specialist_routing']['search_strategy']}")
        
        # Mostrar configuración específica del especialista
        if 'case_analysis' in result['specialist_routing']:
            case_info = result['specialist_routing']['case_analysis']
            print(f"   📋 Elementos del caso: {case_info.get('legal_issues', [])}")
            print(f"   👥 Partes: {case_info.get('stakeholders', [])}")
        
        print(f"   ⏱️  Tiempo de procesamiento: {result['processing_time']:.3f}s")

if __name__ == "__main__":
    demo_intelligent_system()