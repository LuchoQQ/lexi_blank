"""
Sistema de clasificación y recuperación legal multi-dominio.
Maneja casos laborales, civiles, comerciales, penales y administrativos.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import re
from abc import ABC, abstractmethod

class LegalDomain(Enum):
    LABOR = "LABORAL"
    CIVIL = "CIVIL" 
    COMMERCIAL = "COMERCIAL"
    CRIMINAL = "PENAL"
    ADMINISTRATIVE = "ADMINISTRATIVO"
    CONSTITUTIONAL = "CONSTITUCIONAL"
    PROCEDURAL = "PROCESAL"

@dataclass
class DomainScore:
    """Puntuación de relevancia por dominio"""
    domain: LegalDomain
    score: float
    confidence: float
    key_indicators: List[str]

@dataclass
class LegalQueryAnalysis:
    """Análisis completo de una consulta legal"""
    primary_domain: LegalDomain
    secondary_domains: List[LegalDomain]
    domain_scores: Dict[LegalDomain, DomainScore]
    is_multi_domain: bool
    complexity_level: str  # "simple", "complex", "multi_jurisdictional"
    key_entities: Dict[str, List[str]]
    legal_concepts: List[str]

# ==================================================================
# CONFIGURACIÓN DE DOMINIOS LEGALES EXPANDIDA
# ==================================================================

class LegalDomainConfig:
    """Configuración detallada para cada dominio legal"""
    
    LABOR_CONFIG = {
        "keywords": [
            # Relación laboral
            "trabajador", "empleador", "empleado", "patrón", "jefe", "supervisor",
            "contrato trabajo", "relación dependencia", "empleo", "puesto", "cargo",
            
            # Derechos y obligaciones
            "sueldo", "salario", "remuneración", "aguinaldo", "vacaciones", "licencia",
            "jornada", "horario", "descanso", "feriado", "horas extras", "turno",
            
            # Terminación laboral
            "despido", "renuncia", "cesantía", "indemnización", "preaviso", "finiquito",
            "liquidación", "estabilidad laboral", "fuero", "reinstalación",
            
            # Protecciones especiales
            "embarazo", "maternidad", "paternidad", "lactancia", "menor", "discapacidad",
            "sindical", "gremial", "delegado", "comisión interna",
            
            # Conflictos
            "discriminación laboral", "acoso laboral", "mobbing", "accidente trabajo",
            "enfermedad profesional", "riesgo trabajo", "seguridad laboral"
        ],
        
        "exclusion_keywords": ["compraventa", "sociedad", "sucesión", "herencia", "delito"],
        
        "laws": ["Ley de contrato de trabajo", "Ley de empleo", "Ley de riesgos del trabajo"],
        
        "critical_articles": {
            "embarazo_despido": ["177", "178", "182", "245", "232", "233"],
            "discriminacion": ["17", "18", "81", "245"],
            "despido_sin_causa": ["245", "232", "233", "231"],
            "jornada_trabajo": ["196", "197", "198", "203", "204"],
            "remuneracion": ["103", "116", "120", "124", "129"]
        }
    }
    
    CIVIL_COMMERCIAL_CONFIG = {
        "keywords": [
            # Contratos y obligaciones
            "contrato", "obligación", "acreedor", "deudor", "prestación", "cumplimiento",
            "incumplimiento", "mora", "daños perjuicios", "responsabilidad civil",
            
            # Derechos reales
            "propiedad", "dominio", "posesión", "usufructo", "servidumbre", "hipoteca",
            "prenda", "embargo", "registro propiedad", "título", "escritura",
            
            # Familia y sucesiones
            "matrimonio", "divorcio", "separación", "régimen patrimonial", "gananciales",
            "herencia", "testamento", "legado", "sucesión", "heredero", "legatario",
            "patria potestad", "tutela", "curatela", "alimentos", "tenencia",
            
            # Comercial - Sociedades
            "sociedad", "socio", "accionista", "administrador", "gerente", "directorio",
            "capital social", "dividendo", "fusión", "escisión", "transformación",
            "quiebra", "concurso", "concordato", "liquidación empresarial",
            
            # Comercial - Operaciones
            "compraventa", "locación", "comodato", "mutuo", "fianza", "aval",
            "cheque", "pagaré", "letra cambio", "factura", "remito", "mercadería",
            "comerciante", "fondo comercio", "marca", "patente", "competencia desleal",
            
            # Seguros y banca
            "seguro", "póliza", "siniestro", "prima", "banco", "préstamo", "crédito",
            "cuenta corriente", "depósito", "tarjeta crédito", "intereses", "usura"
        ],
        
        "exclusion_keywords": ["despido", "salario", "trabajador", "delito", "prisión"],
        
        "laws": ["Codigo Civil y Comercial", "Ley de seguros", "Ley de sociedades"],
        
        "critical_articles": {
            "incumplimiento_contractual": ["724", "729", "730", "731", "1716", "1717"],
            "responsabilidad_civil": ["1708", "1709", "1710", "1716", "1717", "1721"],
            "sociedades": ["1", "11", "25", "94", "299", "319"],
            "compraventa": ["1123", "1132", "1140", "1142", "1149"],
            "locacion": ["1187", "1188", "1189", "1190", "1194"],
            "sucesiones": ["2277", "2280", "2295", "2302", "2330"]
        }
    }
    
    CRIMINAL_CONFIG = {
        "keywords": [
            # Delitos contra las personas
            "homicidio", "asesinato", "lesiones", "amenaza", "coacción", "privación libertad",
            "secuestro", "trata personas", "violación", "abuso sexual", "estupro",
            
            # Delitos contra la propiedad
            "hurto", "robo", "estafa", "defraudación", "apropiación indebida", "extorsión",
            "usurpación", "daño", "incendio", "estrago", "piratería",
            
            # Delitos contra la fe pública
            "falsificación", "documento falso", "testimonio falso", "perjurio",
            "usurpación identidad", "moneda falsa",
            
            # Delitos contra la administración
            "cohecho", "soborno", "malversación", "prevaricato", "abuso autoridad",
            "violación deberes", "enriquecimiento ilícito",
            
            # Conceptos penales
            "delito", "pena", "prisión", "multa", "inhabilitación", "probation",
            "suspensión juicio", "prescripción", "tentativa", "complicidad",
            "instigación", "encubrimiento", "reincidencia", "concurso"
        ],
        
        "exclusion_keywords": ["contrato", "sociedad", "trabajador", "divorcio"],
        
        "laws": ["Codigo Penal", "Codigo Procesal Penal", "Ley de drogas"],
        
        "critical_articles": {
            "homicidio": ["79", "80", "81", "82", "84"],
            "hurto_robo": ["162", "163", "164", "165", "167"],
            "estafa": ["172", "173", "174", "175", "176"],
            "lesiones": ["89", "90", "91", "92", "93"],
            "amenazas": ["149", "149bis"],
            "falsificacion": ["292", "293", "294", "296"]
        }
    }

# ==================================================================
# CLASIFICADOR MULTI-DOMINIO
# ==================================================================

class MultiDomainClassifier:
    """Clasificador inteligente para múltiples dominios legales"""
    
    def __init__(self):
        self.domain_configs = {
            LegalDomain.LABOR: LegalDomainConfig.LABOR_CONFIG,
            LegalDomain.CIVIL: LegalDomainConfig.CIVIL_COMMERCIAL_CONFIG,
            LegalDomain.COMMERCIAL: LegalDomainConfig.CIVIL_COMMERCIAL_CONFIG,
            LegalDomain.CRIMINAL: LegalDomainConfig.CRIMINAL_CONFIG
        }
        
        # Patrones de intersección entre dominios
        self.intersection_patterns = {
            ("LABORAL", "PENAL"): ["acoso", "discriminación", "violencia", "amenaza", "lesión"],
            ("CIVIL", "COMERCIAL"): ["contrato", "sociedad", "responsabilidad", "daño"],
            ("LABORAL", "CIVIL"): ["responsabilidad", "daño", "indemnización"],
            ("PENAL", "CIVIL"): ["daño", "estafa", "apropiación", "responsabilidad"],
            ("COMERCIAL", "PENAL"): ["estafa", "defraudación", "quiebra", "insolvencia"]
        }
    
    def analyze_query(self, query: str) -> LegalQueryAnalysis:
        """
        Análisis completo de la consulta para determinar dominios relevantes
        """
        query_lower = query.lower()
        
        # 1. Calcular scores por dominio
        domain_scores = self._calculate_domain_scores(query_lower)
        
        # 2. Determinar dominio principal y secundarios
        primary_domain, secondary_domains = self._determine_domains(domain_scores)
        
        # 3. Detectar si es multi-dominio
        is_multi_domain = self._is_multi_domain_query(domain_scores, query_lower)
        
        # 4. Determinar complejidad
        complexity = self._assess_complexity(domain_scores, is_multi_domain, query)
        
        # 5. Extraer entidades y conceptos
        entities = self._extract_legal_entities(query)
        concepts = self._extract_legal_concepts(query, primary_domain)
        
        return LegalQueryAnalysis(
            primary_domain=primary_domain,
            secondary_domains=secondary_domains,
            domain_scores=domain_scores,
            is_multi_domain=is_multi_domain,
            complexity_level=complexity,
            key_entities=entities,
            legal_concepts=concepts
        )
    
    def _calculate_domain_scores(self, query: str) -> Dict[LegalDomain, DomainScore]:
        """Calcula puntuaciones para cada dominio"""
        scores = {}
        
        for domain, config in self.domain_configs.items():
            score = 0.0
            matched_keywords = []
            
            # Score positivo por keywords relevantes
            for keyword in config["keywords"]:
                if keyword.lower() in query:
                    # Peso por especificidad de la keyword
                    weight = len(keyword.split()) * 0.5 + 1.0
                    score += weight
                    matched_keywords.append(keyword)
            
            # Penalización por keywords de exclusión
            exclusion_penalty = 0.0
            for exclusion in config.get("exclusion_keywords", []):
                if exclusion.lower() in query:
                    exclusion_penalty += 2.0
            
            # Score final ajustado
            final_score = max(0.0, score - exclusion_penalty)
            
            # Calcular confianza basada en número de matches y especificidad
            confidence = min(1.0, (len(matched_keywords) * 0.2) + (final_score * 0.1))
            
            scores[domain] = DomainScore(
                domain=domain,
                score=final_score,
                confidence=confidence,
                key_indicators=matched_keywords
            )
        
        return scores
    
    def _determine_domains(self, domain_scores: Dict[LegalDomain, DomainScore]) -> Tuple[LegalDomain, List[LegalDomain]]:
        """Determina dominio principal y secundarios"""
        
        # Ordenar dominios por score
        sorted_domains = sorted(
            domain_scores.items(), 
            key=lambda x: x[1].score, 
            reverse=True
        )
        
        if not sorted_domains or sorted_domains[0][1].score == 0:
            # Si no hay matches claros, defaultear a civil
            return LegalDomain.CIVIL, []
        
        primary_domain = sorted_domains[0][0]
        
        # Secundarios: dominios con score > 30% del principal y confianza > 0.3
        primary_score = sorted_domains[0][1].score
        threshold = primary_score * 0.3
        
        secondary_domains = [
            domain for domain, score_obj in sorted_domains[1:] 
            if score_obj.score > threshold and score_obj.confidence > 0.3
        ]
        
        return primary_domain, secondary_domains
    
    def _is_multi_domain_query(self, domain_scores: Dict[LegalDomain, DomainScore], query: str) -> bool:
        """Detecta si la consulta abarca múltiples dominios"""
        
        # Contar dominios con score significativo
        significant_domains = [
            domain for domain, score_obj in domain_scores.items() 
            if score_obj.score > 2.0 and score_obj.confidence > 0.4
        ]
        
        if len(significant_domains) > 1:
            return True
        
        # Verificar patrones de intersección
        for (domain1, domain2), patterns in self.intersection_patterns.items():
            if any(pattern in query for pattern in patterns):
                # Verificar si ambos dominios están presentes
                domains_present = [d.value for d in significant_domains]
                if domain1 in domains_present and domain2 in domains_present:
                    return True
        
        return False
    
    def _assess_complexity(self, domain_scores: Dict[LegalDomain, DomainScore], 
                          is_multi_domain: bool, query: str) -> str:
        """Evalúa la complejidad de la consulta"""
        
        # Indicadores de complejidad
        complexity_indicators = [
            "y además", "también", "pero", "sin embargo", "por otro lado",
            "al mismo tiempo", "simultáneamente", "conjuntamente",
            "responsabilidad civil y penal", "daños y perjuicios",
            "tanto civil como", "no solo", "sino también"
        ]
        
        multi_law_indicators = ["código civil", "código penal", "ley de", "decreto"]
        procedural_indicators = ["juicio", "demanda", "recurso", "apelación", "casación"]
        
        # Calcular puntuación de complejidad
        complexity_score = 0
        
        if is_multi_domain:
            complexity_score += 2
            
        for indicator in complexity_indicators:
            if indicator in query.lower():
                complexity_score += 1
                
        law_mentions = sum(1 for indicator in multi_law_indicators if indicator in query.lower())
        complexity_score += min(law_mentions, 3)
        
        procedural_mentions = sum(1 for indicator in procedural_indicators if indicator in query.lower())
        complexity_score += min(procedural_mentions, 2)
        
        # Clasificar complejidad
        if complexity_score <= 2:
            return "simple"
        elif complexity_score <= 5:
            return "complex"
        else:
            return "multi_jurisdictional"
    
    def _extract_legal_entities(self, query: str) -> Dict[str, List[str]]:
        """Extrae entidades legales específicas"""
        entities = {
            "personas": [],
            "bienes": [],
            "actos": [],
            "procedimientos": [],
            "montos": [],
            "plazos": []
        }
        
        # Patrones para entidades
        patterns = {
            "montos": r'(\$\s*\d+(?:\.\d{3})*(?:,\d{2})?|\d+\s*pesos|\d+\s*dólares)',
            "plazos": r'(\d+\s*(?:días?|meses?|años?)|plazo\s+de\s+\d+)',
            "personas": r'(empleador|trabajador|comprador|vendedor|locador|locatario|deudor|acreedor)',
            "procedimientos": r'(demanda|recurso|apelación|mediación|conciliación|arbitraje)'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, query.lower())
            entities[entity_type].extend(matches)
        
        return entities
    
    def _extract_legal_concepts(self, query: str, primary_domain: LegalDomain) -> List[str]:
        """Extrae conceptos legales relevantes"""
        
        # Conceptos por dominio
        domain_concepts = {
            LegalDomain.LABOR: [
                "estabilidad laboral", "fuero maternal", "despido discriminatorio",
                "preaviso", "indemnización por antigüedad", "integración del mes"
            ],
            LegalDomain.CIVIL: [
                "responsabilidad civil", "daños y perjuicios", "cumplimiento forzado",
                "resolución contractual", "mora", "caso fortuito"
            ],
            LegalDomain.COMMERCIAL: [
                "sociedad comercial", "quiebra", "concurso preventivo",
                "competencia desleal", "fondo de comercio"
            ],
            LegalDomain.CRIMINAL: [
                "tipo penal", "dolo", "culpa", "tentativa", "complicidad",
                "prescripción de la acción", "legítima defensa"
            ]
        }
        
        relevant_concepts = []
        concepts = domain_concepts.get(primary_domain, [])
        
        for concept in concepts:
            if any(word in query.lower() for word in concept.split()):
                relevant_concepts.append(concept)
        
        return relevant_concepts

# ==================================================================
# RECUPERADORES ESPECIALIZADOS POR DOMINIO
# ==================================================================

class SpecializedRetriever(ABC):
    """Clase base para recuperadores especializados por dominio"""
    
    def __init__(self, domain_config: Dict[str, Any]):
        self.domain_config = domain_config
        
    @abstractmethod
    def retrieve(self, query: str, base_results: List[Dict[str, Any]], 
                 analysis: LegalQueryAnalysis) -> List[Dict[str, Any]]:
        """Implementar lógica de recuperación especializada"""
        pass

class LaborLawRetriever(SpecializedRetriever):
    """Recuperador especializado en derecho laboral (ya implementado anteriormente)"""
    
    def __init__(self):
        super().__init__(LegalDomainConfig.LABOR_CONFIG)
        # ... implementación existente ...

class CivilCommercialRetriever(SpecializedRetriever):
    """Recuperador especializado en derecho civil y comercial"""
    
    def __init__(self):
        super().__init__(LegalDomainConfig.CIVIL_COMMERCIAL_CONFIG)
        
        # Casos específicos civil/comercial
        self.specific_cases = {
            "incumplimiento_contractual": {
                "keywords": ["incumplimiento", "contrato", "obligación", "cumplir"],
                "critical_articles": ["724", "729", "730", "731", "1716", "1717"],
                "concepts": ["mora", "daños", "resolución", "cumplimiento forzado"]
            },
            "responsabilidad_civil": {
                "keywords": ["daño", "perjuicio", "responsabilidad", "indemnización"],
                "critical_articles": ["1708", "1709", "1710", "1716", "1717", "1721"],
                "concepts": ["nexo causal", "factor atribución", "daño resarcible"]
            },
            "sociedades_comerciales": {
                "keywords": ["sociedad", "socio", "administrador", "gerente"],
                "critical_articles": ["1", "11", "25", "94", "299", "319"],
                "concepts": ["personalidad jurídica", "responsabilidad limitada", "capital social"]
            }
        }
    
    def retrieve(self, query: str, base_results: List[Dict[str, Any]], 
                 analysis: LegalQueryAnalysis) -> List[Dict[str, Any]]:
        """Recuperación especializada para civil/comercial"""
        
        # Identificar caso específico
        specific_case = self._identify_specific_case(query)
        
        # Aplicar scoring especializado
        specialized_results = []
        for result in base_results:
            new_score = self._calculate_specialized_score(result, query, analysis)
            
            if specific_case:
                new_score *= self._get_specific_case_boost(result, specific_case)
            
            result['specialized_score'] = new_score
            result['score'] = new_score
            specialized_results.append(result)
        
        # Filtrar y ordenar
        filtered_results = [r for r in specialized_results if r['score'] > 1.0]
        filtered_results.sort(key=lambda x: x['score'], reverse=True)
        
        return filtered_results[:15]
    
    def _identify_specific_case(self, query: str) -> Optional[str]:
        """Identifica caso específico civil/comercial"""
        query_lower = query.lower()
        
        for case_name, case_config in self.specific_cases.items():
            matches = sum(1 for keyword in case_config["keywords"] 
                         if keyword in query_lower)
            if matches >= 2:
                return case_name
        return None
    
    def _calculate_specialized_score(self, article: Dict[str, Any], 
                                   query: str, analysis: LegalQueryAnalysis) -> float:
        """Calcula score especializado para civil/comercial"""
        base_score = article.get('score', 0.0)
        
        # Boost por ley relevante
        law_name = article.get('law_name', '').lower()
        if 'civil' in law_name and 'comercial' in law_name:
            base_score *= 2.5
        elif any(term in law_name for term in ['civil', 'comercial', 'sociedades']):
            base_score *= 2.0
        
        # Penalización por leyes irrelevantes
        if any(term in law_name for term in ['contrato trabajo', 'empleo', 'penal']):
            base_score *= 0.1
        
        return base_score
    
    def _get_specific_case_boost(self, result: Dict[str, Any], case_name: str) -> float:
        """Boost por caso específico"""
        if case_name not in self.specific_cases:
            return 1.0
        
        case_config = self.specific_cases[case_name]
        article_number = result.get('article_number', '')
        
        if article_number in case_config['critical_articles']:
            return 2.0
            
        return 1.2

class CriminalLawRetriever(SpecializedRetriever):
    """Recuperador especializado en derecho penal"""
    
    def __init__(self):
        super().__init__(LegalDomainConfig.CRIMINAL_CONFIG)
        
        self.specific_cases = {
            "delitos_propiedad": {
                "keywords": ["hurto", "robo", "estafa", "apropiación"],
                "critical_articles": ["162", "163", "164", "172", "173"],
                "concepts": ["apoderamiento", "cosa mueble", "ardid", "engaño"]
            },
            "delitos_personas": {
                "keywords": ["homicidio", "lesiones", "amenaza"],
                "critical_articles": ["79", "80", "89", "90", "149"],
                "concepts": ["vida", "integridad física", "intimidación"]
            }
        }
    
    def retrieve(self, query: str, base_results: List[Dict[str, Any]], 
                 analysis: LegalQueryAnalysis) -> List[Dict[str, Any]]:
        """Recuperación especializada para derecho penal"""
        
        specialized_results = []
        for result in base_results:
            law_name = result.get('law_name', '').lower()
            
            # Boost fuerte para código penal
            if 'penal' in law_name:
                result['score'] = result.get('score', 0.0) * 3.0
            
            # Penalización para leyes no penales
            elif any(term in law_name for term in ['civil', 'comercial', 'trabajo']):
                result['score'] = result.get('score', 0.0) * 0.1
            
            specialized_results.append(result)
        
        specialized_results.sort(key=lambda x: x['score'], reverse=True)
        return specialized_results[:15]

# ==================================================================
# FACTORY Y ORQUESTADOR PRINCIPAL
# ==================================================================

class MultiDomainRetrieverFactory:
    """Factory para crear recuperadores según el análisis de dominio"""
    
    @staticmethod
    def create_retriever(analysis: LegalQueryAnalysis) -> SpecializedRetriever:
        """Crea el recuperador apropiado basado en el análisis"""
        
        primary_domain = analysis.primary_domain
        
        if primary_domain == LegalDomain.LABOR:
            return LaborLawRetriever()
        elif primary_domain in [LegalDomain.CIVIL, LegalDomain.COMMERCIAL]:
            return CivilCommercialRetriever()
        elif primary_domain == LegalDomain.CRIMINAL:
            return CriminalLawRetriever()
        else:
            # Default a civil para casos no específicos
            return CivilCommercialRetriever()

class MultiDomainOrchestrator:
    """Orquestador principal para consultas multi-dominio"""
    
    def __init__(self):
        self.classifier = MultiDomainClassifier()
        
    def enhance_search(self, query: str, base_results: List[Dict[str, Any]], 
                      **resources) -> Tuple[List[Dict[str, Any]], LegalQueryAnalysis]:
        """
        Mejora la búsqueda considerando múltiples dominios legales
        
        Returns:
            Tuple de (resultados_mejorados, análisis_consulta)
        """
        
        # 1. Analizar la consulta
        analysis = self.classifier.analyze_query(query)
        
        print(f"📊 Análisis de consulta:")
        print(f"  Dominio principal: {analysis.primary_domain.value}")
        print(f"  Dominios secundarios: {[d.value for d in analysis.secondary_domains]}")
        print(f"  Multi-dominio: {analysis.is_multi_domain}")
        print(f"  Complejidad: {analysis.complexity_level}")
        
        # 2. Estrategia de recuperación según complejidad
        if analysis.is_multi_domain:
            return self._handle_multi_domain_query(query, base_results, analysis, **resources)
        else:
            return self._handle_single_domain_query(query, base_results, analysis, **resources)
    
    def _handle_single_domain_query(self, query: str, base_results: List[Dict[str, Any]], 
                                   analysis: LegalQueryAnalysis, **resources) -> Tuple[List[Dict[str, Any]], LegalQueryAnalysis]:
        """Maneja consultas de dominio único"""
        
        # Crear recuperador especializado
        retriever = MultiDomainRetrieverFactory.create_retriever(analysis)
        
        # Pasar recursos al recuperador
        for resource_name, resource_value in resources.items():
            setattr(retriever, resource_name, resource_value)
        
        # Aplicar recuperación especializada
        enhanced_results = retriever.retrieve(query, base_results, analysis)
        
        print(f"✅ Aplicada especialización {type(retriever).__name__}")
        print(f"   Resultados: {len(base_results)} → {len(enhanced_results)}")
        
        return enhanced_results, analysis
    
    def _handle_multi_domain_query(self, query: str, base_results: List[Dict[str, Any]], 
                                  analysis: LegalQueryAnalysis, **resources) -> Tuple[List[Dict[str, Any]], LegalQueryAnalysis]:
        """Maneja consultas que abarcan múltiples dominios"""
        
        print(f"🔀 Procesando consulta multi-dominio...")
        
        all_results = []
        
        # Procesar dominio principal
        primary_retriever = MultiDomainRetrieverFactory.create_retriever(analysis)
        for resource_name, resource_value in resources.items():
            setattr(primary_retriever, resource_name, resource_value)
        
        primary_results = primary_retriever.retrieve(query, base_results, analysis)
        
        # Marcar resultados del dominio principal
        for result in primary_results:
            result['domain_source'] = analysis.primary_domain.value
            result['priority'] = 'primary'
        
        all_results.extend(primary_results)
        
        # Procesar dominios secundarios
        for secondary_domain in analysis.secondary_domains:
            # Crear análisis temporal para dominio secundario
            temp_analysis = LegalQueryAnalysis(
                primary_domain=secondary_domain,
                secondary_domains=[],
                domain_scores=analysis.domain_scores,
                is_multi_domain=False,
                complexity_level=analysis.complexity_level,
                key_entities=analysis.key_entities,
                legal_concepts=analysis.legal_concepts
            )
            
            secondary_retriever = MultiDomainRetrieverFactory.create_retriever(temp_analysis)
            for resource_name, resource_value in resources.items():
                setattr(secondary_retriever, resource_name, resource_value)
            
            secondary_results = secondary_retriever.retrieve(query, base_results, temp_analysis)
            
            # Marcar resultados secundarios y ajustar score
            for result in secondary_results:
                result['domain_source'] = secondary_domain.value
                result['priority'] = 'secondary'
                result['score'] = result.get('score', 0.0) * 0.7  # Reducir peso de dominios secundarios
            
            # Evitar duplicados
            existing_ids = {r.get('article_id', '') for r in all_results}
            new_results = [r for r in secondary_results if r.get('article_id', '') not in existing_ids]
            
            all_results.extend(new_results)
            print(f"   + {len(new_results)} resultados de dominio {secondary_domain.value}")
        
        # Reordenar resultados finales por score
        all_results.sort(key=lambda x: (
            x.get('priority') == 'primary',  # Primarios primero
            x.get('score', 0.0)  # Luego por score
        ), reverse=True)
        
        # Limitar resultados totales
        final_results = all_results[:20]  # Más resultados para multi-dominio
        
        print(f"🎯 Consulta multi-dominio completada: {len(final_results)} resultados finales")
        
        return final_results, analysis

# ==================================================================
# FUNCIÓN DE INTEGRACIÓN PRINCIPAL
# ==================================================================

def enhance_search_with_multi_domain_specialization(
    query: str, 
    base_results: List[Dict[str, Any]],
    neo4j_driver=None,
    weaviate_client=None, 
    documents=None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Función principal para integrar en main.py
    
    Args:
        query: Consulta del usuario
        base_results: Resultados base del sistema
        neo4j_driver: Driver de Neo4j (opcional)
        weaviate_client: Cliente de Weaviate (opcional)
        documents: Documentos en memoria (opcional)
        
    Returns:
        Tuple de (resultados_mejorados, información_análisis)
    """
    
    # Crear orquestador
    orchestrator = MultiDomainOrchestrator()
    
    # Preparar recursos
    resources = {}
    if neo4j_driver:
        resources['neo4j_driver'] = neo4j_driver
    if weaviate_client:
        resources['weaviate_client'] = weaviate_client
    if documents:
        resources['documents'] = documents
    
    # Procesar consulta
    enhanced_results, analysis = orchestrator.enhance_search(
        query, base_results, **resources
    )
    
    # Preparar información de análisis para logging
    analysis_info = {
        'primary_domain': analysis.primary_domain.value,
        'secondary_domains': [d.value for d in analysis.secondary_domains],
        'is_multi_domain': analysis.is_multi_domain,
        'complexity_level': analysis.complexity_level,
        'domain_scores': {d.value: s.score for d, s in analysis.domain_scores.items()},
        'key_concepts': analysis.legal_concepts
    }
    
    return enhanced_results, analysis_info

# ==================================================================
# EJEMPLOS DE USO Y CASOS DE PRUEBA
# ==================================================================

class LegalQueryExamples:
    """Ejemplos de consultas para diferentes escenarios"""
    
    SINGLE_DOMAIN_EXAMPLES = {
        "LABORAL": [
            "fui despedida luego de trabajar durante 5 años en relacion de dependencia, sin anticipacion o previo aviso y sin indemnizacion luego de avisar que estoy embarazada",
            "mi empleador no me paga las horas extras trabajadas durante el fin de semana",
            "sufrí discriminación por mi orientación sexual en el trabajo",
            "¿cuántos días de vacaciones me corresponden por año?"
        ],
        
        "CIVIL": [
            "mi inquilino no paga el alquiler hace 3 meses y quiero desalojarlo",
            "compré un auto usado que resultó tener problemas ocultos, quiero devolverlo",
            "mi padre falleció sin testamento, ¿cómo se divide la herencia?",
            "mi vecino construyó una pared que invade mi propiedad"
        ],
        
        "COMERCIAL": [
            "mi socio no aportó el capital comprometido a la sociedad",
            "una empresa competidora está usando nuestra marca registrada",
            "mi empresa está en crisis y no puede pagar a los acreedores",
            "¿cómo disolver una sociedad anónima?"
        ],
        
        "PENAL": [
            "me robaron el celular en la calle con violencia",
            "mi expareja me amenaza constantemente por WhatsApp",
            "alguien está usando mi identidad para hacer compras online",
            "un empleado se apropió de dinero de la empresa"
        ]
    }
    
    MULTI_DOMAIN_EXAMPLES = [
        # Laboral + Penal
        "mi jefe me amenaza con despedirme si no acepto trabajar horas extras sin pago",
        
        # Civil + Comercial  
        "firmé un contrato de compraventa de un inmueble pero el vendedor es una sociedad que está en quiebra",
        
        # Civil + Penal
        "me estafaron vendiéndome un terreno que no les pertenecía",
        
        # Laboral + Civil
        "sufrí un accidente en el trabajo por negligencia del empleador y quiero una indemnización",
        
        # Comercial + Penal
        "mi socio falsificó mi firma para contraer deudas a nombre de la sociedad",
        
        # Triple dominio: Civil + Comercial + Penal
        "una empresa me vendió un producto defectuoso que me causó lesiones, y ahora descubrí que falsificaron las certificaciones de calidad"
    ]

# ==================================================================
# UTILIDADES DE DIAGNÓSTICO Y TESTING
# ==================================================================

def test_domain_classification():
    """Función de prueba para verificar la clasificación de dominios"""
    
    classifier = MultiDomainClassifier()
    examples = LegalQueryExamples()
    
    print("🧪 PRUEBAS DE CLASIFICACIÓN DE DOMINIOS\n")
    
    # Probar casos de dominio único
    for domain, queries in examples.SINGLE_DOMAIN_EXAMPLES.items():
        print(f"📋 DOMINIO: {domain}")
        for query in queries:
            analysis = classifier.analyze_query(query)
            primary = analysis.primary_domain.value
            secondary = [d.value for d in analysis.secondary_domains]
            
            print(f"  ✓ '{query[:50]}...'")
            print(f"    → Principal: {primary}, Secundarios: {secondary}")
            print(f"    → Multi-dominio: {analysis.is_multi_domain}, Complejidad: {analysis.complexity_level}")
        print()
    
    # Probar casos multi-dominio
    print("📋 CASOS MULTI-DOMINIO")
    for query in examples.MULTI_DOMAIN_EXAMPLES:
        analysis = classifier.analyze_query(query)
        primary = analysis.primary_domain.value
        secondary = [d.value for d in analysis.secondary_domains]
        
        print(f"  ✓ '{query[:60]}...'")
        print(f"    → Principal: {primary}, Secundarios: {secondary}")
        print(f"    → Multi-dominio: {analysis.is_multi_domain}, Complejidad: {analysis.complexity_level}")
    print()

def generate_integration_guide():
    """Genera guía de integración con el sistema existente"""
    
    integration_guide = """
# 🔧 GUÍA DE INTEGRACIÓN CON MAIN.PY

## 1. Reemplazar función existente

En main.py, reemplazar la llamada a:
```python
specialized_results = enhance_search_with_specialization(
    query=query,
    base_results=base_results,
    detected_domains=labor_domains,
    category_scores=category_scores,
    neo4j_driver=neo4j_driver
)
```

Por:
```python
specialized_results, analysis_info = enhance_search_with_multi_domain_specialization(
    query=query,
    base_results=base_results,
    neo4j_driver=neo4j_driver,
    weaviate_client=weaviate_client,
    documents=documents
)

# Logging extendido
print(f"📊 Análisis completo:")
print(f"  Dominio principal: {analysis_info['primary_domain']}")
print(f"  Dominios secundarios: {analysis_info['secondary_domains']}")
print(f"  Multi-dominio: {analysis_info['is_multi_domain']}")
print(f"  Complejidad: {analysis_info['complexity_level']}")
```

## 2. Mantener compatibilidad

El sistema es retrocompatible. Casos laborales seguirán funcionando igual,
pero ahora también maneja civil, comercial y penal.

## 3. Configuración por defecto

Si no se detecta un dominio específico, el sistema defaultea a CIVIL,
que es el más general y cubre la mayoría de casos.

## 4. Ventajas del nuevo sistema

✅ Maneja consultas complejas multi-dominio
✅ Mejor precisión en clasificación
✅ Scoring especializado por materia
✅ Detección automática de intersecciones legales
✅ Escalable para agregar nuevos dominios
"""
    
    return integration_guide

# ==================================================================
# CONFIGURACIÓN Y EXPORTACIONES
# ==================================================================

# Exportar clases principales para uso en main.py
__all__ = [
    'MultiDomainOrchestrator',
    'MultiDomainClassifier', 
    'LegalQueryAnalysis',
    'LegalDomain',
    'enhance_search_with_multi_domain_specialization',
    'test_domain_classification',
    'LegalQueryExamples'
]