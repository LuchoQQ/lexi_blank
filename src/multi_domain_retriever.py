"""
Sistema de Recuperación Multi-Dominio Simplificado
==================================================

Archivo: src/multi_domain_retriever.py

Sistema modular que integra las configuraciones separadas de dominios y casos.
Arquitectura limpia y extensible para agregar nuevos dominios fácilmente.

VERSION: 2.0 - MODULAR
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re

# Imports de configuraciones modulares
from .domain_configurations import (
    LegalDomain, DOMAIN_BASE_CONFIG, DOMAIN_INTERSECTION_PATTERNS,
    COMPLEXITY_INDICATORS, LINGUISTIC_PATTERNS
)
from .cases.labor_cases import LABOR_SPECIFIC_CASES
from .cases.commercial_criminal_cases import COMMERCIAL_SPECIFIC_CASES, CRIMINAL_SPECIFIC_CASES

# =============================================================================
# CLASES DE DATOS
# =============================================================================

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
    complexity_level: str
    key_entities: Dict[str, List[str]]
    legal_concepts: List[str]
    identified_case: Optional[str] = None

# =============================================================================
# CLASIFICADOR MULTI-DOMINIO
# =============================================================================

class MultiDomainClassifier:
    """Clasificador inteligente para múltiples dominios legales"""
    
    def __init__(self):
        self.domain_configs = DOMAIN_BASE_CONFIG
        self.intersection_patterns = DOMAIN_INTERSECTION_PATTERNS
        
    def analyze_query(self, query: str) -> LegalQueryAnalysis:
        """Análisis completo de la consulta para determinar dominios relevantes"""
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
                    # Peso por especificidad y multiplicador del dominio
                    weight = (len(keyword.split()) * 0.5 + 1.0) * config.get("weight_multiplier", 1.0)
                    score += weight
                    matched_keywords.append(keyword)
            
            # Penalización por keywords de exclusión
            exclusion_penalty = 0.0
            for exclusion in config.get("exclusion_keywords", []):
                if exclusion.lower() in query:
                    exclusion_penalty += 3.0  # Penalización más fuerte
            
            # Score final ajustado
            final_score = max(0.0, score - exclusion_penalty)
            
            # Calcular confianza
            confidence = min(1.0, (len(matched_keywords) * 0.15) + (final_score * 0.08))
            
            scores[domain] = DomainScore(
                domain=domain,
                score=final_score,
                confidence=confidence,
                key_indicators=matched_keywords
            )
        
        return scores
    
    def _determine_domains(self, domain_scores: Dict[LegalDomain, DomainScore]) -> Tuple[LegalDomain, List[LegalDomain]]:
        """Determina dominio principal y secundarios"""
        sorted_domains = sorted(
            domain_scores.items(), 
            key=lambda x: x[1].score, 
            reverse=True
        )
        
        if not sorted_domains or sorted_domains[0][1].score == 0:
            return LegalDomain.CIVIL, []  # Default a civil
        
        primary_domain = sorted_domains[0][0]
        primary_score = sorted_domains[0][1].score
        threshold = primary_score * 0.4  # Más restrictivo
        
        secondary_domains = [
            domain for domain, score_obj in sorted_domains[1:] 
            if score_obj.score > threshold and score_obj.confidence > 0.4
        ]
        
        return primary_domain, secondary_domains
    
    def _is_multi_domain_query(self, domain_scores: Dict[LegalDomain, DomainScore], query: str) -> bool:
        """Detecta si la consulta abarca múltiples dominios"""
        significant_domains = [
            domain for domain, score_obj in domain_scores.items() 
            if score_obj.score > 3.0 and score_obj.confidence > 0.5
        ]
        
        return len(significant_domains) > 1
    
    def _assess_complexity(self, domain_scores: Dict[LegalDomain, DomainScore], 
                          is_multi_domain: bool, query: str) -> str:
        """Evalúa la complejidad de la consulta"""
        complexity_score = 0
        
        if is_multi_domain:
            complexity_score += 2
            
        # Usar patrones lingüísticos de la configuración
        for indicator in LINGUISTIC_PATTERNS["complexity_indicators"]:
            if indicator in query.lower():
                complexity_score += 1
                
        # Clasificar según umbrales
        if complexity_score <= COMPLEXITY_INDICATORS["simple"]["score_threshold"]:
            return "simple"
        elif complexity_score <= COMPLEXITY_INDICATORS["complex"]["score_threshold"]:
            return "complex"
        else:
            return "multi_jurisdictional"
    
    def _extract_legal_entities(self, query: str) -> Dict[str, List[str]]:
        """Extrae entidades legales específicas"""
        entities = {"personas": [], "bienes": [], "montos": [], "plazos": []}
        
        patterns = {
            "montos": r'(\$\s*\d+(?:\.\d{3})*(?:,\d{2})?|\d+\s*pesos)',
            "plazos": r'(\d+\s*(?:días?|meses?|años?))',
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, query.lower())
            entities[entity_type].extend(matches)
        
        return entities
    
    def _extract_legal_concepts(self, query: str, primary_domain: LegalDomain) -> List[str]:
        """Extrae conceptos legales relevantes del dominio principal"""
        # Esta función se puede expandir según necesidad
        return []

# =============================================================================
# RECUPERADORES ESPECIALIZADOS
# =============================================================================

class SpecializedRetriever(ABC):
    """Clase base para recuperadores especializados por dominio"""
    
    def __init__(self, domain: LegalDomain):
        self.domain = domain
        self.domain_config = DOMAIN_BASE_CONFIG.get(domain, {})
        
    @abstractmethod
    def retrieve(self, query: str, base_results: List[Dict[str, Any]], 
                 analysis: LegalQueryAnalysis) -> List[Dict[str, Any]]:
        """Implementar lógica de recuperación especializada"""
        pass
    
    def _calculate_base_score(self, article: Dict[str, Any], query: str) -> float:
        """Calcula score base común a todos los dominios"""
        base_score = article.get('score', 0.0)
        law_name = article.get('law_name', '').lower()
        
        # Boost por ley prioritaria del dom