"""
Módulo para detección de dominios legales en consultas
"""
from typing import Dict, List

# Dominios legales básicos y sus palabras clave
LEGAL_DOMAINS = {
    "Embarazo": [
        "embarazo", "embarazada", "gestación", "maternidad", "lactancia",
        "licencia maternidad", "estado gestación", "periodo lactancia"
    ],
    "Despido": [
        "despido", "despedir", "cesantía", "terminación", "finalización",
        "renuncia", "rescisión", "extinción contrato", "despedida"
    ],
    "Discriminación": [
        "discriminación", "discriminar", "acoso", "maltrato", "persecución",
        "trato diferencial", "hostigamiento", "mobbing"
    ],
    "Remuneración": [
        "sueldo", "salario", "remuneración", "pago", "indemnización",
        "finiquito", "liquidación", "compensación", "aguinaldo"
    ],
    "Jornada": [
        "jornada", "horario", "horas", "descanso", "feriado",
        "turno", "sobretiempo", "horas extras", "franco"
    ],
    "Accidentes": [
        "accidente trabajo", "enfermedad profesional", "lesión",
        "seguridad laboral", "riesgo trabajo", "ART"
    ],
    "Prestaciones": [
        "obra social", "aportes", "jubilación", "AFIP",
        "seguridad social", "prestaciones", "beneficios"
    ],
    "Procesos_Administrativos": [
        "denuncia", "inspección", "procedimiento", "formulario",
        "solicitud", "reclamo", "recurso", "apelación"
    ]
}

def detect_domains_in_query(query: str) -> List[str]:
    """
    Detecta dominios legales en la consulta del usuario.
    
    Args:
        query: Consulta del usuario en texto libre
        
    Returns:
        Lista de dominios legales detectados
    """
    detected_domains = []
    query_lower = query.lower()
    
    for domain, keywords in LEGAL_DOMAINS.items():
        for keyword in keywords:
            if keyword.lower() in query_lower:
                if domain not in detected_domains:
                    detected_domains.append(domain)
                break
    
    return detected_domains

def extract_labor_entities(query: str) -> Dict[str, List[str]]:
    """
    Extrae entidades específicas del ámbito laboral de la consulta.
    
    Args:
        query: Consulta del usuario
        
    Returns:
        Diccionario con entidades laborales extraídas
    """
    entities = {
        "employment_duration": [],
        "dismissal_type": [],
        "compensation": [],
        "special_conditions": [],
        "laws": [],
        "procedures": []
    }
    
    query_lower = query.lower()
    
    # Duración del empleo
    duration_patterns = [
        "años", "año", "meses", "mes", "días", "día"
    ]
    for pattern in duration_patterns:
        if pattern in query_lower:
            # Buscar números cerca del patrón
            words = query_lower.split()
            for i, word in enumerate(words):
                if pattern in word and i > 0:
                    prev_word = words[i-1]
                    if prev_word.isdigit():
                        entities["employment_duration"].append(f"{prev_word} {pattern}")
    
    # Tipo de despido
    dismissal_types = [
        "sin causa", "con causa", "sin preaviso", "sin anticipación",
        "despido directo", "despido indirecto"
    ]
    for dismissal_type in dismissal_types:
        if dismissal_type in query_lower:
            entities["dismissal_type"].append(dismissal_type)
    
    # Compensación
    compensation_terms = [
        "indemnización", "finiquito", "liquidación", "compensación"
    ]
    for term in compensation_terms:
        if term in query_lower:
            entities["compensation"].append(term)
    
    # Condiciones especiales
    special_conditions = [
        "embarazo", "maternidad", "enfermedad", "accidente",
        "sindical", "delegado", "discapacidad"
    ]
    for condition in special_conditions:
        if condition in query_lower:
            entities["special_conditions"].append(condition)
    
    # Leyes mencionadas
    law_patterns = [
        "ley contrato trabajo", "LCT", "código civil",
        "ley empleo", "convenio colectivo"
    ]
    for law in law_patterns:
        if law in query_lower:
            entities["laws"].append(law)
    
    # Procedimientos
    procedure_patterns = [
        "denuncia", "reclamo", "demanda", "carta documento",
        "inspección trabajo", "ministerio trabajo"
    ]
    for procedure in procedure_patterns:
        if procedure in query_lower:
            entities["procedures"].append(procedure)
    
    return entities

def get_domain_priority(domains: List[str]) -> List[str]:
    """
    Ordena los dominios por prioridad según su relevancia típica.
    
    Args:
        domains: Lista de dominios detectados
        
    Returns:
        Lista ordenada por prioridad
    """
    priority_order = [
        "Embarazo",
        "Discriminación", 
        "Despido",
        "Accidentes",
        "Remuneración",
        "Jornada",
        "Prestaciones",
        "Procesos_Administrativos"
    ]
    
    # Ordenar según la prioridad definida
    sorted_domains = []
    for priority_domain in priority_order:
        if priority_domain in domains:
            sorted_domains.append(priority_domain)
    
    # Añadir cualquier dominio no contemplado en la prioridad
    for domain in domains:
        if domain not in sorted_domains:
            sorted_domains.append(domain)
    
    return sorted_domains

def analyze_query_complexity(query: str) -> Dict[str, any]:
    """
    Analiza la complejidad de una consulta legal.
    
    Args:
        query: Consulta del usuario
        
    Returns:
        Diccionario con análisis de complejidad
    """
    detected_domains = detect_domains_in_query(query)
    labor_entities = extract_labor_entities(query)
    
    # Calcular score de complejidad
    complexity_score = 0
    complexity_score += len(detected_domains) * 2  # Múltiples dominios aumentan complejidad
    complexity_score += len([e for entities in labor_entities.values() for e in entities])
    
    # Indicadores de complejidad específicos
    complexity_indicators = [
        "y además", "también", "pero", "sin embargo", "por otro lado",
        "al mismo tiempo", "conjuntamente", "tanto como", "no solo"
    ]
    
    query_lower = query.lower()
    for indicator in complexity_indicators:
        if indicator in query_lower:
            complexity_score += 3
    
    # Determinar nivel de complejidad
    if complexity_score <= 3:
        level = "simple"
    elif complexity_score <= 8:
        level = "moderate"
    else:
        level = "complex"
    
    return {
        "level": level,
        "score": complexity_score,
        "domains": detected_domains,
        "entities": labor_entities,
        "domain_count": len(detected_domains),
        "multi_domain": len(detected_domains) > 1
    }