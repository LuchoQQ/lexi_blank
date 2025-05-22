"""
Casos Comerciales y Penales Específicos
=======================================

Archivo: src/cases/commercial_criminal_cases.py

Definiciones específicas de casos comerciales y penales con sus artículos críticos.
Separado para facilitar revisión legal y expansión por especialistas.

VERSION: 1.0
ULTIMA REVISION: [PENDIENTE]
"""

from typing import Dict, List, Any

# =============================================================================
# CASOS COMERCIALES ESPECÍFICOS
# =============================================================================

COMMERCIAL_SPECIFIC_CASES = {
    
    # =========================================================================
    # SOCIEDADES COMERCIALES
    # =========================================================================
    
    "constitucion_sociedad": {
        "keywords": [
            "constituir sociedad", "formar sociedad", "crear empresa", "estatuto social",
            "contrato social", "SA", "SRL", "sociedad anónima", "responsabilidad limitada",
            "aportes", "capital social", "objeto social", "administración"
        ],
        
        "critical_articles": ["1", "4", "11", "17", "27", "164", "299"],
        
        "concepts": [
            "personalidad jurídica", "responsabilidad limitada", "objeto social",
            "capital mínimo", "publicidad registral", "administración societaria"
        ],
        
        "priority_boost": 2.0,
        "confidence_threshold": 0.7
    },
    
    "conflictos_socios": {
        "keywords": [
            "conflicto socios", "socio no aporta", "abuso mayoría", "exclusión socio",
            "derecho información", "dividendos", "distribución utilidades",
            "reunión socios", "asamblea", "administrador incumple"
        ],
        
        "critical_articles": ["37", "54", "55", "248", "319", "160", "244"],
        
        "concepts": [
            "derecho información", "abuso derecho", "exclusión justa causa",
            "responsabilidad administradores", "interés social", "minoría societaria"
        ],
        
        "priority_boost": 2.2,
        "confidence_threshold": 0.6
    },
    
    "quiebra_concurso": {
        "keywords": [
            "quiebra", "concurso", "cesación pagos", "insolvencia", "crisis empresarial",
            "verificación créditos", "síndico", "continuidad empresa", "cramdown",
            "acuerdo preventivo", "liquidación"
        ],
        
        "critical_articles": ["1", "11", "48", "88", "177", "214", "236"],
        
        "concepts": [
            "cesación de pagos", "verificación de créditos", "período sospecha",
            "continuidad empresa", "acuerdo preventivo extrajudicial", "cramdown"
        ],
        
        "priority_boost": 2.3,
        "confidence_threshold": 0.8
    },
    
    # =========================================================================
    # CONTRATOS COMERCIALES
    # =========================================================================
    
    "incumplimiento_comercial": {
        "keywords": [
            "incumplimiento contrato", "breach", "daños comerciales", "lucro cesante",
            "pérdida chances", "resolución contrato", "mora comercial",
            "cláusula penal", "arras", "seña"
        ],
        
        "critical_articles": ["216", "217", "790", "791", "1056", "1336"],
        
        "concepts": [
            "mora comercial", "cláusula penal", "resolución por incumplimiento",
            "daño emergente comercial", "pérdida de chance", "pacto comisorio"
        ],
        
        "priority_boost": 1.8,
        "confidence_threshold": 0.6
    },
    
    "transferencia_fondo_comercio": {
        "keywords": [
            "transferencia fondo comercio", "venta empresa", "goodwill", "clientela",
            "llave", "habilitación comercial", "publicidad edictos", "oposición acreedores",
            "embargo preventivo", "privilegios"
        ],
        
        "critical_articles": ["1", "2", "8", "9", "10", "11"],
        
        "concepts": [
            "elementos fondo comercio", "publicidad transferencia", 
            "oposición acreedores", "responsabilidad solidaria", "privilegios especiales"
        ],
        
        "priority_boost": 2.0,
        "confidence_threshold": 0.7
    },
    
    # =========================================================================
    # TÍTULOS DE CRÉDITO
    # =========================================================================
    
    "cheques_rechazados": {
        "keywords": [
            "cheque rechazado", "cheque sin fondos", "cuenta corriente cerrada",
            "multa BCRA", "inhabilitación", "protesto", "ejecutar cheque",
            "acción cambiaria", "libramiento cheque sin fondos"
        ],
        
        "critical_articles": ["27", "28", "35", "38", "52", "53", "302"],
        
        "concepts": [
            "libramiento sin fondos", "acción cambiaria directa", "multa BCRA",
            "inhabilitación cuenta corriente", "protesto cheque", "privilegio especial"
        ],
        
        "priority_boost": 2.1,
        "confidence_threshold": 0.8
    },
    
    # =========================================================================
    # PROPIEDAD INTELECTUAL
    # =========================================================================
    
    "marca_patente": {
        "keywords": [
            "marca registrada", "patente", "uso indebido marca", "competencia desleal",
            "violación marca", "modelo utilidad", "diseño industrial",
            "derecho exclusivo", "regalías", "licencia"
        ],
        
        "critical_articles": ["1", "2", "31", "34", "35", "Ley 22.362"],
        
        "concepts": [
            "derecho exclusivo marca", "uso indebido", "competencia desleal",
            "violación derechos intelectuales", "reparación daños", "medidas cautelares"
        ],
        
        "priority_boost": 1.9,
        "confidence_threshold": 0.7
    }
}

# =============================================================================
# CASOS PENALES ESPECÍFICOS  
# =============================================================================

CRIMINAL_SPECIFIC_CASES = {
    
    # =========================================================================
    # DELITOS CONTRA LA PROPIEDAD
    # =========================================================================
    
    "hurto_robo": {
        "keywords": [
            "me robaron", "hurto", "robo", "apoderamiento", "cosa mueble",
            "violencia", "intimidación", "arrebato", "sustracción", "despojo",
            "robo agravado", "poblado", "despoblado", "arma", "banda"
        ],
        
        "critical_articles": ["162", "163", "164", "165", "166", "167"],
        
        "concepts": [
            "apoderamiento ilegítimo", "cosa mueble ajena", "violencia o intimidación",
            "robo agravado", "circunstancias agravantes", "pena privativa libertad"
        ],
        
        "priority_boost": 2.5,
        "confidence_threshold": 0.8
    },
    
    "estafa_defraudacion": {
        "keywords": [
            "estafa", "defraudación", "engaño", "ardid", "me estafaron",
            "fraude", "documento falso", "abuso confianza", "administración fraudulenta",
            "cheque sin fondos", "maniobra fraudulenta"
        ],
        
        "critical_articles": ["172", "173", "174", "175", "176"],
        
        "concepts": [
            "ardid o engaño", "perjuicio patrimonial", "error en la víctima",
            "administración fraudulenta", "abuso de confianza", "documento adulterado"
        ],
        
        "priority_boost": 2.3,
        "confidence_threshold": 0.7
    },
    
    "apropiacion_indebida": {
        "keywords": [
            "apropiación indebida", "se quedó con mi dinero", "no devuelve",
            "retención indebida", "conversión", "abuso confianza",
            "dinero ajeno", "depósito", "mandato", "comisión"
        ],
        
        "critical_articles": ["173"],
        
        "concepts": [
            "conversión en provecho propio", "cosa mueble ajena", 
            "título no traslativo dominio", "abuso de confianza", "ánimo de lucro"
        ],
        
        "priority_boost": 2.2,
        "confidence_threshold": 0.7
    },
    
    # =========================================================================
    # DELITOS CONTRA LAS PERSONAS
    # =========================================================================
    
    "lesiones": {
        "keywords": [
            "lesiones", "golpes", "me pegó", "agresión física", "heridas",
            "lesiones leves", "lesiones graves", "daño cuerpo", "violencia física",
            "golpiza", "pelea", "riña"
        ],
        
        "critical_articles": ["89", "90", "91", "92", "93", "94"],
        
        "concepts": [
            "daño en el cuerpo", "daño en la salud", "lesiones leves",
            "lesiones graves", "lesiones gravísimas", "debilitamiento permanente"
        ],
        
        "priority_boost": 2.4,
        "confidence_threshold": 0.8
    },
    
    "amenazas": {
        "keywords": [
            "amenaza", "me amenaza", "intimidación", "amenaza condicional",
            "mal grave", "WhatsApp amenaza", "mensaje amenazante",
            "amenaza muerte", "amenaza violencia", "amedrentamiento"
        ],
        
        "critical_articles": ["149", "149bis"],
        
        "concepts": [
            "amenaza de mal grave", "mal determinado", "mal posible",
            "intimidación", "perturbación tranquilidad", "amenaza condicional"
        ],
        
        "priority_boost": 2.2,
        "confidence_threshold": 0.7
    },
    
    "homicidio": {
        "keywords": [
            "homicidio", "asesinato", "mató", "muerte", "femicidio",
            "homicidio simple", "homicidio agravado", "parricidio",
            "premeditación", "alevosía", "ensañamiento"
        ],
        
        "critical_articles": ["79", "80", "81", "82", "84"],
        
        "concepts": [
            "muerte de persona", "homicidio simple", "homicidio agravado",
            "circunstancias agravantes", "vínculo", "violencia de género"
        ],
        
        "priority_boost": 3.0,
        "confidence_threshold": 0.9
    },
    
    # =========================================================================
    # DELITOS CONTRA LA LIBERTAD
    # =========================================================================
    
    "privacion_libertad": {
        "keywords": [
            "secuestro", "privación libertad", "retención", "detención ilegal",
            "encierro", "no me dejan salir", "me tienen encerrado",
            "restricción movimiento", "cautiverio"
        ],
        
        "critical_articles": ["141", "142", "142bis", "170"],
        
        "concepts": [
            "privación de libertad", "detención ilegal", "secuestro extorsivo",
            "restricción ambulatoria", "trata de personas", "explotación"
        ],
        
        "priority_boost": 2.8,
        "confidence_threshold": 0.8
    },
    
    # =========================================================================
    # DELITOS CONTRA LA FE PÚBLICA
    # =========================================================================
    
    "falsificacion_documentos": {
        "keywords": [
            "documento falso", "falsificación", "adulteración documento",
            "firma falsa", "falsificar", "documento apócrifo",
            "alteración documento", "uso documento falso"
        ],
        
        "critical_articles": ["292", "293", "294", "296", "297"],
        
        "concepts": [
            "falsificación material", "falsificación ideológica", 
            "documento público", "documento privado", "alteración de la verdad"
        ],
        
        "priority_boost": 2.1,
        "confidence_threshold": 0.7
    }
}

# =============================================================================
# CASOS POR CATEGORÍA
# =============================================================================

COMMERCIAL_CASES_BY_CATEGORY = {
    "sociedades": [
        "constitucion_sociedad", "conflictos_socios", "quiebra_concurso"
    ],
    "contratos": [
        "incumplimiento_comercial", "transferencia_fondo_comercio"
    ],
    "titulos_credito": [
        "cheques_rechazados"
    ],
    "propiedad_intelectual": [
        "marca_patente"
    ]
}

CRIMINAL_CASES_BY_CATEGORY = {
    "delitos_propiedad": [
        "hurto_robo", "estafa_defraudacion", "apropiacion_indebida"
    ],
    "delitos_personas": [
        "lesiones", "amenazas", "homicidio"
    ],
    "delitos_libertad": [
        "privacion_libertad"
    ],
    "delitos_fe_publica": [
        "falsificacion_documentos"
    ]
}

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def get_commercial_case(case_name: str) -> Dict[str, Any]:
    """Obtiene un caso comercial específico"""
    return COMMERCIAL_SPECIFIC_CASES.get(case_name, {})

def get_criminal_case(case_name: str) -> Dict[str, Any]:
    """Obtiene un caso penal específico"""
    return CRIMINAL_SPECIFIC_CASES.get(case_name, {})

def get_all_commercial_articles() -> List[str]:
    """Obtiene todos los artículos críticos comerciales"""
    articles = set()
    for case in COMMERCIAL_SPECIFIC_CASES.values():
        articles.update(case.get("critical_articles", []))
    return sorted(articles)

def get_all_criminal_articles() -> List[str]:
    """Obtiene todos los artículos críticos penales"""
    articles = set()
    for case in CRIMINAL_SPECIFIC_CASES.values():
        articles.update(case.get("critical_articles", []))
    return sorted(articles)

def validate_commercial_cases():
    """Valida casos comerciales"""
    return _validate_cases_structure(COMMERCIAL_SPECIFIC_CASES, "comercial")

def validate_criminal_cases():
    """Valida casos penales"""
    return _validate_cases_structure(CRIMINAL_SPECIFIC_CASES, "penal")

def _validate_cases_structure(cases_dict: Dict, domain_name: str) -> List[str]:
    """Función auxiliar para validar estructura de casos"""
    required_fields = ["keywords", "critical_articles", "concepts", "priority_boost", "confidence_threshold"]
    issues = []
    
    for case_name, case_data in cases_dict.items():
        for field in required_fields:
            if field not in case_data:
                issues.append(f"Caso {domain_name} '{case_name}': Falta campo '{field}'")
            elif field in ["keywords", "critical_articles", "concepts"] and not case_data[field]:
                issues.append(f"Caso {domain_name} '{case_name}': Campo '{field}' está vacío")
    
    return issues

def get_combined_statistics():
    """Obtiene estadísticas combinadas de casos comerciales y penales"""
    commercial_count = len(COMMERCIAL_SPECIFIC_CASES)
    criminal_count = len(CRIMINAL_SPECIFIC_CASES)
    
    commercial_articles = set()
    criminal_articles = set()
    
    for case in COMMERCIAL_SPECIFIC_CASES.values():
        commercial_articles.update(case.get("critical_articles", []))
    
    for case in CRIMINAL_SPECIFIC_CASES.values():
        criminal_articles.update(case.get("critical_articles", []))
    
    return {
        "commercial_cases": commercial_count,
        "criminal_cases": criminal_count,
        "total_cases": commercial_count + criminal_count,
        "commercial_articles": len(commercial_articles),
        "criminal_articles": len(criminal_articles),
        "commercial_categories": len(COMMERCIAL_CASES_BY_CATEGORY),
        "criminal_categories": len(CRIMINAL_CASES_BY_CATEGORY)
    }

# =============================================================================
# VALIDACIÓN Y EXPORTACIONES
# =============================================================================

# Ejecutar validaciones al importar
_commercial_issues = validate_commercial_cases()
_criminal_issues = validate_criminal_cases()

if _commercial_issues:
    print("⚠️ ADVERTENCIAS EN CASOS COMERCIALES:")
    for issue in _commercial_issues:
        print(f"  - {issue}")

if _criminal_issues:
    print("⚠️ ADVERTENCIAS EN CASOS PENALES:")
    for issue in _criminal_issues:
        print(f"  - {issue}")

# Exportaciones principales
__all__ = [
    'COMMERCIAL_SPECIFIC_CASES',
    'CRIMINAL_SPECIFIC_CASES',
    'COMMERCIAL_CASES_BY_CATEGORY',
    'CRIMINAL_CASES_BY_CATEGORY',
    'get_commercial_case',
    'get_criminal_case',
    'get_all_commercial_articles',
    'get_all_criminal_articles',
    'get_combined_statistics'
]

if __name__ == "__main__":
    # Ejecutar cuando se llama directamente para debugging
    print("🏢 CASOS COMERCIALES Y PENALES")
    print("=" * 50)
    
    stats = get_combined_statistics()
    print(f"📊 ESTADÍSTICAS:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n📋 CATEGORÍAS COMERCIALES:")
    for category, cases in COMMERCIAL_CASES_BY_CATEGORY.items():
        print(f"  {category}: {len(cases)} casos")
    
    print(f"\n⚖️ CATEGORÍAS PENALES:")
    for category, cases in CRIMINAL_CASES_BY_CATEGORY.items():
        print(f"  {category}: {len(cases)} casos")
    
    print(f"\n🔍 VALIDACIÓN:")
    total_issues = len(_commercial_issues) + len(_criminal_issues)
    if total_issues == 0:
        print("  ✅ Todos los casos tienen estructura válida")
    else:
        print(f"  ❌ {total_issues} issues encontrados")