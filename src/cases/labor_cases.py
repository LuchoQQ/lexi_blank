"""
Casos Laborales Específicos
===========================

Archivo: src/cases/labor_cases.py

Definiciones específicas de casos laborales con sus artículos críticos.
Separado para facilitar revisión legal y expansión por especialistas.

INSTRUCCIONES PARA ABOGADOS:
- Revisar precisión legal de cada caso
- Validar artículos críticos listados  
- Expandir keywords basado en experiencia práctica
- Añadir nuevos casos según necesidades del despacho

VERSION: 1.0
ULTIMA REVISION: [PENDIENTE]
"""

from typing import Dict, List, Any

# =============================================================================
# CASOS LABORALES ESPECÍFICOS
# =============================================================================

LABOR_SPECIFIC_CASES = {
    
    # =========================================================================
    # DESPIDOS Y EXTINCIÓN CONTRACTUAL
    # =========================================================================
    
    "embarazo_despido": {
        "keywords": [
            # Términos embarazo
            "embarazo", "embarazada", "gestante", "maternidad", "maternal", 
            "gestación", "encinta", "esperando bebé", "en estado", "prenatal", 
            "postnatal", "lactancia", "parto", "nacimiento", "bebé",
            
            # Términos despido
            "despido", "despedida", "cesantía", "desvinculación", "echaron", 
            "terminaron", "finalizaron contrato", "me rajaron", "me sacaron",
            
            # Términos específicos
            "fuero maternal", "estabilidad", "prohibición", "discriminación embarazo",
            "despido discriminatorio", "reinstalación"
        ],
        
        "critical_articles": ["177", "178", "182", "183", "186", "245", "232", "233"],
        
        "concepts": [
            "estabilidad laboral absoluta", "fuero maternal", 
            "presunción despido discriminatorio", "prohibición despido",
            "indemnización agravada 13 meses", "reinstalación obligatoria",
            "licencia maternidad 90 días", "protección especial maternidad"
        ],
        
        "priority_boost": 2.5,
        "confidence_threshold": 0.8
    },
    
    "despido_sin_causa": {
        "keywords": [
            "despido", "despedido", "sin causa", "sin justa causa", "arbitrario",
            "cesantía", "desvinculación", "echaron", "terminaron", "finalizaron",
            "me rajaron", "me sacaron", "quedé sin trabajo", "perdí empleo",
            "preaviso", "indemnización", "antigüedad", "liquidación"
        ],
        
        "critical_articles": ["245", "232", "233", "231", "95"],
        
        "concepts": [
            "indemnización tarifada", "preaviso sustitutivo", "integración mes despido",
            "mejor remuneración mensual", "antigüedad computable", "extinción sin causa"
        ],
        
        "priority_boost": 2.0,
        "confidence_threshold": 0.7
    },
    
    "despido_con_causa": {
        "keywords": [
            "despido con causa", "justa causa", "grave inconducta", "injuria",
            "falta grave", "incumplimiento", "desobediencia", "abandono trabajo",
            "violencia", "hurto", "robo", "agresión", "insubordinación"
        ],
        
        "critical_articles": ["242", "243", "244", "67", "68", "69", "70"],
        
        "concepts": [
            "justa causa", "injuria laboral", "proporcionalidad sanción", 
            "inmediatez", "debido proceso", "derecho defensa"
        ],
        
        "priority_boost": 2.0,
        "confidence_threshold": 0.7
    },
    
    # =========================================================================
    # DISCRIMINACIÓN Y ACOSO  
    # =========================================================================
    
    "discriminacion_general": {
        "keywords": [
            "discriminación", "trato desigual", "género", "sexo", "raza", "religión", 
            "nacionalidad", "orientación sexual", "discapacidad", "edad", 
            "aspecto físico", "ideología", "condición social", "estado civil"
        ],
        
        "critical_articles": ["17", "18", "81", "172", "173", "245"],
        
        "concepts": [
            "igualdad de trato", "principio no discriminación", 
            "dignidad trabajador", "diversidad laboral", "inclusión"
        ],
        
        "priority_boost": 2.2,
        "confidence_threshold": 0.6
    },
    
    "acoso_laboral": {
        "keywords": [
            "acoso", "mobbing", "hostigamiento", "maltrato", "violencia psicológica",
            "persecución", "humillación", "aislamiento", "intimidación",
            "bullying laboral", "ambiente hostil", "presión psicológica"
        ],
        
        "critical_articles": ["75", "78", "242", "246", "66", "68"],
        
        "concepts": [
            "violencia laboral", "dignidad trabajador", "ambiente trabajo saludable",
            "deber protección empleador", "integridad psicofísica"
        ],
        
        "priority_boost": 2.3,
        "confidence_threshold": 0.7
    },
    
    # =========================================================================
    # SALARIOS Y REMUNERACIONES
    # =========================================================================
    
    "falta_pago_salarios": {
        "keywords": [
            "falta pago", "no pagan", "deben salarios", "sueldo atrasado", 
            "mora salarial", "salarios adeudados", "no cobré", "retraso pago",
            "deuda salarial", "atraso sueldos", "impago", "retención salario"
        ],
        
        "critical_articles": ["126", "127", "128", "129", "130", "133", "74"],
        
        "concepts": [
            "intangibilidad salarial", "pago íntegro oportuno", 
            "intereses moratorios", "mora automática", "pronto pago"
        ],
        
        "priority_boost": 2.1,
        "confidence_threshold": 0.6
    },
    
    "horas_extras": {
        "keywords": [
            "horas extras", "sobretiempo", "exceso jornada", "no pagan horas extras",
            "trabajo extra", "horario extendido", "más de 8 horas", "recargo 50%",
            "recargo 100%", "compensación horas"
        ],
        
        "critical_articles": ["197", "198", "199", "200", "201", "202", "206"],
        
        "concepts": [
            "jornada máxima legal", "límites jornada", "recargo 50% primeras 2 horas",
            "recargo 100% siguientes", "autorización ministerial"
        ],
        
        "priority_boost": 1.8,
        "confidence_threshold": 0.6
    },
    
    # =========================================================================
    # REGISTRO Y INFORMALIDAD
    # =========================================================================
    
    "trabajo_no_registrado": {
        "keywords": [
            "trabajo negro", "no registrado", "sin aportes", "en negro",
            "clandestino", "informalidad", "blanqueo", "regularización",
            "AFIP", "sin recibo sueldo", "sin obra social"
        ],
        
        "critical_articles": ["8", "9", "10", "11", "Ley 24.013"],
        
        "concepts": [
            "presunción relación laboral", "fraude laboral", 
            "solidaridad responsables", "indemnización agravada 25%"
        ],
        
        "priority_boost": 2.0,
        "confidence_threshold": 0.8
    },
    
    # =========================================================================
    # ACCIDENTES Y SEGURIDAD
    # =========================================================================
    
    "accidente_trabajo": {
        "keywords": [
            "accidente trabajo", "me lastimé trabajando", "lesión laboral",
            "accidente in itinere", "camino trabajo", "ART", "aseguradora",
            "incapacidad", "enfermedad profesional", "riesgos trabajo"
        ],
        
        "critical_articles": ["75", "Ley 24.557"],
        
        "concepts": [
            "accidente laboral", "accidente in itinere", "enfermedad profesional",
            "incapacidad laboral temporaria", "prestaciones ART"
        ],
        
        "priority_boost": 2.0,
        "confidence_threshold": 0.7
    },
    
    # =========================================================================
    # LIBERTAD SINDICAL
    # =========================================================================
    
    "libertad_sindical": {
        "keywords": [
            "libertad sindical", "afiliación sindical", "actividad gremial",
            "delegado sindical", "representante gremial", "sindicato",
            "discriminación sindical", "persecución gremial", "fuero sindical"
        ],
        
        "critical_articles": ["Ley 23.551", "14 bis CN"],
        
        "concepts": [
            "libertad afiliación", "actividad sindical protegida", 
            "representación gremial", "no discriminación sindical", "tutela sindical"
        ],
        
        "priority_boost": 2.2,
        "confidence_threshold": 0.7
    }
}

# =============================================================================
# CASOS POR CATEGORÍA (PARA NAVEGACIÓN)
# =============================================================================

CASES_BY_CATEGORY = {
    "extincion_contractual": [
        "embarazo_despido", "despido_sin_causa", "despido_con_causa"
    ],
    "discriminacion_acoso": [
        "discriminacion_general", "acoso_laboral"
    ],
    "remuneraciones": [
        "falta_pago_salarios", "horas_extras"
    ],
    "registro_formalidad": [
        "trabajo_no_registrado"
    ],
    "seguridad_salud": [
        "accidente_trabajo"
    ],
    "derechos_sindicales": [
        "libertad_sindical"
    ]
}

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def get_case_by_name(case_name: str) -> Dict[str, Any]:
    """Obtiene un caso específico por nombre"""
    return {
        "total_cases": total_cases,
        "unique_articles": len(all_articles),
        "unique_keywords": len(all_keywords),
        "avg_articles_per_case": round(avg_articles_per_case, 1),
        "avg_keywords_per_case": round(avg_keywords_per_case, 1),
        "categories": len(CASES_BY_CATEGORY),
        "articles_list": sorted(all_articles)
    }

# =============================================================================
# VALIDACIÓN Y EXPORTACIONES
# =============================================================================

# Ejecutar validación al importar
_validation_issues = validate_cases_structure()
if _validation_issues:
    print("⚠️ ADVERTENCIAS EN CASOS LABORALES:")
    for issue in _validation_issues:
        print(f"  - {issue}")

# Exportaciones principales
__all__ = [
    'LABOR_SPECIFIC_CASES',
    'CASES_BY_CATEGORY', 
    'get_case_by_name',
    'get_cases_by_category',
    'get_all_critical_articles',
    'find_cases_by_keyword',
    'get_cases_statistics'
]

if __name__ == "__main__":
    # Ejecutar cuando se llama directamente para debugging
    print("👷 CASOS LABORALES ESPECÍFICOS")
    print("=" * 50)
    
    stats = get_cases_statistics()
    print(f"📊 ESTADÍSTICAS:")
    for key, value in stats.items():
        if key != "articles_list":
            print(f"  {key}: {value}")
    
    print(f"\n📋 CASOS POR CATEGORÍA:")
    for category, cases in CASES_BY_CATEGORY.items():
        print(f"  {category}: {len(cases)} casos")
    
    print(f"\n🔍 VALIDACIÓN:")
    issues = validate_cases_structure()
    if issues:
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ Todos los casos tienen estructura válida") LABOR_SPECIFIC_CASES.get(case_name, {})

def get_cases_by_category(category: str) -> List[Dict[str, Any]]:
    """Obtiene todos los casos de una categoría"""
    case_names = CASES_BY_CATEGORY.get(category, [])
    return [LABOR_SPECIFIC_CASES[name] for name in case_names if name in LABOR_SPECIFIC_CASES]

def get_all_critical_articles() -> List[str]:
    """Obtiene todos los artículos críticos únicos"""
    articles = set()
    for case in LABOR_SPECIFIC_CASES.values():
        articles.update(case.get("critical_articles", []))
    return sorted(articles)

def find_cases_by_keyword(keyword: str) -> List[str]:
    """Encuentra casos que contienen una keyword específica"""
    matching_cases = []
    keyword_lower = keyword.lower()
    
    for case_name, case_data in LABOR_SPECIFIC_CASES.items():
        keywords = case_data.get("keywords", [])
        if any(keyword_lower in kw.lower() for kw in keywords):
            matching_cases.append(case_name)
    
    return matching_cases

def validate_cases_structure():
    """Valida que todos los casos tengan estructura correcta"""
    required_fields = ["keywords", "critical_articles", "concepts", "priority_boost", "confidence_threshold"]
    issues = []
    
    for case_name, case_data in LABOR_SPECIFIC_CASES.items():
        for field in required_fields:
            if field not in case_data:
                issues.append(f"Caso '{case_name}': Falta campo '{field}'")
            elif field in ["keywords", "critical_articles", "concepts"] and not case_data[field]:
                issues.append(f"Caso '{case_name}': Campo '{field}' está vacío")
    
    return issues

def get_cases_statistics():
    """Obtiene estadísticas de los casos definidos"""
    total_cases = len(LABOR_SPECIFIC_CASES)
    all_articles = set()
    all_keywords = set()
    
    for case in LABOR_SPECIFIC_CASES.values():
        all_articles.update(case.get("critical_articles", []))
        all_keywords.update(case.get("keywords", []))
    
    avg_articles_per_case = len(all_articles) / total_cases if total_cases > 0 else 0
    avg_keywords_per_case = len(all_keywords) / total_cases if total_cases > 0 else 0
    
    return