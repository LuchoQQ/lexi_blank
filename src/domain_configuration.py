"""
Configuraciones de Dominios Legales
===================================

Archivo: src/domain_configurations.py

Contiene las configuraciones base para cada dominio legal.
Separado del código principal para facilitar mantenimiento y expansión.

ESTRUCTURA:
- Keywords de identificación
- Leyes aplicables  
- Keywords de exclusión
- Patrones de intersección entre dominios
"""

from typing import Dict, List, Any
from enum import Enum

class LegalDomain(Enum):
    """Dominios legales soportados"""
    LABOR = "LABORAL"
    CIVIL = "CIVIL" 
    COMMERCIAL = "COMERCIAL"
    CRIMINAL = "PENAL"

# =============================================================================
# CONFIGURACIONES BASE POR DOMINIO
# =============================================================================

DOMAIN_BASE_CONFIG = {
    
    # =========================================================================
    # DOMINIO LABORAL
    # =========================================================================
    LegalDomain.LABOR: {
        "keywords": [
            # Relación laboral básica
            "trabajador", "empleador", "empleado", "patrón", "jefe", "supervisor",
            "contrato trabajo", "relación dependencia", "empleo", "puesto", "cargo",
            
            # Remuneración y beneficios
            "sueldo", "salario", "remuneración", "aguinaldo", "vacaciones", "licencia",
            "horas extras", "recibo sueldo", "liquidación", "finiquito",
            
            # Jornada y horarios
            "jornada", "horario", "descanso", "feriado", "turno", "franco",
            "sobretiempo", "descanso semanal",
            
            # Extinción laboral
            "despido", "renuncia", "cesantía", "indemnización", "preaviso",
            "estabilidad laboral", "fuero", "reinstalación",
            
            # Protecciones especiales
            "embarazo", "maternidad", "paternidad", "lactancia", "sindical", "gremial",
            "delegado", "discapacidad", "menor de edad",
            
            # Conflictos laborales
            "discriminación laboral", "acoso laboral", "mobbing", "accidente trabajo",
            "enfermedad profesional", "seguridad laboral", "condiciones trabajo",
            
            # Registro y formalidad
            "trabajo negro", "no registrado", "sin aportes", "blanqueo",
            "obra social", "AFIP", "monotributo"
        ],
        
        "exclusion_keywords": [
            "compraventa", "sociedad comercial", "herencia", "testamento", 
            "delito", "robo", "estafa", "homicidio", "hurto", "prisión"
        ],
        
        "priority_laws": [
            "Ley de contrato de trabajo", "Ley de empleo", "Ley de riesgos del trabajo",
            "Ley sindical", "Convenios colectivos"
        ],
        
        "weight_multiplier": 2.5  # Multiplicador para scoring
    },
    
    # =========================================================================
    # DOMINIO CIVIL
    # =========================================================================
    LegalDomain.CIVIL: {
        "keywords": [
            # Contratos y obligaciones
            "contrato", "obligación", "acreedor", "deudor", "prestación", 
            "cumplimiento", "incumplimiento", "mora", "daños perjuicios",
            
            # Derechos reales
            "propiedad", "dominio", "posesión", "usufructo", "servidumbre", 
            "hipoteca", "prenda", "registro propiedad", "escritura", "título",
            
            # Familia y personas
            "matrimonio", "divorcio", "separación", "régimen patrimonial",
            "patria potestad", "tutela", "curatela", "alimentos", "tenencia",
            "adopción", "filiación",
            
            # Sucesiones
            "herencia", "testamento", "legado", "sucesión", "heredero", 
            "legatario", "porción legítima", "mejora", "desheredación",
            
            # Responsabilidad civil
            "responsabilidad civil", "culpa", "negligencia", "factor atribución",
            "nexo causal", "daño moral", "lucro cesante", "daño emergente",
            
            # Contratos específicos
            "compraventa", "locación", "comodato", "mutuo", "donación", 
            "mandato", "depósito", "fianza", "renta vitalicia"
        ],
        
        "exclusion_keywords": [
            "trabajador", "empleador", "despido", "salario", "delito", 
            "prisión", "sociedad comercial", "quiebra"
        ],
        
        "priority_laws": [
            "Codigo Civil y Comercial", "Código Civil", "Leyes especiales civiles"
        ],
        
        "weight_multiplier": 2.0
    },
    
    # =========================================================================
    # DOMINIO COMERCIAL
    # =========================================================================
    LegalDomain.COMMERCIAL: {
        "keywords": [
            # Sociedades comerciales
            "sociedad", "socio", "accionista", "administrador", "gerente", 
            "directorio", "síndico", "capital social", "dividendo", "aportes",
            
            # Tipos societarios
            "SA", "SRL", "sociedad anónima", "sociedad responsabilidad limitada",
            "SAS", "sociedad comandita", "sociedad colectiva",
            
            # Operaciones societarias
            "fusión", "escisión", "transformación", "disolución societaria",
            "liquidación empresarial", "reducción capital", "aumento capital",
            
            # Crisis empresarial
            "quiebra", "concurso", "concordato", "cesación pagos", 
            "insolvencia", "verificación créditos", "masa concursal",
            
            # Títulos y documentos comerciales
            "cheque", "pagaré", "letra cambio", "factura", "remito",
            "vale", "warrant", "conocimiento embarque",
            
            # Propiedad intelectual e industrial
            "marca", "patente", "modelo utilidad", "diseño industrial",
            "derecho autor", "software", "competencia desleal",
            
            # Contratos comerciales
            "franquicia", "distribución", "concesión", "agencia", 
            "representación comercial", "joint venture", "fideicomiso",
            
            # Actividad comercial
            "comerciante", "fondo comercio", "empresa", "establecimiento",
            "clientela", "llave", "goodwill", "transferencia fondo"
        ],
        
        "exclusion_keywords": [
            "trabajador", "empleador", "despido", "salario", "matrimonio", 
            "divorcio", "delito", "homicidio"
        ],
        
        "priority_laws": [
            "Codigo Civil y Comercial", "Ley de sociedades comerciales", 
            "Ley de concursos y quiebras", "Ley de cheques", "Ley de marcas"
        ],
        
        "weight_multiplier": 2.0
    },
    
    # =========================================================================
    # DOMINIO PENAL
    # =========================================================================
    LegalDomain.CRIMINAL: {
        "keywords": [
            # Delitos contra las personas
            "homicidio", "asesinato", "lesiones", "amenaza", "coacción", 
            "privación libertad", "secuestro", "violación", "abuso sexual",
            
            # Delitos contra la propiedad
            "hurto", "robo", "estafa", "defraudación", "apropiación indebida", 
            "extorsión", "usurpación", "daño", "incendio",
            
            # Delitos contra la fe pública
            "falsificación", "documento falso", "testimonio falso", 
            "usurpación identidad", "moneda falsa",
            
            # Delitos contra la administración
            "cohecho", "soborno", "malversación", "prevaricato", 
            "abuso autoridad", "enriquecimiento ilícito",
            
            # Conceptos penales generales
            "delito", "crimen", "pena", "prisión", "reclusión", "multa", 
            "inhabilitación", "probation", "excarcelación",
            
            # Elementos del delito
            "dolo", "culpa", "tentativa", "complicidad", "instigación", 
            "encubrimiento", "legítima defensa", "estado necesidad",
            
            # Procedimiento penal
            "denuncia", "querella", "sumario", "indagatoria", "procesamiento",
            "sobreseimiento", "juicio oral", "veredicto", "sentencia",
            
            # Prescripción y otros
            "prescripción", "reincidencia", "concurso delitos", "unificación penas"
        ],
        
        "exclusion_keywords": [
            "contrato", "sociedad", "trabajador", "divorcio", "herencia", 
            "compraventa", "locación"
        ],
        
        "priority_laws": [
            "Codigo Penal", "Codigo Procesal Penal", "Ley de drogas", 
            "Ley de violencia género"
        ],
        
        "weight_multiplier": 3.0  # Mayor peso para temas penales por especificidad
    }
}

# =============================================================================
# PATRONES DE INTERSECCIÓN ENTRE DOMINIOS
# =============================================================================

DOMAIN_INTERSECTION_PATTERNS = {
    # Intersección Laboral-Penal
    (LegalDomain.LABOR, LegalDomain.CRIMINAL): {
        "keywords": [
            "acoso laboral", "violencia trabajo", "amenaza despido", 
            "discriminación", "lesión trabajo", "apropiación empleado",
            "estafa laboral", "documento falso trabajo"
        ],
        "description": "Casos donde el conflicto laboral incluye aspectos penales"
    },
    
    # Intersección Civil-Comercial
    (LegalDomain.CIVIL, LegalDomain.COMMERCIAL): {
        "keywords": [
            "contrato comercial", "responsabilidad civil empresa", 
            "daños sociedad", "incumplimiento comercial", "locación comercial",
            "transferencia fondo comercio", "competencia desleal"
        ],
        "description": "Aspectos civiles en relaciones comerciales"
    },
    
    # Intersección Civil-Penal
    (LegalDomain.CIVIL, LegalDomain.CRIMINAL): {
        "keywords": [
            "estafa", "defraudación", "daño doloso", "apropiación", 
            "responsabilidad civil delito", "restitución", "reparación daño"
        ],
        "description": "Responsabilidad civil derivada de delitos"
    },
    
    # Intersección Comercial-Penal
    (LegalDomain.COMMERCIAL, LegalDomain.CRIMINAL): {
        "keywords": [
            "estafa societaria", "vaciamiento empresa", "balances falsos",
            "administración fraudulenta", "quiebra fraudulenta", 
            "falsificación comercial", "competencia desleal ilícita"
        ],
        "description": "Delitos en el ámbito empresarial"
    },
    
    # Intersección Laboral-Civil
    (LegalDomain.LABOR, LegalDomain.CIVIL): {
        "keywords": [
            "accidente trabajo", "responsabilidad civil empleador",
            "daño moral laboral", "indemnización", "nexo causal trabajo"
        ],
        "description": "Responsabilidad civil en relaciones laborales"
    }
}

# =============================================================================
# INDICADORES DE COMPLEJIDAD
# =============================================================================

COMPLEXITY_INDICATORS = {
    "simple": {
        "score_threshold": 2,
        "characteristics": ["un dominio claro", "pocos conceptos legales", "consulta directa"]
    },
    
    "complex": {
        "score_threshold": 5,
        "characteristics": ["múltiples dominios", "varios conceptos", "menciona múltiples leyes"]
    },
    
    "multi_jurisdictional": {
        "score_threshold": float('inf'),
        "characteristics": ["tres o más dominios", "aspectos procedimentales", "referencias cruzadas"]
    }
}

# =============================================================================
# PATRONES LINGUÍSTICOS
# =============================================================================

LINGUISTIC_PATTERNS = {
    "complexity_indicators": [
        "y además", "también", "pero", "sin embargo", "por otro lado",
        "al mismo tiempo", "simultáneamente", "conjuntamente",
        "tanto civil como", "no solo", "sino también"
    ],
    
    "multi_law_indicators": [
        "código civil", "código penal", "ley de", "decreto", "resolución"
    ],
    
    "procedural_indicators": [
        "juicio", "demanda", "recurso", "apelación", "casación", "amparo"
    ],
    
    "urgency_indicators": [
        "urgente", "inmediato", "ya", "rápido", "pronto", "cuanto antes"
    ]
}

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def get_domain_config(domain: LegalDomain) -> Dict[str, Any]:
    """Obtiene la configuración de un dominio específico"""
    return DOMAIN_BASE_CONFIG.get(domain, {})

def get_all_domains() -> List[LegalDomain]:
    """Obtiene lista de todos los dominios disponibles"""
    return list(DOMAIN_BASE_CONFIG.keys())

def get_intersection_patterns(domain1: LegalDomain, domain2: LegalDomain) -> Dict[str, Any]:
    """Obtiene patrones de intersección entre dos dominios"""
    return DOMAIN_INTERSECTION_PATTERNS.get((domain1, domain2), 
                                           DOMAIN_INTERSECTION_PATTERNS.get((domain2, domain1), {}))

def validate_domain_config():
    """Valida que todas las configuraciones de dominio estén completas"""
    required_fields = ["keywords", "exclusion_keywords", "priority_laws", "weight_multiplier"]
    issues = []
    
    for domain, config in DOMAIN_BASE_CONFIG.items():
        for field in required_fields:
            if field not in config:
                issues.append(f"Dominio {domain.value}: Falta campo '{field}'")
            elif field != "weight_multiplier" and not config[field]:
                issues.append(f"Dominio {domain.value}: Campo '{field}' está vacío")
    
    return issues

# Ejecutar validación al importar
_validation_issues = validate_domain_config()
if _validation_issues:
    print("⚠️ ADVERTENCIAS EN CONFIGURACIÓN DE DOMINIOS:")
    for issue in _validation_issues:
        print(f"  - {issue}")

# =============================================================================
# ESTADÍSTICAS Y MÉTRICAS
# =============================================================================

def get_domain_stats():
    """Obtiene estadísticas de las configuraciones de dominio"""
    stats = {}
    
    for domain, config in DOMAIN_BASE_CONFIG.items():
        stats[domain.value] = {
            "keywords_count": len(config.get("keywords", [])),
            "exclusion_keywords_count": len(config.get("exclusion_keywords", [])),
            "priority_laws_count": len(config.get("priority_laws", [])),
            "weight_multiplier": config.get("weight_multiplier", 1.0)
        }
    
    # Estadísticas globales
    total_keywords = sum(len(config.get("keywords", [])) for config in DOMAIN_BASE_CONFIG.values())
    total_intersections = len(DOMAIN_INTERSECTION_PATTERNS)
    
    stats["GLOBAL"] = {
        "total_domains": len(DOMAIN_BASE_CONFIG),
        "total_keywords": total_keywords,
        "total_intersections": total_intersections,
        "avg_keywords_per_domain": total_keywords / len(DOMAIN_BASE_CONFIG) if DOMAIN_BASE_CONFIG else 0
    }
    
    return stats

if __name__ == "__main__":
    # Ejecutar cuando se llama directamente para debugging
    print("🔧 CONFIGURACIONES DE DOMINIOS LEGALES")
    print("=" * 50)
    
    stats = get_domain_stats()
    for domain, data in stats.items():
        print(f"{domain}: {data}")
    
    print("\n🔍 VALIDACIÓN:")
    issues = validate_domain_config()
    if issues:
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ Todas las configuraciones son válidas")