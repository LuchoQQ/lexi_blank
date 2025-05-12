"""
Module containing definitions of legal domains and their related keywords.
This centralizes domain knowledge for better maintainability and provides
specialized functions for Argentine labor law domain analysis.
"""
from typing import List, Dict, Any, Set, Tuple
import re

# Dictionary mapping legal domains to their related keywords
LEGAL_DOMAINS = {
    "Embarazo": [
        "embarazo", "embarazada", "maternidad", "maternal", "gestación", "gestante", 
        "lactancia", "parto", "nacimiento", "licencia", "fuero maternal", "preparto",
        "postparto", "obstetricia", "prenatal", "postnatal", "guardería", "cuna",
        "aborto", "gravidez", "paternidad", "ginecología", "asignación", "maternal",
        "cuidado del recién nacido", "día femenino", "periodo menstrual", "fertilización asistida",
        "adopción", "excedencia", "reducción de jornada por cuidado"
    ],
    "Despido": [
        "despido", "despedido", "desvinculación", "cesantía", "preaviso", "indemnización",
        "antigüedad", "extinción", "contrato", "rescisión", "telegrama", "carta documento",
        "causa justa", "sin causa", "notificación", "aviso", "ruptura", "liquidación final",
        "finiquito", "renuncia", "telegrama laboral", "período de prueba", "certificados",
        "antigüedad", "reincorporación", "abandono de trabajo", "discriminación", "cese",
        "suspensión", "denuncia", "homologación", "acuerdo", "desistimiento", "improcedente",
        "nulo", "certificado de trabajo", "liquidación final", "proporcional", "telegrama colacionado",
        "despido directo", "despido indirecto", "abuso de derecho", "injuria", "prejuicio"
    ],
    "Discriminación": [
        "discriminación", "discriminatorio", "igualdad", "desigualdad", "trato", 
        "género", "sexo", "religión", "raza", "orientación", "discapacidad", "edad",
        "apariencia", "identidad", "transgénero", "ideología", "política", "sindical",
        "nacionalidad", "origen étnico", "extranjero", "inmigrante", "condición social",
        "estado civil", "embarazo", "maternidad", "equidad", "lesión", "violencia",
        "diferencia salarial", "brecha", "puestos directivos", "techo de cristal",
        "representación sindical", "acción afirmativa", "igualdad de oportunidades",
        "inclusión laboral", "diversidad", "prácticas discriminatorias", "mobbing"
    ],
    "Acoso": [
        "acoso", "hostigamiento", "maltrato", "violencia", "dignidad", "integridad",
        "moral", "psicológico", "sexual", "intimidación", "abuso", "mobbing", "bullying",
        "persecución", "agresor", "víctima", "denuncia", "sexting", "grooming", "presión",
        "humillación", "agresión", "ciberbullying", "ambiente hostil", "agresivo",
        "maltrato", "insulto", "difamación", "calumnia", "comentarios inapropiados",
        "tocamientos", "insinuaciones", "proposiciones", "indecentes", "riesgo psicosocial",
        "comportamiento ofensivo", "lascivo", "indecoroso", "perturbador", "testimonios",
        "pruebas", "protocolo de actuación", "prevención", "consentimiento", "superior jerárquico"
    ],
    "Remuneración": [
        "remuneración", "salario", "sueldo", "pago", "aguinaldo", "vacaciones",
        "compensación", "bonificación", "horas extras", "plus", "feriado", "comisión",
        "propina", "viáticos", "vales", "tickets", "beneficios", "gratificación",
        "liquidación", "aumento", "bruto", "neto", "retención", "descuento", "jubilación",
        "obra social", "seguridad social", "impuesto", "ganancias", "recibo de sueldo",
        "depósito", "transferencia", "pago en efectivo", "categoría", "escala salarial",
        "convenio colectivo", "paritarias", "ajuste", "inflación", "bono", "premio",
        "retribución variable", "productividad", "presentismo", "antigüedad", "asignación",
        "SAC", "sueldo anual complementario", "complemento", "retenciones", "pago en negro",
        "monotributo", "autónomo", "régimen", "mínimo vital y móvil", "pago insuficiente"
    ],
    "Jornada": [
        "jornada", "horario", "descanso", "horas", "nocturno", "diurno", "semanal",
        "turnos", "feriado", "vacaciones", "disponibilidad", "rotativo", "cambio",
        "tiempo completo", "medio tiempo", "parcial", "francos", "licencia", "pausa",
        "trabajo efectivo", "fichaje", "fichado", "control horario", "guardia",
        "turnos rotativos", "trabajo remoto", "teletrabajo", "home office", "híbrido",
        "compensación horaria", "banco de horas", "tiempo extra", "sobretiempo",
        "horario reducido", "flexible", "conciliación", "descanso semanal", "fines de semana",
        "turnos nocturnos", "desconexión digital", "pausas", "tolerancia", "entrada",
        "salida", "ausencia", "tardanza", "permiso", "justificación", "marcaciones",
        "horas ordinarias", "calendario laboral", "jordana insalubre", "enfermedad",
        "restricciones horarias", "horas acumuladas", "intensiva", "partida", "adaptación"
    ],
    "Accidentes": [
        "accidente", "enfermedad", "profesional", "riesgo", "seguridad", "incapacidad",
        "laboral", "trabajo", "indemnización", "médico", "prevención", "ART",
        "aseguradora", "alta", "baja", "reposo", "recuperación", "tratamiento",
        "rehabilitación", "secuelas", "incapacidad", "lesión", "crónico", "permanente",
        "transitorio", "parcial", "total", "gran invalidez", "recalificación", "cobertura",
        "denuncia", "accidente in itinere", "trayecto", "comisión médica", "junta médica",
        "peritaje", "dictamen", "riesgos del trabajo", "prevención", "elementos de protección",
        "EPP", "procedimiento", "protocolo", "emergencia", "primeros auxilios", "investigación",
        "accidente fatal", "enfermedades inculpables", "licencia médica", "parte médico"
    ],
    "Licencias": [
        "licencia", "permiso", "ausencia", "descanso", "vacaciones", "enfermedad", 
        "familiar", "estudio", "examen", "matrimonio", "nacimiento", "adopción",
        "fallecimiento", "mudanza", "donación de sangre", "trámites", "médico", 
        "tratamiento", "duelo", "cuidado", "familiar enfermo", "incapacidad", 
        "especial", "extraordinaria", "sin goce de sueldo", "con goce de sueldo",
        "reducción de jornada", "médica", "psiquiátrica", "psicológica", "excedencia",
        "paternidad", "maternidad", "violencia de género", "adaptación escolar",
        "licencia sindical", "ejercicio político", "feriados", "días festivos", 
        "asueto", "compensatorios", "lactancia", "exámenes", "cargos públicos",
        "trámites prenatales", "días de estudio", "técnicas de reproducción"
    ],
    "Salud_Laboral": [
        "salud", "higiene", "seguridad", "riesgo", "prevención", "ergonomía", 
        "psicosocial", "evaluación", "protocolo", "comité", "delegado", "inspector",
        "instalaciones", "protección", "capacitación", "auditoría", "procedimiento",
        "mediciones", "informe", "elementos de protección personal", "EPP", "ART",
        "condiciones", "ambiente", "manipulación", "sustancias", "ruido", "iluminación",
        "ventilación", "temperatura", "vibraciones", "radiaciones", "contaminantes",
        "productos químicos", "tóxicos", "microclima", "carga física", "carga mental",
        "estrés", "burnout", "fatiga", "sobrecarga", "reconocimiento médico", "vigilancia",
        "apto", "plan de emergencia", "evacuación", "simulacro", "primeros auxilios",
        "extintores", "señalización", "riesgo eléctrico", "caídas", "lesiones", "CYMAT"
    ],
    "Contratación": [
        "contrato", "contratación", "relación laboral", "empleo", "alta", "registro",
        "periodo de prueba", "temporal", "indefinido", "plazo fijo", "tiempo determinado",
        "eventual", "obra", "servicio", "pasantía", "práctica", "formación", "aprendizaje",
        "monotributo", "autónomo", "independiente", "freelance", "tercerización", "outsourcing",
        "subcontratación", "falso autónomo", "fraude laboral", "AFIP", "obra social",
        "seguridad social", "aportes", "contribuciones", "régimen", "categoría", "escalafón",
        "grupo profesional", "función", "convenio colectivo", "condiciones", "cláusulas",
        "pacto", "acuerdo", "contrato verbal", "nulidad", "rescisión", "modificación",
        "prórroga", "renovación", "temporario", "estacional", "contrato de temporada"
    ],
    "Sindicatos": [
        "sindicato", "sindical", "gremio", "asociación", "representación", "delegado",
        "comisión interna", "afiliación", "cuota", "huelga", "paro", "asamblea", "conflicto",
        "conciliación", "negociación", "convenio colectivo", "paritarias", "acuerdo",
        "federación", "confederación", "central", "libertad sindical", "personería gremial",
        "inscripción", "tutela", "fuero sindical", "práctica desleal", "sanción", "desafiliación",
        "elecciones", "mandato", "representatividad", "mayoría", "minoría", "seccional",
        "delegación", "estatuto", "CGT", "CTA", "descuentos sindicales", "obra social sindical",
        "actividad gremial", "derechos colectivos", "trabajadores", "medidas de fuerza",
        "conciliación obligatoria", "laudo", "dictamen"
    ],
    "Trabajo_Remoto": [
        "teletrabajo", "remoto", "virtual", "a distancia", "home office", "híbrido",
        "domicilio", "conectividad", "internet", "equipo", "herramientas", "compensación",
        "gastos", "servicios", "electricidad", "comunicación", "disponibilidad", "horario",
        "desconexión", "digital", "presencial", "alternancia", "equipamiento", "ergonomía",
        "silla", "escritorio", "pantalla", "computadora", "notebook", "tecnología",
        "software", "plataforma", "reunión virtual", "videollamada", "supervisión",
        "control", "metas", "objetivos", "productividad", "evaluación", "rendimiento",
        "registro", "voluntario", "reversibilidad", "acuerdo", "adenda", "ley de teletrabajo",
        "regulación", "modalidad", "compensación de gastos", "viáticos", "conectividad"
    ],
    "Prestaciones": [
        "prestación", "beneficio", "ayuda", "subsidio", "asignación", "seguro",
        "cobertura", "jubilación", "pensión", "retiro", "vejez", "invalidez", "muerte",
        "supervivencia", "familiar", "hijo", "escolaridad", "matrimonio", "nacimiento",
        "adopción", "ANSES", "aportes", "contribuciones", "cotización", "sistema", "régimen",
        "fondo", "haberes", "moratoria", "reconocimiento", "servicios", "años", "edad",
        "declaración jurada", "formularios", "solicitud", "prestaciones no contributivas",
        "certificación", "comprobante", "reajuste", "actualización", "movilidad", "mínima",
        "máxima", "régimen general", "especial", "autónomos", "monotributistas", "SIPA",
        "IPS", "PAMI", "obra social", "salud", "urgencias", "emergencias", "tratamientos"
    ],
    "Retribuciones_Especiales": [
        "comisión", "comisiones", "porcentaje", "venta", "producción", "objetivo",
        "bono", "premio", "incentivo", "gratificación", "complemento", "extra", "adicional",
        "propina", "viático", "gastos", "representación", "movilidad", "traslado", "vehículo",
        "combustible", "peaje", "kilometraje", "transporte", "tarjeta", "beneficio", "corporativo",
        "variable", "fijo", "mensual", "trimestral", "anual", "cumplimiento", "meta", "objetivo",
        "productividad", "presentismo", "puntualidad", "asistencia", "título", "idioma",
        "antigüedad", "permanencia", "fidelidad", "stock options", "participación",
        "beneficios", "utilidades", "resultados", "retribución flexible", "plan de compensación"
    ],
    "Capacitación": [
        "capacitación", "formación", "entrenamiento", "curso", "taller", "seminario",
        "congreso", "conferencia", "educación", "desarrollo", "aprendizaje", "habilidad",
        "competencia", "certificación", "título", "diploma", "crédito", "hora", "evaluación",
        "examen", "práctica", "beca", "subsidio", "financiamiento", "convenio", "institución",
        "universidad", "terciario", "profesional", "técnico", "especialización", "postgrado",
        "master", "doctorado", "actualización", "reconversión", "reinserción", "carrera",
        "plan", "programa", "itinerario", "obligatoria", "voluntaria", "interna", "externa",
        "presencial", "virtual", "a distancia", "mixta", "tiempo de trabajo", "fuera de horario"
    ],
    "Jubilación": [
        "jubilación", "jubilar", "retiro", "pension", "vejez", "años de servicio", 
        "ANSES", "SIPA", "aporte", "contribución", "moratoria", "régimen", "sistema",
        "prestación", "beneficio", "haber", "edad", "antigüedad", "servicio", "cotización",
        "caja", "fondo", "capitalización", "reparto", "invalidez", "incapacidad", "muerte",
        "supervivencia", "viudez", "reconocimiento", "cómputo", "simulador", "plan", 
        "prejubilación", "retiro anticipado", "retiro voluntario", "años de aporte", 
        "certificación", "historia laboral", "mínima", "máxima", "reajuste", "movilidad",
        "compatibilidad", "incompatibilidad", "reingreso", "continuidad", "servicios",
        "proporcional", "jubilación parcial", "reducción de jornada", "PUAM", "PBU"
    ],
    "Modalidades_Contractuales": [
        "contrato", "plazo fijo", "obra", "eventual", "tiempo parcial", "a tiempo completo", 
        "por temporada", "permanente", "discontinuo", "por equipo", "grupo", "pasantía",
        "beca", "aprendizaje", "formación", "práctica profesional", "periodo de prueba",
        "prueba", "temporario", "interino", "suplencia", "término", "duración determinada",
        "indeterminada", "autónomo", "freelance", "consultor", "asesor", "prestación",
        "servicio", "locación", "obra", "proyecto", "campaña", "zafra", "cosecha",
        "estacional", "eventual", "a demanda", "jornalero", "relevo", "sustitución",
        "sucesión", "contrato en cadena", "puesto vacante", "reemplazo", "contrato a prueba"
    ],
    "Procesos_Administrativos": [
        "proceso", "administrativo", "procedimiento", "expediente", "trámite", "formulario",
        "solicitud", "presentación", "SECLO", "ministerio", "trabajo", "delegación", "inspección",
        "multa", "sanción", "clausura", "audiencia", "conciliación", "mediación", "negociación",
        "acta", "acuerdo", "homologación", "ratificación", "notificación", "recurso", "apelación",
        "agotamiento", "vía", "reclamación", "previa", "denuncia", "verificación", "citación",
        "comparecencia", "vista", "acceso", "documento", "copias", "digitalización", "firma",
        "certificación", "registro", "inscripción", "baja", "alta", "modificación", "plazo",
        "término", "feria", "inhábil", "prórroga", "vencimiento", "caducidad", "dictamen"
    ],
    "Procesos_Judiciales": [
        "proceso", "judicial", "juicio", "demanda", "juzgado", "tribunal", "juez",
        "laboral", "fuero", "competencia", "jurisdicción", "actor", "demandado", "parte",
        "contestación", "reconvención", "audiencia", "preliminar", "vista", "prueba",
        "testimonial", "documental", "pericial", "informativa", "confesional", "inspección",
        "alegato", "sentencia", "apelación", "cámara", "casación", "corte", "suprema",
        "costas", "honorarios", "perito", "abogado", "procurador", "representante", "poder",
        "legitimación", "personería", "medida", "cautelar", "embargo", "preventivo", "ejecutivo",
        "ejecución", "mandamiento", "notificación", "cédula", "oficio", "exhorto", "edicto",
        "rebeldía", "feria", "plazo", "término", "prescripción", "caducidad"
    ]
}

# Key laws and regulations in Argentine labor law with their common names/references
ARGENTINE_LABOR_LAWS = {
    "LCT": {
        "full_name": "Ley de Contrato de Trabajo",
        "number": "20.744",
        "description": "Norma principal que regula las relaciones laborales en Argentina",
        "key_topics": ["Contrato de Trabajo", "Remuneración", "Jornada", "Despido", "Licencias", "Suspensiones"]
    },
    "LEY_EMPLEO": {
        "full_name": "Ley Nacional de Empleo",
        "number": "24.013",
        "description": "Regula el Sistema Único de Registro Laboral y establece indemnizaciones adicionales",
        "key_topics": ["Registro", "Indemnizaciones", "Trabajo no registrado", "Multas", "Regularización"]
    },
    "LRT": {
        "full_name": "Ley de Riesgos del Trabajo",
        "number": "24.557",
        "description": "Establece el sistema de prevención y reparación de accidentes de trabajo",
        "key_topics": ["ART", "Accidentes", "Enfermedades profesionales", "Incapacidad", "Prestaciones"]
    },
    "LEY_ASOCIACIONES_SINDICALES": {
        "full_name": "Ley de Asociaciones Sindicales",
        "number": "23.551",
        "description": "Regula la constitución y funcionamiento de las asociaciones sindicales",
        "key_topics": ["Sindicatos", "Representación", "Personería", "Tutela", "Convenciones Colectivas"]
    },
    "LEY_TELETRABAJO": {
        "full_name": "Régimen Legal del Contrato de Teletrabajo",
        "number": "27.555",
        "description": "Establece los presupuestos legales mínimos para el teletrabajo",
        "key_topics": ["Teletrabajo", "Remoto", "Conectividad", "Desconexión", "Reversibilidad"]
    },
    "LEY_PYMES": {
        "full_name": "Ley de Pequeñas y Medianas Empresas",
        "number": "24.467",
        "description": "Contiene disposiciones específicas para relaciones laborales en PyMES",
        "key_topics": ["PYMES", "Convenios", "Indemnizaciones", "Flexibilización"]
    }
}

# Common jurisdictional and administrative entities in Argentine labor law
LABOR_ENTITIES = {
    "SECLO": {
        "full_name": "Servicio de Conciliación Laboral Obligatoria",
        "description": "Órgano administrativo de conciliación previa obligatoria en CABA",
        "function": "Conciliación previa obligatoria para demandas laborales en CABA"
    },
    "MTEySS": {
        "full_name": "Ministerio de Trabajo, Empleo y Seguridad Social",
        "description": "Organismo nacional encargado de las políticas laborales",
        "function": "Fiscalización, regulación y políticas públicas laborales"
    },
    "SRT": {
        "full_name": "Superintendencia de Riesgos del Trabajo",
        "description": "Organismo de control del sistema de riesgos laborales",
        "function": "Control y fiscalización de las ART y el sistema de riesgos del trabajo"
    },
    "ANSES": {
        "full_name": "Administración Nacional de la Seguridad Social",
        "description": "Organismo nacional de seguridad social",
        "function": "Administración de jubilaciones, pensiones y asignaciones"
    },
    "FUERO_LABORAL": {
        "full_name": "Justicia Nacional del Trabajo/Tribunales Laborales Provinciales",
        "description": "Tribunales especializados en materia laboral",
        "function": "Resolución judicial de conflictos laborales"
    }
}

# Important calculations in Argentine labor law
LABOR_CALCULATIONS = {
    "indemnizacion_despido": {
        "description": "Indemnización por antigüedad (Art. 245 LCT)",
        "formula": "Mejor remuneración mensual normal y habitual × Años de servicio (o fracción mayor a 3 meses)",
        "components": ["mejor_remuneracion", "antiguedad"],
        "ejemplo": "Empleado con 5 años de antigüedad y mejor sueldo de $100.000: $100.000 × 5 = $500.000"
    },
    "preaviso": {
        "description": "Indemnización por falta de preaviso (Art. 232 LCT)",
        "formula": "Un mes de sueldo (o dos si la antigüedad supera los 5 años)",
        "components": ["remuneracion_mensual", "antiguedad"],
        "ejemplo": "Empleado con 3 años: un mes de sueldo. Con 7 años: dos meses de sueldo."
    },
    "integracion_mes_despido": {
        "description": "Integración mes de despido (Art. 233 LCT)",
        "formula": "Salario diario × días faltantes hasta fin de mes",
        "components": ["salario_diario", "dias_pendientes"],
        "ejemplo": "Despido el día 10 del mes: se pagan los 20 días restantes."
    },
    "sac_proporcional": {
        "description": "SAC proporcional (Aguinaldo - Art. 123 LCT)",
        "formula": "Mejor remuneración del semestre × Días trabajados en el semestre / 180",
        "components": ["mejor_remuneracion_semestre", "dias_trabajados_semestre"],
        "ejemplo": "Si trabajó 90 días en el semestre: Mejor sueldo × 90/180 = 50% del aguinaldo"
    },
    "vacaciones_proporcionales": {
        "description": "Vacaciones no gozadas (Art. 156 LCT)",
        "formula": "Días de vacaciones × Días trabajados en el año / 365",
        "components": ["dias_vacaciones_corresponden", "dias_trabajados_anio"],
        "ejemplo": "Para 14 días de vacaciones, habiendo trabajado 180 días: 14 × 180/365 ≈ 7 días"
    }
}

def detect_domains_in_query(query: str) -> List[str]:
    """
    Detect legal domains present in a query.
    
    Args:
        query: User query text
        
    Returns:
        List of detected domain names
    """
    query_lower = query.lower()
    detected_domains = []
    
    # First, try direct keyword matching
    for domain, keywords in LEGAL_DOMAINS.items():
        for keyword in keywords:
            if keyword.lower() in query_lower:
                if domain not in detected_domains:
                    detected_domains.append(domain)
                break
    
    # If no domains detected, try more sophisticated analysis
    if not detected_domains:
        detected_domains = analyze_query_context(query)
    
    return detected_domains

def analyze_query_context(query: str) -> List[str]:
    """
    Perform more sophisticated analysis of the query to detect legal domains
    when direct keyword matching fails.
    
    Args:
        query: User query text
        
    Returns:
        List of inferred domain names
    """
    query_lower = query.lower()
    detected_domains = []
    
    # Check for temporal patterns (suggesting Jornada or Licencias)
    time_patterns = [
        r'\b\d+\s*(?:hora|horas|día|días|semana|semanas|mes|meses|año|años)\b',
        r'\b(?:lunes|martes|miércoles|jueves|viernes|sábado|domingo)\b',
        r'\b(?:mañana|tarde|noche|madrugada)\b'
    ]
    
    if any(re.search(pattern, query_lower) for pattern in time_patterns):
        if "Jornada" not in detected_domains:
            detected_domains.append("Jornada")
    
    # Check for monetary patterns (suggesting Remuneración)
    money_patterns = [
        r'\$\s*\d+',
        r'\b\d+\s*(?:peso|pesos|dólar|dólares|usd)\b',
        r'\b(?:pago|cobro|sueldo|plata|dinero|monto|importe)\b'
    ]
    
    if any(re.search(pattern, query_lower) for pattern in time_patterns):
        if "Remuneración" not in detected_domains:
            detected_domains.append("Remuneración")
    
    # Check for situation verbs that might indicate Despido
    dismissal_verbs = [
        r'\b(?:despidieron|echaron|desvincularon|cesaron|rescindieron|terminaron)\b',
        r'\b(?:me\s+(?:despidió|echó|desvinculó|cesó))\b',
        r'\b(?:fin|finalización|término|terminación)\s+(?:del|de la|de)\s+(?:contrato|relación|vínculo)\b'
    ]
    
    if any(re.search(pattern, query_lower) for pattern in dismissal_verbs):
        if "Despido" not in detected_domains:
            detected_domains.append("Despido")
    
    # Check for health-related terminology (suggesting Accidentes or Licencias)
    health_patterns = [
        r'\b(?:accidente|enfermedad|lesión|médico|doctor|hospital|clínica|diagnóstico|tratamiento)\b',
        r'\b(?:dolor|molestia|incapacidad|discapacidad|inhabilitación|recuperación)\b'
    ]
    
    if any(re.search(pattern, query_lower) for pattern in health_patterns):
        if "Accidentes" not in detected_domains:
            detected_domains.append("Accidentes")
        if "Licencias" not in detected_domains:
            detected_domains.append("Licencias")
    
    # Default to Laboral general domain if still empty
    if not detected_domains:
        detected_domains = ["Despido"]  # Default to most common query topic
    
    return detected_domains

def identify_legal_references(text: str) -> List[Dict[str, str]]:
    """
    Identifies references to specific legal articles and laws in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of dictionaries with identified law references
    """
    references = []
    
    # Common patterns for article references in Argentine legislation
    article_patterns = [
        r'(?:artículo|art\.?|artículos|arts\.?)\s+(\d+)(?:\s+(?:de la|del)\s+(?:ley|código|decreto)\s+(?:de|nº|n°|número)?\s*([a-zA-Z0-9\.\s]+))?',
        r'(?:ley|código|decreto)\s+(?:nº|n°|número)?\s*([a-zA-Z0-9\.\s]+)(?:\s+en su)?\s+(?:artículo|art\.?)\s+(\d+)',
        r'(?:ley|código|decreto)\s+([a-zA-Z0-9\.\s]+)(?:\s+en su)?\s+(?:artículo|art\.?)\s+(\d+)'
    ]
    
    for pattern in article_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match.groups()) >= 2:
                # Handle different patterns with different group arrangements
                if match.group(1).isdigit():
                    article_num = match.group(1)
                    law_ref = match.group(2) if len(match.groups()) > 1 and match.group(2) else "No especificada"
                else:
                    law_ref = match.group(1)
                    article_num = match.group(2)
                
                # Remove extra spaces and normalize
                law_ref = re.sub(r'\s+', ' ', law_ref).strip() if law_ref else "No especificada"
                
                # Look up in known laws dictionary for standardization
                standardized_law = None
                for law_code, law_info in ARGENTINE_LABOR_LAWS.items():
                    if (law_info["number"] in law_ref or 
                        law_info["full_name"].lower() in law_ref.lower()):
                        standardized_law = law_info["full_name"]
                        break
                
                references.append({
                    "law": standardized_law if standardized_law else law_ref,
                    "article": article_num,
                    "original_text": match.group(0)
                })
    
    return references

def extract_labor_entities(text: str) -> Dict[str, List[str]]:
    """
    Extracts references to labor-specific entities from text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with entity types and their occurrences
    """
    entities = {
        "laws": [],
        "organizations": [],
        "procedures": [],
        "timeframes": []
    }
    
    # Extract law references
    for law_code, law_info in ARGENTINE_LABOR_LAWS.items():
        if law_info["full_name"] in text or law_info["number"] in text or law_code in text.upper():
            if law_info["full_name"] not in entities["laws"]:
                entities["laws"].append(law_info["full_name"])
    
    # Extract organization references
    for org_code, org_info in LABOR_ENTITIES.items():
        if org_info["full_name"] in text or org_code in text.upper():
            if org_info["full_name"] not in entities["organizations"]:
                entities["organizations"].append(org_info["full_name"])
    
    # Extract procedural terms
    procedure_terms = [
        "audiencia", "conciliación", "mediación", "demanda", "juicio", 
        "reclamo", "recurso", "apelación", "telegrama", "carta documento",
        "SECLO", "homologación", "acuerdo", "indemnización", "preaviso"
    ]
    
    for term in procedure_terms:
        if term.lower() in text.lower():
            if term not in entities["procedures"]:
                entities["procedures"].append(term)
    
    # Extract timeframes
    timeframe_patterns = [
        r'\b(\d+)\s+(?:día|días|mes|meses|año|años)\b',
        r'\b(?:plazo|término)\s+de\s+(\d+)\s+(?:día|días|mes|meses|año|años)\b'
    ]
    
    for pattern in timeframe_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if match.group(0) not in entities["timeframes"]:
                entities["timeframes"].append(match.group(0))
    
    return entities

# Dictionary with subdomain relationships - maps from general domain to specific subdomains
DOMAIN_HIERARCHIES = {
    "Laboral": ["Despido", "Remuneración", "Jornada", "Licencias", "Contratación", 
                "Accidentes", "Modalidades_Contractuales", "Sindicatos", "Trabajo_Remoto"],
    "Despido": ["Indemnización", "Preaviso", "Causa_Justa", "Telegrama_Laboral", "Renuncia"],
    "Discriminación": ["Género", "Raza", "Religión", "Discapacidad", "Orientación_Sexual", "Edad", "Nacionalidad"],
    "Remuneración": ["Salario", "Horas_Extras", "Aguinaldo", "Bonificaciones", "Vacaciones", "Retribuciones_Especiales"],
    "Accidentes": ["ART", "Enfermedad_Profesional", "Accidente_In_Itinere", "Incapacidad", "Tratamiento"],
    "Jornada": ["Horario", "Descansos", "Horas_Extras", "Feriados", "Trabajo_Remoto"],
    "Salud_Laboral": ["Riesgos", "Prevención", "Elementos_Protección", "Condiciones_Trabajo", "Evaluación"],
    "Embarazo": ["Licencia_Maternidad", "Fuero_Maternal", "Lactancia", "Adaptación", "Discriminación_Gestante"]
}