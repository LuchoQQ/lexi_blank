"""
Asistente legal que utiliza el sistema de ranking especializado
y GPT-4-mini para generar respuestas contextualizadas a consultas legales.
"""
import os
import sys
import argparse
import time
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import openai

# Cargar variables de entorno
load_dotenv()

# Importar módulos del sistema
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config_loader import load_config
from main import search_query, check_connections

# Configuración de la API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o-mini"  # Utilizamos gpt-4o-mini como modelo por defecto
MAX_TOKENS = 4000  # Limitamos la longitud de la respuesta
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

def create_legal_prompt(query: str, results: List[Dict[str, Any]], max_articles: int = 15) -> str:
    """
    Crear un prompt estructurado con la consulta y los artículos más relevantes para GPT-4-mini.
    El prompt incluye instrucciones específicas para el contexto legal argentino.
    
    Args:
        query: Consulta del usuario
        results: Lista de resultados rankeados
        max_articles: Número máximo de artículos a incluir
        
    Returns:
        Prompt estructurado para enviar al modelo
    """
    # Limitar a los artículos más relevantes
    top_results = results[:max_articles]
    
    # Detectar dominios legales en la consulta
    from src.legal_domains import detect_domains_in_query, ARGENTINE_LABOR_LAWS
    domains = detect_domains_in_query(query)
    domain_str = ", ".join(domains) if domains else "general laboral"
    
    # Buscar términos específicos para extraer contexto adicional
    context_clues = {
        "embarazo": "embarazo",
        "antigüedad": r"\b(\d+)\s+(año|años|mes|meses)",
        "sueldo": r"\$\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s+(?:pesos|dolares)",
        "indemnización": "indemnización|indemnizaciones|liquidación|finiquito"
    }
    
    user_context = {}
    for context_key, pattern in context_clues.items():
        import re
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            if context_key == "antigüedad":
                # Extraer tiempo de antigüedad
                value = match.group(0)
                user_context["antigüedad"] = value
            elif context_key == "sueldo":
                # Extraer monto mencionado
                value = match.group(0)
                user_context["sueldo"] = value
            else:
                user_context[context_key] = True
    
    # Iniciar el prompt con la consulta y contexto específico
    prompt = f"""Eres un asistente legal especializado en derecho laboral argentino. Debes proporcionar asesoramiento legal preciso y detallado sobre la siguiente consulta utilizando únicamente la información presente en los artículos proporcionados:

CONSULTA DEL USUARIO: {query}

CONTEXTO DETECTADO:
- Dominio legal principal: {domain_str}
"""

    # Añadir contexto específico detectado si existe
    if user_context:
        prompt += "- Información específica detectada:\n"
        for key, value in user_context.items():
            if key == "antigüedad":
                prompt += f"  * Antigüedad mencionada: {value}\n"
            elif key == "sueldo":
                prompt += f"  * Monto salarial mencionado: {value}\n"
            elif key == "embarazo" and value:
                prompt += f"  * Situación de embarazo mencionada\n"
            elif key == "indemnización" and value:
                prompt += f"  * Consulta sobre indemnizaciones o liquidación final\n"

    prompt += """
NORMATIVA LABORAL ARGENTINA APLICABLE:
- Ley de Contrato de Trabajo (Ley 20.744)
- Ley Nacional de Empleo (Ley 24.013)
- Ley de Riesgos del Trabajo (Ley 24.557)
- Convenios Colectivos de Trabajo según la actividad
"""

    # Agregar cada artículo con su ley, número e información
    prompt += "\nA continuación se presentan los artículos legales más relevantes para tu consulta:\n\n"
    
    for i, result in enumerate(top_results, 1):
        law_name = result.get("law_name", "Ley no especificada")
        article_number = result.get("article_number", "")
        category = result.get("category", "")
        content = result.get("content", "").strip()
        relevance = result.get("score", 0)
        
        prompt += f"[ARTÍCULO {i}] {law_name}, Artículo {article_number} (Relevancia: {relevance:.2f})\n"
        prompt += f"Categoría: {category}\n"
        prompt += f"{content}\n\n"
    
    # Agregar instrucciones específicas para la generación de la respuesta con mejoras para contexto argentino
    prompt += """Basándote ÚNICAMENTE en los artículos proporcionados, construye una respuesta DETALLADA Y ESTRUCTURADA con las siguientes secciones:

1. ANÁLISIS DETALLADO: 
   - Explica cómo las leyes y artículos se aplican específicamente a la situación
   - Incluye TODOS los conceptos indemnizatorios que corresponden según la normativa:
     * Indemnización por antigüedad (Art. 245 LCT) = Mejor remuneración mensual × años de servicio
     * Preaviso (Art. 232 LCT) = 1 mes de sueldo (menor a 5 años) o 2 meses (mayor a 5 años)
     * Integración del mes de despido (Art. 233 LCT) = días faltantes hasta fin de mes
     * SAC proporcional = Mejor sueldo semestral × días trabajados / 180
     * Vacaciones proporcionales = Días vacaciones × días trabajados / 365
     * Indemnización Art. 2 Ley 25.323 (duplicación) si corresponde
     * Indemnización Art. 8 Ley 24.013 (empleo no registrado) si corresponde
     * Indemnización Art. 15 Ley 24.013 (registro parcial) si corresponde
   - Especifica EXACTAMENTE la base de cálculo para cada concepto
   - Incluye TODOS los montos, porcentajes y plazos exactos mencionados en los artículos
   - Calcula ejemplos numéricos concretos cuando sea posible (ej. "Si el trabajador ganaba $50,000...")
   
2. PROCEDIMIENTO RECOMENDADO EN ARGENTINA:
   - Enumera los pasos prácticos que debe seguir la persona, en orden cronológico
   - Menciona TODOS los plazos legales para cada acción (especifica días hábiles o corridos)
   - Explica el plazo de prescripción para realizar el reclamo (2 años en materia laboral)
   - Incluye instancias específicas argentinas:
     * Servicio de Conciliación Laboral Obligatoria (SECLO) en CABA
     * Delegaciones del Ministerio de Trabajo
     * Fuero Laboral (Juzgados Nacionales/Provinciales del Trabajo)
   - Detalla el procedimiento según la jurisdicción en Argentina
   
3. RECOMENDACIONES LEGALES:
   - Ofrece consejos sobre documentación necesaria según la legislación argentina
   - Especifica qué pruebas debería reunir el consultante (recibos, testigos, etc.)
   - Menciona alternativas de solución (ej. acuerdo, audiencia SECLO, juicio)
   - Advierte sobre posibles defensas del empleador en el contexto argentino
   - Advierte sobre posibles particularidades de convenios colectivos

IMPORTANTE:
- Utiliza SOLAMENTE la información de los artículos proporcionados
- Cita los artículos específicos para cada afirmación legal que hagas (Por ejemplo: "Según el Art. 231 de la LCT...")
- INCLUYE TODOS LOS PLAZOS Y VALORES NUMÉRICOS mencionados en los artículos (días, meses, años, montos)
- MENCIONA INSTITUCIONES ESPECÍFICAS ARGENTINAS cuando corresponda (SECLO, Ministerio de Trabajo, etc.)
- PROPORCIONA EJEMPLOS NUMÉRICOS para ilustrar cálculos cuando sea posible
- EXPLICA el cálculo de TODAS las indemnizaciones que corresponden legalmente 
- Si la información proporcionada es insuficiente para algún aspecto, indícalo claramente
- Usa un lenguaje claro y accesible, evitando jerga legal excesiva
- No inventes información que no esté respaldada por los artículos proporcionados

SITUACIONES ESPECÍFICAS:

PARA CASOS DE DESPIDO:
- Aclara la diferencia entre despido con justa causa (sin indemnización) y sin justa causa (con indemnización)
- Explica los requisitos del preaviso y su indemnización sustitutiva
- Detalla la forma válida de comunicar y recibir un despido en Argentina (telegrama laboral/carta documento)

PARA CASOS DE LICENCIAS:
- Especifica los días correspondientes según la legislación argentina
- Aclara si la licencia es con o sin goce de sueldo
- Detalla los requisitos de aviso previo y documentación

PARA CASOS DE ACCIDENTES DE TRABAJO:
- Explica el procedimiento de denuncia ante la ART
- Detalla las prestaciones que corresponden (médicas y económicas)
- Informa sobre los plazos de reclamación y vías administrativas

PARA CASOS DE REMUNERACIÓN:
- Diferencia conceptos remunerativos y no remunerativos
- Explica plazos de pago según normativa argentina (4° día hábil)
- Detalla vías de reclamo por falta/retraso de pago

Tu respuesta:
"""
    
    return prompt

def call_gpt_model(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = MAX_TOKENS) -> Optional[str]:
    """
    Llamar a la API de OpenAI para obtener una respuesta.
    
    Args:
        prompt: Prompt estructurado
        model: Modelo a utilizar
        max_tokens: Número máximo de tokens para la respuesta
        
    Returns:
        Respuesta del modelo o None en caso de error
    """
    if not OPENAI_API_KEY:
        print("Error: No se encontró la clave de API de OpenAI. Configure la variable de entorno OPENAI_API_KEY.")
        return None
    
    try:
        # Configurar la API key
        openai.api_key = OPENAI_API_KEY
        
        # Llamar a la API
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un asistente legal especializado que ayuda a personas con problemas legales. Basas tus respuestas exclusivamente en la información legal proporcionada."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.6  # Aumentamos ligeramente la temperatura para respuestas más detalladas
        )
        
        # Extraer la respuesta
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error al llamar a la API de OpenAI: {str(e)}")
        return None

def process_legal_query(
    query: str, 
    config_path: str = DEFAULT_CONFIG_PATH,
    data_path: str = DEFAULT_DATA_PATH,
    model: str = DEFAULT_MODEL,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Procesar una consulta legal completa: búsqueda, ranking y respuesta generada.
    
    Args:
        query: Consulta del usuario
        config_path: Ruta a la configuración del sistema
        data_path: Ruta a los datos
        model: Modelo a utilizar
        save_results: Si se deben guardar los resultados
        
    Returns:
        Diccionario con los resultados del proceso
    """
    print(f"\n=== Procesando consulta legal: '{query}' ===\n")
    start_time = time.time()
    
    # Cargar configuración del sistema
    config = load_config(config_path)
    if not config:
        return {
            "success": False,
            "message": "Error al cargar la configuración del sistema",
            "query": query,
            "response": None,
        }
    
    # Verificar conexiones
    weaviate_client, neo4j_driver = check_connections(config)
    
    # Cargar documentos para búsqueda léxica si está habilitada
    try:
        from src.data_loader import load_json_data
        print(f"Cargando documentos desde {data_path}...")
        documents = load_json_data(data_path)
        print(f"✓ Cargados {len(documents)} documentos")
    except Exception as e:
        print(f"Error al cargar documentos: {str(e)}")
        documents = None
    
    # Paso 1: Realizar búsqueda con ranking especializado
    print("\nRealizando búsqueda con ranking especializado...")
    search_start_time = time.time()
    results = search_query(query, config, weaviate_client, neo4j_driver, documents)
    search_time = time.time() - search_start_time
    print(f"Búsqueda completada en {search_time:.2f} segundos, {len(results)} resultados")
    
    if not results:
        return {
            "success": True,
            "message": "No se encontraron resultados relevantes",
            "query": query,
            "response": "No se encontraron artículos legales relevantes para tu consulta. Por favor, intenta reformular tu pregunta o consulta con un abogado para obtener asesoramiento personalizado.",
            "results": []
        }
    
    # Cerrar la conexión Neo4j si está abierta
    if neo4j_driver:
        neo4j_driver.close()
    
    # Paso 2: Generar prompt estructurado
    print("\nGenerando prompt para el modelo...")
    prompt = create_legal_prompt(query, results)
    print("\n" + "=" * 80)
    print("PROMPT ENVIADO AL MODELO:")
    print("=" * 80)
    print(prompt)
    print("=" * 80 + "\n")
    
    # Paso 3: Llamar al modelo
    print(f"\nLlamando al modelo {model}...")
    model_start_time = time.time()
    response = call_gpt_model(prompt, model=model)
    model_time = time.time() - model_start_time
    
    if not response:
        return {
            "success": False,
            "message": "Error al generar respuesta legal",
            "query": query,
            "results": results
        }
    
    print(f"Respuesta generada en {model_time:.2f} segundos")
    
    # Paso 4: Guardar resultados si está habilitado
    if save_results:
        try:
            # Crear directorio si no existe
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            
            # Guardar timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Guardar la respuesta
            response_path = os.path.join(results_dir, f"legal_response_{timestamp}.txt")
            with open(response_path, "w", encoding="utf-8") as f:
                f.write(f"Consulta: {query}\n\n")
                f.write("RESPUESTA LEGAL:\n")
                f.write("==============\n\n")
                f.write(response)
                f.write("\n\n")
                f.write("==============\n\n")
                f.write("PROMPT UTILIZADO:\n")
                f.write("==============\n\n")
                f.write(prompt)
                f.write("\n\n==============\n\n")
                f.write("ARTÍCULOS UTILIZADOS:\n\n")
                
                # Guardar los artículos utilizados
                for i, result in enumerate(results[:10], 1):
                    f.write(f"{i}. {result.get('law_name', 'N/A')}, Artículo {result.get('article_number', 'N/A')}\n")
                    f.write(f"   Relevancia: {result.get('score', 0):.2f}\n")
                    f.write(f"   Categoría: {result.get('category', 'N/A')}\n\n")
            
            print(f"Respuesta guardada en {response_path}")
        except Exception as e:
            print(f"Error al guardar la respuesta: {str(e)}")
    
    # Completar y retornar los resultados
    total_time = time.time() - start_time
    return {
        "success": True,
        "query": query,
        "response": response,
        "results": results[:10],
        "processing_time": {
            "search": search_time,
            "model": model_time,
            "total": total_time
        }
    }

def format_response_output(result: Dict[str, Any], show_articles: bool = True) -> str:
    """
    Formatear la respuesta para presentarla al usuario.
    
    Args:
        result: Resultado del procesamiento
        show_articles: Si se deben mostrar los artículos utilizados
        
    Returns:
        Texto formateado con la respuesta
    """
    if not result.get("success", False):
        return f"Error: {result.get('message', 'Error desconocido')}"
    
    query = result.get("query", "")
    response = result.get("response", "")
    results = result.get("results", [])
    times = result.get("processing_time", {})
    
    # Formatear la respuesta
    formatted = "\n" + "=" * 80 + "\n"
    formatted += " RESPUESTA LEGAL\n"
    formatted += "=" * 80 + "\n\n"
    
    formatted += f"Consulta: {query}\n\n"
    formatted += response + "\n\n"
    
    # Incluir información sobre los artículos utilizados
    if show_articles and results:
        formatted += "-" * 60 + "\n"
        formatted += "ARTÍCULOS PRINCIPALES UTILIZADOS:\n"
        formatted += "-" * 60 + "\n\n"
        
        for i, result in enumerate(results, 1):
            law_name = result.get("law_name", "N/A")
            article_number = result.get("article_number", "N/A")
            relevance = result.get("score", 0)
            
            formatted += f"{i}. {law_name}, Artículo {article_number}\n"
            formatted += f"   Relevancia: {relevance:.2f}\n\n"
    
    # Incluir información sobre los tiempos de procesamiento
    formatted += "-" * 60 + "\n"
    formatted += f"Tiempo de búsqueda: {times.get('search', 0):.2f} segundos\n"
    formatted += f"Tiempo de generación: {times.get('model', 0):.2f} segundos\n"
    formatted += f"Tiempo total: {times.get('total', 0):.2f} segundos\n"
    formatted += "=" * 80 + "\n"
    
    return formatted

def main():
    """Función principal del programa."""
    parser = argparse.ArgumentParser(description="Asistente Legal con Ranking Especializado y GPT-4-mini")
    parser.add_argument("--query", type=str, help="Consulta legal del usuario")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Ruta al archivo de configuración del sistema")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Ruta al directorio de datos")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Modelo de OpenAI a utilizar")
    parser.add_argument("--no-save", action="store_true", help="No guardar resultados en archivos")
    parser.add_argument("--no-articles", action="store_true", help="No mostrar artículos en la respuesta")
    parser.add_argument("--interactive", action="store_true", help="Iniciar modo interactivo")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Número máximo de tokens para la respuesta")
    args = parser.parse_args()
    
    # Verificar la clave de API
    if not OPENAI_API_KEY:
        print("\nError: No se encontró la clave de API de OpenAI.")
        print("Configure la variable de entorno OPENAI_API_KEY o cree un archivo .env con OPENAI_API_KEY=su_clave")
        return
    
    # Modo interactivo
    if args.interactive:
        print("\n=== Asistente Legal Interactivo ===")
        print("Escriba 'salir' o 'exit' para terminar\n")
        
        while True:
            query = input("\nConsulta legal: ")
            
            if query.lower() in ['salir', 'exit', 'quit']:
                print("¡Hasta pronto!")
                break
                
            if not query.strip():
                continue
                
            result = process_legal_query(
                query,
                config_path=args.config,
                data_path=args.data,
                model=args.model,
                save_results=not args.no_save
            )
            
            print(format_response_output(result, not args.no_articles))
        
        return
    
    # Modo de consulta única
    if not args.query:
        print("\n=== Asistente Legal con Ranking Especializado y GPT-4-mini ===")
        print("Utilice --query para realizar una consulta o --interactive para modo interactivo")
        print("Ejemplo: python legal_gpt_assistant.py --query \"Me despidieron sin aviso previo después de 5 años\"")
        parser.print_help()
        return
    
    # Procesar la consulta
    result = process_legal_query(
        args.query,
        config_path=args.config,
        data_path=args.data,
        model=args.model,
        save_results=not args.no_save
    )
    
    # Formatear y mostrar la respuesta
    print(format_response_output(result, not args.no_articles))

if __name__ == "__main__":
    main()