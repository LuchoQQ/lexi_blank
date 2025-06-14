"""
API REST para el Sistema de Recuperación Legal con Asesor GPT
Expone endpoints para realizar consultas legales y obtener asesoramiento.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import os
import time
from datetime import datetime
import uvicorn
import openai
from openai import OpenAI
from dotenv import load_dotenv
from src.openai_question_classifier import EnhancedLegalSystemWithOpenAI


try:
    from src.api_integration import EnhancedAPIConsultaHandler
    INTELLIGENT_SYSTEM_AVAILABLE = True
    print("✅ Sistema inteligente disponible")
except ImportError:
    INTELLIGENT_SYSTEM_AVAILABLE = False
    print("⚠️ Sistema inteligente no disponible, usando sistema tradicional")
# Cargar variables de entorno desde .env
load_dotenv()
enhanced_legal_system = None

# Flag para habilitar/deshabilitar el clasificador OpenAI
USE_OPENAI_CLASSIFIER = os.getenv("USE_OPENAI_CLASSIFIER", "true").lower() in ["true", "1", "t"]

# Importar módulos del sistema
from src.config_loader import load_config
from src.data_loader import load_json_data
from src.weaviate_utils import connect_weaviate
from src.neo4j_utils import connect_neo4j
from main import search_query_neutral, check_connections

# Configuración de la aplicación FastAPI
app = FastAPI(
    title="Sistema de Recuperación Legal",
    description="API para búsqueda de documentos legales con asesoramiento GPT",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Solo necesitamos estos modelos ahora
class ConsultaRequest(BaseModel):
    query: str = Field(..., description="Consulta legal del usuario", min_length=10, max_length=1000)
    top_n: int = Field(default=15, description="Número máximo de artículos para análisis", ge=1, le=50)

class ConsultaResponse(BaseModel):
    response: str  # Solo la respuesta de GPT como string

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, bool]
    version: str

# Variables globales para conexiones y configuración
config = None
weaviate_client = None
neo4j_driver = None
documents = None
openai_client = None

@app.on_event("startup")
async def startup_event():
    """Inicializar conexiones y cargar configuración al iniciar la API."""
    global config, weaviate_client, neo4j_driver, documents, openai_client, enhanced_handler, enhanced_legal_system
    
    print("🚀 Iniciando Sistema de Recuperación Legal...")
    
    # ===== 1. CARGAR CONFIGURACIÓN =====
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    config = load_config(config_path)
    if not config:
        raise Exception("No se pudo cargar la configuración")
    
    # ===== 2. CONFIGURAR CLIENTE OPENAI =====
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️ ADVERTENCIA: OPENAI_API_KEY no configurada. El asesoramiento GPT estará deshabilitado.")
        openai_client = None
    else:
        try:
            openai_client = OpenAI(api_key=openai_api_key)
            response = openai_client.models.list()
            print("✅ Cliente OpenAI configurado correctamente")
        except Exception as e:
            print(f"❌ Error configurando OpenAI: {str(e)}")
            openai_client = None
    
    # ===== 3. VERIFICAR CONEXIONES A BASES DE DATOS =====
    try:
        weaviate_client, neo4j_driver = check_connections(config)
        print("✅ Conexiones a bases de datos verificadas")
    except Exception as e:
        print(f"⚠️ Error en conexiones a BD: {str(e)}")
    
    # ===== 4. CARGAR DOCUMENTOS (ESTO FALTABA!) =====
    try:
        data_path = os.getenv("DATA_PATH", "data")
        documents = load_json_data(data_path)
        print(f"✅ Cargados {len(documents)} documentos legales")
    except Exception as e:
        print(f"⚠️ Error cargando documentos: {str(e)}")
        documents = []
    
    # ===== 5. INICIALIZAR SISTEMA INTELIGENTE EXISTENTE =====
    if INTELLIGENT_SYSTEM_AVAILABLE:
        try:
            # Tu configuración existente de Llama
            llama_config = {
                "ollama_model": "llama2",
                "ollama_url": "http://localhost:11434",
                "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY"),
                "local_model": "microsoft/DialoGPT-medium"
            }
            
            enhanced_handler = EnhancedAPIConsultaHandler(
                config=config,
                weaviate_client=weaviate_client,
                neo4j_driver=neo4j_driver,
                documents=documents,
                openai_client=openai_client,
                llama_config=llama_config
            )
            
            print("🧠 Sistema inteligente de clasificación inicializado")
            
        except Exception as e:
            print(f"⚠️ Error inicializando sistema inteligente: {str(e)}")
            print("🔄 Continuando con sistema tradicional")
            enhanced_handler = None
    
    # ===== 6. NUEVO: INICIALIZAR CLASIFICADOR OPENAI =====
    if openai_client:
        try:
            classifier_config = {
                "use_openai_classifier": True,   # Activar OpenAI
                "use_llama_fallback": False,     # Desactivar Llama por ahora (para Mac)
                "openai_model": "gpt-4o-mini"   # Modelo económico
            }
            
            enhanced_legal_system = EnhancedLegalSystemWithOpenAI(
                config=classifier_config,
                openai_client=openai_client
            )
            
            print("🤖 Clasificador de preguntas OpenAI inicializado")
            
        except Exception as e:
            print(f"⚠️ Error inicializando clasificador OpenAI: {str(e)}")
            enhanced_legal_system = None
    else:
        print("⚠️ Clasificador OpenAI no disponible (sin API key)")
        enhanced_legal_system = None
    
    print("🎉 Sistema inicializado correctamente")

@app.on_event("shutdown")
async def shutdown_event():
    """Cerrar conexiones al terminar la API."""
    global neo4j_driver
    
    if neo4j_driver:
        neo4j_driver.close()
        print("🔒 Conexiones cerradas correctamente")

def generate_gpt_advice(query: str, articles: List[Dict[str, Any]]) -> str:
    """
    Generar asesoramiento legal usando GPT basado en los artículos encontrados.
    Retorna solo el texto de la respuesta.
    """
    if not openai_client:
        return "Lo siento, el servicio de asesoramiento legal no está disponible en este momento. Por favor, consulte con un abogado especializado."
    
    try:
        # Preparar contexto con los artículos más relevantes
        relevant_articles_text = ""
        
        # Usar máximo 8 artículos para evitar exceder límites de tokens
        top_articles = articles[:8]
        
        for i, article in enumerate(top_articles, 1):
            law_name = article.get('law_name', 'Ley no especificada')
            article_num = article.get('article_number', 'N/A')
            content = article.get('content', '')[:600]  # Limitar contenido
            
            relevant_articles_text += f"\n--- Artículo {i} ({law_name} - Art. {article_num}) ---\n{content}\n"
        
        # Crear prompt optimizado para GPT
        system_prompt = """Eres un asistente legal especializado en derecho argentino. Proporciona asesoramiento legal claro y práctico basado ÚNICAMENTE en los artículos de ley proporcionados.

INSTRUCCIONES:
- Analiza la situación legal del usuario
- Explica qué derechos le asisten según los artículos
- Proporciona recomendaciones específicas y pasos a seguir
- Cita específicamente los artículos que respaldan tu análisis
- Usa un lenguaje claro y accesible
- Incluye advertencias sobre plazos legales importantes
- Termina con un disclaimer apropiado

FORMATO DE RESPUESTA:
1. Análisis de la situación
2. Derechos que le asisten
3. Recomendaciones específicas
4. Pasos a seguir
5. Advertencias importantes
6. Disclaimer legal"""

        user_prompt = f"""CONSULTA: "{query}"

ARTÍCULOS LEGALES APLICABLES:
{relevant_articles_text}

Proporciona un análisis legal completo y recomendaciones prácticas basándote exclusivamente en estos artículos."""

        # Llamar a la API de OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1200,
            temperature=0.2,  # Respuestas más consistentes y precisas
            timeout=30.0
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error generando asesoramiento GPT: {str(e)}")
        return f"Lo siento, hubo un error al generar el asesoramiento legal. Error técnico: {str(e)}. Por favor, consulte con un abogado especializado para obtener asesoramiento específico sobre su situación."

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz con información básica de la API."""
    return {
        "message": "Sistema de Recuperación Legal - API REST",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verificar el estado de salud del sistema y sus conexiones."""
    services_status = {
        "weaviate": weaviate_client is not None,
        "neo4j": neo4j_driver is not None,
        "openai": openai_client is not None,
        "documents_loaded": documents is not None and len(documents) > 0
    }
    
    overall_status = "healthy" if all([
        services_status["documents_loaded"],
        any([services_status["weaviate"], services_status["neo4j"]])  # Al menos una BD funcionando
    ]) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        services=services_status,
        version="1.0.0"
    )

@app.post("/consulta", response_model=ConsultaResponse)
async def realizar_consulta(request: ConsultaRequest):
    """
    Realizar una consulta legal con clasificación automática opcional.
    """
    start_time = time.time()
    
    try:
        # ===== VALIDACIONES EXISTENTES =====
        if not documents:
            return ConsultaResponse(
                response="Lo siento, el sistema no está disponible en este momento. Los documentos legales no se han cargado correctamente."
            )
        
        if not weaviate_client and not neo4j_driver:
            return ConsultaResponse(
                response="Lo siento, el sistema de búsqueda no está disponible. Por favor, contacte al administrador."
            )
        
        if not openai_client:
            return ConsultaResponse(
                response="Lo siento, el servicio de asesoramiento legal no está disponible en este momento. La API de OpenAI no está configurada."
            )
        
        print(f"🔍 Procesando consulta: '{request.query}'")
        
        # ===== NUEVO: CLASIFICAR PREGUNTA (OPCIONAL) =====
        dialogue_title = "Consulta Legal"  # Título por defecto
        question_type = "legal_advice"      # Tipo por defecto
        
        if enhanced_legal_system:
            try:
                print("🤖 Clasificando pregunta con OpenAI...")
                classification_result = enhanced_legal_system.classify_question_smart(request.query)
                question_type = classification_result['question_type']
                dialogue_title = classification_result['dialogue_title']
                
                print(f"   📋 Clasificada como: {question_type}")
                print(f"   📝 Título: {dialogue_title}")
                
            except Exception as e:
                print(f"   ⚠️ Error en clasificación: {str(e)}")
                # Continuar sin clasificación
        else:
            print("🤖 Clasificador no disponible, usando flujo tradicional")
        
        # ===== USAR TU SISTEMA EXISTENTE =====
        # Intentar con sistema inteligente primero
        if INTELLIGENT_SYSTEM_AVAILABLE and enhanced_handler:
            try:
                print("🧠 Usando sistema inteligente...")
                intelligent_response = enhanced_handler.process_intelligent_consulta(
                    request.query, request.top_n
                )
                
                # Agregar título si hay clasificación
                if dialogue_title != "Consulta Legal":
                    final_response = f"**{dialogue_title}**\n\n{intelligent_response}"
                else:
                    final_response = intelligent_response
                
                execution_time = time.time() - start_time
                print(f"✅ Consulta inteligente procesada en {execution_time:.2f}s")
                
                return ConsultaResponse(response=final_response)
                
            except Exception as e:
                print(f"⚠️ Error en sistema inteligente: {str(e)}")
                print("🔄 Fallback a sistema tradicional...")
        
        # ===== FALLBACK: SISTEMA TRADICIONAL =====
        print("⚖️ Usando sistema tradicional...")
        
        # Modificar configuración temporalmente
        temp_config = config.copy()
        temp_config.setdefault("retrieval", {})["top_n"] = request.top_n
        
        # Ejecutar búsqueda tradicional
        search_results = search_query_neutral(
            request.query, 
            temp_config, 
            weaviate_client, 
            neo4j_driver, 
            documents
        )
        
        if not search_results:
            return ConsultaResponse(
                response=f"**{dialogue_title}**\n\nNo se encontraron artículos legales relevantes para su consulta. Le recomiendo reformular su pregunta o consultar directamente con un abogado especializado."
            )
        
        # Generar asesoramiento GPT tradicional
        print("🤖 Generando asesoramiento con GPT tradicional...")
        gpt_response = generate_gpt_advice(request.query, search_results)
        
        # Agregar título si hay clasificación
        if dialogue_title != "Consulta Legal":
            final_response = f"**{dialogue_title}**\n\n{gpt_response}"
        else:
            final_response = gpt_response
        
        execution_time = time.time() - start_time
        print(f"✅ Consulta tradicional procesada en {execution_time:.2f}s")
        
        return ConsultaResponse(response=final_response)
        
    except Exception as e:
        print(f"❌ Error procesando consulta: {str(e)}")
        return ConsultaResponse(
            response=f"Lo siento, ocurrió un error al procesar su consulta: {str(e)}. Por favor, intente nuevamente o consulte con un abogado especializado."
        )


async def generate_adapted_gpt_response(query: str, articles: List[Dict[str, Any]], 
                                      question_type: str, dialogue_title: str) -> str:
    """
    Generar asesoramiento legal usando GPT adaptado al tipo de pregunta.
    """
    if not openai_client:
        return f"**{dialogue_title}**\n\nLo siento, el servicio de asesoramiento legal no está disponible en este momento."
    
    try:
        # Preparar contexto con los artículos
        relevant_articles_text = ""
        top_articles = articles[:8]  # Máximo 8 artículos
        
        for i, article in enumerate(top_articles, 1):
            law_name = article.get('law_name', 'Ley no especificada')
            article_num = article.get('article_number', 'N/A')
            content = article.get('content', '')[:600]
            
            relevant_articles_text += f"\n--- Artículo {i} ({law_name} - Art. {article_num}) ---\n{content}\n"
        
        # ===== PROMPTS ADAPTADOS SEGÚN TIPO DE PREGUNTA =====
        
        if question_type == "articles_search":
            # Para búsqueda de artículos específicos
            system_prompt = """Eres un experto jurista argentino especializado en explicar artículos legales específicos. 

INSTRUCCIONES:
- Explica claramente el contenido del artículo solicitado
- Proporciona el texto completo y su interpretación
- Explica en qué situaciones aplica
- Usa lenguaje claro y accesible
- Cita exactamente los artículos encontrados

FORMATO:
📄 **ARTÍCULO SOLICITADO**
[Explicación del artículo específico]

💡 **INTERPRETACIÓN**
[Qué significa en términos prácticos]

🔍 **CUÁNDO APLICA**
[Situaciones donde es relevante]

⚠️ **CONSIDERACIONES IMPORTANTES**
[Aspectos clave a tener en cuenta]"""

        elif question_type == "case_analysis":
            # Para análisis de casos personales
            system_prompt = """Eres un abogado especializado en derecho argentino que analiza casos personales específicos.

INSTRUCCIONES:
- Analiza la situación legal del usuario con profundidad
- Identifica qué derechos le asisten según los artículos
- Proporciona recomendaciones específicas y pasos concretos
- Incluye información sobre plazos legales críticos
- Considera el contexto emocional y urgencia
- Proporciona información sobre recursos disponibles

FORMATO:
🔍 **ANÁLISIS DE SU SITUACIÓN**
[Análisis detallado de la situación legal]

⚖️ **SUS DERECHOS**
[Derechos específicos con base legal]

📋 **RECOMENDACIONES ESPECÍFICAS**
[Acciones concretas a tomar]

⏰ **PLAZOS IMPORTANTES**
[Plazos legales críticos]

🆘 **PRÓXIMOS PASOS**
[Pasos específicos ordenados por prioridad]

⚠️ **ADVERTENCIAS LEGALES**
[Advertencias sobre plazos y riesgos]"""

        else:  # legal_advice (consulta general)
            # Para consultas generales educativas
            system_prompt = """Eres un experto en derecho argentino que proporciona información educativa clara y práctica.

INSTRUCCIONES:
- Proporciona información educativa completa
- Explica conceptos legales de manera accesible
- Incluye ejemplos prácticos relevantes
- Menciona los aspectos más importantes
- Proporciona una base sólida para entender el tema

FORMATO:
📚 **INFORMACIÓN LEGAL**
[Explicación completa del tema]

💡 **CONCEPTOS CLAVE**
[Conceptos fundamentales explicados]

📖 **BASE LEGAL**
[Artículos y leyes que respaldan la información]

🔍 **EJEMPLOS PRÁCTICOS**
[Ejemplos que ilustren los conceptos]

📋 **PUNTOS IMPORTANTES**
[Aspectos clave a recordar]

➡️ **SIGUIENTES PASOS**
[Qué hacer si necesita más información específica]"""
        
        user_prompt = f"""CONSULTA: "{query}"

ARTÍCULOS LEGALES APLICABLES:
{relevant_articles_text}

Por favor, proporciona una respuesta completa basándote exclusivamente en estos artículos y tu conocimiento del derecho argentino."""

        # Llamar a OpenAI con el prompt adaptado
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1200,
            temperature=0.2,
            timeout=30.0
        )
        
        gpt_content = response.choices[0].message.content.strip()
        
        # Agregar título al inicio
        final_response = f"**{dialogue_title}**\n\n{gpt_content}"
        
        # Agregar disclaimer al final
        final_response += f"\n\n---\n💼 **Disclaimer Legal**: Esta información es de carácter educativo. Para asesoramiento específico sobre su situación, consulte con un abogado especializado."
        
        return final_response
        
    except Exception as e:
        print(f"Error generando respuesta adaptada: {str(e)}")
        return f"**{dialogue_title}**\n\nLo siento, hubo un error al generar el asesoramiento legal. Error: {str(e)}. Por favor, consulte con un abogado especializado."

@app.get("/consulta/ejemplo")
async def ejemplo_consulta():
    """Endpoint con ejemplos de consultas para testing."""
    ejemplos = {
        "consultas_ejemplo": [
            "fui despedida sin indemnización por estar embarazada",
            "me hacen trabajar más de 8 horas sin pagar extras",
            "mi jefe me discrimina por mi edad",
            "no me pagaron la liquidación final",
            "puedo divorciarme sin el consentimiento de mi esposo",
            "mi vecino construyó en mi terreno"
        ],
        "formato_request": {
            "query": "tu consulta legal aquí",
            "top_n": 15
        },
        "respuesta": "Solo texto del asesoramiento legal de GPT",
        "ejemplo_curl": """curl -X POST "http://localhost:8000/consulta" \\
     -H "Content-Type: application/json" \\
     -d '{"query": "fui despedida por embarazo", "top_n": 10}'"""
    }
    return ejemplos

@app.get("/stats")
async def obtener_estadisticas():
    """Obtener estadísticas del sistema."""
    if not documents:
        return {"error": "No hay documentos cargados"}
    
    # Calcular estadísticas básicas
    total_docs = len(documents)
    laws = set()
    categories = set()
    
    for doc in documents:
        if doc.get('law_name'):
            laws.add(doc['law_name'])
        if doc.get('category'):
            categories.add(doc['category'])
    
    return {
        "total_documentos": total_docs,
        "total_leyes": len(laws),
        "total_categorias": len(categories),
        "leyes_disponibles": list(laws),
        "categorias_disponibles": list(categories),
        "servicios_activos": {
            "weaviate": weaviate_client is not None,
            "neo4j": neo4j_driver is not None,
            "openai": openai_client is not None
        }
    }


@app.get("/sistema/status")
async def sistema_status():
    """Estado del sistema inteligente vs tradicional."""
    status = {
        "sistema_tradicional": {
            "disponible": bool(documents and openai_client),
            "documentos_cargados": len(documents) if documents else 0,
            "openai_configurado": bool(openai_client),
            "weaviate_disponible": bool(weaviate_client),
            "neo4j_disponible": bool(neo4j_driver)
        },
        "sistema_inteligente": {
            "disponible": INTELLIGENT_SYSTEM_AVAILABLE and bool(enhanced_handler),
            "llama_status": None
        }
    }
    
    if enhanced_handler:
        try:
            status["sistema_inteligente"]["llama_status"] = enhanced_handler.intelligent_system.get_llama_status()
        except:
            status["sistema_inteligente"]["llama_status"] = "Error obteniendo status"
    
    return status

@app.post("/sistema/clasificar")
async def solo_clasificar(request: dict):
    """Solo clasificar la consulta sin ejecutar búsqueda completa."""
    if not enhanced_handler:
        return {"error": "Sistema inteligente no disponible"}
    
    try:
        query = request.get("query", "")
        if not query:
            return {"error": "Query requerido"}
        
        # Solo clasificar, no buscar
        classification_result = enhanced_handler.intelligent_system.process_query_with_real_llama(query)
        
        return {
            "query": query,
            "tipo_detectado": classification_result["classification"]["query_type"],
            "urgencia": classification_result["classification"]["urgency_level"],
            "confianza": classification_result["classification"]["confidence"],
            "dominios_legales": classification_result["classification"]["legal_domains"],
            "especialista_recomendado": classification_result["specialist_routing"]["specialist_type"],
            "metodo_clasificacion": classification_result["classification"].get("classification_method", "unknown"),
            "llama_usado": classification_result.get("llama_available", False)
        }
    except Exception as e:
        return {"error": f"Error en clasificación: {str(e)}"}

if __name__ == "__main__":
    # Configuración para desarrollo
    port = int(os.getenv("PORT", 3500))  # Cambiado a puerto 3500 para evitar conflictos
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Iniciando servidor en {host}:{port}")
    print(f"📖 Documentación disponible en: http://localhost:{port}/docs")
    print(f"🔍 Ejemplo de consulta: http://localhost:{port}/consulta/ejemplo")
    
    uvicorn.run(
        "api:app",  # Usando el nombre correcto del archivo actual
        host=host,
        port=port,
        reload=True,  # Solo para desarrollo
        log_level="info"
    )