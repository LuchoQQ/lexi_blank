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

# Cargar variables de entorno desde .env
load_dotenv()

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
    global config, weaviate_client, neo4j_driver, documents, openai_client
    
    print("🚀 Iniciando Sistema de Recuperación Legal...")
    
    # Cargar configuración
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    config = load_config(config_path)
    if not config:
        raise Exception("No se pudo cargar la configuración")
    
    # Configurar cliente OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️ ADVERTENCIA: OPENAI_API_KEY no configurada. El asesoramiento GPT estará deshabilitado.")
        openai_client = None
    else:
        try:
            openai_client = OpenAI(api_key=openai_api_key)
            # Probar conexión
            response = openai_client.models.list()
            print("✅ Cliente OpenAI configurado correctamente")
        except Exception as e:
            print(f"❌ Error configurando OpenAI: {str(e)}")
            openai_client = None
    
    # Verificar conexiones a bases de datos
    try:
        weaviate_client, neo4j_driver = check_connections(config)
        print("✅ Conexiones a bases de datos verificadas")
    except Exception as e:
        print(f"⚠️ Error en conexiones a BD: {str(e)}")
    
    # Cargar documentos
    try:
        data_path = os.getenv("DATA_PATH", "data")
        documents = load_json_data(data_path)
        print(f"✅ Cargados {len(documents)} documentos legales")
    except Exception as e:
        print(f"⚠️ Error cargando documentos: {str(e)}")
        documents = []
    
    print("🎉 Sistema inicializado correctamente")

@app.on_event("shutdown")
async def shutdown_event():
    """Cerrar conexiones al terminar la API."""
    global neo4j_driver
    
    if neo4j_driver:
        neo4j_driver.close()
        print("🔒 Conexiones cerradas correctamente")

async def generate_gpt_advice(query: str, articles: List[Dict[str, Any]]) -> str:
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
    Realizar una consulta legal y obtener asesoramiento GPT.
    Retorna solo la respuesta del asesor legal.
    """
    start_time = time.time()
    
    try:
        # Validar que el sistema esté funcionando
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
        
        # Realizar búsqueda usando el sistema existente
        print(f"🔍 Procesando consulta: '{request.query}'")
        
        # Modificar configuración temporalmente
        temp_config = config.copy()
        temp_config.setdefault("retrieval", {})["top_n"] = request.top_n
        
        # Ejecutar búsqueda
        search_results = search_query_neutral(
            request.query, 
            temp_config, 
            weaviate_client, 
            neo4j_driver, 
            documents
        )
        
        if not search_results:
            return ConsultaResponse(
                response="No se encontraron artículos legales relevantes para su consulta. Le recomiendo reformular su pregunta o consultar directamente con un abogado especializado."
            )
        
        # Generar asesoramiento GPT
        print("🤖 Generando asesoramiento con GPT...")
        gpt_response = await generate_gpt_advice(request.query, search_results)
        
        execution_time = time.time() - start_time
        print(f"✅ Consulta procesada en {execution_time:.2f}s")
        
        return ConsultaResponse(response=gpt_response)
        
    except Exception as e:
        print(f"❌ Error procesando consulta: {str(e)}")
        return ConsultaResponse(
            response=f"Lo siento, ocurrió un error al procesar su consulta: {str(e)}. Por favor, intente nuevamente o consulte con un abogado especializado."
        )

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