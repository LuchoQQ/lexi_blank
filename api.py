import time
import json
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import os
import logging
from dotenv import load_dotenv

# Importar el procesador de consultas legales
from legal_gpt_assistant import process_legal_query

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("legal_assistant_api")

# Inicializar la aplicación FastAPI
app = FastAPI(title="API de Asistente Legal GPT", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajusta esto en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir modelos de datos
class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = "gpt-4o-mini"  # Modelo por defecto

class ArticleCited(BaseModel):
    law_name: str
    article_number: str
    content: str
    relevance: float

class ValidationInfo(BaseModel):
    factual_accuracy: Optional[float] = None
    unvalidated_citations: Optional[List[Dict[str, Any]]] = None
    resultado: Optional[str] = None
    total_articulos: Optional[int] = None

class LegalAssistantResponse(BaseModel):
    # Campos principales
    assistant: Optional[str] = None
    user: Optional[str] = None
    
    # Métricas del modelo LLM
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    response_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    model_id: Optional[str] = None
    
    # Métricas de la base de conocimiento
    document_count: Optional[int] = None
    query_time_ms: Optional[float] = None
    
    # Información de validación legal
    articles_cited: Optional[List[ArticleCited]] = None
    validation: Optional[ValidationInfo] = None
    
    # Contenido de la respuesta
    response: Optional[str] = None
    
    # Campos de error
    error: Optional[str] = None
    error_type: Optional[str] = None

# Autenticación básica mediante API key
def verify_api_key(x_api_key: str = Header(None)):
    expected_api_key = os.getenv("API_KEY")
    if expected_api_key and x_api_key != expected_api_key:
        raise HTTPException(
            status_code=401, 
            detail="API key inválida"
        )
    return x_api_key

# Endpoint principal para procesar consultas
@app.post("/query", response_model=LegalAssistantResponse)
async def api_process_query(
    request: QueryRequest, 
    api_key: str = Depends(verify_api_key)
):
    start_time = time.time()
    
    try:
        logger.info(f"Procesando consulta: {request.query}")
        
        # Usar el procesador de consultas legales de legal_gpt_assistant.py
        result = process_legal_query(
            query=request.query,
            model=request.model,
            save_results=True  # Guardar resultados para análisis posterior
        )
        
        # Si el procesamiento fue exitoso
        if result.get("success"):
            # Preparar los artículos citados para la respuesta
            articles_cited = []
            for article in result.get("results", []):
                articles_cited.append(ArticleCited(
                    law_name=article.get("law_name", "N/A"),
                    article_number=article.get("article_number", "N/A"),
                    content=article.get("content", ""),
                    relevance=article.get("score", 0)
                ))
            
            # Calcular métricas de tokens si están disponibles
            processing_time = result.get("processing_time", {})
            
            # Preparar la respuesta completa
            response = LegalAssistantResponse(
                user=request.query,
                assistant=result.get("response"),
                response=result.get("response"),
                model_id=request.model,
                articles_cited=articles_cited,
                total_time_ms=processing_time.get("total", 0) * 1000,  # Convertir a ms
                response_time_ms=processing_time.get("model", 0) * 1000,  # Convertir a ms
                query_time_ms=processing_time.get("search", 0) * 1000,  # Convertir a ms
                document_count=len(result.get("results", [])),
                validation=ValidationInfo(
                    resultado="OK" if articles_cited else "Información limitada",
                    total_articulos=len(articles_cited)
                )
            )
        else:
            # Si hubo un error en el procesamiento
            response = LegalAssistantResponse(
                user=request.query,
                error=result.get("message", "Error desconocido al procesar la consulta"),
                error_type="ProcessingError",
                total_time_ms=(time.time() - start_time) * 1000  # Convertir a ms
            )
        
        total_time = time.time() - start_time
        logger.info(f"Consulta procesada en {total_time:.2f} segundos")
        return response
        
    except Exception as e:
        logger.error(f"Error al procesar consulta: {str(e)}", exc_info=True)
        
        return LegalAssistantResponse(
            user=request.query,
            error=str(e),
            error_type=type(e).__name__,
            total_time_ms=(time.time() - start_time) * 1000  # Convertir a ms
        )

# Para pruebas o ejecución directa
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Iniciando servidor en {host}:{port}")
    
    # Para usar reload, debemos usar la cadena de importación
    if os.getenv("RELOAD", "true").lower() == "true":
        uvicorn.run(
            "api:app", 
            host=host,
            port=port,
            reload=True
        )
    else:
        # Modo sin recarga automática
        uvicorn.run(app, host=host, port=port)