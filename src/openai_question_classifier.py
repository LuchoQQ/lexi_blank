"""
openai_question_classifier.py

Clasificador de consultas legales usando OpenAI GPT como alternativa a Llama.
Integra perfectamente con el sistema existente y permite desactivar fácilmente.
"""

import json
import time
import os
from typing import Dict, Any, List, Optional
from openai import OpenAI

class OpenAIQuestionClassifier:
    """
    Clasificador de consultas legales usando OpenAI GPT
    """
    
    def __init__(self, openai_client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = openai_client
        self.model = model
        self.available = bool(openai_client)
        
        # Prompt optimizado para clasificación de 3 categorías
        self.classification_prompt = self._build_classification_prompt()
        
    def _build_classification_prompt(self) -> str:
        """Construye el prompt optimizado para OpenAI"""
        return """Eres un experto jurista argentino especializado en clasificar consultas legales. Tu tarea es analizar la consulta del usuario y clasificarla en UNA de estas 3 categorías EXACTAS:

**TIPOS DE CONSULTA (elegir solo UNO):**

1. **articles_search**: Búsqueda específica de artículos legales o normas
   - Ejemplos: "¿Cuál es el artículo 14 del código penal?", "Muéstrame el artículo 75 de la LCT", "Artículo sobre despidos"
   - Palabras clave: "artículo", "art.", "código", "ley", número específico

2. **case_analysis**: Análisis de una situación legal específica del usuario (caso personal)
   - Ejemplos: "Fui despedido sin indemnización", "Mi jefe me discrimina por embarazo", "Me deben salarios"
   - Palabras clave: "fui", "me", "mi jefe", "mi empleador", situaciones personales

3. **legal_advice**: Consulta general sobre temas legales (información educativa)
   - Ejemplos: "¿Cuáles son mis derechos laborales?", "¿Cómo funciona el divorcio?", "Derechos del trabajador"
   - Palabras clave: "cuáles son", "cómo", "qué es", consultas generales

**INSTRUCCIONES ESTRICTAS:**
1. Debes elegir EXACTAMENTE uno de estos 3 tipos: "articles_search", "case_analysis", "legal_advice"
2. Crea un título descriptivo y específico para la conversación (máximo 50 caracteres)
3. El título debe reflejar el contenido específico de la consulta
4. Responde ÚNICAMENTE con el JSON solicitado, sin texto adicional

**FORMATO DE RESPUESTA (JSON únicamente):**
{
    "question_type": "articles_search",
    "dialogue_title": "Artículo 14 Código Penal"
}

**CONSULTA A CLASIFICAR:**
{query}"""

    def classify_question(self, query: str) -> Dict[str, str]:
        """
        Clasifica la consulta usando OpenAI GPT
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Dict con question_type y dialogue_title
        """
        if not self.available:
            raise Exception("OpenAI client no disponible")
        
        print(f"🤖 Clasificando con OpenAI: '{query[:50]}...'")
        start_time = time.time()
        
        try:
            # Llamar a OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Eres un clasificador experto de consultas legales. Responde ÚNICAMENTE con JSON válido."},
                    {"role": "user", "content": self.classification_prompt.format(query=query)}
                ],
                max_tokens=150,
                temperature=0.1,  # Muy baja para respuestas consistentes
                timeout=15.0
            )
            
            # Extraer respuesta
            classification_text = response.choices[0].message.content.strip()
            
            # Parsear JSON
            classification_data = self._parse_openai_response(classification_text)
            
            # Validar y normalizar
            validated_data = self._validate_classification(classification_data, query)
            
            elapsed_time = time.time() - start_time
            print(f"✅ Clasificación OpenAI completada en {elapsed_time:.2f}s")
            print(f"   🏷️ Tipo: {validated_data['question_type']}")
            print(f"   📝 Título: {validated_data['dialogue_title']}")
            
            return validated_data
            
        except Exception as e:
            print(f"❌ Error en clasificación OpenAI: {str(e)}")
            # Fallback a clasificación de reglas
            return self._rule_based_fallback(query)
    
    def _parse_openai_response(self, response_text: str) -> Dict[str, str]:
        """Parsea la respuesta JSON de OpenAI"""
        try:
            # Limpiar respuesta
            response_clean = response_text.strip()
            
            # Buscar JSON en la respuesta
            json_start = response_clean.find('{')
            json_end = response_clean.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No se encontró JSON válido en respuesta de OpenAI")
            
            json_str = response_clean[json_start:json_end]
            
            # Parsear JSON
            parsed_data = json.loads(json_str)
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parseando JSON de OpenAI: {str(e)}")
            print(f"Respuesta recibida: {response_text}")
            raise
        except Exception as e:
            print(f"❌ Error procesando respuesta de OpenAI: {str(e)}")
            raise
    
    def _validate_classification(self, data: Dict[str, str], original_query: str) -> Dict[str, str]:
        """Valida y normaliza los datos de clasificación"""
        
        # Tipos válidos
        valid_types = ["articles_search", "case_analysis", "legal_advice"]
        
        # Validar question_type
        question_type = data.get("question_type", "").lower().strip()
        if question_type not in valid_types:
            print(f"⚠️ Tipo inválido '{question_type}', usando fallback")
            question_type = self._guess_type_from_query(original_query)
        
        # Validar y limpiar dialogue_title
        dialogue_title = data.get("dialogue_title", "").strip()
        if not dialogue_title or len(dialogue_title) < 3:
            dialogue_title = self._generate_title_from_query(original_query, question_type)
        
        # Limitar longitud del título
        if len(dialogue_title) > 50:
            dialogue_title = dialogue_title[:47] + "..."
        
        return {
            "question_type": question_type,
            "dialogue_title": dialogue_title
        }
    
    def _guess_type_from_query(self, query: str) -> str:
        """Adivina el tipo basado en patrones en la consulta"""
        query_lower = query.lower()
        
        # Buscar artículos específicos
        if any(pattern in query_lower for pattern in ['artículo', 'art.', 'art ', 'código']):
            return "articles_search"
        
        # Buscar situaciones personales
        if any(pattern in query_lower for pattern in ['fui', 'me ', 'mi jefe', 'mi empleador', 'despidieron', 'discriminan']):
            return "case_analysis"
        
        # Por defecto, consulta general
        return "legal_advice"
    
    def _generate_title_from_query(self, query: str, question_type: str) -> str:
        """Genera un título basado en la consulta y el tipo"""
        query_words = query.split()[:6]  # Primeras 6 palabras
        base_title = " ".join(query_words)
        
        # Prefijos por tipo
        prefixes = {
            "articles_search": "Búsqueda: ",
            "case_analysis": "Caso: ",
            "legal_advice": "Consulta: "
        }
        
        prefix = prefixes.get(question_type, "")
        title = f"{prefix}{base_title}"
        
        return title[:50]
    
    def _rule_based_fallback(self, query: str) -> Dict[str, str]:
        """Clasificación de fallback cuando OpenAI falla"""
        print("🔄 Usando clasificación de reglas como fallback...")
        
        question_type = self._guess_type_from_query(query)
        dialogue_title = self._generate_title_from_query(query, question_type)
        
        return {
            "question_type": question_type,
            "dialogue_title": dialogue_title
        }


class EnhancedLegalSystemWithOpenAI:
    """
    Sistema legal mejorado que puede usar OpenAI o Llama según disponibilidad
    """
    
    def __init__(self, config: Dict[str, Any], openai_client: Optional[OpenAI] = None):
        self.config = config
        self.openai_client = openai_client
        
        # Flags de configuración
        self.use_openai_classifier = config.get("use_openai_classifier", True)  # Por defecto usar OpenAI
        self.use_llama_fallback = config.get("use_llama_fallback", True)       # Llama como fallback
        
        # Inicializar clasificadores
        self.openai_classifier = None
        self.llama_system = None
        
        # Configurar OpenAI classifier
        if self.use_openai_classifier and openai_client:
            try:
                self.openai_classifier = OpenAIQuestionClassifier(
                    openai_client=openai_client,
                    model=config.get("openai_model", "gpt-4o-mini")
                )
                print("🤖 OpenAI classifier inicializado correctamente")
            except Exception as e:
                print(f"❌ Error inicializando OpenAI classifier: {str(e)}")
        
        # Configurar Llama fallback
        if self.use_llama_fallback:
            try:
                from .llama_integration import EnhancedIntelligentLegalSystem
                self.llama_system = EnhancedIntelligentLegalSystem(
                    config.get("llama_config", {})
                )
                print("🦙 Llama system inicializado como fallback")
            except Exception as e:
                print(f"⚠️ Llama system no disponible: {str(e)}")
    
    def classify_question_smart(self, query: str) -> Dict[str, Any]:
        """
        Clasifica la consulta usando el mejor método disponible
        
        Returns:
            Dict con question_type, dialogue_title y metadata adicional
        """
        print(f"\n🧠 CLASIFICACIÓN INTELIGENTE: '{query[:50]}...'")
        
        # Intentar con OpenAI primero (si está configurado)
        if self.use_openai_classifier and self.openai_classifier:
            try:
                result = self.openai_classifier.classify_question(query)
                
                # Expandir resultado con metadata
                return {
                    "question_type": result["question_type"],
                    "dialogue_title": result["dialogue_title"],
                    "classification_method": "openai",
                    "confidence": 0.9,  # Alta confianza para OpenAI
                    "timestamp": time.time()
                }
                
            except Exception as e:
                print(f"❌ Error con OpenAI classifier: {str(e)}")
                print("🔄 Intentando con fallback...")
        
        # Fallback a Llama (si está disponible)
        if self.use_llama_fallback and self.llama_system:
            try:
                llama_result = self.llama_system.process_query_with_real_llama(query)
                
                # Convertir resultado de Llama al formato esperado
                classification = llama_result["classification"]
                
                # Mapear tipos de Llama a los 3 tipos simplificados
                question_type = self._map_llama_type_to_simple(classification["query_type"])
                dialogue_title = self._generate_title_from_llama_result(query, classification)
                
                return {
                    "question_type": question_type,
                    "dialogue_title": dialogue_title,
                    "classification_method": "llama_fallback",
                    "confidence": classification.get("confidence", 0.7),
                    "timestamp": time.time(),
                    "llama_details": classification  # Detalles adicionales de Llama
                }
                
            except Exception as e:
                print(f"❌ Error con Llama fallback: {str(e)}")
        
        # Último fallback: clasificación de reglas simples
        print("🔄 Usando clasificación de reglas como último recurso...")
        return self._ultimate_fallback_classification(query)
    
    def _map_llama_type_to_simple(self, llama_type: str) -> str:
        """Mapea los tipos de Llama a los 3 tipos simplificados"""
        mapping = {
            "article_lookup": "articles_search",
            "case_analysis": "case_analysis", 
            "general_consultation": "legal_advice",
            "procedural_guidance": "legal_advice",
            "comparative_analysis": "legal_advice"
        }
        
        return mapping.get(llama_type, "legal_advice")
    
    def _generate_title_from_llama_result(self, query: str, classification: Dict[str, Any]) -> str:
        """Genera título basado en resultado de Llama"""
        # Usar dominios legales de Llama para crear un título más específico
        domains = classification.get("legal_domains", [])
        query_type = classification.get("query_type", "")
        
        # Crear título basado en el dominio y tipo
        if domains:
            primary_domain = domains[0].capitalize()
            if query_type == "article_lookup":
                title = f"Artículo - {primary_domain}"
            elif query_type == "case_analysis":
                title = f"Caso {primary_domain}"
            else:
                title = f"Consulta {primary_domain}"
        else:
            # Fallback a primeras palabras de la consulta
            words = query.split()[:4]
            title = " ".join(words)
        
        return title[:50]
    
    def _ultimate_fallback_classification(self, query: str) -> Dict[str, Any]:
        """Clasificación de último recurso usando reglas simples"""
        query_lower = query.lower()
        
        # Detectar tipo
        if any(pattern in query_lower for pattern in ['artículo', 'art.', 'código']):
            question_type = "articles_search"
            title = "Búsqueda de Artículo"
        elif any(pattern in query_lower for pattern in ['fui', 'me ', 'mi jefe', 'despidieron']):
            question_type = "case_analysis"
            title = "Análisis de Caso"
        else:
            question_type = "legal_advice"
            title = "Consulta Legal"
        
        # Mejorar título con palabras de la consulta
        query_words = query.split()[:3]
        if query_words:
            title = f"{title}: {' '.join(query_words)}"
        
        return {
            "question_type": question_type,
            "dialogue_title": title[:50],
            "classification_method": "simple_rules",
            "confidence": 0.5,
            "timestamp": time.time()
        }
    
    def get_classifier_status(self) -> Dict[str, Any]:
        """Obtiene el estado de los clasificadores disponibles"""
        return {
            "openai_available": bool(self.openai_classifier),
            "llama_available": bool(self.llama_system),
            "openai_enabled": self.use_openai_classifier,
            "llama_fallback_enabled": self.use_llama_fallback,
            "active_method": self._get_active_method()
        }
    
    def _get_active_method(self) -> str:
        """Determina cuál será el método activo de clasificación"""
        if self.use_openai_classifier and self.openai_classifier:
            return "openai_primary"
        elif self.use_llama_fallback and self.llama_system:
            return "llama_only"
        else:
            return "rules_only"


# ========== INTEGRACIÓN CON API EXISTENTE ==========

def integrate_openai_classifier_with_api():
    """
    Código para integrar el clasificador OpenAI con la API existente
    """
    
    integration_code = '''
# ===== MODIFICACIONES EN api.py =====

# 1. Agregar import al inicio del archivo
from .openai_question_classifier import EnhancedLegalSystemWithOpenAI

# 2. Modificar startup_event() para incluir el clasificador OpenAI
enhanced_legal_system = None

@app.on_event("startup") 
async def startup_event():
    """Inicializar conexiones y cargar configuración al iniciar la API."""
    global config, weaviate_client, neo4j_driver, documents, openai_client, enhanced_legal_system
    
    # ... tu código existente ...
    
    # ===== NUEVO: INICIALIZAR SISTEMA CON OPENAI CLASSIFIER =====
    try:
        # Configuración del clasificador
        classifier_config = {
            "use_openai_classifier": True,  # Usar OpenAI como principal
            "use_llama_fallback": True,     # Llama como fallback
            "openai_model": "gpt-4o-mini",  # Modelo a usar
            "llama_config": {
                "ollama_model": "llama2",
                "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY"),
                "local_model": "microsoft/DialoGPT-medium"
            }
        }
        
        enhanced_legal_system = EnhancedLegalSystemWithOpenAI(
            config=classifier_config,
            openai_client=openai_client
        )
        
        print("🤖 Sistema de clasificación inteligente inicializado (OpenAI + Llama)")
        
    except Exception as e:
        print(f"⚠️ Error inicializando clasificador: {str(e)}")
        enhanced_legal_system = None

# 3. Agregar endpoint para clasificación de preguntas
@app.post("/classify")
async def classify_question(request: dict):
    """Clasifica una pregunta en las 3 categorías principales."""
    try:
        query = request.get("query", "")
        if not query:
            return {"error": "Query requerido"}
        
        if not enhanced_legal_system:
            return {"error": "Sistema de clasificación no disponible"}
        
        # Clasificar pregunta
        result = enhanced_legal_system.classify_question_smart(query)
        
        return {
            "success": True,
            "result": {
                "question_type": result["question_type"],
                "dialogue_title": result["dialogue_title"]
            },
            "metadata": {
                "classification_method": result["classification_method"],
                "confidence": result["confidence"],
                "timestamp": result["timestamp"]
            }
        }
        
    except Exception as e:
        return {"error": f"Error en clasificación: {str(e)}"}

# 4. Agregar endpoint para estado del clasificador
@app.get("/classifier/status")
async def get_classifier_status():
    """Obtiene el estado del sistema de clasificación."""
    if not enhanced_legal_system:
        return {"error": "Sistema de clasificación no inicializado"}
    
    status = enhanced_legal_system.get_classifier_status()
    
    return {
        "status": status,
        "available_methods": {
            "openai": "Clasificación rápida y precisa con GPT",
            "llama": "Clasificación avanzada con Llama (fallback)",
            "rules": "Clasificación básica por reglas (último recurso)"
        }
    }

# 5. Modificar endpoint /consulta para incluir clasificación automática
@app.post("/consulta", response_model=ConsultaResponse)
async def realizar_consulta(request: ConsultaRequest):
    """
    Realizar una consulta legal con clasificación automática de pregunta.
    """
    start_time = time.time()
    
    try:
        # ... validaciones existentes ...
        
        print(f"🧠 Procesando consulta: '{request.query}'")
        
        # NUEVO: Clasificar pregunta automáticamente
        classification_result = None
        if enhanced_legal_system:
            try:
                classification_result = enhanced_legal_system.classify_question_smart(request.query)
                print(f"   📋 Clasificada como: {classification_result['question_type']}")
                print(f"   📝 Título: {classification_result['dialogue_title']}")
            except Exception as e:
                print(f"   ⚠️ Error en clasificación: {str(e)}")
        
        # Continuar con la lógica existente de búsqueda y respuesta...
        # (tu código actual)
        
        # Agregar información de clasificación a la respuesta
        if classification_result:
            response_text = f"**{classification_result['dialogue_title']}**\\n\\n{gpt_response}"
        else:
            response_text = gpt_response
        
        execution_time = time.time() - start_time
        print(f"✅ Consulta procesada en {execution_time:.2f}s")
        
        return ConsultaResponse(response=response_text)
        
    except Exception as e:
        print(f"❌ Error procesando consulta: {str(e)}")
        return ConsultaResponse(
            response=f"Lo siento, ocurrió un error al procesar su consulta: {str(e)}"
        )
'''
    
    return integration_code


# ========== EJEMPLOS DE USO ==========

def test_openai_classifier():
    """Función para probar el clasificador OpenAI"""
    
    # Configuración de prueba
    config = {
        "use_openai_classifier": True,
        "use_llama_fallback": False,  # Desactivar Llama para esta prueba
        "openai_model": "gpt-4o-mini"
    }
    
    # Crear cliente OpenAI de prueba (necesitarás tu API key)
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Inicializar sistema
    system = EnhancedLegalSystemWithOpenAI(config, openai_client)
    
    # Consultas de prueba
    test_queries = [
        "¿Cuál es el artículo 14 del código penal?",
        "Fui despedida sin indemnización por estar embarazada", 
        "¿Cuáles son mis derechos como trabajador?",
        "Artículo 75 de la constitución nacional",
        "Mi jefe me discrimina por mi edad",
        "¿Cómo presento una denuncia laboral?"
    ]
    
    print("🧪 PROBANDO CLASIFICADOR OPENAI")
    print("="*50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\\n{'='*40}")
        print(f"PRUEBA {i}: {query}")
        print(f"{'='*40}")
        
        try:
            result = system.classify_question_smart(query)
            
            print(f"✅ RESULTADO:")
            print(f"   🏷️ Tipo: {result['question_type']}")
            print(f"   📝 Título: {result['dialogue_title']}")
            print(f"   🔧 Método: {result['classification_method']}")
            print(f"   📊 Confianza: {result['confidence']:.2f}")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")


if __name__ == "__main__":
    print("🤖 CLASIFICADOR OPENAI PARA CONSULTAS LEGALES")
    print("="*50)
    print("\\nEste módulo proporciona:")
    print("✅ Clasificación en 3 categorías: articles_search, case_analysis, legal_advice")
    print("✅ Títulos automáticos para conversaciones")
    print("✅ Integración con sistema existente")
    print("✅ Fallback a Llama si OpenAI falla")
    print("✅ Fácil activación/desactivación")
    
    print("\\n🔧 Para integrar:")
    print("1. Agregar este archivo a tu proyecto")
    print("2. Modificar api.py según el código de integración")
    print("3. Configurar OPENAI_API_KEY en tu .env")
    print("4. Usar endpoint /classify para clasificar preguntas")
    
    # Mostrar código de integración
    print("\\n" + "="*50)
    print("CÓDIGO DE INTEGRACIÓN:")
    print("="*50)
    integration_code = integrate_openai_classifier_with_api()
    print(integration_code)