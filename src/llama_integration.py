"""
llama_integration.py

Implementación real de integración con Llama para clasificación de consultas legales.
Incluye múltiples opciones: Ollama local, API de Hugging Face, y Transformers local.
"""

import json
import requests
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re
import os

# Para instalación local de transformers (opcional)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class LlamaProvider:
    """Proveedor base para diferentes implementaciones de Llama"""
    
    def __init__(self):
        self.available = False
        self.model_name = ""
    
    def classify_query(self, query: str, prompt: str) -> str:
        raise NotImplementedError

class OllamaLlamaProvider(LlamaProvider):
    """Proveedor usando Ollama local"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model_name: str = "llama2"):
        super().__init__()
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Verifica si Ollama está disponible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                if any(self.model_name in model for model in available_models):
                    print(f"✅ Ollama disponible con modelo {self.model_name}")
                    return True
                else:
                    print(f"⚠️ Ollama disponible pero modelo {self.model_name} no encontrado")
                    print(f"   Modelos disponibles: {available_models}")
                    return False
            return False
        except Exception as e:
            print(f"❌ Ollama no disponible: {str(e)}")
            return False
    
    def classify_query(self, query: str, prompt: str) -> str:
        """Clasifica consulta usando Ollama"""
        if not self.available:
            raise Exception("Ollama no está disponible")
        
        full_prompt = prompt.format(query=query)
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 800,  # CAMBIAR: usar num_predict en lugar de max_tokens
                "stop": ["}"]  # AGREGAR: parar en el cierre del JSON
            }
        }
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120  # CAMBIAR: aumentar de 60 a 120 segundos
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                raise Exception(f"Error Ollama: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Error llamando a Ollama: {str(e)}")

class HuggingFaceLlamaProvider(LlamaProvider):
    """Proveedor usando API de Hugging Face"""
    
    def __init__(self, api_key: str, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.available = bool(api_key)
        
        if self.available:
            print(f"✅ Hugging Face API configurado con {model_name}")
    
    def classify_query(self, query: str, prompt: str) -> str:
        """Clasifica consulta usando Hugging Face API"""
        if not self.available:
            raise Exception("API key de Hugging Face no disponible")
        
        full_prompt = prompt.format(query=query)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": 0.1,
                "max_new_tokens": 800,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
                return str(result)
            else:
                raise Exception(f"Error HuggingFace API: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Error llamando a HuggingFace: {str(e)}")

class LocalLlamaProvider(LlamaProvider):
    """Proveedor usando transformers local"""
    
    def __init__(self, model_name: str = "distilgpt2"):  # CAMBIAR: modelo más simple
        super().__init__()
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.available = TRANSFORMERS_AVAILABLE and self._load_model()
    
    def _load_model(self) -> bool:
        """Carga el modelo local"""
        try:
            print(f"🔄 Cargando modelo simple {self.model_name}...")
            
            # Usar pipeline simple sin configuraciones complejas
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=-1,  # FORZAR CPU
                return_full_text=False  # Solo texto generado
            )
        
            print(f"✅ Modelo local {self.model_name} cargado")
            return True
        
        except Exception as e:
            print(f"❌ Error cargando modelo: {str(e)}")
            return False
    
    def classify_query(self, query: str, prompt: str) -> str:
        """Clasifica consulta usando modelo local"""
        if not self.available:
            raise Exception("Modelo local no disponible")
        
        full_prompt = prompt.format(query=query)
        
        try:
            result = self.pipeline(
                full_prompt,
                max_length=len(full_prompt) + 500,
                temperature=0.1,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
                do_sample=True
            )
            
            generated_text = result[0]["generated_text"]
            # Extraer solo la parte generada (después del prompt)
            response = generated_text[len(full_prompt):].strip()
            
            return response
            
        except Exception as e:
            raise Exception(f"Error en generación local: {str(e)}")

class RealLlamaQueryClassifier:
    """
    Clasificador de consultas que usa Llama real con múltiples proveedores
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.providers = []
        self.active_provider = None
        
        # Configurar proveedores según disponibilidad
        self._setup_providers()
        
        # Template de prompt optimizado para Llama
        self.classification_prompt = self._build_llama_prompt_template()
    
    def _setup_providers(self):
        """Configura proveedores de Llama en orden de preferencia"""
        
        # 1. Intentar Ollama local (preferido para velocidad y privacidad)
        try:
            ollama_provider = OllamaLlamaProvider(
                model_name=self.config.get("ollama_model", "llama2")
            )
            if ollama_provider.available:
                self.providers.append(ollama_provider)
                print("✅ Ollama configurado como proveedor principal")
        except Exception as e:
            print(f"⚠️ No se pudo configurar Ollama: {str(e)}")
        
        # 2. Hugging Face API (si hay API key)
        hf_api_key = self.config.get("huggingface_api_key") or os.getenv("HUGGINGFACE_API_KEY")
        if hf_api_key:
            try:
                hf_provider = HuggingFaceLlamaProvider(
                    api_key=hf_api_key,
                    model_name=self.config.get("hf_model", "meta-llama/Llama-2-7b-chat-hf")
                )
                self.providers.append(hf_provider)
                print("✅ Hugging Face API configurado como proveedor")
            except Exception as e:
                print(f"⚠️ No se pudo configurar Hugging Face: {str(e)}")
        
        # 3. Modelo local con transformers (fallback)
        if TRANSFORMERS_AVAILABLE:
            try:
                local_provider = LocalLlamaProvider(
                    model_name=self.config.get("local_model", "microsoft/DialoGPT-medium")
                )
                if local_provider.available:
                    self.providers.append(local_provider)
                    print("✅ Modelo local configurado como proveedor")
            except Exception as e:
                print(f"⚠️ No se pudo configurar modelo local: {str(e)}")
        
        # Seleccionar proveedor activo
        if self.providers:
            self.active_provider = self.providers[0]
            print(f"🎯 Proveedor activo: {type(self.active_provider).__name__}")
        else:
            print("❌ No hay proveedores de Llama disponibles")
    
    def _build_llama_prompt_template(self) -> str:
        """Construye template de prompt optimizado para Llama"""
        return """<s>[INST] Eres un experto jurista argentino. Analiza esta consulta legal y clasifícala.

TIPOS DE CONSULTA:
- article_lookup: Búsqueda específica de artículos (ej: "artículo 14", "art. 20")
- case_analysis: Situación personal del usuario (ej: "fui despedido", "me discriminan")
- general_consultation: Consulta general (ej: "cuáles son mis derechos")
- procedural_guidance: Cómo hacer algo (ej: "cómo presento denuncia")
- comparative_analysis: Comparaciones (ej: "diferencias entre...")

URGENCIA:
- low: Consultas académicas o artículos específicos
- medium: Consultas generales
- high: Situaciones urgentes
- critical: Emergencias

DOMINIOS:
- constitucional: Constitución Nacional, derechos constitucionales
- penal: Código Penal, delitos
- civil: Código Civil, contratos, propiedad
- laboral: Trabajo, LCT, despidos
- familia: Divorcio, custodia
- administrativo: Trámites, Estado

CONSULTA: "{query}"

Responde SOLO este JSON:
{{"query_type": "article_lookup", "urgency_level": "low", "confidence": 0.95, "legal_domains": ["constitucional"], "key_entities": [], "emotional_indicators": [], "specific_articles": [], "reasoning": "Búsqueda específica de artículo constitucional", "recommended_specialist": "article_specialist"}} [/INST]"""

    
    def classify_query_with_real_llama(self, query: str) -> Dict[str, Any]:
        """Clasifica consulta usando Llama real"""
        if not self.active_provider:
            raise Exception("No hay proveedores de Llama disponibles")
        
        print(f"🦙 Clasificando con Llama: '{query[:50]}...'")
        start_time = time.time()
        
        try:
            # Intentar con proveedor activo
            llama_response = self.active_provider.classify_query(query, self.classification_prompt)
            
            # Parsear respuesta
            classification_data = self._parse_llama_json_response(llama_response)
            
            elapsed_time = time.time() - start_time
            print(f"✅ Clasificación Llama completada en {elapsed_time:.2f}s")
            
            return classification_data
            
        except Exception as e:
            print(f"❌ Error con proveedor principal: {str(e)}")
            
            # Intentar con proveedores alternativos
            for i, provider in enumerate(self.providers[1:], 1):
                try:
                    print(f"🔄 Intentando con proveedor alternativo {i}...")
                    llama_response = provider.classify_query(query, self.classification_prompt)
                    classification_data = self._parse_llama_json_response(llama_response)
                    
                    # Actualizar proveedor activo si este funciona
                    self.active_provider = provider
                    
                    elapsed_time = time.time() - start_time
                    print(f"✅ Clasificación con proveedor alternativo en {elapsed_time:.2f}s")
                    
                    return classification_data
                    
                except Exception as e2:
                    print(f"❌ Error con proveedor alternativo {i}: {str(e2)}")
                    continue
            
            # Si todos los proveedores fallan, usar fallback de reglas
            print("🔄 Todos los proveedores Llama fallaron, usando clasificación de reglas...")
            return self._rule_based_fallback(query)
    
    def _parse_llama_json_response(self, llama_response: str) -> Dict[str, Any]:
        """Parsea respuesta JSON de Llama"""
        try:
            # Limpiar respuesta de Llama
            response_clean = llama_response.strip()
            
            # Buscar JSON en la respuesta
            json_start = response_clean.find('{')
            json_end = response_clean.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No se encontró JSON válido en respuesta de Llama")
            
            json_str = response_clean[json_start:json_end]
            
            # Limpiar JSON de posibles caracteres problemáticos
            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # Caracteres de control
            json_str = re.sub(r',\s*}', '}', json_str)  # Comas finales
            json_str = re.sub(r',\s*]', ']', json_str)  # Comas en arrays
            
            # Parsear JSON
            response_data = json.loads(json_str)
            
            # Validar campos requeridos
            required_fields = ["query_type", "urgency_level", "confidence", "legal_domains"]
            for field in required_fields:
                if field not in response_data:
                    response_data[field] = self._get_default_value(field)
            
            # Normalizar valores
            response_data = self._normalize_classification_data(response_data)
            
            return response_data
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parseando JSON de Llama: {str(e)}")
            print(f"Respuesta recibida: {llama_response[:200]}...")
            
            # Intentar extracción manual de campos clave
            return self._manual_extraction_fallback(llama_response)
            
        except Exception as e:
            print(f"❌ Error procesando respuesta de Llama: {str(e)}")
            raise
    
    def _get_default_value(self, field: str) -> Any:
        """Obtiene valor por defecto para campos faltantes"""
        defaults = {
            "query_type": "general_consultation",
            "urgency_level": "medium",
            "confidence": 0.5,
            "legal_domains": ["laboral"],
            "key_entities": [],
            "emotional_indicators": [],
            "specific_articles": [],
            "reasoning": "Clasificación automática",
            "recommended_specialist": "general_counselor"
        }
        return defaults.get(field, "")
    
    def _normalize_classification_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normaliza datos de clasificación"""
        # Normalizar query_type
        query_type = data.get("query_type", "").lower()
        valid_types = ["article_lookup", "case_analysis", "general_consultation", 
                      "procedural_guidance", "comparative_analysis"]
        
        if query_type not in valid_types:
            data["query_type"] = "general_consultation"
        else:
            data["query_type"] = query_type
        
        # Normalizar urgency_level
        urgency = data.get("urgency_level", "").lower()
        valid_urgencies = ["low", "medium", "high", "critical"]
        
        if urgency not in valid_urgencies:
            data["urgency_level"] = "medium"
        else:
            data["urgency_level"] = urgency
        
        # Asegurar tipos correctos
        data["confidence"] = float(data.get("confidence", 0.5))
        data["legal_domains"] = data.get("legal_domains", []) if isinstance(data.get("legal_domains"), list) else ["laboral"]
        data["key_entities"] = data.get("key_entities", []) if isinstance(data.get("key_entities"), list) else []
        data["emotional_indicators"] = data.get("emotional_indicators", []) if isinstance(data.get("emotional_indicators"), list) else []
        data["specific_articles"] = data.get("specific_articles", []) if isinstance(data.get("specific_articles"), list) else []
        
        # Mapear especialista recomendado
        specialist_mapping = {
            "article_lookup": "article_specialist",
            "case_analysis": "case_analyst", 
            "procedural_guidance": "procedural_guide",
            "general_consultation": "general_counselor",
            "comparative_analysis": "general_counselor"
        }
        
        data["recommended_specialist"] = specialist_mapping.get(data["query_type"], "general_counselor")
        
        return data
    
    def _manual_extraction_fallback(self, llama_response: str) -> Dict[str, Any]:
        """Extracción manual cuando el JSON falla"""
        print("🔄 Intentando extracción manual de respuesta Llama...")
        
        response_lower = llama_response.lower()
        
        # Detectar tipo de consulta
        if re.search(r'artículo\s+\d+|art\.\s*\d+', response_lower):
            query_type = "article_lookup"
        elif any(indicator in response_lower for indicator in ['fui', 'me', 'despidieron', 'discriminan']):
            query_type = "case_analysis"
        elif any(proc in response_lower for proc in ['cómo', 'procedimiento', 'pasos']):
            query_type = "procedural_guidance"
        else:
            query_type = "general_consultation"
        
        # Detectar urgencia
        if any(urgent in response_lower for urgent in ['critical', 'emergencia']):
            urgency = "critical"
        elif any(urgent in response_lower for urgent in ['high', 'urgente', 'despido']):
            urgency = "high"
        elif any(medium in response_lower for medium in ['medium', 'medio']):
            urgency = "medium"
        else:
            urgency = "low"
        
        # Detectar dominios
        domains = []
        if any(term in response_lower for term in ['trabajo', 'laboral', 'empleado']):
            domains.append("laboral")
        if any(term in response_lower for term in ['civil', 'contrato', 'propiedad']):
            domains.append("civil")
        if any(term in response_lower for term in ['penal', 'delito']):
            domains.append("penal")
        
        if not domains:
            domains = ["laboral"]
        
        return {
            "query_type": query_type,
            "urgency_level": urgency,
            "confidence": 0.7,  # Confianza reducida para extracción manual
            "legal_domains": domains,
            "key_entities": [],
            "emotional_indicators": [],
            "specific_articles": [],
            "reasoning": "Extracción manual de respuesta Llama",
            "recommended_specialist": self._get_default_value("recommended_specialist")
        }
    
    def _rule_based_fallback(self, query: str) -> Dict[str, Any]:
        """Clasificación de reglas CORREGIDA cuando Llama no está disponible"""
        print("🔄 Usando clasificación de reglas como fallback...")
        
        query_lower = query.lower()
        
        # Detectar tipo de consulta
        article_patterns = [
            r'artículo\s*\d+',
            r'articulo\s*\d+',
            r'art\.?\s*\d+'
        ]
        
        is_article_request = any(re.search(pattern, query_lower) for pattern in article_patterns)
        
        if is_article_request:
            query_type = "article_lookup"
            urgency = "low"
            specialist = "article_specialist"
        elif any(indicator in query_lower for indicator in ['fui', 'me', 'despidieron', 'discriminan']):
            query_type = "case_analysis"
            urgency = "high"
            specialist = "case_analyst"
        elif any(proc in query_lower for proc in ['cómo', 'como', 'procedimiento', 'denuncia']):
            query_type = "procedural_guidance"
            urgency = "medium"
            specialist = "procedural_guide"
        else:
            query_type = "general_consultation"
            urgency = "low"
            specialist = "general_counselor"
        
        # MEJORAR: Detectar dominios más precisamente
        domains = []
        
        # Detectar dominio constitucional PRIMERO
        if any(term in query_lower for term in ['constitución', 'constitucion', 'constitucional', 'carta magna']):
            domains = ['constitucional']
        elif any(term in query_lower for term in ['penal', 'código penal', 'codigo penal', 'delito']):
            domains = ['penal']
        elif any(term in query_lower for term in ['civil', 'código civil', 'codigo civil']):
            domains = ['civil']
        elif any(term in query_lower for term in ['trabajo', 'laboral', 'empleado', 'despido', 'lct']):
            domains = ['laboral']
        elif any(term in query_lower for term in ['familia', 'divorcio', 'custodia']):
            domains = ['familia']
        else:
            # Para artículos sin contexto, usar 'general'
            domains = ['general']
        
        return {
            "query_type": query_type,
            "urgency_level": urgency,
            "confidence": 0.8,
            "legal_domains": domains,
            "key_entities": [],
            "emotional_indicators": [],
            "specific_articles": self._extract_articles_regex(query),
            "reasoning": f"Clasificación basada en reglas. Artículo: {is_article_request}, Dominio: {domains}",
            "recommended_specialist": specialist
        }
    
    def _extract_articles_regex(self, query: str) -> List[str]:
        """Extrae artículos específicos usando regex"""
        articles = []
        patterns = [
            r'artículo\s+(\d+)(?:\s+del?\s+(.+?))?',
            r'art\.?\s*(\d+)(?:\s+(.+?))?'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, query.lower())
            for match in matches:
                article_num = match.group(1)
                law_context = match.group(2) if len(match.groups()) > 1 and match.group(2) else ""
                
                if law_context:
                    articles.append(f"Art. {article_num} {law_context.strip()}")
                else:
                    articles.append(f"Art. {article_num}")
        
        return articles

# ========== INTEGRACIÓN CON EL SISTEMA EXISTENTE ==========

class EnhancedIntelligentLegalSystem:
    """
    Sistema legal inteligente mejorado que usa Llama real
    """
    
    def __init__(self, llama_config: Optional[Dict[str, Any]] = None):
        self.llama_config = llama_config or {}
        
        # Inicializar clasificador con Llama real
        try:
            self.real_llama_classifier = RealLlamaQueryClassifier(self.llama_config)
            self.use_real_llama = bool(self.real_llama_classifier.active_provider)
            
            if self.use_real_llama:
                print("🦙 Sistema inicializado con Llama REAL")
            else:
                print("⚠️ Llama no disponible, usando clasificación de reglas")
                
        except Exception as e:
            print(f"❌ Error inicializando Llama: {str(e)}")
            self.use_real_llama = False
            self.real_llama_classifier = None
        
        # Importar router del sistema existente
        from .legal_query_classifier import SpecialistRouter
        self.router = SpecialistRouter()
    
    def process_query_with_real_llama(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Procesa consulta usando Llama real si está disponible"""
        if context is None:
            context = {}
        
        print(f"\n🦙 SISTEMA LLAMA REAL - Procesando: '{query}'\n")
        
        # 1. Clasificar con Llama real o fallback
        start_time = time.time()
        
        if self.use_real_llama:
            try:
                classification_data = self.real_llama_classifier.classify_query_with_real_llama(query)
                classification_method = "llama_real"
            except Exception as e:
                print(f"❌ Error con Llama real: {str(e)}")
                classification_data = self.real_llama_classifier._rule_based_fallback(query)
                classification_method = "rule_fallback"
        else:
            classification_data = self._simple_rule_classification(query)
            classification_method = "simple_rules"
        
        classification_time = time.time() - start_time
        
        # 2. Crear objeto de clasificación compatible
        from .legal_query_classifier import QueryClassification, QueryType, UrgencyLevel
        
        classification = QueryClassification(
            query_type=QueryType(classification_data["query_type"]),
            urgency_level=UrgencyLevel(classification_data["urgency_level"]),
            confidence=classification_data["confidence"],
            legal_domains=classification_data["legal_domains"],
            key_entities=classification_data["key_entities"],
            emotional_indicators=classification_data["emotional_indicators"],
            specific_articles=classification_data["specific_articles"],
            reasoning=classification_data["reasoning"],
            recommended_specialist=classification_data["recommended_specialist"]
        )
        
        print(f"📊 CLASIFICACIÓN LLAMA COMPLETADA ({classification_time:.2f}s):")
        print(f"   🔧 Método: {classification_method}")
        print(f"   🏷️  Tipo: {classification.query_type.value}")
        print(f"   🚨 Urgencia: {classification.urgency_level.value}")
        print(f"   📈 Confianza: {classification.confidence:.2f}")
        print(f"   ⚖️  Dominios: {', '.join(classification.legal_domains)}")
        print(f"   🎯 Especialista: {classification.recommended_specialist}")
        
        # 3. Enrutar a especialista
        specialist_response = self.router.route_query(classification, query, context)
        
        # 4. Preparar respuesta completa
        complete_response = {
            "classification": {
                "query_type": classification.query_type.value,
                "urgency_level": classification.urgency_level.value,
                "confidence": classification.confidence,
                "legal_domains": classification.legal_domains,
                "reasoning": classification.reasoning,
                "classification_method": classification_method
            },
            "specialist_routing": specialist_response,
            "processing_time": classification_time,
            "timestamp": time.time(),
            "llama_available": self.use_real_llama
        }
        
        print(f"\n🎯 ROUTING COMPLETADO:")
        print(f"   🤖 Especialista: {specialist_response['specialist_type']}")
        print(f"   🔍 Estrategia: {specialist_response['search_strategy']}")
        print(f"   📋 Formato: {specialist_response['response_format']}")
        
        return complete_response
    
    def _simple_rule_classification(self, query: str) -> Dict[str, Any]:
        """Clasificación simple de reglas cuando Llama no está disponible"""
        query_lower = query.lower()
        
        # Tipo de consulta
        if re.search(r'artículo?\s+\d+|art\.?\s*\d+', query_lower):
            query_type = "article_lookup"
            urgency = "low"
        elif any(word in query_lower for word in ['fui', 'me', 'despidieron']):
            query_type = "case_analysis"
            urgency = "high"
        else:
            query_type = "general_consultation"
            urgency = "medium"
        
        return {
            "query_type": query_type,
            "urgency_level": urgency,
            "confidence": 0.5,
            "legal_domains": ["laboral"],
            "key_entities": [],
            "emotional_indicators": [],
            "specific_articles": [],
            "reasoning": "Clasificación simple de reglas",
            "recommended_specialist": "general_counselor"
        }
    
    def get_llama_status(self) -> Dict[str, Any]:
        """Obtiene estado de Llama y proveedores disponibles"""
        status = {
            "llama_available": self.use_real_llama,
            "active_provider": None,
            "available_providers": [],
            "configuration": self.llama_config
        }
        
        if self.real_llama_classifier:
            if self.real_llama_classifier.active_provider:
                status["active_provider"] = type(self.real_llama_classifier.active_provider).__name__
                status["model_name"] = getattr(self.real_llama_classifier.active_provider, 'model_name', 'unknown')
            
            status["available_providers"] = [
                {
                    "type": type(provider).__name__,
                    "available": provider.available,
                    "model": getattr(provider, 'model_name', 'unknown')
                }
                for provider in self.real_llama_classifier.providers
            ]
        
        return status

# ========== CONFIGURACIÓN E INSTALACIÓN ==========

def setup_llama_environment():
    """Guía para configurar el entorno Llama"""
    print("🦙 GUÍA DE CONFIGURACIÓN DE LLAMA")
    print("="*50)
    
    print("\n1. OLLAMA LOCAL (RECOMENDADO):")
    print("   - Instalar Ollama: https://ollama.ai/")
    print("   - Descargar modelo: `ollama pull llama2`")
    print("   - Ventajas: Rápido, privado, sin límites")
    
    print("\n2. HUGGING FACE API:")
    print("   - Obtener API key: https://huggingface.co/settings/tokens")
    print("   - Configurar: export HUGGINGFACE_API_KEY='tu-api-key'")
    print("   - Ventajas: No requiere instalación local")
    
    print("\n3. MODELO LOCAL CON TRANSFORMERS:")
    print("   - Instalar: pip install transformers torch")
    print("   - Ventajas: Control total, sin dependencias externas")
    
    print("\n4. CONFIGURACIÓN EN CÓDIGO:")
    print("""
llama_config = {
    "ollama_model": "llama2",
    "huggingface_api_key": "tu-api-key",
    "hf_model": "meta-llama/Llama-2-7b-chat-hf",
    "local_model": "microsoft/DialoGPT-medium"
}

system = EnhancedIntelligentLegalSystem(llama_config)
""")

def test_llama_integration():
    """Prueba la integración completa con Llama"""
    print("🧪 PRUEBA DE INTEGRACIÓN LLAMA")
    print("="*40)
    
    # Configuración de prueba
    test_config = {
        "ollama_model": "llama2",
        "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY"),
        "local_model": "microsoft/DialoGPT-medium"
    }
    
    # Crear sistema
    system = EnhancedIntelligentLegalSystem(test_config)
    
    # Obtener estado
    status = system.get_llama_status()
    print(f"🔍 Estado de Llama: {status}")
    
    # Consultas de prueba
    test_queries = [
        "¿Cuál es el artículo 14 del código penal?",
        "Fui despedida sin indemnización por estar embarazada",
        "¿Cómo presento una denuncia laboral?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*30}")
        print(f"PRUEBA {i}: {query}")
        print(f"{'='*30}")
        
        try:
            result = system.process_query_with_real_llama(query)
            print(f"✅ Procesamiento exitoso")
            print(f"   Tipo: {result['classification']['query_type']}")
            print(f"   Método: {result['classification']['classification_method']}")
            print(f"   Confianza: {result['classification']['confidence']:.2f}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # Mostrar guía de configuración
    setup_llama_environment()
    
    print("\n" + "="*50)
    print("Ejecutando prueba de integración...")
    print("="*50)
    
    # Ejecutar prueba
    test_llama_integration()