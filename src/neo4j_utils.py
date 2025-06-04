"""
Versión mejorada de neo4j_utils.py que integra el Graph RAG Avanzado optimizado
y aprovecha mejor los metadatos ricos (tags, references, penalty).
"""
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import re
import contextlib
import json

def connect_neo4j(uri: str, username: str, password: str) -> GraphDatabase.driver:
    """
    Connect to Neo4j database.
    
    Args:
        uri: URI of the Neo4j instance
        username: Username for authentication
        password: Password for authentication
        
    Returns:
        Neo4j driver instance
    """
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        # Verify connection
        with driver.session() as session:
            session.run("RETURN 1")
        return driver
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Neo4j: {str(e)}")

@contextlib.contextmanager
def get_session(driver):
    """
    Context manager for Neo4j sessions to ensure proper closing.
    
    Args:
        driver: Neo4j driver instance
        
    Yields:
        Neo4j session
    """
    session = None
    try:
        session = driver.session()
        yield session
    finally:
        if session:
            session.close()

def create_enhanced_neo4j_nodes(driver: GraphDatabase.driver, documents: List[Dict[str, Any]]) -> List[str]:
    """
    Crea nodos Article mejorados aprovechando todos los metadatos disponibles.
    Incluye tags, references, penalty y otros campos ricos.
    """
    created_ids = []
    
    with get_session(driver) as session:
        for doc in documents:
            # Extraer metadatos completos
            metadata = doc.get("metadata", {})
            references = doc.get("references", [])
            
            # Generar article_id
            article_number = metadata.get("article", "")
            code = metadata.get("code", "")
            article_id = f"{code}_{article_number}" if code and article_number else ""
            
            if not article_id:
                continue
            
            # Preparar propiedades enriquecidas
            properties = {
                "article_id": article_id,
                "content": doc.get("content", ""),
                "law_name": code,
                "article_number": article_number,
                "category": metadata.get("chapter", ""),
                "section": metadata.get("section", ""),
                "source": metadata.get("section", ""),
                # Nuevos campos enriquecidos
                "tags": json.dumps(metadata.get("tags", [])),  # Almacenar como JSON
                "penalty": json.dumps(metadata.get("penalty", [])),
                "references_count": len(references),
                "has_penalties": len(metadata.get("penalty", [])) > 0,
                "tag_count": len(metadata.get("tags", []))
            }
            
            # Crear o actualizar el nodo Article con propiedades enriquecidas
            query = """
            MERGE (a:Article {article_id: $article_id})
            SET a.content = $content,
                a.law_name = $law_name,
                a.article_number = $article_number,
                a.category = $category,
                a.section = $section,
                a.source = $source,
                a.tags = $tags,
                a.penalty = $penalty,
                a.references_count = $references_count,
                a.has_penalties = $has_penalties,
                a.tag_count = $tag_count
            RETURN a.article_id
            """
            
            try:
                result = session.run(query, **properties)
                record = result.single()
                if record:
                    created_ids.append(record[0])
            except Exception as e:
                print(f"Error creando nodo {article_id}: {str(e)}")
                continue
    
    print(f"✅ Creados {len(created_ids)} nodos Article enriquecidos")
    return created_ids

def create_enhanced_relationships(driver: GraphDatabase.driver, documents: List[Dict[str, Any]]) -> None:
    """
    Crea relaciones enriquecidas aprovechando tags, references y otros metadatos.
    """
    print("🔗 Creando relaciones enriquecidas...")
    
    with get_session(driver) as session:
        # 1. Relaciones por Tags compartidos (MUY IMPORTANTE)
        print("   📌 Creando relaciones por tags compartidos...")
        create_tag_relationships(session, documents)
        
        # 2. Relaciones por Referencias explícitas
        print("   🔗 Creando relaciones por referencias...")
        create_reference_relationships(session, documents)
        
        # 3. Relaciones por Penalties similares
        print("   ⚖️ Creando relaciones por penalties...")
        create_penalty_relationships(session, documents)
        
        # 4. Relaciones por Sección/Chapter
        print("   📂 Creando relaciones por sección...")
        create_section_relationships(session)
        
        # 5. Relaciones semánticas avanzadas
        print("   🧠 Creando relaciones semánticas avanzadas...")
        create_advanced_semantic_relationships(session)

def create_tag_relationships(session, documents: List[Dict[str, Any]]) -> None:
    """
    Crea relaciones basadas en tags compartidos entre artículos.
    Los tags son muy valiosos para determinar temas relacionados.
    """
    # Mapear tags a artículos
    tag_to_articles = {}
    
    for doc in documents:
        metadata = doc.get("metadata", {})
        tags = metadata.get("tags", [])
        article_id = f"{metadata.get('code', '')}_{metadata.get('article', '')}"
        
        if not article_id.strip("_"):
            continue
            
        for tag in tags:
            if tag and len(tag.strip()) > 2:  # Filtrar tags muy cortos
                tag_normalized = tag.lower().strip()
                if tag_normalized not in tag_to_articles:
                    tag_to_articles[tag_normalized] = []
                tag_to_articles[tag_normalized].append(article_id)
    
    # Crear relaciones entre artículos que comparten tags
    for tag, article_ids in tag_to_articles.items():
        if len(article_ids) > 1 and len(article_ids) <= 20:  # Solo si hay al menos 2 artículos con el mismo tag
            # Crear nodo Tag si no existe
            try:
                query_create_tag = """
                MERGE (t:Tag {name: $tag_name})
                SET t.article_count = $article_count
                """
                session.run(query_create_tag, tag_name=tag, article_count=len(article_ids))
                
                # Conectar artículos al tag
                for article_id in article_ids:
                    query_connect = """
                    MATCH (a:Article {article_id: $article_id})
                    MATCH (t:Tag {name: $tag_name})
                    MERGE (a)-[r:HAS_TAG]->(t)
                    """
                    session.run(query_connect, article_id=article_id, tag_name=tag)
                
                # Crear relaciones directas entre artículos que comparten tags importantes
                for i, art1 in enumerate(article_ids):
                    for art2 in article_ids[i+1:]:
                        query_shared_tag = """
                        MATCH (a1:Article {article_id: $art1})
                        MATCH (a2:Article {article_id: $art2})
                        MERGE (a1)-[r:SHARES_TAG {tag: $tag_name, strength: 1.0}]->(a2)
                        """
                        session.run(query_shared_tag, art1=art1, art2=art2, tag_name=tag)
            except Exception as e:
                print(f"Error creando relaciones para tag {tag}: {str(e)}")
                continue

def create_reference_relationships(session, documents: List[Dict[str, Any]]) -> None:
    """
    Crea relaciones basadas en referencias explícitas entre artículos.
    """
    for doc in documents:
        metadata = doc.get("metadata", {})
        references = doc.get("references", [])
        source_article_id = f"{metadata.get('code', '')}_{metadata.get('article', '')}"
        
        if not source_article_id.strip("_"):
            continue
            
        for ref in references:
            if isinstance(ref, dict):
                # Si la referencia es un diccionario con metadatos
                ref_code = ref.get("code", "")
                ref_article = ref.get("article", "")
                ref_article_id = f"{ref_code}_{ref_article}" if ref_code and ref_article else ""
            elif isinstance(ref, str):
                # Si la referencia es solo un string, intentar parsearlo
                ref_article_id = ref.strip()
            else:
                continue
            
            if ref_article_id and ref_article_id != source_article_id:
                # Crear relación de referencia explícita
                query_reference = """
                MATCH (source:Article {article_id: $source_id})
                MATCH (target:Article {article_id: $target_id})
                MERGE (source)-[r:REFERENCES {type: 'explicit', strength: 3.0}]->(target)
                """
                try:
                    session.run(query_reference, source_id=source_article_id, target_id=ref_article_id)
                except:
                    # Si el artículo referenciado no existe, crear un nodo placeholder
                    query_placeholder = """
                    MERGE (placeholder:Article {article_id: $ref_id})
                    SET placeholder.is_placeholder = true
                    
                    WITH placeholder
                    MATCH (source:Article {article_id: $source_id})
                    MERGE (source)-[r:REFERENCES {type: 'placeholder', strength: 1.0}]->(placeholder)
                    """
                    session.run(query_placeholder, ref_id=ref_article_id, source_id=source_article_id)

def create_penalty_relationships(session, documents: List[Dict[str, Any]]) -> None:
    """
    Crea relaciones entre artículos que tienen penalties similares o relacionados.
    """
    # Agrupar artículos por tipos de penalty
    penalty_groups = {}
    
    for doc in documents:
        metadata = doc.get("metadata", {})
        penalties = metadata.get("penalty", [])
        article_id = f"{metadata.get('code', '')}_{metadata.get('article', '')}"
        
        if not article_id.strip("_") or not penalties:
            continue
            
        for penalty in penalties:
            if isinstance(penalty, dict):
                penalty_type = penalty.get("type", "").lower()
                penalty_amount = penalty.get("amount", "")
            elif isinstance(penalty, str):
                penalty_type = penalty.lower()
                penalty_amount = ""
            else:
                continue
                
            if penalty_type:
                if penalty_type not in penalty_groups:
                    penalty_groups[penalty_type] = []
                penalty_groups[penalty_type].append({
                    'article_id': article_id,
                    'amount': penalty_amount
                })
    
    # Crear relaciones entre artículos con penalties similares
    for penalty_type, articles in penalty_groups.items():
        if len(articles) > 1 and len(articles) <= 15:  # Limitar para evitar demasiadas relaciones
            try:
                # Crear nodo PenaltyType
                query_penalty_type = """
                MERGE (pt:PenaltyType {name: $penalty_type})
                SET pt.article_count = $article_count
                """
                session.run(query_penalty_type, penalty_type=penalty_type, article_count=len(articles))
                
                # Conectar artículos al tipo de penalty
                for article_info in articles:
                    query_connect_penalty = """
                    MATCH (a:Article {article_id: $article_id})
                    MATCH (pt:PenaltyType {name: $penalty_type})
                    MERGE (a)-[r:HAS_PENALTY_TYPE {amount: $amount}]->(pt)
                    """
                    session.run(query_connect_penalty, 
                               article_id=article_info['article_id'],
                               penalty_type=penalty_type,
                               amount=str(article_info['amount']))
                
                # Crear relaciones directas entre artículos con mismo tipo de penalty
                for i, art1 in enumerate(articles):
                    for art2 in articles[i+1:]:
                        query_similar_penalty = """
                        MATCH (a1:Article {article_id: $art1_id})
                        MATCH (a2:Article {article_id: $art2_id})
                        MERGE (a1)-[r:SIMILAR_PENALTY {penalty_type: $penalty_type, strength: 2.0}]->(a2)
                        """
                        session.run(query_similar_penalty, 
                                   art1_id=art1['article_id'],
                                   art2_id=art2['article_id'],
                                   penalty_type=penalty_type)
            except Exception as e:
                print(f"Error creando relaciones para penalty {penalty_type}: {str(e)}")
                continue

def create_section_relationships(session) -> None:
    """
    Crea relaciones jerárquicas basadas en secciones y capítulos.
    """
    try:
        # Relaciones dentro de la misma sección
        query_same_section = """
        MATCH (a1:Article), (a2:Article)
        WHERE a1.section = a2.section 
        AND a1.section IS NOT NULL 
        AND a1.section <> ''
        AND a1.article_id <> a2.article_id
        AND NOT EXISTS((a1)-[:SAME_SECTION]-(a2))
        MERGE (a1)-[r:SAME_SECTION {strength: 1.5}]->(a2)
        """
        session.run(query_same_section)
        
        # Relaciones dentro del mismo capítulo
        query_same_chapter = """
        MATCH (a1:Article), (a2:Article)
        WHERE a1.category = a2.category 
        AND a1.category IS NOT NULL 
        AND a1.category <> ''
        AND a1.article_id <> a2.article_id
        AND a1.law_name <> a2.law_name  
        AND NOT EXISTS((a1)-[:SAME_CHAPTER]-(a2))
        MERGE (a1)-[r:SAME_CHAPTER {strength: 1.2}]->(a2)
        """
        session.run(query_same_chapter)
    except Exception as e:
        print(f"Error creando relaciones de sección: {str(e)}")

def create_advanced_semantic_relationships(session) -> None:
    """
    Crea relaciones semánticas avanzadas basadas en patrones de contenido.
    """
    try:
        # 1. Relaciones por patrones de definición
        query_definitions = """
        MATCH (def:Article)
        WHERE toLower(def.content) CONTAINS 'se entiende por' 
           OR toLower(def.content) CONTAINS 'definición'
           OR toLower(def.content) CONTAINS 'concepto'
        
        MATCH (related:Article)
        WHERE related.article_id <> def.article_id
        AND any(word IN split(toLower(def.content), ' ') 
                WHERE word IN split(toLower(related.content), ' ') 
                AND length(word) > 4)
        
        MERGE (def)-[r:DEFINES_CONCEPT {strength: 2.5}]->(related)
        """
        session.run(query_definitions)
    except Exception as e:
        print(f"Error en relaciones de definición: {str(e)}")
    
    try:
        # 2. Relaciones por patrones de excepción
        query_exceptions = """
        MATCH (exc:Article)
        WHERE toLower(exc.content) CONTAINS 'excepción'
           OR toLower(exc.content) CONTAINS 'salvo'
           OR toLower(exc.content) CONTAINS 'sin perjuicio'
        
        MATCH (general:Article)
        WHERE general.article_id <> exc.article_id
        AND general.category = exc.category
        AND NOT toLower(general.content) CONTAINS 'excepción'
        
        MERGE (exc)-[r:EXCEPTION_TO {strength: 2.8}]->(general)
        """
        session.run(query_exceptions)
    except Exception as e:
        print(f"Error en relaciones de excepción: {str(e)}")

def search_neo4j_enhanced(driver: GraphDatabase.driver, query: str, limit: int = 15) -> List[Dict[str, Any]]:
    """
    Búsqueda mejorada que usa el sistema optimizado de Graph RAG.
    """
    print(f"🔍 Iniciando búsqueda Neo4j optimizada para: '{query}'")
    
    try:
        # Importar el sistema optimizado
        from .advanced_graph_rag_optimized import optimized_advanced_neo4j_search
        
        # Usar el Graph RAG optimizado como método principal
        optimized_results = optimized_advanced_neo4j_search(driver, query, limit)
        
        if optimized_results:
            return optimized_results
        else:
            # Fallback a búsqueda tradicional
            return search_neo4j_traditional(driver, query, limit)
            
    except ImportError:
        print("Sistema optimizado no disponible, usando búsqueda tradicional...")
        return search_neo4j_traditional(driver, query, limit)
    except Exception as e:
        print(f"Error en búsqueda optimizada: {str(e)}")
        return search_neo4j_traditional(driver, query, limit)

def search_neo4j_traditional(driver: GraphDatabase.driver, query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Búsqueda tradicional mejorada que aprovecha los nuevos metadatos.
    """
    results = []
    query_words = [word.lower() for word in query.split() if len(word) > 2]
    
    if not query_words:
        return results
    
    with get_session(driver) as session:
        cypher_query = """
        MATCH (a:Article)
        WHERE """ + " OR ".join([f"toLower(a.content) CONTAINS ${i}" for i in range(len(query_words))]) + """
        
        WITH a,
             // Score base por coincidencias en contenido
             size([word IN $words WHERE toLower(a.content) CONTAINS word]) as content_matches,
             
             // Boost por tags relevantes
             CASE 
                WHEN any(word IN $words WHERE toLower(a.tags) CONTAINS word)
                THEN 3.0 ELSE 0.0 
             END as tag_boost,
             
             // Boost por penalties (importante para consultas sobre sanciones)
             CASE 
                WHEN any(word IN ['sanción', 'pena', 'multa'] WHERE word IN $words)
                AND a.has_penalties = true
                THEN 2.0 ELSE 0.0 
             END as penalty_boost,
             
             // Boost por referencias (artículos muy referenciados son importantes)
             CASE 
                WHEN a.references_count > 0 THEN a.references_count * 0.1 
                ELSE 0.0 
             END as reference_boost
        
        WITH a, 
             (toFloat(content_matches) / size($words)) * 5.0 + 
             tag_boost + 
             penalty_boost + 
             reference_boost as relevance_score
        
        WHERE relevance_score > 0.5
        
        RETURN a.article_id as article_id,
               a.content as content,
               a.law_name as law_name,
               a.article_number as article_number,
               a.category as category,
               a.source as source,
               relevance_score as score
        
        ORDER BY relevance_score DESC
        LIMIT $limit
        """
        
        params = {
            **{str(i): word for i, word in enumerate(query_words)},
            'words': query_words,
            'limit': limit
        }
        
        try:
            result = session.run(cypher_query, params)
            for record in result:
                results.append({
                    "article_id": record["article_id"],
                    "content": record["content"],
                    "law_name": record["law_name"],
                    "article_number": record["article_number"],
                    "category": record["category"],
                    "source": record["source"],
                    "score": float(record["score"]) * 3,  # Escalar para comparar con otros métodos
                    "method": "neo4j_traditional_enhanced"
                })
        except Exception as e:
            print(f"Error en búsqueda tradicional: {str(e)}")
    
    return results

def check_data_exists(driver: GraphDatabase.driver) -> bool:
    """
    Checks if data already exists in the Neo4j database.
    """
    with get_session(driver) as session:
        query = """
        MATCH (a:Article)
        RETURN count(a) as article_count
        """
        result = session.run(query)
        record = result.single()
        if record and record["article_count"] > 0:
            return True
    return False

def clear_neo4j_data(driver: GraphDatabase.driver) -> None:
    """
    Clears all data from the Neo4j database.
    """
    print("ADVERTENCIA: Eliminando todos los datos de Neo4j...")
    
    with get_session(driver) as session:
        query = """
        MATCH (n)
        DETACH DELETE n
        """
        session.run(query)
        print("Todos los datos eliminados de la base de datos Neo4j.")

def setup_enhanced_neo4j_data(driver: GraphDatabase.driver, documents: List[Dict[str, Any]]) -> None:
    """
    Configura Neo4j con el sistema mejorado que aprovecha todos los metadatos.
    """
    print("\n=== Configurando Neo4j con Sistema Mejorado ===")
    
    # Verificar si ya existen datos
    if check_data_exists(driver):
        print("Ya existen datos en Neo4j.")
        choice = input("¿Desea limpiar y recargar todos los datos? (s/n): ")
        if choice.lower() == 's':
            clear_neo4j_data(driver)
        else:
            print("Manteniendo datos existentes y agregando relaciones mejoradas...")
    
    # 1. Crear nodos enriquecidos
    print("📊 Creando nodos Article enriquecidos...")
    article_ids = create_enhanced_neo4j_nodes(driver, documents)
    
    # 2. Crear relaciones tradicionales (leyes, etc.)
    print("🏛️ Creando relaciones básicas de leyes...")
    create_basic_law_relationships(driver, documents)
    
    # 3. Crear relaciones enriquecidas
    print("🚀 Creando relaciones enriquecidas...")
    create_enhanced_relationships(driver, documents)
    
    print(f"\n✅ Configuración mejorada completada:")
    print(f"   - {len(article_ids)} artículos enriquecidos")
    print(f"   - Relaciones por tags, referencias y penalties")
    print(f"   - Sistema Graph RAG optimizado activado")

def create_basic_law_relationships(driver: GraphDatabase.driver, documents: List[Dict[str, Any]]) -> None:
    """
    Crea relaciones básicas de leyes y códigos.
    """
    law_articles = {}
    
    for doc in documents:
        metadata = doc.get("metadata", {})
        law_name = metadata.get("code", "")
        article_number = metadata.get("article", "")
        article_id = f"{law_name}_{article_number}" if law_name and article_number else ""
        
        if law_name and article_id:
            if law_name not in law_articles:
                law_articles[law_name] = []
            law_articles[law_name].append(article_id)
    
    with get_session(driver) as session:
        for law_name, article_ids in law_articles.items():
            try:
                # Crear nodo Law
                query_law = """
                MERGE (l:Law {name: $law_name})
                SET l.article_count = $article_count
                """
                session.run(query_law, law_name=law_name, article_count=len(article_ids))
                
                # Crear relaciones CONTAINS en lotes
                for i in range(0, len(article_ids), 100):
                    batch_ids = article_ids[i:i+100]
                    query_contains = """
                    MATCH (l:Law {name: $law_name})
                    UNWIND $article_ids as article_id
                    MATCH (a:Article {article_id: article_id})
                    MERGE (l)-[r:CONTAINS]->(a)
                    """
                    session.run(query_contains, law_name=law_name, article_ids=batch_ids)
            except Exception as e:
                print(f"Error creando relaciones para ley {law_name}: {str(e)}")
                continue

# Funciones de compatibilidad para mantener la API existente
def create_neo4j_nodes(driver: GraphDatabase.driver, documents: List[Dict[str, Any]]) -> List[str]:
    """Función de compatibilidad que usa el sistema mejorado."""
    return create_enhanced_neo4j_nodes(driver, documents)

def search_neo4j(driver: GraphDatabase.driver, query_params: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Búsqueda optimizada en Neo4j que evita timeouts y mejora performance.
    """
    results = []
    
    # Extraer query string de los parámetros
    query = ""
    if "keywords" in query_params:
        query = " ".join(query_params["keywords"])
    
    if not query:
        return results
    
    print(f"🔍 Búsqueda Neo4j optimizada para: '{query[:50]}{'...' if len(query) > 50 else ''}'")
    
    # Optimizar términos de búsqueda
    query_words = [word.lower() for word in query.split() if len(word) > 3]
    # Limitar a máximo 5 términos para evitar consultas muy complejas
    query_words = query_words[:5]
    
    if not query_words:
        return results
    
    with get_session(driver) as session:
        try:
            # Consulta optimizada con timeout y límites
            cypher_query = """
            MATCH (a:Article)
            WHERE """ + " OR ".join([f"toLower(a.content) CONTAINS $word{i}" for i in range(len(query_words))]) + """
            
            WITH a,
                 // Contar coincidencias de términos
                 size([word IN $words WHERE toLower(a.content) CONTAINS word]) as term_matches,
                 
                 // Boost por metadatos si existen
                 CASE 
                    WHEN a.has_penalties = true AND any(word IN $words WHERE word IN ['sanción', 'pena', 'multa']) 
                    THEN 2.0 
                    ELSE 0.0 
                 END as penalty_boost,
                 
                 // Boost por tags si existen
                 CASE 
                    WHEN a.tag_count > 2 THEN 1.0 
                    ELSE 0.0 
                 END as tag_boost
            
            WITH a, 
                 (toFloat(term_matches) / size($words)) * 10.0 + penalty_boost + tag_boost as relevance_score
            
            WHERE relevance_score > 1.0
            
            RETURN a.article_id as article_id,
                   a.content as content,
                   a.law_name as law_name,
                   a.article_number as article_number,
                   a.category as category,
                   a.source as source,
                   relevance_score as score
            
            ORDER BY relevance_score DESC
            LIMIT $limit
            """
            
            # Preparar parámetros
            params = {
                **{f'word{i}': word for i, word in enumerate(query_words)},
                'words': query_words,
                'limit': limit
            }
            
            # Ejecutar con timeout de 10 segundos
            result = session.run(cypher_query, params, timeout=10)
            
            for record in result:
                article = {
                    "article_id": record["article_id"],
                    "content": record["content"],
                    "law_name": record["law_name"],
                    "article_number": record["article_number"],
                    "category": record["category"],
                    "source": record["source"],
                    "score": float(record["score"]) * 3,  # Escalar para comparar con otros métodos
                    "method": "neo4j_optimized"
                }
                results.append(article)
                
            print(f"   ✅ Neo4j encontró {len(results)} resultados")
            
        except Exception as e:
            print(f"   ❌ Error en Neo4j (usando fallback): {str(e)}")
            # Fallback a búsqueda más simple
            try:
                simple_query = """
                MATCH (a:Article)
                WHERE toLower(a.content) CONTAINS $main_term
                RETURN a.article_id as article_id,
                       a.content as content,
                       a.law_name as law_name,
                       a.article_number as article_number,
                       a.category as category,
                       a.source as source,
                       1.0 as score
                ORDER BY a.article_id
                LIMIT $limit
                """
                
                # Usar el término más largo como término principal
                main_term = max(query_words, key=len) if query_words else query.split()[0].lower()
                
                result = session.run(simple_query, {
                    'main_term': main_term,
                    'limit': min(limit, 10)
                }, timeout=5)
                
                for record in result:
                    article = {
                        "article_id": record["article_id"],
                        "content": record["content"],
                        "law_name": record["law_name"],
                        "article_number": record["article_number"],
                        "category": record["category"],
                        "source": record["source"],
                        "score": float(record["score"]),
                        "method": "neo4j_simple_fallback"
                    }
                    results.append(article)
                    
                print(f"   🔄 Fallback Neo4j: {len(results)} resultados")
                
            except Exception as e2:
                print(f"   ❌ Error en fallback Neo4j: {str(e2)}")
    
    return results