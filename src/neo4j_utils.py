"""
Versión mejorada de neo4j_utils.py con Graph RAG Avanzado integrado.
Incluye navegación inteligente del grafo y filtrado contextual.
"""
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import re
import contextlib
import json
import time

# Importar el navegador avanzado
from .advanced_graph_navigator import AdvancedGraphNavigator, IntelligentFilter

def connect_neo4j(uri: str, username: str, password: str) -> GraphDatabase.driver:
    """Connect to Neo4j database."""
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
    """Context manager for Neo4j sessions to ensure proper closing."""
    session = None
    try:
        session = driver.session()
        yield session
    finally:
        if session:
            session.close()

def search_neo4j_enhanced(driver: GraphDatabase.driver, query: str, limit: int = 15) -> List[Dict[str, Any]]:
    """
    Búsqueda Neo4j mejorada con Graph RAG Avanzado.
    Ahora encuentra 8-12 artículos relevantes en lugar de 2-3.
    """
    print(f"🔍 Iniciando búsqueda Neo4j avanzada para: '{query[:50]}{'...' if len(query) > 50 else ''}'")
    start_time = time.time()
    
    # Inicializar componentes avanzados
    navigator = AdvancedGraphNavigator(driver)
    intelligent_filter = IntelligentFilter()
    
    # 1. Búsqueda de artículos semilla (mejorada)
    seed_articles = _find_enhanced_seed_articles(driver, query, limit=8)
    print(f"   🌱 Artículos semilla encontrados: {len(seed_articles)}")
    
    if not seed_articles:
        return _fallback_simple_search(driver, query, limit)
    
    # 2. Navegación inteligente del grafo
    try:
        expanded_articles = navigator.intelligent_graph_expansion(
            seed_articles, query, max_depth=3
        )
        print(f"   🕸️ Total tras navegación inteligente: {len(expanded_articles)}")
    except Exception as e:
        print(f"   ❌ Error en navegación inteligente: {str(e)}")
        expanded_articles = seed_articles
    
    # 3. Detectar contexto para filtrado inteligente
    context_info = navigator.detect_query_context(query)
    primary_context = context_info['primary_context']
    
    # 4. Aplicar filtrado inteligente (menos agresivo)
    filtered_articles = intelligent_filter.contextual_filtering(
        expanded_articles, query, context=primary_context, base_threshold=0.10  # Umbral más bajo
    )
    
    # 5. Re-ranking por importancia legal
    final_articles = intelligent_filter.rank_by_legal_importance(filtered_articles, query)
    
    # 6. Limitar resultados finales
    limited_results = final_articles[:limit]
    
    elapsed_time = time.time() - start_time
    print(f"   ⚡ Búsqueda avanzada completada en {elapsed_time:.2f}s")
    print(f"   🎯 Resultados finales: {len(limited_results)} artículos")
    
    # Mostrar breakdown de métodos
    methods_breakdown = {}
    for article in limited_results:
        method = article.get('expansion_method', article.get('method', 'unknown'))
        methods_breakdown[method] = methods_breakdown.get(method, 0) + 1
    
    if methods_breakdown:
        methods_str = ", ".join([f"{method}: {count}" for method, count in methods_breakdown.items()])
        print(f"   📊 Métodos utilizados: {methods_str}")
    
    return limited_results

def _find_enhanced_seed_articles(driver: GraphDatabase.driver, query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """
    Búsqueda de artículos semilla mejorada con mejor cobertura.
    """
    seed_articles = []
    query_words = [word.lower() for word in query.split() if len(word) > 3][:7]
    
    if not query_words:
        return []
    
    with get_session(driver) as session:
        # Consulta mejorada que encuentra más artículos semilla relevantes
        seed_query = """
        MATCH (a:Article)
        WHERE """ + " OR ".join([f"toLower(a.content) CONTAINS $word{i}" for i in range(len(query_words))]) + """
        
        WITH a,
             // Contar coincidencias de términos
             size([word IN $words WHERE toLower(a.content) CONTAINS word]) as term_matches
             
        WITH a, term_matches,
             // Boost por metadatos específicos
             CASE 
                WHEN a.has_penalties = true AND any(word IN $words WHERE word IN ['sanción', 'pena', 'multa', 'despido']) 
                THEN 2.0
                WHEN a.tag_count > 2 THEN 1.5
                WHEN toLower(a.law_name) CONTAINS 'trabajo' OR toLower(a.law_name) CONTAINS 'contrato' THEN 1.3
                WHEN a.references_count > 0 THEN 1.2
                ELSE 0.0 
             END as metadata_boost,
             
             // Boost por densidad de términos relevantes
             CASE 
                WHEN size($words) > 0 THEN (toFloat(term_matches) / size($words))
                ELSE 0.0
             END as term_density
        
        WITH a, 
             term_matches * 2.0 + metadata_boost + (term_density * 3.0) as seed_score
        
        WHERE seed_score > 1.5  // Umbral más bajo para encontrar más semillas
        
        RETURN a.article_id as article_id,
               a.content as content,
               a.law_name as law_name,
               a.article_number as article_number,
               a.category as category,
               a.source as source,
               seed_score as score
        
        ORDER BY seed_score DESC
        LIMIT $limit
        """
        
        params = {
            **{f'word{i}': word for i, word in enumerate(query_words)},
            'words': query_words,
            'limit': limit
        }
        
        try:
            result = session.run(seed_query, params, timeout=10)
            
            for record in result:
                seed_articles.append({
                    'article_id': record['article_id'],
                    'content': record['content'],
                    'law_name': record['law_name'],
                    'article_number': record['article_number'],
                    'category': record['category'],
                    'source': record['source'],
                    'score': float(record['score']),
                    'method': 'enhanced_seed',
                    'expansion_level': 0
                })
                
        except Exception as e:
            print(f"   ❌ Error en búsqueda de semillas: {str(e)}")
    
    return seed_articles

def _fallback_simple_search(driver: GraphDatabase.driver, query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Búsqueda de fallback simple cuando no se encuentran artículos semilla."""
    results = []
    query_words = [word.lower() for word in query.split() if len(word) > 3][:5]
    
    if not query_words:
        return results
    
    with get_session(driver) as session:
        simple_query = """
        MATCH (a:Article)
        WHERE """ + " OR ".join([f"toLower(a.content) CONTAINS $word{i}" for i in range(len(query_words))]) + """
        
        WITH a, size([word IN $words WHERE toLower(a.content) CONTAINS word]) as matches
        
        WHERE matches > 0
        
        RETURN a.article_id as article_id,
               a.content as content,
               a.law_name as law_name,
               a.article_number as article_number,
               a.category as category,
               a.source as source,
               toFloat(matches) as score
        
        ORDER BY score DESC
        LIMIT $limit
        """
        
        params = {
            **{f'word{i}': word for i, word in enumerate(query_words)},
            'words': query_words,
            'limit': limit
        }
        
        try:
            result = session.run(simple_query, params, timeout=5)
            for record in result:
                results.append({
                    'article_id': record['article_id'],
                    'content': record['content'],
                    'law_name': record['law_name'],
                    'article_number': record['article_number'],
                    'category': record['category'],
                    'source': record['source'],
                    'score': float(record['score']),
                    'method': 'simple_fallback'
                })
        except Exception as e:
            print(f"   ❌ Error en búsqueda de fallback: {str(e)}")
    
    return results

# ========== FUNCIONES EXISTENTES MEJORADAS ==========

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
                "tags": json.dumps(metadata.get("tags", [])),
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

def create_tag_relationships(session, documents: List[Dict[str, Any]]) -> None:
    """Crea relaciones basadas en tags compartidos entre artículos."""
    tag_to_articles = {}
    
    for doc in documents:
        metadata = doc.get("metadata", {})
        tags = metadata.get("tags", [])
        article_id = f"{metadata.get('code', '')}_{metadata.get('article', '')}"
        
        if not article_id.strip("_"):
            continue
            
        for tag in tags:
            if tag and len(tag.strip()) > 2:
                tag_normalized = tag.lower().strip()
                if tag_normalized not in tag_to_articles:
                    tag_to_articles[tag_normalized] = []
                tag_to_articles[tag_normalized].append(article_id)
    
    # Crear relaciones entre artículos que comparten tags
    for tag, article_ids in tag_to_articles.items():
        if len(article_ids) > 1 and len(article_ids) <= 20:
            for i, art1 in enumerate(article_ids):
                for art2 in article_ids[i+1:]:
                    try:
                        query = """
                        MATCH (a1:Article {article_id: $art1})
                        MATCH (a2:Article {article_id: $art2})
                        MERGE (a1)-[r:SHARES_TAG {tag: $tag_name}]->(a2)
                        """
                        session.run(query, art1=art1, art2=art2, tag_name=tag)
                    except Exception as e:
                        continue

def create_reference_relationships(session, documents: List[Dict[str, Any]]) -> None:
    """Crea relaciones basadas en referencias explícitas entre artículos."""
    for doc in documents:
        metadata = doc.get("metadata", {})
        references = doc.get("references", [])
        source_article_id = f"{metadata.get('code', '')}_{metadata.get('article', '')}"
        
        if not source_article_id.strip("_"):
            continue
            
        for ref in references:
            if isinstance(ref, dict):
                ref_code = ref.get("code", "")
                ref_article = ref.get("article", "")
                ref_article_id = f"{ref_code}_{ref_article}" if ref_code and ref_article else ""
            elif isinstance(ref, str):
                ref_article_id = ref.strip()
            else:
                continue
            
            if ref_article_id and ref_article_id != source_article_id:
                try:
                    query = """
                    MATCH (source:Article {article_id: $source_id})
                    MATCH (target:Article {article_id: $target_id})
                    MERGE (source)-[r:REFERENCES]->(target)
                    """
                    session.run(query, source_id=source_article_id, target_id=ref_article_id)
                except Exception as e:
                    continue

def create_penalty_relationships(session, documents: List[Dict[str, Any]]) -> None:
    """Crea relaciones entre artículos con penalties similares."""
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
            elif isinstance(penalty, str):
                penalty_type = penalty.lower()
            else:
                continue
                
            if penalty_type:
                if penalty_type not in penalty_groups:
                    penalty_groups[penalty_type] = []
                penalty_groups[penalty_type].append(article_id)
    
    # Crear relaciones entre artículos con penalties similares
    for penalty_type, article_ids in penalty_groups.items():
        if len(article_ids) > 1 and len(article_ids) <= 10:
            for i, art1 in enumerate(article_ids):
                for art2 in article_ids[i+1:]:
                    try:
                        query = """
                        MATCH (a1:Article {article_id: $art1_id})
                        MATCH (a2:Article {article_id: $art2_id})
                        MERGE (a1)-[r:SIMILAR_PENALTY {penalty_type: $penalty_type}]->(a2)
                        """
                        session.run(query, art1_id=art1, art2_id=art2, penalty_type=penalty_type)
                    except Exception as e:
                        continue

def create_section_relationships(session) -> None:
    """Crea relaciones jerárquicas basadas en secciones y capítulos."""
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

# ========== FUNCIONES DE COMPATIBILIDAD ==========

def check_data_exists(driver: GraphDatabase.driver) -> bool:
    """Checks if data already exists in the Neo4j database."""
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
    """Clears all data from the Neo4j database."""
    print("ADVERTENCIA: Eliminando todos los datos de Neo4j...")
    
    with get_session(driver) as session:
        query = """
        MATCH (n)
        DETACH DELETE n
        """
        session.run(query)
        print("Todos los datos eliminados de la base de datos Neo4j.")

def setup_enhanced_neo4j_data(driver: GraphDatabase.driver, documents: List[Dict[str, Any]]) -> None:
    """Configura Neo4j con el sistema mejorado que aprovecha todos los metadatos."""
    print("\n=== Configurando Neo4j con Sistema Graph RAG Avanzado ===")
    
    # Verificar si ya existen datos
    if check_data_exists(driver):
        print("Ya existen datos en Neo4j.")
        choice = input("¿Desea limpiar y recargar todos los datos? (s/n): ")
        if choice.lower() == 's':
            clear_neo4j_data(driver)
        else:
            print("Manteniendo datos existentes...")
    
    # 1. Crear nodos enriquecidos
    print("📊 Creando nodos Article enriquecidos...")
    article_ids = create_enhanced_neo4j_nodes(driver, documents)
    
    # 2. Crear relaciones básicas de leyes
    print("🏛️ Creando relaciones básicas...")
    create_basic_law_relationships(driver, documents)
    
    # 3. Crear relaciones enriquecidas
    print("🚀 Creando relaciones enriquecidas...")
    create_enhanced_relationships(driver, documents)
    
    print(f"\n✅ Configuración Graph RAG Avanzado completada:")
    print(f"   - {len(article_ids)} artículos enriquecidos")
    print(f"   - Relaciones por tags, referencias y penalties")
    print(f"   - Sistema de navegación inteligente activado")

def create_basic_law_relationships(driver: GraphDatabase.driver, documents: List[Dict[str, Any]]) -> None:
    """Crea relaciones básicas de leyes y códigos."""
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
    Función de compatibilidad que usa el sistema Graph RAG Avanzado.
    """
    # Extraer query string de los parámetros
    query = ""
    if "keywords" in query_params:
        query = " ".join(query_params["keywords"])
    
    if not query:
        return []
    
    # Usar el sistema avanzado
    return search_neo4j_enhanced(driver, query, limit)

def search_neo4j_traditional(driver: GraphDatabase.driver, query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Búsqueda tradicional como fallback."""
    return _fallback_simple_search(driver, query, limit)