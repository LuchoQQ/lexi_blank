"""
Module for integrating with Neo4j graph database.
"""
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional

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

def create_neo4j_nodes(driver: GraphDatabase.driver, documents: List[Dict[str, Any]]) -> List[str]:
    """
    Create Article nodes in Neo4j from document data.
    
    Args:
        driver: Neo4j driver instance
        documents: List of document dictionaries
        
    Returns:
        List of created article IDs
    """
    created_ids = []
    
    with driver.session() as session:
        for doc in documents:
            # Extract document properties
            article_id = doc.get("article_id", "")
            if not article_id:
                continue
                
            # Create properties map
            properties = {
                "article_id": article_id,
                "content": doc.get("content", ""),
                "law_name": doc.get("law_name", ""),
                "article_number": doc.get("article_number", ""),
                "category": doc.get("category", ""),
                "source": doc.get("source", "")
            }
            
            # Create or merge the Article node
            query = """
            MERGE (a:Article {article_id: $article_id})
            SET a.content = $content,
                a.law_name = $law_name,
                a.article_number = $article_number,
                a.category = $category,
                a.source = $source
            RETURN a.article_id
            """
            
            result = session.run(query, **properties)
            record = result.single()
            if record:
                created_ids.append(record[0])
    
    return created_ids

def create_law_relationship(driver: GraphDatabase.driver, law_name: str, article_ids: List[str]) -> None:
    """
    Create a Law node and establish CONTAINS relationships with Article nodes.
    
    Args:
        driver: Neo4j driver instance
        law_name: Name of the law
        article_ids: List of article IDs to connect to the law
    """
    if not law_name or not article_ids:
        return
        
    with driver.session() as session:
        # Create Law node
        query = """
        MERGE (l:Law {name: $law_name})
        RETURN l
        """
        session.run(query, law_name=law_name)
        
        # Create relationships between Law and Articles
        for article_id in article_ids:
            query = """
            MATCH (l:Law {name: $law_name})
            MATCH (a:Article {article_id: $article_id})
            MERGE (l)-[r:CONTAINS]->(a)
            RETURN r
            """
            session.run(query, law_name=law_name, article_id=article_id)
    
    print(f"Created Law node '{law_name}' with relationships to {len(article_ids)} articles.")

def create_cross_law_relationships(driver: GraphDatabase.driver, documents: List[Dict[str, Any]]) -> None:
    """
    Crea relaciones entre diferentes códigos y leyes basadas en referencias temáticas.
    
    Args:
        driver: Instancia del driver de Neo4j
        documents: Lista de documentos con artículos legales
    """
    print("Creando relaciones entre diferentes códigos y leyes...")
    
    # Extraer categorías y temas de los documentos
    law_categories = {}
    for doc in documents:
        law_name = doc.get("law_name")
        categories = doc.get("categories", [])
        topics = doc.get("topics", [])
        
        if not law_name:
            continue
            
        if law_name not in law_categories:
            law_categories[law_name] = {"categories": set(), "topics": set()}
            
        # Añadir categorías y temas
        if isinstance(categories, list):
            law_categories[law_name]["categories"].update(categories)
        elif isinstance(categories, str):
            law_categories[law_name]["categories"].add(categories)
            
        if isinstance(topics, list):
            law_categories[law_name]["topics"].update(topics)
        elif isinstance(topics, str):
            law_categories[law_name]["topics"].add(topics)
    
    # Crear relaciones basadas en categorías y temas compartidos
    with driver.session() as session:
        # 1. Crear relaciones basadas en categorías compartidas
        for law1, data1 in law_categories.items():
            for law2, data2 in law_categories.items():
                if law1 != law2:
                    # Encontrar categorías compartidas
                    shared_categories = data1["categories"].intersection(data2["categories"])
                    if shared_categories:
                        for category in shared_categories:
                            if category:  # Asegurarse de que la categoría no esté vacía
                                query = """
                                MATCH (l1:Law {name: $law1})
                                MATCH (l2:Law {name: $law2})
                                MERGE (l1)-[r:RELATED_BY_CATEGORY {category: $category}]->(l2)
                                RETURN r
                                """
                                session.run(query, law1=law1, law2=law2, category=category)
                                print(f"Creada relación RELATED_BY_CATEGORY entre '{law1}' y '{law2}' por categoría '{category}'")
                    
                    # Encontrar temas compartidos
                    shared_topics = data1["topics"].intersection(data2["topics"])
                    if shared_topics:
                        for topic in shared_topics:
                            if topic:  # Asegurarse de que el tema no esté vacío
                                query = """
                                MATCH (l1:Law {name: $law1})
                                MATCH (l2:Law {name: $law2})
                                MERGE (l1)-[r:RELATED_BY_TOPIC {topic: $topic}]->(l2)
                                RETURN r
                                """
                                session.run(query, law1=law1, law2=law2, topic=topic)
                                print(f"Creada relación RELATED_BY_TOPIC entre '{law1}' y '{law2}' por tema '{topic}'")
    
    # 2. Crear relaciones basadas en referencias explícitas entre artículos
    with driver.session() as session:
        for doc in documents:
            article_id = doc.get("article_id")
            law_name = doc.get("law_name")
            references = doc.get("references", [])
            
            if not article_id or not law_name or not references:
                continue
                
            for ref in references:
                ref_article = ref.get("article")
                ref_law = ref.get("law")
                
                if not ref_article or not ref_law:
                    continue
                    
                # Crear relación entre artículos
                query = """
                MATCH (a1:Article {article_id: $article_id})
                MATCH (a2:Article)
                WHERE a2.article_number = $ref_article AND EXISTS {
                    MATCH (l:Law {name: $ref_law})-[:CONTAINS]->(a2)
                }
                MERGE (a1)-[r:REFERENCES]->(a2)
                RETURN r
                """
                session.run(query, article_id=article_id, ref_article=ref_article, ref_law=ref_law)
                
                # Crear relación entre leyes si no existe
                query = """
                MATCH (l1:Law {name: $law1})
                MATCH (l2:Law {name: $law2})
                WHERE l1 <> l2
                MERGE (l1)-[r:REFERENCES_LAW]->(l2)
                RETURN r
                """
                session.run(query, law1=law_name, law2=ref_law)
                print(f"Creada relación REFERENCES_LAW entre '{law_name}' y '{ref_law}'")

def create_thematic_relationships(driver: GraphDatabase.driver) -> None:
    """
    Crea relaciones temáticas entre artículos basadas en palabras clave compartidas.
    
    Args:
        driver: Instancia del driver de Neo4j
    """
    print("Creando relaciones temáticas entre artículos...")
    
    with driver.session() as session:
        # Crear índice de texto en el contenido de los artículos si no existe
        # La sintaxis varía según la versión de Neo4j
        try:
            # Intentar primero con la sintaxis de Neo4j 4.x+
            query = """
            CREATE FULLTEXT INDEX FOR (a:Article) ON EACH [a.content]
            """
            session.run(query)
        except Exception as e:
            print(f"Error al crear índice de texto completo: {str(e)}")
            try:
                # Intentar con sintaxis alternativa para versiones anteriores
                query = """
                CALL db.index.fulltext.createNodeIndex('article_content', ['Article'], ['content'])
                """
                session.run(query)
                print("Índice de texto completo creado con sintaxis alternativa")
            except Exception as e2:
                print(f"Error al crear índice de texto completo (alternativo): {str(e2)}")
                print("Continuando sin índice de texto completo...")
        
        # Identificar artículos con contenido similar y crear relaciones
        # En lugar de usar apoc.text.similarity, usamos una comparación más básica
        # que debería funcionar en todas las versiones de Neo4j
        try:
            query = """
            MATCH (a1:Article)
            MATCH (a2:Article)
            WHERE id(a1) < id(a2) 
            AND a1.content IS NOT NULL 
            AND a2.content IS NOT NULL
            AND a1.law_name <> a2.law_name
            WITH a1, a2, 
                 size([word IN split(toLower(a1.content), ' ') WHERE word IN split(toLower(a2.content), ' ')]) AS commonWords,
                 size(split(a1.content, ' ')) AS words1,
                 size(split(a2.content, ' ')) AS words2
            WHERE commonWords > 5
            WITH a1, a2, toFloat(commonWords) / (words1 + words2 - commonWords) AS similarity
            WHERE similarity > 0.1
            MERGE (a1)-[r:THEMATICALLY_RELATED {score: similarity}]->(a2)
            RETURN count(r) as relationshipsCreated
            """
            result = session.run(query)
            record = result.single()
            if record:
                print(f"Creadas {record['relationshipsCreated']} relaciones temáticas entre artículos")
        except Exception as e:
            print(f"Error al crear relaciones temáticas: {str(e)}")
            print("Intentando con una consulta más simple...")
            
            try:
                # Consulta más simple que debería funcionar en todas las versiones
                query = """
                MATCH (a1:Article)
                MATCH (a2:Article)
                WHERE id(a1) < id(a2) 
                AND a1.law_name <> a2.law_name
                AND a1.content CONTAINS a2.article_number
                MERGE (a1)-[r:MENTIONS]->(a2)
                RETURN count(r) as relationshipsCreated
                """
                result = session.run(query)
                record = result.single()
                if record:
                    print(f"Creadas {record['relationshipsCreated']} relaciones MENTIONS entre artículos")
            except Exception as e2:
                print(f"Error al crear relaciones simples: {str(e2)}")

def search_neo4j(driver: GraphDatabase.driver, query_params: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for articles in Neo4j based on query parameters.
    
    Args:
        driver: Neo4j driver instance
        query_params: Dictionary of search parameters (law_name, category, etc.)
        limit: Maximum number of results to return
        
    Returns:
        List of matching articles with their properties
    """
    results = []
    
    with driver.session() as session:
        # Construir consulta Cypher
        cypher_query = (
            "MATCH (a:Article) "
        )
        
        # Agregar condiciones de búsqueda
        where_clauses = []
        params = {}
        
        # Búsqueda por palabras clave en el contenido
        if "keywords" in query_params and query_params["keywords"]:
            keywords = query_params["keywords"]
            keyword_conditions = []
            
            for i, keyword in enumerate(keywords):
                param_name = f"keyword{i}"
                keyword_conditions.append(f"a.content CONTAINS ${param_name}")
                params[param_name] = keyword
            
            if keyword_conditions:
                where_clauses.append("(" + " OR ".join(keyword_conditions) + ")")
        
        # Búsqueda por ley
        if "law_name" in query_params and query_params["law_name"]:
            where_clauses.append("a.law_name = $law_name")
            params["law_name"] = query_params["law_name"]
        
        # Búsqueda por categoría
        if "category" in query_params and query_params["category"]:
            where_clauses.append("a.category = $category")
            params["category"] = query_params["category"]
        
        # Búsqueda por número de artículo
        if "article_number" in query_params and query_params["article_number"]:
            where_clauses.append("a.article_number = $article_number")
            params["article_number"] = query_params["article_number"]
        
        # Agregar cláusulas WHERE si existen
        if where_clauses:
            cypher_query += "WHERE " + " AND ".join(where_clauses) + " "
        
        # Completar consulta con retorno y límite
        cypher_query += (
            "RETURN a.article_number as article_number, "
            "a.content as content, "
            "a.law_name as law_name, "
            "a.category as category, "
            "1.0 as score "
            "LIMIT $limit"
        )
        
        # Agregar parámetro de límite
        params["limit"] = limit
        
        # Ejecutar consulta
        result = session.run(cypher_query, params)
        
        for record in result:
            article = {
                "article_number": record["article_number"],
                "content": record["content"],
                "law_name": record["law_name"],
                "category": record["category"],
                "score": record["score"]
            }
            results.append(article)
    
    return results

def check_data_exists(driver: GraphDatabase.driver) -> bool:
    """
    Verifica si ya existen datos en la base de datos Neo4j.
    
    Args:
        driver: Instancia del driver de Neo4j
        
    Returns:
        True si ya existen datos, False en caso contrario
    """
    with driver.session() as session:
        # Verificar si existen nodos Article
        query = """
        MATCH (a:Article)
        RETURN count(a) as article_count
        """
        result = session.run(query)
        record = result.single()
        if record and record["article_count"] > 0:
            return True
            
        # Verificar si existen nodos Law
        query = """
        MATCH (l:Law)
        RETURN count(l) as law_count
        """
        result = session.run(query)
        record = result.single()
        if record and record["law_count"] > 0:
            return True
            
    return False
