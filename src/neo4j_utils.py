"""
Module for integrating with Neo4j graph database.
"""
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import re
import contextlib

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
    
    with get_session(driver) as session:
        for doc in documents:
            # Extract document properties
            article_id = doc.get("article_id", "")
            if not article_id:
                # Generate article_id if not present
                law_name = doc.get("law_name", "unknown")
                article_number = doc.get("article_number", "")
                if not article_number and "metadata" in doc:
                    article_number = doc["metadata"].get("article", "")
                if law_name and article_number:
                    article_id = f"{law_name}_{article_number}"
                else:
                    continue
                
            # Handle metadata if it exists
            law_name = doc.get("law_name", "")
            article_number = doc.get("article_number", "")
            category = doc.get("category", "")
            source = doc.get("source", "")
            
            # Extract from metadata if available
            if "metadata" in doc:
                metadata = doc["metadata"]
                if not law_name and "code" in metadata:
                    law_name = metadata["code"]
                if not article_number and "article" in metadata:
                    article_number = metadata["article"]
                if not category and "chapter" in metadata:
                    category = metadata["chapter"]
                if not source and "section" in metadata:
                    source = metadata["section"]
            
            # Create properties map
            properties = {
                "article_id": article_id,
                "content": doc.get("content", ""),
                "law_name": law_name,
                "article_number": article_number,
                "category": category,
                "source": source
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
        
    with get_session(driver) as session:
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

def create_topic_relationships_from_tags(driver: GraphDatabase.driver, documents: List[Dict[str, Any]]) -> None:
    """
    Create Topic nodes and relationships automatically from document tags.
    
    Args:
        driver: Neo4j driver instance
        documents: List of document dictionaries
    """
    print("Creating topic nodes and relationships from document tags...")
    
    # Collect all unique tags from documents
    all_tags = set()
    article_tags = {}
    
    for doc in documents:
        article_id = doc.get("article_id", "")
        if not article_id:
            # Generate article_id if not present
            law_name = doc.get("law_name", "unknown")
            article_number = doc.get("article_number", "")
            if not article_number and "metadata" in doc:
                article_number = doc["metadata"].get("article", "")
            if law_name and article_number:
                article_id = f"{law_name}_{article_number}"
            else:
                continue
                
        tags = []
        
        # Extract tags from document directly or from metadata
        if "tags" in doc:
            tags.extend(doc["tags"])
        
        if "metadata" in doc and "tags" in doc["metadata"]:
            if isinstance(doc["metadata"]["tags"], list):
                tags.extend(doc["metadata"]["tags"])
            elif isinstance(doc["metadata"]["tags"], str):
                tags.append(doc["metadata"]["tags"])
        
        # If still no tags, use category or chapter info if available
        if not tags:
            category = doc.get("category", "")
            if not category and "metadata" in doc:
                category = doc["metadata"].get("chapter", "")
            
            if category:
                tags.append(category)
        
        # Store tags for this article
        if tags:
            article_tags[article_id] = tags
            # Add to global set of tags
            for tag in tags:
                if tag and isinstance(tag, str):
                    all_tags.add(tag)
    
    print(f"Collected {len(all_tags)} unique tags from documents")
    
    if not all_tags:
        print("No tags found. Trying to generate topics from content...")
        # Extract potential topics from content using simple pattern matching
        topic_patterns = [
            r'(?:sobre|acerca de|referente a|relativo a|sobre la|sobre el)\s+([A-Za-zÀ-ÿ\s]+?)\.',  # "sobre X."
            r'(?:en caso de|cuando ocurra)\s+([A-Za-zÀ-ÿ\s]+?)[,\.]',  # "en caso de X."
            r'(?:en materia de|en el ámbito de)\s+([A-Za-zÀ-ÿ\s]+?)[,\.]'  # "en materia de X."
        ]
        
        for doc in documents:
            article_id = doc.get("article_id", "")
            if not article_id:
                continue
                
            content = doc.get("content", "")
            if not content:
                continue
                
            # Find potential topics in content
            potential_topics = []
            for pattern in topic_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if len(match) > 3 and len(match.split()) < 5:  # Reasonable topic length
                        potential_topics.append(match.strip())
            
            if potential_topics:
                article_tags[article_id] = potential_topics
                for topic in potential_topics:
                    all_tags.add(topic)
        
        print(f"Generated {len(all_tags)} topics from content patterns")
    
    with get_session(driver) as session:
        # Create Topic nodes for each tag
        for tag in all_tags:
            query = """
            MERGE (t:Topic {name: $tag})
            RETURN t
            """
            session.run(query, tag=tag)
        
        print(f"Created {len(all_tags)} Topic nodes")
        
        # Connect articles to their topics
        for article_id, tags in article_tags.items():
            for tag in tags:
                if tag and isinstance(tag, str):
                    query = """
                    MATCH (a:Article {article_id: $article_id})
                    MATCH (t:Topic {name: $tag})
                    MERGE (a)-[r:HAS_TOPIC]->(t)
                    RETURN r
                    """
                    session.run(query, article_id=article_id, tag=tag)
        
        print(f"Connected articles to their respective topics")
        
        # Create relationships between topics that appear together
        query = """
        MATCH (a:Article)-[:HAS_TOPIC]->(t1:Topic)
        MATCH (a)-[:HAS_TOPIC]->(t2:Topic)
        WHERE t1 <> t2
        MERGE (t1)-[r:RELATED_TO]->(t2)
        RETURN count(DISTINCT r) as relCount
        """
        result = session.run(query)
        record = result.single()
        
        if record:
            print(f"Created {record['relCount']} relationships between related topics")
    
    print("Topic relationships created successfully based on document tags.")

def create_thematic_relationships(driver: GraphDatabase.driver, documents: List[Dict[str, Any]]) -> None:
    """
    Creates thematic relationships between articles based on common topics and content analysis.
    
    Args:
        driver: Neo4j driver instance
        documents: List of document dictionaries
    """
    print("Creating thematic relationships between articles...")
    
    # Create thematic relationships based on content similarity
    with get_session(driver) as session:
        # Find articles with similar content patterns
        common_legal_patterns = [
            r"(?:derechos|obligaciones|deberes)",
            r"(?:prohibiciones|prohibido|prohibe)",
            r"(?:sanciones|penas|multas|castigos)",
            r"(?:procedimiento|proceso|trámite)",
            r"(?:autoridad|competencia|jurisdicción)"
        ]
        
        # For each pattern, find articles that match and create relationships
        for pattern in common_legal_patterns:
            # Find articles matching the pattern
            matching_articles = []
            for doc in documents:
                article_id = doc.get("article_id")
                if not article_id:
                    continue
                    
                content = doc.get("content", "")
                if not content:
                    continue
                    
                if re.search(pattern, content, re.IGNORECASE):
                    matching_articles.append(article_id)
            
            # Create relationships between matching articles
            if len(matching_articles) > 1:
                pattern_name = pattern.replace("(?:", "").replace(")", "").replace("|", "_")
                for i in range(len(matching_articles)):
                    for j in range(i+1, len(matching_articles)):
                        query = """
                        MATCH (a1:Article {article_id: $article_id1})
                        MATCH (a2:Article {article_id: $article_id2})
                        MERGE (a1)-[r:CONTENT_PATTERN_MATCH {pattern: $pattern}]->(a2)
                        RETURN r
                        """
                        session.run(query, 
                                   article_id1=matching_articles[i], 
                                   article_id2=matching_articles[j],
                                   pattern=pattern_name)
    
    print("Thematic relationships created successfully.")

def create_cross_law_relationships(driver: GraphDatabase.driver, documents: List[Dict[str, Any]]) -> None:
    """
    Creates relationships between different codes and laws based on thematic references.
    
    Args:
        driver: Neo4j driver instance
        documents: List of documents with legal articles
    """
    print("Creating relationships between different codes and laws...")
    
    # Extract categories and topics from documents
    law_categories = {}
    for doc in documents:
        law_name = doc.get("law_name")
        if not law_name and "metadata" in doc:
            law_name = doc["metadata"].get("code")
            
        if not law_name:
            continue
            
        # Get categories
        categories = []
        if "category" in doc and doc["category"]:
            categories.append(doc["category"])
        if "metadata" in doc and "chapter" in doc["metadata"]:
            categories.append(doc["metadata"]["chapter"])
        
        # Get topics
        topics = []
        if "tags" in doc and isinstance(doc["tags"], list):
            topics.extend(doc["tags"])
        if "metadata" in doc and "tags" in doc["metadata"]:
            if isinstance(doc["metadata"]["tags"], list):
                topics.extend(doc["metadata"]["tags"])
            elif isinstance(doc["metadata"]["tags"], str):
                topics.append(doc["metadata"]["tags"])
            
        if law_name not in law_categories:
            law_categories[law_name] = {"categories": set(), "topics": set()}
            
        # Add categories and topics
        for category in categories:
            if category:
                law_categories[law_name]["categories"].add(category)
                
        for topic in topics:
            if topic:
                law_categories[law_name]["topics"].add(topic)
    
    # Create relationships based on shared categories and topics
    with get_session(driver) as session:
        # Create relationships based on shared categories
        for law1, data1 in law_categories.items():
            for law2, data2 in law_categories.items():
                if law1 != law2:
                    # Find shared categories
                    shared_categories = data1["categories"].intersection(data2["categories"])
                    if shared_categories:
                        for category in shared_categories:
                            if category:  # Ensure category is not empty
                                query = """
                                MATCH (l1:Law {name: $law1})
                                MATCH (l2:Law {name: $law2})
                                MERGE (l1)-[r:RELATED_BY_CATEGORY {category: $category}]->(l2)
                                RETURN r
                                """
                                session.run(query, law1=law1, law2=law2, category=category)
                    
                    # Find shared topics
                    shared_topics = data1["topics"].intersection(data2["topics"])
                    if shared_topics:
                        for topic in shared_topics:
                            if topic:  # Ensure topic is not empty
                                query = """
                                MATCH (l1:Law {name: $law1})
                                MATCH (l2:Law {name: $law2})
                                MERGE (l1)-[r:RELATED_BY_TOPIC {topic: $topic}]->(l2)
                                RETURN r
                                """
                                session.run(query, law1=law1, law2=law2, topic=topic)
    
    print("Cross-law relationships created successfully.")

def create_semantic_content_relationships(driver: GraphDatabase.driver) -> None:
    """
    Creates relationships between articles based on semantic content analysis.
    
    Args:
        driver: Neo4j driver instance
    """
    print("Creating semantic content relationships between articles...")
    
    with get_session(driver) as session:
        # Create indices for better performance
        try:
            # Create index on article_id if it doesn't exist
            query = """
            CREATE INDEX article_id_index IF NOT EXISTS FOR (a:Article) ON (a.article_id)
            """
            session.run(query)
            print("Index on article_id created")
        except Exception as e:
            print(f"Error creating article_id index: {str(e)}")
            try:
                # Try with alternative syntax
                query = """
                CREATE INDEX ON :Article(article_id)
                """
                session.run(query)
                print("Index on article_id created with alternative syntax")
            except Exception as e2:
                print(f"Error creating article_id index (alternative): {str(e2)}")
        
        # 1. Create relationships between articles in the same category
        try:
            query = """
            MATCH (a1:Article)
            MATCH (a2:Article)
            WHERE id(a1) < id(a2) 
            AND a1.category = a2.category
            AND a1.category IS NOT NULL AND a1.category <> ''
            AND a1.law_name <> a2.law_name
            MERGE (a1)-[r:SAME_CATEGORY]->(a2)
            RETURN count(r) as relCount
            """
            result = session.run(query)
            record = result.single()
            if record:
                print(f"Created {record['relCount']} SAME_CATEGORY relationships")
        except Exception as e:
            print(f"Error creating SAME_CATEGORY relationships: {str(e)}")
        
        # 2. Create relationships based on common text patterns
        try:
            # Simpler approach using keywords
            common_legal_terms = [
                "derecho", "obligación", "contrato", "persona", "responsabilidad", 
                "propiedad", "plazo", "demanda", "resolución", "sentencia", "sanción",
                "pena", "delito", "tribunal", "juez", "procedimiento", "recurso"
            ]
            
            for term in common_legal_terms[:5]:  # Limit to first 5 terms
                query = f"""
                MATCH (a1:Article) WHERE toLower(a1.content) CONTAINS '{term}'
                MATCH (a2:Article) WHERE toLower(a2.content) CONTAINS '{term}' AND id(a1) < id(a2)
                AND a1.law_name <> a2.law_name
                MERGE (a1)-[r:SHARES_CONCEPT {{concept: '{term}'}}]->(a2)
                RETURN count(r) as relCount
                LIMIT 500
                """
                result = session.run(query)
                record = result.single()
                if record:
                    print(f"Created {record['relCount']} SHARES_CONCEPT relationships for term '{term}'")
        except Exception as e:
            print(f"Error creating SHARES_CONCEPT relationships: {str(e)}")
        
        # 3. Create relationships based on references in the content
        try:
            query = """
            MATCH (a1:Article)
            MATCH (a2:Article)
            WHERE a1 <> a2
            AND a1.article_number IS NOT NULL
            AND a2.content CONTAINS a1.article_number
            MERGE (a2)-[r:REFERENCES_ARTICLE]->(a1)
            RETURN count(r) as relCount
            LIMIT 1000
            """
            result = session.run(query)
            record = result.single()
            if record:
                print(f"Created {record['relCount']} REFERENCES_ARTICLE relationships")
        except Exception as e:
            print(f"Error creating REFERENCES_ARTICLE relationships: {str(e)}")
    
    print("Semantic content relationships created successfully.")

def detect_query_topics(query: str, driver: GraphDatabase.driver) -> List[str]:
    """
    Detect topics related to the query by analyzing the database structure.
    
    Args:
        query: User query
        driver: Neo4j driver instance
        
    Returns:
        List of relevant topics found in the database
    """
    topics = []
    
    with get_session(driver) as session:
        # Find topics that might be relevant to the query by looking at words in common
        query_words = [word.lower() for word in query.split() if len(word) > 3]
        
        if not query_words:
            return topics
            
        # Query to find topics that have articles with content containing query words
        cypher_query = """
        MATCH (t:Topic)<-[:HAS_TOPIC]-(a:Article)
        WHERE 
        """ + " OR ".join([f"toLower(a.content) CONTAINS ${i}" for i in range(len(query_words))]) + """
        RETURN t.name as topic, count(a) as relevance
        ORDER BY relevance DESC
        LIMIT 5
        """
        
        # Create parameters dictionary
        params = {str(i): word for i, word in enumerate(query_words)}
        
        try:
            result = session.run(cypher_query, params)
            for record in result:
                topics.append(record["topic"])
        except Exception as e:
            print(f"Error detecting topics: {str(e)}")
            
            # Fallback: find articles that match query words and return their categories
            try:
                fallback_query = """
                MATCH (a:Article)
                WHERE 
                """ + " OR ".join([f"toLower(a.content) CONTAINS ${i}" for i in range(len(query_words))]) + """
                RETURN DISTINCT a.category as category
                LIMIT 5
                """
                
                result = session.run(fallback_query, params)
                for record in result:
                    if record["category"]:
                        topics.append(record["category"])
            except Exception as e2:
                print(f"Error with fallback topic detection: {str(e2)}")
    
    return topics

def search_neo4j(driver: GraphDatabase.driver, query_params: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for articles in Neo4j based on query parameters.
    
    Args:
        driver: Neo4j driver instance
        query_params: Dictionary of search parameters (keywords, topics, etc.)
        limit: Maximum number of results to return
        
    Returns:
        List of matching articles with their properties
    """
    results = []
    
    with get_session(driver) as session:
        # Build Cypher query
        cypher_query = "MATCH (a:Article) "
        
        # Add search conditions
        where_clauses = []
        params = {}
        
        # Keyword search in content
        if "keywords" in query_params and query_params["keywords"]:
            keywords = query_params["keywords"]
            keyword_conditions = []
            
            for i, keyword in enumerate(keywords):
                param_name = f"keyword{i}"
                keyword_conditions.append(f"toLower(a.content) CONTAINS toLower(${param_name})")
                params[param_name] = keyword
            
            if keyword_conditions:
                where_clauses.append("(" + " OR ".join(keyword_conditions) + ")")
        
        # Search by law
        if "law_name" in query_params and query_params["law_name"]:
            where_clauses.append("a.law_name = $law_name")
            params["law_name"] = query_params["law_name"]
        
        # Search by category
        if "category" in query_params and query_params["category"]:
            where_clauses.append("a.category = $category")
            params["category"] = query_params["category"]
        
        # Add WHERE clauses if they exist
        if where_clauses:
            cypher_query += "WHERE " + " AND ".join(where_clauses) + " "
        
        # Add return statement with scoring
        cypher_query += (
            "RETURN a.article_id as article_id, "
            "a.article_number as article_number, "
            "a.content as content, "
            "a.law_name as law_name, "
            "a.category as category, "
            "a.source as source, "
            "1.0 as score "
            "ORDER BY score DESC "
            "LIMIT $limit"
        )
        
        # Add limit parameter
        params["limit"] = limit
        
        try:
            # Execute query
            result = session.run(cypher_query, params)
            
            for record in result:
                article = {
                    "article_id": record["article_id"],
                    "article_number": record["article_number"],
                    "content": record["content"],
                    "law_name": record["law_name"],
                    "category": record["category"],
                    "source": record["source"],
                    "score": record["score"]
                }
                results.append(article)
        except Exception as e:
            print(f"Error executing Neo4j search: {str(e)}")
    
    return results

def check_data_exists(driver: GraphDatabase.driver) -> bool:
    """
    Checks if data already exists in the Neo4j database.
    
    Args:
        driver: Neo4j driver instance
        
    Returns:
        True if data exists, False otherwise
    """
    with get_session(driver) as session:
        # Check if Article nodes exist
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
    Use with caution - this will delete all nodes and relationships.
    
    Args:
        driver: Neo4j driver instance
    """
    print("ADVERTENCIA: Eliminando todos los datos de Neo4j...")
    
    with get_session(driver) as session:
        # Delete all nodes and relationships
        query = """
        MATCH (n)
        DETACH DELETE n
        """
        session.run(query)
        print("All data cleared from Neo4j database.")

def create_legal_domain_relationships(driver: GraphDatabase.driver, documents: List[Dict[str, Any]]) -> None:
    """
    Creates domain-specific thematic relationships for legal concepts.
    
    Args:
        driver: Neo4j driver instance
        documents: List of document dictionaries
    """
    # Import legal domains from our simplified module
    from src.legal_domains import LEGAL_DOMAINS
    
    print("Creating legal domain relationships...")
    
    # Create domain nodes first
    with get_session(driver) as session:
        for domain, keywords in LEGAL_DOMAINS.items():
            # Create domain node
            query = """
            MERGE (d:LegalDomain {name: $domain})
            SET d.keywords = $keywords
            RETURN d
            """
            session.run(query, domain=domain, keywords=keywords)
            print(f"Created LegalDomain node: {domain}")
    
    # Connect articles to domains based on content analysis
    article_domains = {}
    
    # First pass: analyze documents and determine domain relevance
    for doc in documents:
        article_id = doc.get("article_id", "")
        if not article_id:
            # Generate article_id if not present
            law_name = doc.get("law_name", "unknown")
            article_number = doc.get("article_number", "")
            if not article_number and "metadata" in doc:
                article_number = doc["metadata"].get("article", "")
            if law_name and article_number:
                article_id = f"{law_name}_{article_number}"
            else:
                continue
        
        # Get content for analysis
        content = doc.get("content", "").lower()
        
        # Check relevance to each domain
        article_domains[article_id] = []
        
        for domain, keywords in LEGAL_DOMAINS.items():
            relevance_score = 0
            
            # Check content for keyword matches
            for keyword in keywords:
                if keyword.lower() in content:
                    relevance_score += 1
            
            # If article is sufficiently relevant to domain, add to mapping
            if relevance_score >= 1:  # Threshold can be adjusted
                article_domains[article_id].append((domain, relevance_score))
    
    # Second pass: create relationships in database
    with get_session(driver) as session:
        for article_id, domains in article_domains.items():
            for domain, score in domains:
                # Create RELATED_TO relationship with relevance score
                query = """
                MATCH (a:Article {article_id: $article_id})
                MATCH (d:LegalDomain {name: $domain})
                MERGE (a)-[r:RELATED_TO]->(d)
                SET r.relevance = $score
                RETURN r
                """
                session.run(query, article_id=article_id, domain=domain, score=score)
        
        # Create relationships between related domains
        query = """
        MATCH (d1:LegalDomain)
        MATCH (d2:LegalDomain)
        WHERE d1 <> d2
        WITH d1, d2
        MATCH (a:Article)-[:RELATED_TO]->(d1)
        MATCH (a)-[:RELATED_TO]->(d2)
        WITH d1, d2, count(a) as common
        WHERE common > 0
        MERGE (d1)-[r:RELATED_DOMAIN]->(d2)
        SET r.strength = common
        RETURN d1.name, d2.name, r.strength
        """
        result = session.run(query)
        for record in result:
            print(f"Connected domains {record[0]} and {record[1]} with strength {record[2]}")
    
    print("Legal domain relationships created successfully.")

def search_by_legal_domain(driver: GraphDatabase.driver, query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search for articles related to specific legal domains detected in the query.
    
    Args:
        driver: Neo4j driver instance
        query: User query text
        limit: Maximum number of results to return
        
    Returns:
        List of relevant articles with their properties
    """
    # Import detection function from our simplified module
    from src.legal_domains import detect_domains_in_query
    
    results = []
    
    # Detect domains in query
    detected_domains = detect_domains_in_query(query)
    
    if not detected_domains:
        print("No se detectaron dominios legales específicos en la consulta")
        return results
    
    print(f"Dominios legales detectados en la consulta: {detected_domains}")
    
    with get_session(driver) as session:
        # For each detected domain, find related articles
        for domain in detected_domains:
            # Query for articles directly related to the domain
            query_cypher = """
            MATCH (d:LegalDomain {name: $domain})<-[r:RELATED_TO]-(a:Article)
            WITH a, r.relevance as domainScore
            
            // Calculate content relevance to the search query
            WITH a, domainScore,
                 CASE WHEN toLower(a.content) CONTAINS toLower($searchTerm) THEN 5.0 ELSE 0.0 END +
                 CASE WHEN toLower(a.category) CONTAINS toLower($searchTerm) THEN 3.0 ELSE 0.0 END as queryScore
            
            // Weight practical articles higher
            WITH a, domainScore, queryScore,
                 CASE 
                    WHEN toLower(a.content) CONTAINS "procedimiento" THEN 2.0
                    WHEN toLower(a.content) CONTAINS "plazo" THEN 1.5
                    WHEN toLower(a.content) CONTAINS "días" THEN 1.2
                    ELSE 1.0
                 END as practicalBoost
            
            // Calculate final score
            WITH a, (domainScore * 2.0) + queryScore + practicalBoost as finalScore
            
            RETURN a.article_id as article_id, 
                   a.content as content,
                   a.law_name as law_name,
                   a.article_number as article_number,
                   a.category as category,
                   a.source as source,
                   finalScore as relevance
            ORDER BY finalScore DESC
            LIMIT $limit
            """
            
            result = session.run(query_cypher, domain=domain, searchTerm=query, limit=limit)
            
            # Process results
            for record in result:
                article = {
                    "article_id": record["article_id"],
                    "content": record["content"],
                    "law_name": record["law_name"],
                    "article_number": record["article_number"],
                    "category": record["category"],
                    "source": record["source"],
                    "score": float(record["relevance"]),
                    "domain": domain
                }
                
                # Check if article is already in results
                if not any(r["article_id"] == article["article_id"] for r in results):
                    results.append(article)
    
    # Sort results by score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Limit to requested number
    return results[:limit]