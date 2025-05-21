#!/usr/bin/env python3
"""
Script unificado para la configuración del sistema de recuperación de documentos legales.
Este script se encarga de:
1. Instalar dependencias necesarias
2. Configurar servicios Docker (Neo4j y Weaviate)
3. Cargar datos y embeddings en Weaviate
4. Crear nodos y relaciones en el grafo Neo4j

Uso:
    python setup_system.py --all      # Ejecutar todos los pasos
    python setup_system.py --deps     # Solo instalar dependencias
    python setup_system.py --docker   # Solo configurar Docker
    python setup_system.py --data     # Solo cargar datos
    python setup_system.py --test     # Ejecutar búsqueda de prueba
"""

import os
import sys
import time
import subprocess
import argparse
import importlib
import json
import yaml
import platform
import hashlib
import pickle
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import codecs

# Definir rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Asegurar que los directorios necesarios existan
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Dependencias requeridas
REQUIRED_PACKAGES = [
    "weaviate-client==3.26.0",
    "neo4j==5.14.0",
    "rank_bm25==0.2.2",
    "pyyaml==6.0.1",
    "numpy>=2.0.0",
    "sentence-transformers==2.2.2",
    "requests>=2.28.0"
]

def run_command(command, cwd=None):
    """Ejecuta un comando y muestra su salida en tiempo real."""
    print(f"Ejecutando: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=cwd
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def install_package(package, options=""):
    """Instala un paquete usando pip."""
    command = f"{sys.executable} -m pip install {options} {package}"
    return run_command(command)

def check_dependencies():
    """Verifica e instala las dependencias necesarias."""
    print("\n=== Verificando dependencias ===")
    
    # Verificar versión de Python
    python_version = sys.version_info
    print(f"Versión de Python detectada: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Verificar cada dependencia
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        package_name = package.split("==")[0].split(">=")[0]
        try:
            importlib.import_module(package_name)
            print(f"✓ {package} está instalado")
        except ImportError:
            print(f"✗ {package} no está instalado")
            missing_packages.append(package)
    
    # Instalar dependencias faltantes
    if missing_packages:
        print("\n=== Instalando dependencias faltantes ===")
        for package in missing_packages:
            print(f"\n{'='*50}")
            print(f"Instalando {package}...")
            print(f"{'='*50}")
            
            # Caso especial para sentence-transformers
            if package.startswith("sentence-transformers"):
                success = install_sentence_transformers()
            else:
                # Intentar con diferentes estrategias para otros paquetes
                strategies = [
                    ("", ""),  # Sin opciones
                    ("", "--no-build-isolation"),
                    ("", "--no-cache-dir")
                ]
                
                success = False
                for pkg, options in strategies:
                    package_to_install = pkg or package
                    print(f"\nIntentando instalar {package_to_install} {options}")
                    result = install_package(package_to_install, options)
                    if result == 0:
                        success = True
                        print(f"Instalación exitosa de {package_to_install}")
                        break
            
            if not success:
                print(f"\nNo se pudo instalar {package}")
                if package.startswith("sentence-transformers"):
                    print("Se usará una alternativa para generar embeddings.")
                else:
                    print("El sistema funcionará con capacidades limitadas.")
                    if package_name in ['weaviate-client', 'neo4j']:
                        return False
        
        print("\nInstalación de dependencias completada.")
    else:
        print("\nTodas las dependencias ya están instaladas.")
    
    # Verificar si podemos usar sentence-transformers o una alternativa
    try:
        import sentence_transformers
        print("✓ sentence-transformers está instalado correctamente")
    except ImportError:
        print("✗ sentence-transformers no está instalado. Se usará una alternativa.")
        try:
            # Verificar si tenemos las bibliotecas necesarias para la alternativa
            import torch
            import numpy
            print("✓ Se puede usar torch y numpy como alternativa para embeddings")
        except ImportError:
            print("✗ No se encontró torch, que es necesario para la alternativa")
            print("Intentando instalar torch...")
            result = install_package("torch")
            if result != 0:
                print("No se pudo instalar torch. El sistema tendrá funcionalidad limitada.")
    
    return True

def install_sentence_transformers():
    """Intenta instalar sentence-transformers con manejo especial para problemas de CMake."""
    
    # Primero intentar la instalación normal
    print("Intentando instalar sentence-transformers...")
    result = install_package("sentence-transformers")
    if result == 0:
        return True
        
    # Si falla, intentar con wheel pre-compilado
    print("\nInstalación estándar falló. Intentando alternativas...")
    print("Intentando instalar wheel de sentence-transformers pre-compilado...")
    
    # Intentar instalar necesidades básicas primero
    dependencies = [
        "torch", 
        "transformers", 
        "numpy", 
        "scikit-learn", 
        "scipy", 
        "nltk"
    ]
    
    for dep in dependencies:
        print(f"Instalando dependencia: {dep}")
        install_package(dep)
    
    # Intentar instalar sentencepiece binario (wheel) en lugar de compilarlo
    print("Intentando instalar sentencepiece desde un wheel pre-compilado...")
    result = install_package("--only-binary=:all: sentencepiece")
    
    if result == 0:
        # Si sentencepiece se instala, intentar sentence-transformers nuevamente
        print("Intentando instalar sentence-transformers ahora que sentencepiece está instalado...")
        result = install_package("sentence-transformers")
        if result == 0:
            return True
    
    print("No se pudo instalar sentence-transformers.")
    print("El sistema usará una implementación alternativa para embeddings.")
    
    # Crear una implementación alternativa básica para embeddings
    alt_embeddings_path = os.path.join(SCRIPT_DIR, "src", "alt_embeddings.py")
    os.makedirs(os.path.dirname(alt_embeddings_path), exist_ok=True)
    
    with open(alt_embeddings_path, 'w', encoding='utf-8') as f:
        f.write("""'''
Implementación alternativa para embeddings cuando sentence-transformers no está disponible.
Utiliza modelos de torch para generar embeddings básicos.
'''
import torch
import numpy as np

class SimpleEmbedder:
    '''Clase alternativa simple para embeddings.'''
    
    def __init__(self, model_name_or_path=None):
        # Ignoramos el nombre del modelo ya que estamos usando un enfoque alternativo
        print(f"Utilizando SimpleEmbedder en lugar de {model_name_or_path}")
        self.tokenizer = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def _tokenize(self, text):
        '''Tokenización simple en palabras.'''
        return text.lower().split()
        
    def encode(self, texts, batch_size=32, **kwargs):
        '''
        Genera embeddings simples para textos.
        Utiliza un hash consistente para generar vectores pseudo-aleatorios.
        '''
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        dim = 384  # Dimensión similar a modelos pequeños como MiniLM
        
        for text in texts:
            # Generar un embedding simple pero determinista basado en el hash del texto
            text_hash = hash(text) % (2**32)
            np.random.seed(text_hash)
            
            # Generar un vector con valores entre -1 y 1
            vec = np.random.uniform(-1, 1, dim)
            
            # Normalizar a norma unitaria como lo hacen los modelos de embedding reales
            vec = vec / np.linalg.norm(vec)
            
            embeddings.append(vec)
        
        if len(texts) == 1:
            return embeddings[0]
        return np.array(embeddings)

def load_embedder(model_name_or_path):
    '''Carga el embedder alternativo.'''
    return SimpleEmbedder(model_name_or_path)
""")
    
    print(f"✓ Implementación alternativa para embeddings creada en {alt_embeddings_path}")
    return False

def check_docker():
    """Verifica si Docker está instalado y en ejecución."""
    print("\n=== Verificando Docker ===")
    
    # Verificar si Docker está instalado
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Docker está instalado:")
            print(result.stdout.strip())
        else:
            print("✗ Docker no está instalado o no se puede acceder desde la línea de comandos.")
            return False
    except FileNotFoundError:
        print("✗ Docker no está instalado o no está en el PATH del sistema.")
        return False
    
    # Verificar si Docker está en ejecución
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["docker", "info"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Docker está en ejecución")
                return True
            else:
                print("✗ Docker no está en ejecución")
                return False
        else:  # Linux/Mac
            result = subprocess.run(["docker", "info"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Docker está en ejecución")
                return True
            else:
                print("✗ Docker no está en ejecución")
                return False
    except Exception as e:
        print(f"✗ Error al verificar el estado de Docker: {str(e)}")
        return False

def setup_neo4j():
    """Configura y ejecuta Neo4j usando Docker."""
    print("\n=== Configurando Neo4j ===")
    
    # Verificar si ya existe un contenedor de Neo4j
    result = subprocess.run(["docker", "ps", "-a", "--filter", "name=neo4j-lexi"], 
                           capture_output=True, text=True)
    
    if "neo4j-lexi" in result.stdout:
        print("Ya existe un contenedor de Neo4j. Verificando su estado...")
        
        # Verificar si el contenedor está en ejecución
        result = subprocess.run(["docker", "ps", "--filter", "name=neo4j-lexi"],
                               capture_output=True, text=True)
        
        if "neo4j-lexi" in result.stdout:
            print("El contenedor Neo4j ya está en ejecución.")
            return True
        else:
            # Si el contenedor existe pero no está en ejecución, intentar iniciarlo
            print("El contenedor Neo4j existe pero no está en ejecución. Iniciándolo...")
            result = subprocess.run(["docker", "start", "neo4j-lexi"], check=False)
            
            if result.returncode == 0:
                print("Contenedor Neo4j iniciado.")
                # Esperar para dar tiempo a Neo4j a iniciar
                time.sleep(5)
                return True
            else:
                print("Error al iniciar el contenedor Neo4j.")
                return False
    
    # Crear directorio para datos de Neo4j si no existe
    neo4j_data_dir = os.path.join(SCRIPT_DIR, "neo4j_data")
    os.makedirs(neo4j_data_dir, exist_ok=True)
    print(f"Directorio de datos Neo4j: {neo4j_data_dir}")
    
    # Convertir la ruta a formato compatible con Docker
    docker_path = neo4j_data_dir
    if platform.system() == "Windows":
        # En Windows, convertir la ruta a formato Docker (usando / en lugar de \)
        docker_path = docker_path.replace("\\", "/")
        # Asegurarse de que la ruta tenga el formato correcto para montaje en Docker
        if ":" in docker_path:
            drive, path = docker_path.split(":", 1)
            docker_path = f"/{drive.lower()}{path}"
    
    print(f"Ruta para montar en Docker: {docker_path}")
    
    # Ejecutar Neo4j con Docker
    print("Iniciando Neo4j con Docker...")
    
    cmd = [
        "docker", "run", "--name", "neo4j-lexi",
        "-p", "7474:7474", "-p", "7687:7687",
        "-e", "NEO4J_AUTH=neo4j/password",  # Contraseña inicial
        "-e", "NEO4J_dbms_memory_heap_max__size=1G",  # Limitar uso de memoria
        "-e", "NEO4J_dbms_memory_pagecache_size=512M",
        "-v", f"{docker_path}:/data",
        "--restart", "unless-stopped",  # Reiniciar automáticamente si falla
        "-d", "neo4j:5.15.0"  # Versión específica de Neo4j
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode == 0:
        print("\nNeo4j se ha iniciado correctamente.")
        print("Puede acceder a la interfaz de Neo4j en http://localhost:7474")
        print("Usuario: neo4j")
        print("Contraseña: password")
        
        # Esperar un momento para dar tiempo a Neo4j a iniciar
        print("Esperando 15 segundos para que Neo4j se inicialice...")
        time.sleep(15)
        return True
    else:
        print("\nError al iniciar Neo4j.")
        print(process.stderr)
        return False

def setup_weaviate():
    """Configura y ejecuta Weaviate usando Docker Compose."""
    print("\n=== Configuración de Weaviate ===")
    
    # Verificar si ya existe un contenedor de Weaviate
    result = subprocess.run(["docker", "ps", "-a", "--filter", "name=weaviate"],
                           capture_output=True, text=True)
    
    if "weaviate" in result.stdout:
        print("Ya existe un contenedor de Weaviate. Verificando si está en ejecución...")
        
        # Verificar si el contenedor está en ejecución
        result = subprocess.run(["docker", "ps", "--filter", "name=weaviate"],
                               capture_output=True, text=True)
        
        if "weaviate" in result.stdout:
            print("El contenedor Weaviate ya está en ejecución.")
            return True
    
    # Crear archivo docker-compose.yml para Weaviate
    docker_compose_path = os.path.join(SCRIPT_DIR, "docker-compose.yml")
    
    weaviate_compose = """version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:1.24.5
    ports:
      - 8080:8080
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: ''
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  weaviate_data:
"""
    
    try:
        with open(docker_compose_path, 'w') as file:
            file.write(weaviate_compose)
        print(f"Archivo docker-compose.yml creado en: {docker_compose_path}")
        
        # Ejecutar docker-compose up
        print("Iniciando Weaviate con Docker Compose...")
        result = run_command(f"docker-compose -f {docker_compose_path} up -d")
        
        if result == 0:
            print("Weaviate se ha iniciado correctamente.")
            print("Puede acceder a la API de Weaviate en http://localhost:8080")
            
            # Esperar unos segundos para asegurarse de que Weaviate esté listo
            print("Esperando 15 segundos para que Weaviate se inicialice...")
            time.sleep(15)
            return True
        else:
            print("Error al iniciar Weaviate.")
            return False
    except Exception as e:
        print(f"Error al configurar Weaviate: {str(e)}")
        return False

"""
Script unificado para la configuración del sistema de recuperación de documentos legales.
Este script se encarga de:
1. Instalar dependencias necesarias
2. Configurar servicios Docker (Neo4j y Weaviate)
3. Cargar datos y embeddings en Weaviate
4. Crear nodos y relaciones en el grafo Neo4j

Uso:
    python setup_system.py --all      # Ejecutar todos los pasos
    python setup_system.py --deps     # Solo instalar dependencias
    python setup_system.py --docker   # Solo configurar Docker
    python setup_system.py --data     # Solo cargar datos
    python setup_system.py --test     # Ejecutar búsqueda de prueba
"""

import os
import sys
import time
import subprocess
import argparse
import importlib
import json
import yaml
import platform
import hashlib
import pickle
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import codecs

# Definir rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Directorio raíz del proyecto (un nivel arriba)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Mostrar información sobre rutas
print(f"Directorio del script: {SCRIPT_DIR}")
print(f"Directorio raíz del proyecto: {PROJECT_ROOT}")
print(f"Ruta de datos: {DATA_DIR}")

# Asegurar que los directorios necesarios existan
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Dependencias requeridas
REQUIRED_PACKAGES = [
    "weaviate-client==3.26.0",
    "neo4j==5.14.0",
    "rank_bm25==0.2.2",
    "pyyaml==6.0.1",
    "numpy>=2.0.0",
    "sentence-transformers==2.2.2",
    "requests>=2.28.0"
]

def run_command(command, cwd=None):
    """Ejecuta un comando y muestra su salida en tiempo real."""
    print(f"Ejecutando: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=cwd
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def install_package(package, options=""):
    """Instala un paquete usando pip."""
    command = f"{sys.executable} -m pip install {options} {package}"
    return run_command(command)

def check_dependencies():
    """Verifica e instala las dependencias necesarias."""
    print("\n=== Verificando dependencias ===")
    
    # Verificar versión de Python
    python_version = sys.version_info
    print(f"Versión de Python detectada: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Verificar cada dependencia
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        package_name = package.split("==")[0].split(">=")[0]
        try:
            importlib.import_module(package_name)
            print(f"✓ {package} está instalado")
        except ImportError:
            print(f"✗ {package} no está instalado")
            missing_packages.append(package)
    
    # Instalar dependencias faltantes
    if missing_packages:
        print("\n=== Instalando dependencias faltantes ===")
        for package in missing_packages:
            print(f"\n{'='*50}")
            print(f"Instalando {package}...")
            print(f"{'='*50}")
            
            # Caso especial para sentence-transformers
            if package.startswith("sentence-transformers"):
                success = install_sentence_transformers()
            else:
                # Intentar con diferentes estrategias para otros paquetes
                strategies = [
                    ("", ""),  # Sin opciones
                    ("", "--no-build-isolation"),
                    ("", "--no-cache-dir")
                ]
                
                success = False
                for pkg, options in strategies:
                    package_to_install = pkg or package
                    print(f"\nIntentando instalar {package_to_install} {options}")
                    result = install_package(package_to_install, options)
                    if result == 0:
                        success = True
                        print(f"Instalación exitosa de {package_to_install}")
                        break
            
            if not success:
                print(f"\nNo se pudo instalar {package}")
                if package.startswith("sentence-transformers"):
                    print("Se usará una alternativa para generar embeddings.")
                else:
                    print("El sistema funcionará con capacidades limitadas.")
                    if package_name in ['weaviate-client', 'neo4j']:
                        return False
        
        print("\nInstalación de dependencias completada.")
    else:
        print("\nTodas las dependencias ya están instaladas.")
    
    # Verificar si podemos usar sentence-transformers o una alternativa
    try:
        import sentence_transformers
        print("✓ sentence-transformers está instalado correctamente")
    except ImportError:
        print("✗ sentence-transformers no está instalado. Se usará una alternativa.")
        try:
            # Verificar si tenemos las bibliotecas necesarias para la alternativa
            import torch
            import numpy
            print("✓ Se puede usar torch y numpy como alternativa para embeddings")
        except ImportError:
            print("✗ No se encontró torch, que es necesario para la alternativa")
            print("Intentando instalar torch...")
            result = install_package("torch")
            if result != 0:
                print("No se pudo instalar torch. El sistema tendrá funcionalidad limitada.")
    
    return True

def install_sentence_transformers():
    """Intenta instalar sentence-transformers con manejo especial para problemas de CMake."""
    
    # Primero intentar la instalación normal
    print("Intentando instalar sentence-transformers...")
    result = install_package("sentence-transformers")
    if result == 0:
        return True
        
    # Si falla, intentar con wheel pre-compilado
    print("\nInstalación estándar falló. Intentando alternativas...")
    print("Intentando instalar wheel de sentence-transformers pre-compilado...")
    
    # Intentar instalar necesidades básicas primero
    dependencies = [
        "torch", 
        "transformers", 
        "numpy", 
        "scikit-learn", 
        "scipy", 
        "nltk"
    ]
    
    for dep in dependencies:
        print(f"Instalando dependencia: {dep}")
        install_package(dep)
    
    # Intentar instalar sentencepiece binario (wheel) en lugar de compilarlo
    print("Intentando instalar sentencepiece desde un wheel pre-compilado...")
    result = install_package("--only-binary=:all: sentencepiece")
    
    if result == 0:
        # Si sentencepiece se instala, intentar sentence-transformers nuevamente
        print("Intentando instalar sentence-transformers ahora que sentencepiece está instalado...")
        result = install_package("sentence-transformers")
        if result == 0:
            return True
    
    print("No se pudo instalar sentence-transformers.")
    print("El sistema usará una implementación alternativa para embeddings.")
    
    # Crear una implementación alternativa básica para embeddings
    alt_embeddings_path = os.path.join(SCRIPT_DIR, "src", "alt_embeddings.py")
    os.makedirs(os.path.dirname(alt_embeddings_path), exist_ok=True)
    
    with open(alt_embeddings_path, 'w', encoding='utf-8') as f:
        f.write("""'''
Implementación alternativa para embeddings cuando sentence-transformers no está disponible.
Utiliza modelos de torch para generar embeddings básicos.
'''
import torch
import numpy as np

class SimpleEmbedder:
    '''Clase alternativa simple para embeddings.'''
    
    def __init__(self, model_name_or_path=None):
        # Ignoramos el nombre del modelo ya que estamos usando un enfoque alternativo
        print(f"Utilizando SimpleEmbedder en lugar de {model_name_or_path}")
        self.tokenizer = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def _tokenize(self, text):
        '''Tokenización simple en palabras.'''
        return text.lower().split()
        
    def encode(self, texts, batch_size=32, **kwargs):
        '''
        Genera embeddings simples para textos.
        Utiliza un hash consistente para generar vectores pseudo-aleatorios.
        '''
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        dim = 384  # Dimensión similar a modelos pequeños como MiniLM
        
        for text in texts:
            # Generar un embedding simple pero determinista basado en el hash del texto
            text_hash = hash(text) % (2**32)
            np.random.seed(text_hash)
            
            # Generar un vector con valores entre -1 y 1
            vec = np.random.uniform(-1, 1, dim)
            
            # Normalizar a norma unitaria como lo hacen los modelos de embedding reales
            vec = vec / np.linalg.norm(vec)
            
            embeddings.append(vec)
        
        if len(texts) == 1:
            return embeddings[0]
        return np.array(embeddings)

def load_embedder(model_name_or_path):
    '''Carga el embedder alternativo.'''
    return SimpleEmbedder(model_name_or_path)
""")
    
    print(f"✓ Implementación alternativa para embeddings creada en {alt_embeddings_path}")
    return False

def check_docker():
    """Verifica si Docker está instalado y en ejecución."""
    print("\n=== Verificando Docker ===")
    
    # Verificar si Docker está instalado
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Docker está instalado:")
            print(result.stdout.strip())
        else:
            print("✗ Docker no está instalado o no se puede acceder desde la línea de comandos.")
            return False
    except FileNotFoundError:
        print("✗ Docker no está instalado o no está en el PATH del sistema.")
        return False
    
    # Verificar si Docker está en ejecución
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["docker", "info"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Docker está en ejecución")
                return True
            else:
                print("✗ Docker no está en ejecución")
                return False
        else:  # Linux/Mac
            result = subprocess.run(["docker", "info"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Docker está en ejecución")
                return True
            else:
                print("✗ Docker no está en ejecución")
                return False
    except Exception as e:
        print(f"✗ Error al verificar el estado de Docker: {str(e)}")
        return False

def setup_neo4j():
    """Configura y ejecuta Neo4j usando Docker."""
    print("\n=== Configurando Neo4j ===")
    
    # Verificar si ya existe un contenedor de Neo4j
    result = subprocess.run(["docker", "ps", "-a", "--filter", "name=neo4j-lexi"], 
                           capture_output=True, text=True)
    
    if "neo4j-lexi" in result.stdout:
        print("Ya existe un contenedor de Neo4j. Verificando su estado...")
        
        # Verificar si el contenedor está en ejecución
        result = subprocess.run(["docker", "ps", "--filter", "name=neo4j-lexi"],
                               capture_output=True, text=True)
        
        if "neo4j-lexi" in result.stdout:
            print("El contenedor Neo4j ya está en ejecución.")
            return True
        else:
            # Si el contenedor existe pero no está en ejecución, intentar iniciarlo
            print("El contenedor Neo4j existe pero no está en ejecución. Iniciándolo...")
            result = subprocess.run(["docker", "start", "neo4j-lexi"], check=False)
            
            if result.returncode == 0:
                print("Contenedor Neo4j iniciado.")
                # Esperar para dar tiempo a Neo4j a iniciar
                time.sleep(5)
                return True
            else:
                print("Error al iniciar el contenedor Neo4j.")
                return False
    
    # Crear directorio para datos de Neo4j si no existe
    neo4j_data_dir = os.path.join(SCRIPT_DIR, "neo4j_data")
    os.makedirs(neo4j_data_dir, exist_ok=True)
    print(f"Directorio de datos Neo4j: {neo4j_data_dir}")
    
    # Convertir la ruta a formato compatible con Docker
    docker_path = neo4j_data_dir
    if platform.system() == "Windows":
        # En Windows, convertir la ruta a formato Docker (usando / en lugar de \)
        docker_path = docker_path.replace("\\", "/")
        # Asegurarse de que la ruta tenga el formato correcto para montaje en Docker
        if ":" in docker_path:
            drive, path = docker_path.split(":", 1)
            docker_path = f"/{drive.lower()}{path}"
    
    print(f"Ruta para montar en Docker: {docker_path}")
    
    # Ejecutar Neo4j con Docker
    print("Iniciando Neo4j con Docker...")
    
    cmd = [
        "docker", "run", "--name", "neo4j-lexi",
        "-p", "7474:7474", "-p", "7687:7687",
        "-e", "NEO4J_AUTH=neo4j/password",  # Contraseña inicial
        "-e", "NEO4J_dbms_memory_heap_max__size=1G",  # Limitar uso de memoria
        "-e", "NEO4J_dbms_memory_pagecache_size=512M",
        "-v", f"{docker_path}:/data",
        "--restart", "unless-stopped",  # Reiniciar automáticamente si falla
        "-d", "neo4j:5.15.0"  # Versión específica de Neo4j
    ]
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode == 0:
        print("\nNeo4j se ha iniciado correctamente.")
        print("Puede acceder a la interfaz de Neo4j en http://localhost:7474")
        print("Usuario: neo4j")
        print("Contraseña: password")
        
        # Esperar un momento para dar tiempo a Neo4j a iniciar
        print("Esperando 15 segundos para que Neo4j se inicialice...")
        time.sleep(15)
        return True
    else:
        print("\nError al iniciar Neo4j.")
        print(process.stderr)
        return False

def setup_weaviate():
    """Configura y ejecuta Weaviate usando Docker Compose."""
    print("\n=== Configuración de Weaviate ===")
    
    # Verificar si ya existe un contenedor de Weaviate
    result = subprocess.run(["docker", "ps", "-a", "--filter", "name=weaviate"],
                           capture_output=True, text=True)
    
    if "weaviate" in result.stdout:
        print("Ya existe un contenedor de Weaviate. Verificando si está en ejecución...")
        
        # Verificar si el contenedor está en ejecución
        result = subprocess.run(["docker", "ps", "--filter", "name=weaviate"],
                               capture_output=True, text=True)
        
        if "weaviate" in result.stdout:
            print("El contenedor Weaviate ya está en ejecución.")
            return True
    
    # Crear archivo docker-compose.yml para Weaviate
    docker_compose_path = os.path.join(SCRIPT_DIR, "docker-compose.yml")
    
    weaviate_compose = """version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:1.24.5
    ports:
      - 8080:8080
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: ''
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  weaviate_data:
"""
    
    try:
        with open(docker_compose_path, 'w') as file:
            file.write(weaviate_compose)
        print(f"Archivo docker-compose.yml creado en: {docker_compose_path}")
        
        # Ejecutar docker-compose up
        print("Iniciando Weaviate con Docker Compose...")
        result = run_command(f"docker-compose -f {docker_compose_path} up -d")
        
        if result == 0:
            print("Weaviate se ha iniciado correctamente.")
            print("Puede acceder a la API de Weaviate en http://localhost:8080")
            
            # Esperar unos segundos para asegurarse de que Weaviate esté listo
            print("Esperando 15 segundos para que Weaviate se inicialice...")
            time.sleep(15)
            return True
        else:
            print("Error al iniciar Weaviate.")
            return False
    except Exception as e:
        print(f"Error al configurar Weaviate: {str(e)}")
        return False

def load_config():
    """Carga la configuración desde YAML o crea un archivo por defecto."""
    print("\n=== Cargando configuración ===")
    
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                print(f"Configuración cargada desde {CONFIG_PATH}")
                return config
        except Exception as e:
            print(f"Error al cargar configuración: {str(e)}")
    
    # Configuración por defecto
    config = {
        "weaviate": {
            "enabled": True,
            "url": "http://localhost:8080",
            "api_key": None,
            "collection_name": "ArticulosLegales",
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "use_cache": True
        },
        "neo4j": {
            "enabled": True,
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password"
        },
        "bm25": {
            "enabled": True
        },
        "retrieval": {
            "top_n": 15,
            "weights": [0.4, 0.4, 0.2],  # vectorial, grafo, léxico
            "save_results": True,
            "results_dir": "results"
        }
    }
    
    # Guardar configuración
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)
        print(f"Configuración por defecto creada en {CONFIG_PATH}")
        return config
    except Exception as e:
        print(f"Error al crear archivo de configuración: {str(e)}")
        return config

def load_json_data(directory_path=DATA_DIR):
    """Carga todos los archivos JSON del directorio especificado."""
    print(f"\n=== Cargando datos desde {directory_path} ===")
    
    documents = []
    
    # Verificar si el directorio existe
    if not os.path.isdir(directory_path):
        print(f"Directorio no encontrado: {directory_path}")
        return []
    
    # Iterar por todos los archivos en el directorio
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            try:
                # Probar diferentes codificaciones
                encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1']
                data = None
                
                for encoding in encodings:
                    try:
                        with codecs.open(file_path, 'r', encoding=encoding) as file:
                            content = file.read()
                            data = json.loads(content)
                            print(f"Archivo {filename} cargado con codificación {encoding}")
                            break
                    except UnicodeDecodeError:
                        continue
                
                if data is None:
                    print(f"Error: No se pudo decodificar el archivo {filename}")
                    continue
                    
                print(f"Archivo cargado: {filename}")
                
                # Si los datos son una lista, extender documentos con sus elementos
                if isinstance(data, list):
                    print(f"  Encontrados {len(data)} documentos en formato lista")
                    documents.extend(data)
                # Si es un documento único, agregarlo a la lista
                else:
                    print(f"  Encontrado un documento único")
                    documents.append(data)
            except json.JSONDecodeError:
                print(f"Error: Formato JSON inválido en el archivo {filename}")
            except Exception as e:
                print(f"Error al cargar el archivo {filename}: {str(e)}")
    
    # Convertir documentos al formato estándar esperado por el sistema
    standardized_docs = []
    for doc in documents:
        # Verificar si el documento tiene los campos requeridos
        if 'content' in doc:
            # Crear un documento estandarizado
            std_doc = {
                'content': doc['content'],
                'article_id': '',
                'law_name': '',
                'article_number': '',
                'category': '',
                'source': ''
            }
            
            # Extraer metadatos si están disponibles
            if 'metadata' in doc:
                metadata = doc['metadata']
                if 'article' in metadata:
                    std_doc['article_number'] = metadata['article']
                    std_doc['article_id'] = f"{metadata.get('code', 'unknown')}_{metadata['article']}"
                if 'code' in metadata:
                    std_doc['law_name'] = metadata['code']
                if 'chapter' in metadata:
                    std_doc['category'] = metadata['chapter']
                if 'section' in metadata:
                    std_doc['source'] = metadata['section']
            
            standardized_docs.append(std_doc)
    
    print(f"Estandarizados {len(standardized_docs)} documentos")
    
    return standardized_docs

def connect_weaviate(config):
    """Conecta con Weaviate y verifica la conexión."""
    print("\n=== Conectando con Weaviate ===")
    
    weaviate_url = config["weaviate"].get("url", "http://localhost:8080")
    weaviate_api_key = config["weaviate"].get("api_key")
    
    try:
        import weaviate
        from weaviate.auth import AuthApiKey
        
        auth_config = AuthApiKey(api_key=weaviate_api_key) if weaviate_api_key else None
        
        client = weaviate.Client(
            url=weaviate_url,
            auth_client_secret=auth_config
        )
        
        # Verificar conexión
        if not client.is_ready():
            raise Exception("Weaviate no está listo")
            
        print(f"✓ Conexión exitosa con Weaviate en {weaviate_url}")
        return client
    except Exception as e:
        print(f"✗ Error al conectar con Weaviate: {str(e)}")
        return None

def connect_neo4j(config):
    """Conecta con Neo4j y verifica la conexión."""
    print("\n=== Conectando con Neo4j ===")
    
    neo4j_uri = config["neo4j"].get("uri", "bolt://localhost:7687")
    neo4j_username = config["neo4j"].get("username", "neo4j")
    neo4j_password = config["neo4j"].get("password", "password")
    
    try:
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        
        # Verificar conexión
        with driver.session() as session:
            session.run("RETURN 1")
        
        print(f"✓ Conexión exitosa con Neo4j en {neo4j_uri}")
        return driver
    except Exception as e:
        print(f"✗ Error al conectar con Neo4j: {str(e)}")
        return None

def create_weaviate_schema(client, collection_name):
    """Crea esquema para colección de artículos legales en Weaviate."""
    try:
        print(f"Verificando esquema para colección '{collection_name}'...")
        
        # Verificar si la colección ya existe
        schema = client.schema.get()
        classes = [c["class"] for c in schema["classes"]] if "classes" in schema else []
        
        if collection_name in classes:
            print(f"La colección '{collection_name}' ya existe.")
            return
    except Exception as e:
        print(f"Error al verificar esquema: {str(e)}")
    
    # Definir propiedades para artículos legales
    properties = [
        {
            "name": "content",
            "description": "Contenido completo del artículo",
            "dataType": ["text"]
        },
        {
            "name": "article_id",
            "description": "Identificador único del artículo",
            "dataType": ["string"]
        },
        {
            "name": "law_name",
            "description": "Nombre de la ley o código",
            "dataType": ["string"]
        },
        {
            "name": "article_number",
            "description": "Número de artículo dentro de la ley",
            "dataType": ["string"]
        },
        {
            "name": "category",
            "description": "Categoría o sección de la ley",
            "dataType": ["string"]
        },
        {
            "name": "source",
            "description": "Fuente del artículo",
            "dataType": ["string"]
        }
    ]
    
    # Definir esquema para artículos legales
    schema = {
        "class": collection_name,
        "description": "Artículos legales de varios códigos y leyes",
        "vectorizer": "none",  # Proporcionaremos nuestros propios vectores
        "properties": properties
    }
    
    try:
        client.schema.create_class(schema)
        print(f"✓ Se creó con éxito la colección '{collection_name}' en Weaviate.")
    except Exception as e:
        # Intentar con el método alternativo si el primero falla
        try:
            client.schema.create({"classes": [schema]})
            print(f"✓ Se creó con éxito la colección '{collection_name}' en Weaviate (método alternativo).")
        except Exception as e2:
            raise Exception(f"Error al crear esquema: {str(e)} / {str(e2)}")

def generate_embeddings(documents, embedding_model, cache_dir=CACHE_DIR, use_cache=True):
    """Genera embeddings para documentos con soporte para caché."""
    # Crear directorio de caché si no existe
    if use_cache and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Generar un nombre de archivo para la caché basado en los documentos y el modelo
    cache_id = hashlib.md5(f"{embedding_model}_{len(documents)}".encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"embeddings_{cache_id}.pkl")
    
    # Verificar si la caché existe y es válida
    if use_cache and os.path.exists(cache_file):
        try:
            print(f"Cargando embeddings desde caché: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error al cargar caché: {str(e)}")
    
    # Intentar cargar el modelo de embedding
    print(f"Generando embeddings usando modelo: {embedding_model}")
    
    try:
        # Intentar usar sentence-transformers si está disponible
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embedding_model)
        
        # Generar embeddings
        result = []
        for i, doc in enumerate(documents):
            # Extraer contenido del documento
            content = doc.get("content", "")
            if not content:
                print(f"Advertencia: Documento en índice {i} no tiene contenido, omitiendo.")
                continue
                
            # Generar embedding
            embedding = model.encode(content).tolist()
            result.append((doc, embedding))
            
            if i % 100 == 0 and i > 0:
                print(f"Generados embeddings para {i} documentos...")
    except ImportError:
        print("SentenceTransformer no está disponible, usando implementación alternativa...")
        
        # Usar la implementación alternativa
        try:
            # Intentar importar la implementación alternativa
            sys.path.append(os.path.join(SCRIPT_DIR, "src"))
            from alt_embeddings import load_embedder
            
            model = load_embedder(embedding_model)
            
            # Generar embeddings con la alternativa
            result = []
            for i, doc in enumerate(documents):
                content = doc.get("content", "")
                if not content:
                    print(f"Advertencia: Documento en índice {i} no tiene contenido, omitiendo.")
                    continue
                    
                # Generar embedding con la implementación alternativa
                embedding = model.encode(content).tolist()
                result.append((doc, embedding))
                
                if i % 100 == 0 and i > 0:
                    print(f"Generados embeddings alternativos para {i} documentos...")
        except Exception as e:
            print(f"Error usando implementación alternativa: {str(e)}")
            print("Usando embeddings aleatorios como último recurso...")
            
            # Como último recurso, generar embeddings aleatorios
            import numpy as np
            dim = 384  # Dimensión estándar para modelos pequeños
            
            result = []
            for i, doc in enumerate(documents):
                content = doc.get("content", "")
                if not content:
                    continue
                
                # Generar vector aleatorio pero determinista basado en el hash del contenido
                hash_val = hash(content) % (2**32)
                np.random.seed(hash_val)
                vec = np.random.uniform(-1, 1, dim)
                vec = vec / np.linalg.norm(vec)  # Normalizar a norma unitaria
                
                result.append((doc, vec.tolist()))
                
                if i % 100 == 0 and i > 0:
                    print(f"Generados embeddings aleatorios para {i} documentos...")
    
    # Guardar en caché si está habilitado
    if use_cache and result:
        try:
            print(f"Guardando embeddings en caché: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"Error al guardar caché: {str(e)}")
    
    print(f"✓ Generados {len(result)} embeddings")
    return result

def store_embeddings_weaviate(client, collection_name, documents, embedding_model, use_cache=True):
    """Genera embeddings para documentos y los almacena en Weaviate."""
    # Generar o cargar embeddings
    doc_embeddings = generate_embeddings(
        documents, 
        embedding_model=embedding_model,
        cache_dir=CACHE_DIR,
        use_cache=use_cache
    )
    
    # Crear un proceso por lotes
    with client.batch as batch:
        batch.batch_size = 100
        
        for i, (doc, embedding) in enumerate(doc_embeddings):
            # Extraer contenido y metadatos del documento
            content = doc.get("content", "")
            if not content:
                continue
                
            # Preparar propiedades
            properties = {
                "content": content,
                "article_id": doc.get("article_id", f"article_{i}"),
                "law_name": doc.get("law_name", ""),
                "article_number": doc.get("article_number", ""),
                "category": doc.get("category", ""),
                "source": doc.get("source", "")
            }
            
            # Añadir el objeto con su embedding
            batch.add_data_object(
                data_object=properties,
                class_name=collection_name,
                vector=embedding
            )
            
            if i % 100 == 0 and i > 0:
                print(f"Procesados {i} documentos...")
                
    print(f"✓ Almacenados {len(doc_embeddings)} documentos con embeddings en Weaviate.")

def create_neo4j_nodes(driver, documents):
    """Crea nodos de Article en Neo4j a partir de los datos de documentos."""
    created_ids = []
    
    with driver.session() as session:
        for doc in documents:
            # Extraer propiedades del documento
            article_id = doc.get("article_id", "")
            if not article_id:
                # Generar article_id si no está presente
                law_name = doc.get("law_name", "unknown")
                article_number = doc.get("article_number", "")
                if not article_number and "metadata" in doc:
                    article_number = doc["metadata"].get("article", "")
                if law_name and article_number:
                    article_id = f"{law_name}_{article_number}"
                else:
                    continue
                
            # Manejar metadatos si existen
            law_name = doc.get("law_name", "")
            article_number = doc.get("article_number", "")
            category = doc.get("category", "")
            source = doc.get("source", "")
            
            # Extraer de metadatos si están disponibles
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
            
            # Crear mapa de propiedades
            properties = {
                "article_id": article_id,
                "content": doc.get("content", ""),
                "law_name": law_name,
                "article_number": article_number,
                "category": category,
                "source": source
            }
            
            # Crear o fusionar el nodo Article
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

def create_law_relationship(driver, law_name, article_ids):
    """Crea un nodo Law y establece relaciones CONTAINS con nodos Article."""
    if not law_name or not article_ids:
        return
        
    with driver.session() as session:
        # Crear nodo Law
        query = """
        MERGE (l:Law {name: $law_name})
        RETURN l
        """
        session.run(query, law_name=law_name)
        
        # Crear relaciones entre Law y Articles
        for article_id in article_ids:
            query = """
            MATCH (l:Law {name: $law_name})
            MATCH (a:Article {article_id: $article_id})
            MERGE (l)-[r:CONTAINS]->(a)
            RETURN r
            """
            session.run(query, law_name=law_name, article_id=article_id)
    
    print(f"Creado nodo Law '{law_name}' con relaciones a {len(article_ids)} artículos.")

def create_semantic_content_relationships(driver):
    """Crea relaciones entre artículos basadas en análisis de contenido semántico."""
    print("\n=== Creando relaciones semánticas de contenido entre artículos ===")
    
    with driver.session() as session:
        # Crear índices para mejor rendimiento
        try:
            # Crear índice en article_id si no existe
            query = """
            CREATE INDEX article_id_index IF NOT EXISTS FOR (a:Article) ON (a.article_id)
            """
            session.run(query)
            print("Índice en article_id creado")
        except Exception as e:
            print(f"Error al crear índice article_id: {str(e)}")
            try:
                # Intentar con sintaxis alternativa
                query = """
                CREATE INDEX ON :Article(article_id)
                """
                session.run(query)
                print("Índice en article_id creado con sintaxis alternativa")
            except Exception as e2:
                print(f"Error al crear índice article_id (alternativa): {str(e2)}")
        
        # 1. Crear relaciones entre artículos en la misma categoría
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
                print(f"Creadas {record['relCount']} relaciones SAME_CATEGORY")
        except Exception as e:
            print(f"Error al crear relaciones SAME_CATEGORY: {str(e)}")
        
        # 2. Crear relaciones basadas en patrones de texto comunes
        try:
            # Enfoque más simple usando términos legales comunes
            common_legal_terms = [
                "derecho", "obligación", "contrato", "persona", "responsabilidad", 
                "propiedad", "plazo", "demanda", "resolución", "sentencia", "sanción",
                "pena", "delito", "tribunal", "juez", "procedimiento", "recurso"
            ]
            
            for term in common_legal_terms[:5]:  # Limitar a primeros 5 términos
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
                    print(f"Creadas {record['relCount']} relaciones SHARES_CONCEPT para término '{term}'")
        except Exception as e:
            print(f"Error al crear relaciones SHARES_CONCEPT: {str(e)}")
        
        # 3. Crear relaciones basadas en referencias en el contenido
        try:
            query = """
            MATCH (a1:Article)
            MATCH (a2:Article)
            WHERE a1 <> a2
            AND a1.article_number IS NOT NULL
            AND a2.content CONTAINS a1.article_number
            MERGE (a2)-[r:REFERENCES_ARTICLE]->(a1)
            RETURN count(r) as relCount
            LIMIT 500
            """
            result = session.run(query)
            record = result.single()
            if record:
                print(f"Creadas {record['relCount']} relaciones REFERENCES_ARTICLE")
        except Exception as e:
            print(f"Error al crear relaciones REFERENCES_ARTICLE: {str(e)}")
    
    print("Relaciones de contenido semántico creadas correctamente.")

def create_cross_law_relationships(driver, documents):
    """Crea relaciones entre diferentes códigos y leyes basadas en referencias temáticas."""
    print("\n=== Creando relaciones entre diferentes códigos y leyes ===")
    
    # Extraer categorías y temas de los documentos
    law_categories = {}
    for doc in documents:
        law_name = doc.get("law_name")
        if not law_name and "metadata" in doc:
            law_name = doc["metadata"].get("code")
            
        if not law_name:
            continue
            
        # Obtener categorías
        categories = []
        if "category" in doc and doc["category"]:
            categories.append(doc["category"])
        if "metadata" in doc and "chapter" in doc["metadata"]:
            categories.append(doc["metadata"]["chapter"])
        
        # Obtener temas
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
            
        # Añadir categorías y temas
        for category in categories:
            if category:
                law_categories[law_name]["categories"].add(category)
                
        for topic in topics:
            if topic:
                law_categories[law_name]["topics"].add(topic)
    
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
                            if category:  # Asegurar que la categoría no esté vacía
                                query = """
                                MATCH (l1:Law {name: $law1})
                                MATCH (l2:Law {name: $law2})
                                MERGE (l1)-[r:RELATED_BY_CATEGORY {category: $category}]->(l2)
                                RETURN r
                                """
                                session.run(query, law1=law1, law2=law2, category=category)
                    
                    # Encontrar temas compartidos
                    shared_topics = data1["topics"].intersection(data2["topics"])
                    if shared_topics:
                        for topic in shared_topics:
                            if topic:  # Asegurar que el tema no esté vacío
                                query = """
                                MATCH (l1:Law {name: $law1})
                                MATCH (l2:Law {name: $law2})
                                MERGE (l1)-[r:RELATED_BY_TOPIC {topic: $topic}]->(l2)
                                RETURN r
                                """
                                session.run(query, law1=law1, law2=law2, topic=topic)
    
    print("Relaciones entre códigos y leyes creadas correctamente.")

def setup_neo4j_data(driver, documents):
    """Configura Neo4j y carga documentos si es necesario."""
    print("\n=== Configurando datos en Neo4j ===")
    
    # Verificar si ya existen datos en Neo4j
    with driver.session() as session:
        # Verificar si existen nodos Article
        query = """
        MATCH (a:Article)
        RETURN count(a) as article_count
        """
        result = session.run(query)
        record = result.single()
        if record and record["article_count"] > 0:
            print(f"Ya existen {record['article_count']} nodos Article en Neo4j.")
            choice = input("¿Desea continuar y crear más relaciones? (s/n): ")
            if choice.lower() != 's':
                return
    
    # Crear nodos de artículos
    print("Creando nodos de artículos en Neo4j...")
    article_ids = create_neo4j_nodes(driver, documents)
    print(f"✓ Creados {len(article_ids)} nodos de artículos")
    
    # Agrupar artículos por ley
    law_articles = {}
    for doc in documents:
        # Extraer law_name del documento o de metadata si está disponible
        law_name = doc.get("law_name")
        if not law_name and "metadata" in doc:
            law_name = doc["metadata"].get("code")
            
        article_id = doc.get("article_id")
        if not article_id and "metadata" in doc:
            # Crear article_id a partir de código y número de artículo si está disponible
            code = doc["metadata"].get("code")
            article_num = doc["metadata"].get("article")
            if code and article_num:
                article_id = f"{code}_{article_num}"
                
        if law_name and article_id:
            if law_name not in law_articles:
                law_articles[law_name] = []
            law_articles[law_name].append(article_id)
    
    # Crear nodos de leyes y relaciones
    print("Creando nodos de leyes y relaciones...")
    for law_name, article_ids in law_articles.items():
        create_law_relationship(driver, law_name, article_ids)
    
    # Crear relaciones semánticas basadas en contenido
    create_semantic_content_relationships(driver)
    
    # Crear relaciones entre códigos y leyes
    create_cross_law_relationships(driver, documents)
    
    print("\n✓ Configuración de datos en Neo4j completada")

def setup_weaviate_data(client, config, documents):
    """Configura Weaviate y carga documentos si es necesario."""
    print("\n=== Configurando datos en Weaviate ===")
    
    collection_name = config["weaviate"].get("collection_name", "ArticulosLegales")
    embedding_model = config["weaviate"].get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
    use_cache = config["weaviate"].get("use_cache", True)
    
    # Crear esquema si no existe
    create_weaviate_schema(client, collection_name)
    
    # Almacenar embeddings (se generarán solo si no están en caché)
    print("Almacenando documentos con embeddings en Weaviate...")
    print(f"Usando caché: {'Sí' if use_cache else 'No'}")
    
    try:
        store_embeddings_weaviate(
            client,
            collection_name,
            documents,
            embedding_model=embedding_model,
            use_cache=use_cache
        )
        print("✓ Documentos almacenados correctamente en Weaviate")
    except Exception as e:
        print(f"Error al almacenar documentos en Weaviate: {str(e)}")

def run_test_search(config, weaviate_client, neo4j_driver, documents):
    """Ejecuta una búsqueda de prueba."""
    print("\n=== Ejecutando búsqueda de prueba ===")
    
    # Importar la función de búsqueda principal
    try:
        sys.path.append(SCRIPT_DIR)
        from main import search_query
        
        # Consulta de ejemplo
        query = "estafa defraudación incumplimiento contractual daños y perjuicios"
        print(f"Consulta de prueba: '{query}'")
        
        # Ejecutar búsqueda
        results = search_query(query, config, weaviate_client, neo4j_driver, documents)
        
        # Mostrar resultados resumidos
        print(f"\nEncontrados {len(results)} resultados:")
        for i, result in enumerate(results[:5], 1):  # Mostrar solo los primeros 5
            print(f"\nResultado #{i} (Relevancia: {result.get('score', 0):.2f})")
            print(f"Ley/Código: {result.get('law_name', 'N/A')}")
            print(f"Artículo: {result.get('article_number', 'N/A')}")
            
            # Mostrar snippet del contenido
            content = result.get('content', 'Sin contenido')
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"Contenido: {content}")
        
        if len(results) > 5:
            print(f"\n... y {len(results) - 5} resultados más.")
        
        print("\n✓ Búsqueda de prueba completada exitosamente")
        return True
    except Exception as e:
        print(f"Error al ejecutar búsqueda de prueba: {str(e)}")
        return False

def main():
    """Función principal del programa."""
    parser = argparse.ArgumentParser(description="Configurar el sistema de recuperación de documentos legales")
    parser.add_argument("--all", action="store_true", help="Ejecutar todos los pasos de configuración")
    parser.add_argument("--deps", action="store_true", help="Verificar e instalar dependencias")
    parser.add_argument("--docker", action="store_true", help="Configurar servicios Docker")
    parser.add_argument("--neo4j", action="store_true", help="Cargar datos en Neo4j")
    parser.add_argument("--weaviate", action="store_true", help="Cargar datos en Weaviate")
    parser.add_argument("--data", action="store_true", help="Cargar datos en ambas bases de datos")
    parser.add_argument("--test", action="store_true", help="Ejecutar búsqueda de prueba")
    parser.add_argument("--skip-checks", action="store_true", help="Omitir verificaciones de dependencias")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Directorio de datos")
    parser.add_argument("--create-example", action="store_true", help="Crear datos de ejemplo si no hay datos")
    
    args = parser.parse_args()
    
    # Si no se proporcionan argumentos, asumir --all
    if not any([args.all, args.deps, args.docker, args.neo4j, args.weaviate, args.data, args.test, args.create_example]):
        args.all = True
    
    # Verificar dependencias (a menos que se indique --skip-checks)
    if (args.all or args.deps) and not args.skip_checks:
        if not check_dependencies():
            print("Se han detectado problemas con las dependencias.")
            proceed = input("¿Desea continuar de todos modos? (s/n): ")
            if proceed.lower() != 's':
                print("Configuración cancelada.")
                return
    
    # Verificar y configurar Docker
    if args.all or args.docker:
        if not args.skip_checks and not check_docker():
            print("Error: Docker no está instalado o no está en ejecución.")
            print("Instalando solo las dependencias de Python...")
            check_dependencies()
            return
        
        if not setup_neo4j():
            print("Error: No se pudo configurar Neo4j.")
            if not args.all:  # Si es --all, continuar con otros pasos
                return
        
        if not setup_weaviate():
            print("Error: No se pudo configurar Weaviate.")
            if not args.all:  # Si es --all, continuar con otros pasos
                return
    
    # Cargar configuración
    config = load_config()
    
    # Crear datos de ejemplo si se solicita
    if args.create_example:
        print("\n=== Creando datos de ejemplo ===")
        example_file_path = os.path.join(DATA_DIR, 'ejemplo.json')
        example_data = [
            {
                "content": "Este es un artículo de ejemplo sobre responsabilidad contractual. El que incumpliere una obligación contractual deberá indemnizar los daños y perjuicios causados.",
                "metadata": {
                    "code": "CODIGO_CIVIL",
                    "article": "1101",
                    "chapter": "Responsabilidad Civil"
                }
            },
            {
                "content": "Los contratos se perfeccionan por el mero consentimiento, y desde entonces obligan, no sólo al cumplimiento de lo expresamente pactado, sino también a todas las consecuencias que, según su naturaleza, sean conformes a la buena fe, al uso y a la ley.",
                "metadata": {
                    "code": "CODIGO_CIVIL",
                    "article": "1258",
                    "chapter": "Contratos"
                }
            },
            {
                "content": "El empleador está obligado a indemnizar al trabajador cuando éste sufra un accidente durante la prestación de sus servicios o con ocasión de los mismos.",
                "metadata": {
                    "code": "LEY_CONTRATO_TRABAJO",
                    "article": "75",
                    "chapter": "Accidentes de Trabajo"
                }
            }
        ]
        
        try:
            with open(example_file_path, 'w', encoding='utf-8') as f:
                json.dump(example_data, f, ensure_ascii=False, indent=2)
            print(f"Archivo de ejemplo creado en: {example_file_path}")
        except Exception as e:
            print(f"Error al crear archivo de ejemplo: {str(e)}")
    
    # Cargar datos
    documents = None
    if args.all or args.data or args.neo4j or args.weaviate or args.test:
        documents = load_json_data(args.data_dir)
        if not documents:
            print("Error: No se pudieron cargar documentos.")
            create_example = input("¿Desea crear datos de ejemplo para continuar? (s/n): ")
            if create_example.lower() == 's':
                # Crear archivo de ejemplo
                example_file_path = os.path.join(DATA_DIR, 'ejemplo.json')
                example_data = [
                    {
                        "content": "Este es un artículo de ejemplo sobre responsabilidad contractual. El que incumpliere una obligación contractual deberá indemnizar los daños y perjuicios causados.",
                        "metadata": {
                            "code": "CODIGO_CIVIL",
                            "article": "1101",
                            "chapter": "Responsabilidad Civil"
                        }
                    },
                    {
                        "content": "Los contratos se perfeccionan por el mero consentimiento, y desde entonces obligan, no sólo al cumplimiento de lo expresamente pactado, sino también a todas las consecuencias que, según su naturaleza, sean conformes a la buena fe, al uso y a la ley.",
                        "metadata": {
                            "code": "CODIGO_CIVIL",
                            "article": "1258",
                            "chapter": "Contratos"
                        }
                    }
                ]
                
                try:
                    with open(example_file_path, 'w', encoding='utf-8') as f:
                        json.dump(example_data, f, ensure_ascii=False, indent=2)
                    print(f"Archivo de ejemplo creado en: {example_file_path}")
                    # Volver a cargar los datos
                    documents = load_json_data(args.data_dir)
                except Exception as e:
                    print(f"Error al crear archivo de ejemplo: {str(e)}")
                    return
            else:
                return
    
    # Conectar con Neo4j y cargar datos
    neo4j_driver = None
    if args.all or args.data or args.neo4j or args.test:
        neo4j_driver = connect_neo4j(config)
        if not neo4j_driver and (args.neo4j or args.test):
            print("Error: No se pudo conectar con Neo4j.")
            print("Si Neo4j está en ejecución, verifique la configuración en config.yaml")
            if not args.all:  # Si es --all, continuar con otros pasos
                return
        
        if neo4j_driver and (args.all or args.data or args.neo4j):
            try:
                setup_neo4j_data(neo4j_driver, documents)
            except Exception as e:
                print(f"Error al configurar datos en Neo4j: {str(e)}")
                if not args.all:
                    return
    
    # Conectar con Weaviate y cargar datos
    weaviate_client = None
    if args.all or args.data or args.weaviate or args.test:
        weaviate_client = connect_weaviate(config)
        if not weaviate_client and (args.weaviate or args.test):
            print("Error: No se pudo conectar con Weaviate.")
            print("Si Weaviate está en ejecución, verifique la configuración en config.yaml")
            if not args.all:
                return
        
        if weaviate_client and (args.all or args.data or args.weaviate):
            try:
                setup_weaviate_data(weaviate_client, config, documents)
            except Exception as e:
                print(f"Error al configurar datos en Weaviate: {str(e)}")
                if not args.all:
                    return
    
    # Ejecutar búsqueda de prueba
    if args.all or args.test:
        try:
            run_test_search(config, weaviate_client, neo4j_driver, documents)
        except Exception as e:
            print(f"Error al ejecutar búsqueda de prueba: {str(e)}")
    
    # Cerrar conexiones
    if neo4j_driver:
        neo4j_driver.close()
        print("Conexión de Neo4j cerrada.")
    
    print("\n=== Configuración completada ===")
    print("El sistema está listo para su uso.")
    print("Puede ejecutar búsquedas con: python main.py --query \"su consulta aquí\"")

if __name__ == "__main__":
    main()