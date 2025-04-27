"""
Script para configurar y ejecutar Weaviate usando Docker.
"""
import subprocess
import sys
import time
import os
import platform

def check_docker_installed():
    """Verifica si Docker está instalado y en ejecución."""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("Docker está instalado:")
            print(result.stdout.strip())
            return True
        else:
            print("Docker no está instalado o no se puede acceder desde la línea de comandos.")
            return False
    except FileNotFoundError:
        print("Docker no está instalado o no está en el PATH del sistema.")
        return False

def check_docker_running():
    """Verifica si Docker está en ejecución."""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["docker", "info"], capture_output=True, text=True)
            return result.returncode == 0
        else:  # Linux/Mac
            result = subprocess.run(["docker", "info"], capture_output=True, text=True)
            return result.returncode == 0
    except Exception as e:
        print(f"Error al verificar el estado de Docker: {str(e)}")
        return False

def run_weaviate_docker():
    """Ejecuta Weaviate usando Docker Compose."""
    # Crear el archivo docker-compose.yml para Weaviate
    docker_compose_content = """
version: '3.4'
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
    
    # Guardar el archivo docker-compose.yml
    docker_compose_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docker-compose.yml")
    with open(docker_compose_path, "w", encoding="utf-8") as f:
        f.write(docker_compose_content)
    
    print(f"Archivo docker-compose.yml creado en {docker_compose_path}")
    
    # Ejecutar docker-compose up
    try:
        print("Iniciando Weaviate con Docker Compose...")
        process = subprocess.Popen(
            ["docker-compose", "up", "-d"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Mostrar la salida en tiempo real
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print("Weaviate se ha iniciado correctamente.")
            print("Puede acceder a la interfaz de Weaviate en http://localhost:8080")
            return True
        else:
            print("Error al iniciar Weaviate.")
            return False
    except Exception as e:
        print(f"Error al ejecutar Docker Compose: {str(e)}")
        return False

def check_weaviate_status():
    """Verifica si Weaviate está en funcionamiento."""
    import time
    import requests
    
    max_retries = 10
    retry_interval = 3  # segundos
    
    print("Verificando el estado de Weaviate...")
    
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8080/v1/.well-known/ready")
            if response.status_code == 200:
                print("Weaviate está en funcionamiento.")
                return True
            else:
                print(f"Weaviate aún no está listo (intento {i+1}/{max_retries}).")
        except requests.exceptions.ConnectionError:
            print(f"No se puede conectar a Weaviate (intento {i+1}/{max_retries}).")
        
        if i < max_retries - 1:
            print(f"Esperando {retry_interval} segundos antes de volver a intentar...")
            time.sleep(retry_interval)
    
    print("No se pudo conectar a Weaviate después de varios intentos.")
    return False

def main():
    """Función principal."""
    print("=== Configuración de Weaviate ===")
    
    # Verificar si Docker está instalado
    if not check_docker_installed():
        print("\nPor favor, instale Docker antes de continuar.")
        print("Puede descargar Docker Desktop desde: https://www.docker.com/products/docker-desktop")
        return
    
    # Verificar si Docker está en ejecución
    if not check_docker_running():
        print("\nDocker no está en ejecución. Por favor, inicie Docker y vuelva a ejecutar este script.")
        return
    
    # Ejecutar Weaviate con Docker Compose
    if run_weaviate_docker():
        # Verificar si Weaviate está en funcionamiento
        if check_weaviate_status():
            print("\nWeaviate está listo para ser utilizado.")
            print("Ahora puede ejecutar el sistema de recuperación de documentos legales.")
        else:
            print("\nWeaviate se inició, pero no responde. Verifique los logs de Docker para más información.")
    
if __name__ == "__main__":
    main()
