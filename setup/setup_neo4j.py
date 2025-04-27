"""
Script para configurar y ejecutar Neo4j usando Docker.
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

def run_neo4j_docker():
    """Ejecuta Neo4j usando Docker."""
    # Verificar si ya existe un contenedor de Neo4j
    try:
        result = subprocess.run(["docker", "ps", "-a", "--filter", "name=neo4j-lexi"], capture_output=True, text=True)
        if "neo4j-lexi" in result.stdout:
            print("Ya existe un contenedor de Neo4j. Verificando si está en ejecución...")
            
            # Verificar si el contenedor está en ejecución
            result = subprocess.run(["docker", "ps", "--filter", "name=neo4j-lexi"], capture_output=True, text=True)
            if "neo4j-lexi" in result.stdout:
                print("El contenedor Neo4j ya está en ejecución.")
                return True
            else:
                print("El contenedor Neo4j existe pero no está en ejecución. Iniciándolo...")
                subprocess.run(["docker", "start", "neo4j-lexi"], check=True)
                print("Contenedor Neo4j iniciado.")
                return True
    except Exception as e:
        print(f"Error al verificar el estado del contenedor Neo4j: {str(e)}")
    
    # Crear directorio para datos de Neo4j si no existe
    neo4j_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neo4j_data")
    os.makedirs(neo4j_data_dir, exist_ok=True)
    
    # Ejecutar Neo4j con Docker
    try:
        print("Iniciando Neo4j con Docker...")
        
        # Convertir la ruta a formato compatible con Docker
        if platform.system() == "Windows":
            # En Windows, convertir la ruta a formato Docker (usando / en lugar de \)
            neo4j_data_dir = neo4j_data_dir.replace("\\", "/")
            # Asegurarse de que la ruta tenga el formato correcto para montaje en Docker
            if ":" in neo4j_data_dir:
                drive, path = neo4j_data_dir.split(":", 1)
                neo4j_data_dir = f"/{drive.lower()}{path}"
        
        cmd = [
            "docker", "run", "--name", "neo4j-lexi",
            "-p", "7474:7474", "-p", "7687:7687",
            "-e", "NEO4J_AUTH=neo4j/password",
            "-v", f"{neo4j_data_dir}:/data",
            "-d", "neo4j:5.15.0"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Mostrar la salida en tiempo real
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print("Neo4j se ha iniciado correctamente.")
            print("Puede acceder a la interfaz de Neo4j en http://localhost:7474")
            print("Usuario: neo4j")
            print("Contraseña: password")
            return True
        else:
            print("Error al iniciar Neo4j.")
            return False
    except Exception as e:
        print(f"Error al ejecutar Docker: {str(e)}")
        return False

def check_neo4j_status():
    """Verifica si Neo4j está en funcionamiento."""
    import time
    import requests
    
    max_retries = 10
    retry_interval = 3  # segundos
    
    print("Verificando el estado de Neo4j...")
    
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:7474")
            if response.status_code == 200:
                print("Neo4j está en funcionamiento.")
                return True
            else:
                print(f"Neo4j aún no está listo (intento {i+1}/{max_retries}).")
        except requests.exceptions.ConnectionError:
            print(f"No se puede conectar a Neo4j (intento {i+1}/{max_retries}).")
        
        if i < max_retries - 1:
            print(f"Esperando {retry_interval} segundos antes de volver a intentar...")
            time.sleep(retry_interval)
    
    print("No se pudo conectar a Neo4j después de varios intentos.")
    return False

def main():
    """Función principal."""
    print("=== Configuración de Neo4j ===")
    
    # Verificar si Docker está instalado
    if not check_docker_installed():
        print("\nPor favor, instale Docker antes de continuar.")
        print("Puede descargar Docker Desktop desde: https://www.docker.com/products/docker-desktop")
        return
    
    # Verificar si Docker está en ejecución
    if not check_docker_running():
        print("\nDocker no está en ejecución. Por favor, inicie Docker y vuelva a ejecutar este script.")
        return
    
    # Ejecutar Neo4j con Docker
    if run_neo4j_docker():
        # Verificar si Neo4j está en funcionamiento
        if check_neo4j_status():
            print("\nNeo4j está listo para ser utilizado.")
            print("Ahora puede ejecutar el sistema de recuperación de documentos legales.")
        else:
            print("\nNeo4j se inició, pero no responde. Verifique los logs de Docker para más información.")
    
if __name__ == "__main__":
    main()
