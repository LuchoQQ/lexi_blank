"""
Script para instalar las dependencias del proyecto de manera controlada.
"""
import subprocess
import sys
import time

def run_command(command):
    """Ejecuta un comando y muestra su salida en tiempo real."""
    print(f"Ejecutando: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def install_package(package, options=""):
    """Instala un paquete usando pip."""
    command = f"{sys.executable} -m pip install {options} {package}"
    return run_command(command)

def main():
    """Función principal para instalar las dependencias."""
    print("Instalando dependencias para el sistema de recuperación de documentos legales...")
    
    # Instalar dependencias básicas primero
    packages = [
        "weaviate-client==3.26.0",
        "neo4j==5.14.0",
        "rank_bm25==0.2.2",
        "pyyaml==6.0.1",
        "numpy==1.24.4"
    ]
    
    for package in packages:
        print(f"\n{'='*50}")
        print(f"Instalando {package}...")
        print(f"{'='*50}")
        result = install_package(package)
        if result != 0:
            print(f"Error al instalar {package}. Código de salida: {result}")
            choice = input("¿Desea continuar con las siguientes instalaciones? (s/n): ")
            if choice.lower() != 's':
                return
    
    # Instalar sentence-transformers con opciones especiales
    print(f"\n{'='*50}")
    print("Instalando sentence-transformers...")
    print(f"{'='*50}")
    
    # Intentar primero con una versión específica
    result = install_package("sentence-transformers==2.0.0")
    
    if result != 0:
        print("No se pudo instalar sentence-transformers==2.0.0")
        print("Intentando con --no-build-isolation...")
        result = install_package("sentence-transformers==2.0.0", "--no-build-isolation")
    
    if result != 0:
        print("No se pudo instalar sentence-transformers con --no-build-isolation")
        print("Intentando con --only-binary=:all:...")
        result = install_package("sentence-transformers==2.0.0", "--only-binary=:all:")
    
    if result != 0:
        print("No se pudo instalar sentence-transformers con --only-binary")
        print("Intentando con la última versión disponible...")
        result = install_package("sentence-transformers")
    
    if result != 0:
        print("\nNo se pudo instalar sentence-transformers.")
        print("El sistema funcionará con capacidades limitadas.")
        print("Para búsqueda vectorial, se usará una función de embedding simplificada.")
        
        # Preguntar si desea usar la versión simplificada
        choice = input("¿Desea usar una versión simplificada sin sentence-transformers? (s/n): ")
        if choice.lower() == 's':
            print("Se usará una versión simplificada.")
            # Aquí se podría modificar automáticamente los archivos necesarios
        else:
            print("Instalación cancelada.")
            return
    else:
        print("\nTodas las dependencias se instalaron correctamente.")
    
    print("\nPuede ejecutar el sistema con: python main.py")

if __name__ == "__main__":
    main()
