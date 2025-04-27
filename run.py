#!/usr/bin/env python
"""
Script para ejecutar el sistema de recuperación de documentos legales.
Este es un punto de entrada simplificado para el sistema.
"""
import os
import sys
import argparse

def main():
    """Función principal para ejecutar el sistema."""
    parser = argparse.ArgumentParser(description="Sistema de Recuperación de Documentos Legales")
    parser.add_argument("--setup", action="store_true", help="Configurar el sistema antes de ejecutarlo")
    parser.add_argument("--query", type=str, help="Consulta para buscar documentos legales")
    args = parser.parse_args()
    
    # Si se solicita configurar el sistema
    if args.setup:
        print("Configurando el sistema...")
        setup_script = os.path.join("setup", "setup_system.py")
        os.system(f"{sys.executable} {setup_script}")
        return
    
    # Ejecutar el sistema con la consulta proporcionada
    cmd = f"{sys.executable} main.py"
    if args.query:
        cmd += f" --query \"{args.query}\""
    
    os.system(cmd)

if __name__ == "__main__":
    main()
