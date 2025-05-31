# Sistema de Recuperación de Documentos Legales

Sistema simplificado para búsqueda y recuperación de documentos legales utilizando técnicas de procesamiento de lenguaje natural y bases de datos gráficas y vectoriales.

## Estructura del Proyecto

```
lexi_blank/
├── main.py                    # Script principal
├── config.yaml               # Configuración del sistema
├── test_system.py            # Script de prueba
├── data/                     # Documentos legales en JSON
│   ├── codigo_penal.json
│   ├── ley_empleo.json
│   └── ley_contrato_empleo.json
├── setup/                    # Scripts de configuración
│   ├── setup_system.py
│   └── docker-compose.yml
├── src/                      # Módulos del sistema
│   ├── config_loader.py      # Carga de configuración
│   ├── data_loader.py        # Carga de datos JSON
│   ├── legal_domains.py      # Detección de dominios legales
│   ├── neo4j_utils.py        # Utilidades para Neo4j
│   └── weaviate_utils.py     # Utilidades para Weaviate
└── cache/                    # Cache de embeddings (se crea automáticamente)
```

## Instalación Rápida

### 1. Instalar Dependencias

```bash
# Instalar dependencias básicas
pip install neo4j==5.14.0 weaviate-client==3.26.0 pyyaml numpy rank_bm25

# Opcional: sentence-transformers para mejores embeddings
pip install sentence-transformers
```

### 2. Configurar Servicios

```bash
# Configurar Docker y servicios
python setup/setup_system.py --all
```

### 3. Probar el Sistema

```bash
# Ejecutar script de prueba
python test_system.py

# Si todo está bien, probar una búsqueda
python main.py --query "despido sin indemnización por embarazo"
```

## Uso Básico

### Búsqueda Simple

```bash
python main.py --query "fui despedida sin indemnización por embarazo"
```

### Configurar Sistema

```bash
python main.py --setup
```

### Limpiar Base de Datos

```bash
python main.py --clear-neo4j
```

## Características

### ✅ Funciones Implementadas

- **Búsqueda Multi-modal**: Combina búsqueda vectorial, por grafo y léxica
- **Detección de Dominios Legales**: Identifica automáticamente temas como embarazo, despido, discriminación
- **Clasificación Legal**: Categoriza consultas en áreas del derecho (laboral, civil, penal, etc.)
- **Cache de Embeddings**: Reutiliza embeddings generados para mayor eficiencia
- **Fallback para Embeddings**: Funciona sin sentence-transformers usando embeddings simples

### 🎯 Dominios Legales Soportados

- **Embarazo**: Protección laboral durante embarazo y maternidad
- **Despido**: Terminación de contratos laborales
- **Discriminación**: Acoso y trato diferencial
- **Remuneración**: Salarios, indemnizaciones y compensaciones
- **Jornada**: Horarios de trabajo y descansos
- **Accidentes**: Accidentes de trabajo y enfermedades profesionales
- **Prestaciones**: Obras sociales y beneficios
- **Procesos Administrativos**: Denuncias y procedimientos

## Configuración

El archivo `config.yaml` controla el comportamiento del sistema:

```yaml
weaviate:
  enabled: true
  url: http://localhost:8080
  collection_name: ArticulosLegales
  embedding_model: paraphrase-multilingual-MiniLM-L12-v2
  use_cache: true

neo4j:
  enabled: true
  uri: bolt://localhost:7687
  username: neo4j
  password: password

bm25:
  enabled: true

retrieval:
  top_n: 15
  weights: [0.4, 0.4, 0.2]  # vectorial, grafo, léxico
  save_results: true
  results_dir: results
```

## Solución de Problemas

### Error de Importación

Si obtienes errores de importación:

```bash
# Verificar estructura de archivos
python test_system.py

# Reinstalar dependencias
pip install --force-reinstall neo4j weaviate-client pyyaml numpy
```

### Problemas con Docker

```bash
# Verificar que Docker esté corriendo
docker ps

# Reiniciar servicios
python setup/setup_system.py --docker
```

### Sin sentence-transformers

El sistema funciona sin sentence-transformers usando embeddings simples:

```bash
# Solo con dependencias básicas
pip install neo4j weaviate-client pyyaml numpy rank_bm25
python main.py --query "tu consulta"
```

## Ejemplos de Consultas

### Consultas Laborales

```bash
# Despido por embarazo
python main.py --query "fui despedida sin indemnización por estar embarazada"

# Problemas de horario
python main.py --query "me hacen trabajar más de 8 horas sin pagar extras"

# Accidente de trabajo
python main.py --query "me lastimé en el trabajo y no me dan ART"
```

### Consultas Civiles

```bash
# Incumplimiento de contrato
python main.py --query "no me pagaron lo acordado en el contrato"

# Problemas de propiedad
python main.py --query "mi vecino construyó en mi terreno"
```

## Extensión del Sistema

### Añadir Nuevos Dominios

Edita `src/legal_domains.py`:

```python
LEGAL_DOMAINS = {
    "TuNuevoDominio": ["palabra1", "palabra2", "palabra3"],
    # ... otros dominios
}
```

### Añadir Nuevas Categorías

Edita `main.py` en la sección `LEGAL_CATEGORIES`:

```python
LEGAL_CATEGORIES = {
    "TU_CATEGORIA": ["keyword1", "keyword2"],
    # ... otras categorías
}
```

## Arquitectura Simplificada

1. **main.py**: Orquesta todo el sistema
2. **legal_domains.py**: Detecta dominios específicos en consultas
3. **neo4j_utils.py**: Maneja búsquedas por grafo y relaciones
4. **weaviate_utils.py**: Maneja búsquedas vectoriales semánticas
5. **data_loader.py**: Carga y estandariza documentos JSON

## Soporte

Para problemas o preguntas:

1. Ejecuta `python test_system.py` para diagnosticar
2. Revisa los logs en consola
3. Verifica que Docker esté corriendo
4. Asegúrate de que los archivos JSON estén en `/data`

## Limitaciones Conocidas

- Funciona mejor con consultas en español
- Requiere Docker para funcionalidad completa
- Los embeddings simples son menos precisos que sentence-transformers
- Base de datos debe configurarse antes del primer uso