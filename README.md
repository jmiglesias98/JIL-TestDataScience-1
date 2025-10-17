# JIL-TestDataScience-1

Este repositorio contiene un proyecto de ciencia de datos centrado en evaluar la efectividad de una campaña de marketing de un banco portugués. El objetivo principal es analizar cómo las acciones de la campaña afectan la decisión de los clientes contactados para contratar un depósito, y proporcionar herramientas predictivas y analíticas que puedan apoyar la planificación de futuras campañas.

Entre las herramientas desarrolladas se incluyen:

- **Modelo de propensión a la contratación de depósitos**: permite predecir la probabilidad de que un cliente contrate un depósito, y puede ejecutarse de manera periódica, tanto en modo batch como online, asegurando un uso eficiente y escalable en entornos de producción.

- **Simulador “What-If**: proporciona al equipo de marketing la posibilidad de realizar análisis de escenarios basados en los resultados históricos y las predicciones del modelo (https://jil-testdatascience-1-app.streamlit.app/).

El proyecto combina análisis exploratorio de datos, construcción y evaluación de modelos predictivos, y está preparado para un despliegue reproducible mediante contenedores Docker, lo que facilita su integración en entornos productivos y el uso colaborativo.

## Estructura del Proyecto

| Carpeta / Archivo      | Descripción |
|:---------------------:|:------------|
| `notebooks/`           | Contiene notebooks de Jupyter donde se realiza: exploración y limpieza de datos, entrenamiento y validación de modelos predictivos, evaluación de métricas de desempeño, interpretabilidad y explicabilidad de los modelos. Además, incluye un archivo HTML con un resumen detallado del contexto del proyecto, procedimientos y resultados obtenidos. |
| `src/`                 | Scripts y funciones listas para ejecutar el modelo entrenado en un entorno de producción, integrando DevOps mediante contenedores Docker. Incluye la API y utilidades para preprocesar datos y generar predicciones. |
| `models/`              | Contiene el modelo definitivo entrenado, listo para su uso en predicciones o para integrarse en otros entornos. |
| `data/`                | Carpeta para almacenar los datasets utilizados en el proyecto, organizada en subcarpetas según su estado: por ejemplo, `raw` para datos originales y `processed` para datos preprocesados. |
| `.devcontainer/`       | Configuración de contenedor para asegurar un entorno de desarrollo reproducible, útil para colaborar en el proyecto sin problemas de compatibilidad. |
| `Dockerfile`           | Define la imagen Docker que incluye todas las dependencias y configuraciones necesarias para ejecutar el proyecto de manera consistente. |
| `requirements.txt`     | Lista de bibliotecas de Python con versiones específicas para garantizar compatibilidad y reproducibilidad. |

## Devops

A continuación se enumeran los distintos scripts utilizados en la operativización del producto.

| Archivo                     | Descripción |
|-----------------------------|-------------|
| `__init__.py`               | Archivo de inicialización del paquete `src`, permite importar módulos como un paquete de Python. |
| `app.py`                    | Implementa la API RESTful (probablemente con FastAPI) para interactuar con el modelo entrenado y exponer endpoints de predicción. |
| `data_cleaner.py`           | Contiene funciones para limpieza y preparación de datos antes del análisis o predicción. |
| `preprocesador_dinamico.py`| Contiene un preprocesador flexible/dinámico que adapta los datos a la estructura requerida por el modelo. |
| `load_model.py`             | Se encarga de cargar el modelo entrenado desde almacenamiento persistente para su uso en predicciones. |
| `inference.py`              | Funciones para realizar inferencias/predicciones usando el modelo entrenado. |
| `streamlit_app.py`          | Aplicación web interactiva usando Streamlit para que el equipo de Marketing pueda realizar simulaciones What-If. |

## Requisitos

- Python 3.8 o superior
- Docker

## Instalación y Ejecución

| Paso | Comando |
|:----:|:-------|
| Descargar repositorio zip y descomprimir | `-` |
| Posicionarse en el directorio del proyecto | `cd "C:\Users\Juanmi Iglesias\Desktop\JIL-TestDataScience-1-main"` |
| Construir la imagen Docker | `docker build -t jil-testdatascience-1 .` |
| Ejecutar el contenedor con la aplicación | `docker run -it --rm -p 8000:8000 jil-testdatascience-1 uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload` |
| Realizar predicciones sobre un CSV de clientes | `curl -X POST "http://localhost:8000/predict_csv" ^ -F "file=@C:\Users\Juanmi Iglesias\Desktop\JIL-TestDataScience-1-main\data\raw\clientes_20251016.csv" ^ -o "C:\Users\Juanmi Iglesias\Desktop\JIL-TestDataScience-1-main\data\processed\clientes_20251016_predicciones.csv"` |

NOTA: Se debe sustituir C:\Users\Juanmi Iglesias\Desktop\ por la ruta en la que cada uno vaya a trabajar.
