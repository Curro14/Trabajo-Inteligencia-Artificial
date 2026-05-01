# Tasador Inteligente y Análisis del Ecosistema Airbnb en Sevilla
Una aproximación híbrida mediante Machine Learning, Big Data e Interoperabilidad R-Python

## 1. Introducción y propósito
Este proyecto nace de una pregunta concreta: ¿por qué resulta tan difícil saber si un alojamiento en Sevilla está bien o mal valorado? El mercado vacacional genera una asimetría de información notable: anfitriones que infravaloran propiedades con buena reputación, turistas que pagan de más en barrios que no justifican ese precio, y los tasadores convencionales no la resuelven porque, sencillamente, no incorporan el contexto.
La plataforma desarrollada aborda esa limitación de forma directa. No se trata de predecir precios a partir de habitaciones y baños. El modelo cruza variables físicas del inmueble con datos socioeconómicos del INE, renta media por distrito, y con un índice de satisfacción construido sobre las reseñas textuales de los huéspedes. La combinación de estas tres capas es lo que da sentido estadístico a la estimación.

## 2. Arquitectura de datos y criterios de diseño
La volumetría del proyecto es heterogénea: desde tablas de pocos miles de registros hasta el archivo calendar.csv.zip, que prácticamente alcanza los tres millones de filas. Gestionar esa diferencia de escala sin que colapse la memoria del sistema fue uno de los problemas prácticos más relevantes del desarrollo.
El repositorio separa intencionadamente dos capas de datos:
data/ — producción. Contiene los artefactos ya procesados: el modelo KNN serializado, el escalador de variables y los datasets refinados listos para la aplicación. También aquí se almacenan las visualizaciones exportadas desde R, en alta resolución, para evitar regenerarlas en cada carga.
data_raw/ — auditoría e histórico. Guarda los CSVs originales sin tocar, los polígonos GIS y las tablas intermedias de limpieza. Esta separación no es un capricho organizativo: permite que cualquier evaluador reconstruya la cadena de transformación completa desde el dato crudo hasta su entrada al modelo, facilitando una auditoría real del proceso. Hubo versiones intermedias de esa limpieza que costaron bastante depurar —principalmente por conflictos de codificación en los CSV del INE y por coordenadas fuera de rango en algunos listings— y ese rastro ha quedado documentado.

## 3. Implementación técnica
Machine Learning e ingeniería de características
El modelo elegido es un regresor KNN, entrenado con StandardScaler para equilibrar variables de escala muy distinta —coordenadas geográficas frente a renta per cápita, por ejemplo—. La decisión de usar KNN frente a otras alternativas responde a su interpretabilidad: en un contexto académico, poder explicar que "este alojamiento se tasa comparándolo con sus vecinos más similares" tiene mayor valor pedagógico que un ensemble.

El ajuste de pesos merece mención aparte. La estadística descriptiva inicial mostró que el score de sentimiento y la condición de Superhost correlacionaban con el precio de forma más consistente de lo esperado. El modelo los incorpora con ponderación específica, lo que reduce el sesgo predictivo que aparecía cuando se trataban como variables ordinarias más.
Procesamiento de Lenguaje Natural

Las reseñas de los huéspedes contienen información que las estrellas no capturan. Un alojamiento puede mantener una media de 4,8 sobre 5 con comentarios que revelan problemas recurrentes de ruido o limpieza. El análisis de sentimiento —implementado sobre los textos en bruto— produce un índice continuo que el modelo usa para distinguir entre reputación cosmética y reputación real. La diferencia en la estimación de precio puede ser considerable.

Big Data con Dask
El análisis de tres millones de registros permite evaluar el compromiso entre la agilidad de Pandas y la arquitectura escalable de Dask. En este escenario, la paridad de tiempos se explica por este volumen de datos, el coste de coordinación del grafo de tareas de Dask iguala al rendimiento bruto en memoria de Pandas. No obstante, Dask asegura la viabilidad del proceso mediante lazy loading y computación paralela ante volúmenes que superan la capacidad de la RAM, cuantificando el punto de inflexión técnica entre el procesamiento secuencial y la gestión eficiente de recursos.

Interoperabilidad R-Python
Python gestiona la lógica de la aplicación, los mapas interactivos y la interfaz en Streamlit. R hace otro trabajo: el análisis estadístico offline que cruza presión turística con renta per cápita por distrito. ggplot2 produce esas visualizaciones con una calidad gráfica que Plotly no iguala para este tipo de análisis multivariante estático. Los scripts de R se ejecutan en local y exportan las imágenes al directorio data/; desde ahí, Streamlit las carga directamente. No es la integración más elegante del mundo, pero es estable y reproducible.


## 4. Guía de instalación y ejecución (Reproducibilidad)
Para garantizar una reproducción exacta del entorno de desarrollo y evitar conflictos de versiones, la gestión de paquetes y dependencias se ha implementado mediante uv, un gestor de alto rendimiento para Python.

### Creación del entorno e instalación de dependencias
El proyecto incluye los archivos pyproject.toml y uv.lock, que definen la estructura y las versiones exactas de las librerías necesarias. Para levantar el proyecto desde cero, ejecuta los siguientes comandos en la terminal:

#### 1) Clonar el repositorio:
git clone <https://github.com/Curro14/Trabajo-Inteligencia-Artificial>
cd Trabajo-Inteligencia-Artificial

#### 2) Crear el entorno virtual:
uv venv

#### 3)Sincronizar e instalar dependencias:
uv sync
(Este comando leerá el archivo uv.lock e instalará exactamente las mismas versiones de las librerías utilizadas en el desarrollo).

## 5. Ejecución y Acceso Público

Una vez configurado el entorno, la interfaz analítica se puede desplegar localmente ejecutando el script principal mediante Streamlit:
### 1. Activar el entorno virtual
En Windows: .venv\Scripts\activate
En macOS/Linux: source .venv/bin/activate

### 2. Correr la aplicación
streamlit run app.py

Para facilitar la evaluación del proyecto sin necesidad de realizar la instalación local, la plataforma se encuentra desplegada y operativa en la siguiente URL pública:
Enlace a la web: https://tu-app-sevilla.streamlit.app
