# Trabajo-Inteligencia-Artificial

# Análisis del Impacto Turístico: Airbnb & Eventos en Sevilla 

Este proyecto analiza la dinámica de precios y ocupación de los alojamientos turísticos en Sevilla, cruzando datos de **Inside Airbnb** con indicadores socioeconómicos del **INE**. Nuestro enfoque principal es medir el "Efecto Fiestas" (Semana Santa y Feria) y su correlación con la rentabilidad por barrios.

## 🚀 Hoja de Ruta del Proyecto
Siguiendo las directrices de implementación, el proyecto se divide en:

1.  **Limpieza e Integración:** Carga optimizada de datasets (manejo de LFS y compresión ZIP) y unión espacial (*Merge*) entre calendarios y listings.
2.  **Ingeniería de Fechas:** Creación de etiquetas temporales para identificar eventos clave de Sevilla (2024/2025) y variables de estacionalidad.
3.  **Análisis de Inflación:** Cálculo del diferencial de precios en Feria de Abril y Semana Santa vs. periodos normales.
4.  **Business Intelligence:** Estimación de facturación mensual por anfitrión y análisis de la profesionalización del sector (Grandes Tenedores).
5.  **Visualización Avanzada:** Mapas de calor de demanda y series temporales de precios.

## 🛠️ Requisitos Técnicos
Para asegurar la **reproducibilidad** del análisis, se debe utilizar el entorno virtual configurado:
* **Entorno:** Python 3.9+
* **Librerías principales:** `pandas`, `geopandas`, `matplotlib`, `seaborn`.
* **Configuración:** ```bash
  python3 -m venv .venv
  source .venv/bin/activate  # En Mac/Linux
  pip install -r requirements.txt