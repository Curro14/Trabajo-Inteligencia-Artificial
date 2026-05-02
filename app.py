import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import utils as ut
import plotly.express as px
import streamlit.components.v1 as components
# -----------------------------------------------------------------------------
# SECCION 1: CONFIGURACION DE LA PAGINA
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Tasador Airbnb Sevilla", layout="wide")

# -----------------------------------------------------------------------------
# SECCION 2: GESTION DE RUTAS Y CARGA DE DATOS
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(BASE_DIR, "data")):
    PATH_DATA = os.path.join(BASE_DIR, "data")
else:
    PATH_DATA = os.path.join(BASE_DIR, "..", "data")

try:
    knn_model = joblib.load(os.path.join(PATH_DATA, 'knn_model.pkl'))
    scaler = joblib.load(os.path.join(PATH_DATA, 'scaler.pkl'))
    dict_barrios = joblib.load(os.path.join(PATH_DATA, 'dict_barrios.pkl'))
    
    # 👇 --- AÑADE ESTA LÍNEA EXACTAMENTE AQUÍ --- 👇
    df_viviendas = pd.read_csv(os.path.join(PATH_DATA, 'dataset_ia_final.csv'))
    
except FileNotFoundError:
    st.error(f"Error: No se encuentran los archivos del modelo en {PATH_DATA}")
    st.stop()

# -----------------------------------------------------------------------------
# SECCION 3: ENCABEZADO Y ESTRUCTURA DE PESTAÑAS
# -----------------------------------------------------------------------------
st.title("Tasador de Mercado Inteligente - Sevilla")

tab1, tab2, tab3 = st.tabs(["💰 Tasador de Precios", "📍 Explorador de Mercado", "📊 Análisis de Festividades"])

# =============================================================================
# PESTAÑA 1: CALCULADORA DE PRECIOS (MODELO KNN)
# =============================================================================
with tab1:
    col_izq, col_der = st.columns([1, 1], gap="large")

    # --- Columna Izquierda: Datos Básicos ---
    with col_izq:
        st.subheader("Ubicación y Capacidad")
        
        lista_barrios = sorted(list(dict_barrios.keys()))
        barrio_seleccionado = st.selectbox("Barrio de Sevilla", lista_barrios)
        
        tipos_alojamiento = ["Piso/Casa completa", "Habitación privada", "Habitación compartida"]
        tipo_alojamiento = st.selectbox("Tipo de alquiler", tipos_alojamiento)
        map_tipo = {"Piso/Casa completa": 3, "Habitación privada": 2, "Habitación compartida": 1}
        
        c1, c2, c3 = st.columns(3)
        accommodates = c1.number_input("Huéspedes", 1, 16, 2)
        bedrooms = c2.number_input("Dormitorios", 1, 10, 1)
        bathrooms = c3.number_input("Baños", 0.5, 10.0, 1.0, step=0.5)

        st.subheader("Reputación")
        
        c4, c5 = st.columns(2)
        is_superhost = c4.toggle("Eres Superhost")
        
        opciones_reviews = [
            "Ninguna (Nuevo)", "1 a 10 reseñas", "11 a 50 reseñas", 
            "51 a 150 reseñas", "151 a 300 reseñas", "Mas de 300 reseñas"
        ]
        rango_reviews = c5.selectbox("Volumen histórico de reseñas", opciones_reviews)
        
        map_reviews = {
            "Ninguna (Nuevo)": 0, "1 a 10 reseñas": 5, "11 a 50 reseñas": 30,
            "51 a 150 reseñas": 100, "151 a 300 reseñas": 225, "Mas de 300 reseñas": 400
        }
        
        score_sentimiento = st.slider("Nota media de reseñas (0-1)", 0.0, 1.0, 0.85)

    # --- Columna Derecha: Amenities ---
    with col_der:
        st.subheader("Comodidades (Amenities)")
        st.markdown("Selecciona los extras que tiene el alojamiento:")
        
        c6, c7 = st.columns(2)
        has_ac = c6.checkbox("Aire Acondicionado", value=True)
        has_pool = c6.checkbox("Piscina")
        has_elevator = c6.checkbox("Ascensor")
        
        has_parking = c7.checkbox("Parking (Incluido o de pago)")
        has_balcony = c7.checkbox("Terraza o Balcon")
        has_workspace = c7.checkbox("Zona de trabajo (Escritorio)")

    st.divider()

    # --- Motor de Prediccion ---
    if st.button("Calcular Precio Óptimo", type="primary", use_container_width=True):
        lat = dict_barrios[barrio_seleccionado]['latitude']
        lon = dict_barrios[barrio_seleccionado]['longitude']
        renta = dict_barrios[barrio_seleccionado]['renta_media']
        
        v_superhost = 1 if is_superhost else 0
        v_reviews_num = map_reviews[rango_reviews] 
        v_pool = 1 if has_pool else 0
        v_ac = 1 if has_ac else 0
        v_parking = 1 if has_parking else 0
        v_elevator = 1 if has_elevator else 0
        v_balcony = 1 if has_balcony else 0
        v_workspace = 1 if has_workspace else 0

        input_data = np.array([[
            lat, lon, accommodates, bedrooms, bathrooms, map_tipo[tipo_alojamiento], 
            renta, score_sentimiento, v_superhost, v_reviews_num,
            v_pool, v_ac, v_parking, v_elevator, v_balcony, v_workspace
        ]])
        
        input_scaled = scaler.transform(input_data)
        
        # Aplicacion de pesos (Ajustado segun entrenamiento)
        input_scaled[:, 7] *= 2.0  
        input_scaled[:, 8] *= 2.0  
        
        prediccion = knn_model.predict(input_scaled)[0]
        
        st.success(f"Precio Sugerido: {prediccion:.2f} € / noche")
        st.info(f"Rango competitivo en tu zona: {prediccion*0.85:.2f}€ - {prediccion*1.15:.2f}€")
        
        # --- EXPLICABILIDAD DEL MODELO ---
        st.divider()
        st.subheader("🔍 ¿Por qué este precio? (Auditoría del Modelo)")
        st.write("El algoritmo estima tu precio basándose en estos alojamientos similares:")
        
        # Le pasamos directamente df_viviendas (que ya cargaste al principio de tu app.py)
        vecinos_df, fig_pca = ut.generar_explicabilidad_knn(knn_model, df_viviendas, input_scaled)
        
        # Mostramos la tabla
        columnas = ['price', 'accommodates', 'bedrooms', 'bathrooms_text', 'Distancia']
        st.dataframe(vecinos_df[columnas], width='stretch')
        
        # Mostramos el gráfico
        st.write("**Mapa de Similitud (PCA):** Proyección en 2D del mercado.")
        st.plotly_chart(fig_pca, width='stretch')




# =============================================================================
# PESTAÑA 2: BUSCADOR INTERACTIVO PARA USUARIOS (AIRBNB STYLE)
# =============================================================================
with tab2:
    st.header("🏡 Encuentra tu alojamiento ideal")
    st.markdown("Ajusta los filtros según tus necesidades para tus próximas vacaciones. Las gráficas te ayudarán a entender cómo está el mercado.")

    ruta_dataset_final = os.path.join(PATH_DATA, 'dataset_ia_final.csv')
    
    if os.path.exists(ruta_dataset_final):
        # 1. Carga y Preparación de Datos
        df_viviendas = pd.read_csv(ruta_dataset_final)
        
        # Limpieza básica de columnas clave
        if 'price' in df_viviendas.columns and df_viviendas['price'].dtype == 'object':
            df_viviendas['price'] = pd.to_numeric(df_viviendas['price'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        df_viviendas['price'] = df_viviendas['price'].fillna(df_viviendas['price'].mean())
        
        df_viviendas['accommodates'] = df_viviendas['accommodates'].fillna(2)
        df_viviendas['bedrooms'] = df_viviendas['bedrooms'].fillna(1)
        
        # Extraer número de baños si es texto (ej. "1.5 baths")
        if 'bathrooms_text' in df_viviendas.columns and 'bathrooms_num' not in df_viviendas.columns:
            import re
            def extraer_banos(texto):
                if pd.isna(texto): return 1.0
                numeros = re.findall(r"[-+]?\d*\.\d+|\d+", str(texto))
                return float(numeros[0]) if numeros else 1.0
            df_viviendas['bathrooms_num'] = df_viviendas['bathrooms_text'].apply(extraer_banos)
        elif 'bathrooms' in df_viviendas.columns:
            df_viviendas['bathrooms_num'] = df_viviendas['bathrooms'].fillna(1)
        else:
            df_viviendas['bathrooms_num'] = 1.0 # Fallback de seguridad

        # ---------------------------------------------------------------------
        # 2. PANEL DE FILTROS "ESTILO AIRBNB"
        # ---------------------------------------------------------------------
        with st.expander("🔍 FILTROS DE BÚSQUEDA", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                distritos_unicos = sorted(df_viviendas['neighbourhood_group_cleansed'].dropna().unique())
                opciones_distritos = ["Todos los distritos"] + distritos_unicos
                distrito_sel = st.selectbox("📍 ¿Dónde quieres quedarte?", opciones_distritos)
                
                max_p = int(df_viviendas['price'].max()) if not df_viviendas.empty else 800
                rango_precio = st.slider("💰 Presupuesto por noche (€):", 0, 800, (20, 150))
                
            with col2:
                min_huespedes = st.number_input("👥 ¿Cuántos sois?", min_value=1, max_value=16, value=2)
                min_habs = st.number_input("🛏️ Habitaciones mínimas", min_value=1, max_value=10, value=1)
                
            with col3:
                min_banos = st.number_input("🚿 Baños mínimos", min_value=1.0, max_value=10.0, value=1.0, step=0.5)
                
                # Buscador de texto libre para comodidades (Piscinas, WiFi, etc)
                amenidades = st.multiselect("✨ Imprescindibles:", ["Piscina", "Aire acondicionado", "Parking", "Ascensor"])

        # ---------------------------------------------------------------------
        # 3. LÓGICA DE FILTRADO
        # ---------------------------------------------------------------------
        # Copia para no modificar el original
        df_filtrado = df_viviendas.copy()
        
        if distrito_sel != "Todos los distritos":
            df_filtrado = df_filtrado[df_filtrado['neighbourhood_group_cleansed'] == distrito_sel]
            
        df_filtrado = df_filtrado[
            (df_filtrado['price'].between(rango_precio[0], rango_precio[1])) &
            (df_filtrado['accommodates'] >= min_huespedes) &
            (df_filtrado['bedrooms'] >= min_habs) &
            (df_filtrado['bathrooms_num'] >= min_banos)
        ]
        
        # Filtro de texto para amenidades (si la columna amenities existe)
        if amenidades and 'amenities' in df_filtrado.columns:
            am_lo = df_filtrado['amenities'].str.lower().fillna('')
            if "Piscina" in amenidades: df_filtrado = df_filtrado[am_lo.str.contains('pool')]
            if "Aire acondicionado" in amenidades: df_filtrado = df_filtrado[am_lo.str.contains('air conditioning|ac')]
            if "Parking" in amenidades: df_filtrado = df_filtrado[am_lo.str.contains('parking')]
            if "Ascensor" in amenidades: df_filtrado = df_filtrado[am_lo.str.contains('elevator')]

        st.divider()

        # ---------------------------------------------------------------------
        # 4. MUESTRA DE RESULTADOS Y GRÁFICAS
        # ---------------------------------------------------------------------
        if not df_filtrado.empty:
            # Resumen visual
            st.success(f"¡Hemos encontrado **{len(df_filtrado)} alojamientos** que encajan con tu búsqueda!")
            
            # El mapa toma todo el ancho para que sea fácil explorar
            fig_mapa = ut.generar_mapa_interactivo(df_filtrado)
            st.plotly_chart(fig_mapa, use_container_width=True)  
                      
            st.markdown("### 📊 Conoce tu mercado antes de reservar")
            st.markdown("Descubre qué zonas son más baratas y qué tipo de pisos hay disponibles con tus filtros actuales.")

            # Fila 1 de gráficas
            g1, g2 = st.columns(2)
            with g1: st.plotly_chart(ut.grafico_precio_medio_barrio(df_filtrado), use_container_width=True)
            with g2: st.plotly_chart(ut.grafico_distribucion_precios(df_filtrado), use_container_width=True)

            # Fila 2 de gráficas
            g3, g4 = st.columns(2)
            with g3: st.plotly_chart(ut.grafico_oferta_por_barrio(df_filtrado), use_container_width=True)
            with g4: st.plotly_chart(ut.grafico_tipo_habitaciones(df_filtrado), use_container_width=True)

            # Fila 3 de gráficas
            g5, g6 = st.columns(2)
            with g5: st.plotly_chart(ut.grafico_top_barrios_baratos(df_filtrado), use_container_width=True)
            with g6: st.plotly_chart(ut.grafico_capacidad_vs_precio(df_filtrado), use_container_width=True)

        else:
            st.warning("🥲 Vaya, no hay alojamientos con esas características exactas. Prueba a subir el presupuesto o quitar algún requisito (como la piscina).")

    else:
        st.error("No se encontró `dataset_ia_final.csv`. Verifica la carpeta data.")

# =============================================================================
# PESTAÑA 3: ANÁLISIS DE MERCADO E IA
# =============================================================================
with tab3:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    st.header("📈 Análisis Avanzado e Inteligencia Artificial")
    
    # --- PARTE 1: GALERÍA R ---
    st.subheader("📊 Análisis de Mercado y Sentimiento (R + ggplot2)")
    st.markdown("Visualizaciones estáticas generadas cruzando datos inmobiliarios con rentas del INE y análisis de Procesamiento de Lenguaje Natural (NLP).")
    
    rutas_r = ut.obtener_rutas_graficos_r()
    
    # Comprobamos si las gráficas de R ya han sido generadas
    if os.path.exists(rutas_r.get("precios_cajas", "")):
        # Organizadas en 3 filas y 2 columnas exactas
        c1, c2 = st.columns(2)
        with c1:
            st.image(rutas_r["precios_cajas"], width="stretch")
            st.image(rutas_r["capacidad"], width="stretch")
            st.image(rutas_r["ocupacion"], width="stretch")
            
        with c2:
            st.image(rutas_r["precios_barras"], width="stretch")
            st.image(rutas_r["renta"], width="stretch")
            st.image(rutas_r["nlp"], width="stretch")
    else:
        st.info("💡 Ejecuta el script `analisis_completo.R` para generar las 6 gráficas estáticas.")

    st.divider()
    
    # --- PARTE 2: COMPARATIVA PANDAS VS DASK ---
    st.subheader("🚀 Duelo de Rendimiento: Pandas vs Dask (Big Data)")
    st.markdown("Comparativa procesando **más de 3 millones de registros** de disponibilidad usando procesamiento secuencial frente a distribuido.")

    # Ruta corregida: Eliminamos st.secrets y apuntamos a "data" local
    ruta_cal_zip = os.path.join("data", 'calendar.csv.zip')

    if os.path.exists(ruta_cal_zip):
        if st.button('🔥 Iniciar Comparativa de Velocidad', type="primary"):
            col_pan, col_das = st.columns(2)
            
            with col_pan:
                st.write("🐢 **Pandas (Secuencial)**")
                with st.spinner('Pandas está sufriendo...'):
                    tiempo_pandas = ut.benchmark_pandas(ruta_cal_zip)
                st.success(f"Tiempo Pandas: {tiempo_pandas:.2f} segundos")
                st.caption("Carga todo el archivo en un solo núcleo de la CPU.")
            
            with col_das:
                st.write("🚀 **Dask (Multiprocesamiento)**")
                with st.spinner('Dask está repartiendo el trabajo...'):
                    import time
                    start_d = time.time()
                    _ = ut.demostrar_procesamiento_big_data(ruta_cal_zip)
                    tiempo_dask = time.time() - start_d
                st.error(f"Tiempo Dask: {tiempo_dask:.2f} segundos")
                st.caption("Divide el archivo y usa todos los núcleos en paralelo.")

            # Conclusión matemática
            mejora = (tiempo_pandas / tiempo_dask) if tiempo_dask > 0 else 1
            st.info(f"💡 Dask ha sido **{mejora:.1f} veces más rápido** al paralelizar las tareas.")
            
            # Gráfico de barras visual
            fig_comp, ax_comp = plt.subplots(figsize=(8, 3))
            sns.barplot(x=['Pandas (Secuencial)', 'Dask (Paralelo)'], 
                        y=[tiempo_pandas, tiempo_dask], 
                        palette=['green', 'red'], ax=ax_comp)
            ax_comp.set_ylabel("Segundos (menos es mejor)")
            st.pyplot(fig_comp)
    else:
        st.warning("No se encontró el archivo de calendario 'calendar.csv.zip' en la carpeta data.")