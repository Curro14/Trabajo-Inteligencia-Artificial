import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import utils as ut
import plotly.express as px

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
except FileNotFoundError:
    st.error(f"Error: No se encuentran los archivos del modelo en {PATH_DATA}")
    st.stop()

# -----------------------------------------------------------------------------
# SECCION 3: ENCABEZADO Y ESTRUCTURA DE PESTAÑAS
# -----------------------------------------------------------------------------
st.title("Tasador de Mercado Inteligente - Sevilla")
st.markdown("Algoritmo de K-Nearest Neighbors entrenado con datos reales de Airbnb.")

tab1, tab2, tab3 = st.tabs(["💰 Tasador de Precios", "📍 Explorador de Mercado", "📊 Análisis de Festividades"])

# =============================================================================
# PESTAÑA 1: CALCULADORA DE PRECIOS (MODELO KNN)
# =============================================================================
with tab1:
    col_izq, col_der = st.columns([1, 1], gap="large")

    # --- Columna Izquierda: Datos Básicos ---
    with col_izq:
        st.subheader("Ubicacion y Capacidad")
        
        lista_barrios = sorted(list(dict_barrios.keys()))
        barrio_seleccionado = st.selectbox("Barrio de Sevilla", lista_barrios)
        
        tipos_alojamiento = ["Piso/Casa completa", "Habitacion privada", "Habitacion compartida"]
        tipo_alojamiento = st.selectbox("Tipo de alquiler", tipos_alojamiento)
        map_tipo = {"Piso/Casa completa": 3, "Habitacion privada": 2, "Habitacion compartida": 1}
        
        c1, c2, c3 = st.columns(3)
        accommodates = c1.number_input("Huespedes", 1, 16, 2)
        bedrooms = c2.number_input("Dormitorios", 1, 10, 1)
        bathrooms = c3.number_input("Baños", 0.5, 10.0, 1.0, step=0.5)

        st.subheader("Reputacion")
        
        c4, c5 = st.columns(2)
        is_superhost = c4.toggle("Eres Superhost")
        
        opciones_reviews = [
            "Ninguna (Nuevo)", "1 a 10 resenas", "11 a 50 resenas", 
            "51 a 150 resenas", "151 a 300 resenas", "Mas de 300 resenas"
        ]
        rango_reviews = c5.selectbox("Volumen historico de resenas", opciones_reviews)
        
        map_reviews = {
            "Ninguna (Nuevo)": 0, "1 a 10 resenas": 5, "11 a 50 resenas": 30,
            "51 a 150 resenas": 100, "151 a 300 resenas": 225, "Mas de 300 resenas": 400
        }
        
        score_sentimiento = st.slider("Nota media de resenas (0-1)", 0.0, 1.0, 0.85)

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
    if st.button("Calcular Precio Optimo", type="primary", use_container_width=True):
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


# =============================================================================
# PESTAÑA 2: MAPA DINÁMICO POR DISTRITO
# =============================================================================
with tab2:
    st.subheader("📍 Mapa Dinámico por Distrito y Profesionalización")

    # 1. Cargar el dataset específico
    ruta_dataset_final = os.path.join(PATH_DATA, 'dataset_ia_final.csv')
    
    if os.path.exists(ruta_dataset_final):
        # Cargamos el dataframe
        df_viviendas = pd.read_csv(ruta_dataset_final)
        
        # Limpieza rápida: Plotly necesita que el precio sea numérico y no tenga nulos para el tamaño (size)
        df_viviendas['price'] = df_viviendas['price'].fillna(df_viviendas['price'].mean())

        # Clasificamos usando la función de utils.py
        df_mapa = ut.clasificar_propietario(df_viviendas.copy())

        # 2. Selector de Distrito (Columna: neighbourhood_group_cleansed)
        distritos = sorted(df_mapa['neighbourhood_group_cleansed'].unique())
        distrito_sel = st.selectbox("Selecciona un Distrito para explorar:", distritos)

        # 3. Selector de Tipo de Propietario
        tipo_sel = st.multiselect("Filtrar por tipo de anfitrión:", 
                                ['Particular', 'Empresa'], 
                                default=['Particular', 'Empresa'])

        # 4. Filtrar datos según selección
        df_filtrado = df_mapa[
            (df_mapa['neighbourhood_group_cleansed'] == distrito_sel) & 
            (df_mapa['tipo_propietario'].isin(tipo_sel))
        ]

        # 5. Crear el Mapa con Plotly
        if not df_filtrado.empty:
            fig_map = px.scatter_mapbox(
                df_filtrado, 
                lat="latitude", 
                lon="longitude", 
                color="tipo_propietario", 
                size="price",             # El tamaño del punto depende del precio
                hover_name="neighbourhood_cleansed", 
                hover_data={
                    "price": ":.2f", 
                    "room_type": True, 
                    "tipo_propietario": True,
                    "calculated_host_listings_count": True,
                    "latitude": False,
                    "longitude": False
                },
                color_discrete_map={'Particular': '#2ecc71', 'Empresa': '#e74c3c'}, # Verde y Rojo
                zoom=13, 
                height=600
            )

            fig_map.update_layout(mapbox_style="carto-positron")
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

            st.plotly_chart(fig_map, use_container_width=True)
            
            # 6. Estadísticas detalladas del distrito
            st.markdown(f"### Análisis de {distrito_sel}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Alojamientos", len(df_filtrado))
            with col2:
                n_empresas = len(df_filtrado[df_filtrado['tipo_propietario'] == 'Empresa'])
                st.metric("Gestionados por Empresas", n_empresas)
            with col3:
                pct_empresa = (n_empresas / len(df_filtrado)) * 100 if len(df_filtrado) > 0 else 0
                st.metric("% Profesionalización", f"{pct_empresa:.1f}%")
        else:
            st.warning("No hay datos que coincidan con los filtros seleccionados.")
    else:
        st.error(f"No se encontró el archivo '{ruta_dataset_final}'. Asegúrate de que el CSV esté en la carpeta data.")

    # --- Aquí puedes mantener el código que ya tenías para los mapas de R si quieres mostrarlos debajo ---
# =============================================================================
# PESTAÑA 3: ANÁLISIS DE OCUPACIÓN Y EVENTOS
# =============================================================================
with tab3:
    st.header("Análisis de Impacto: Eventos y Festividades")
    st.markdown("""
    En esta sección analizamos cómo afectan los eventos especiales (Semana Santa, Feria, Navidad, ...) 
    a la ocupación y disponibilidad de los alojamientos en Sevilla.
    """)

    # --- Carga de datos específica para esta pestaña ---
    @st.cache_data # Usamos caché para que no tarde cada vez que cambies de pestaña
    def cargar_datos_analisis():
        ruta_cal = os.path.join(PATH_DATA, 'calendar_limpio.csv.zip')
        if os.path.exists(ruta_cal):
            df = pd.read_csv(ruta_cal, compression='zip', parse_dates=['date'])
            # Aplicamos la lógica de utils.py
            df = ut.preparar_fechas_y_eventos(df)
            # Aseguramos que la columna ocupado existe (1 si 'f', 0 si 't')
            if 'ocupado' not in df.columns:
                df['ocupado'] = df['available'].apply(lambda x: 0 if x == True or x == 't' else 1)
            return df
        return None

    df_analisis = cargar_datos_analisis()

    if df_analisis is not None:
        # --- BLOQUE 1: Estadísticas Dinámicas ---
        st.subheader("Métricas por Festividad")
        
        # Calculamos la media global para comparar
        promedio_total = df_analisis['ocupado'].mean() * 100
        
        # Obtenemos la lista de eventos únicos (quitando 'Normal')
        eventos_especiales = [e for e in df_analisis['evento'].unique() if e != 'Normal']
        
        # 1. Primera fila: Metrica Global siempre visible
        st.metric("Ocupación Media Anual (Sevilla)", f"{promedio_total:.1f}%")
        st.write("---")

        # 2. Filas Dinámicas para todos los eventos (Feria, SS, Puentes, Navidad...)
        # Creamos columnas de 3 en 3
        cols = st.columns(3)
        for i, evento in enumerate(eventos_especiales):
            with cols[i % 3]:
                # Calculamos ocupación de ese evento concreto
                ocu_evento = df_analisis[df_analisis['evento'] == evento]['ocupado'].mean() * 100
                delta_val = ocu_evento - promedio_total
                
                st.metric(
                    label=f"Ocupación {evento}", 
                    value=f"{ocu_evento:.1f}%", 
                    delta=f"{delta_val:.1f}% vs Media"
                )

        st.divider()

        # --- BLOQUE 2: Gráfico de Barras ---
        st.subheader("Ocupación por Periodo y Festividad")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        resumen_ocu = (df_analisis.groupby('periodo')['ocupado'].mean() * 100).sort_values(ascending=False)
        grafico = sns.barplot(x=resumen_ocu.index, y=resumen_ocu.values, palette="mako", ax=ax_bar)
        for p in grafico.patches:
            grafico.annotate(f"{p.get_height():.1f}%", 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha = 'center', va = 'center', 
                             xytext = (0, 9), 
                             textcoords = 'offset points',
                             fontweight='bold')
        plt.xticks(rotation=45)
        plt.ylabel("% Ocupación")
        st.pyplot(fig_bar)

        st.divider()

        # --- BLOQUE 3: Heatmap ---
        st.subheader("Mapa de Calor: Disponibilidad por Mes y Día")
        # Aquí llamamos directamente a la lógica del heatmap
        temp_df = df_analisis.copy()
        temp_df['mes_nombre'] = temp_df['date'].dt.month_name()
        temp_df['dia_nombre'] = temp_df['date'].dt.day_name()
        
        pivot = temp_df.pivot_table(index='mes_nombre', columns='dia_nombre', values='ocupado', aggfunc='mean') * 100
        meses = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex(index=meses, columns=dias)

        fig_heat, ax_heat = plt.subplots(figsize=(12, 7))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap='YlOrRd', ax=ax_heat)
        st.pyplot(fig_heat)
        
        st.divider()
        # --- BLOQUE 4: Dask vs Pandas ---
        st.subheader("🚀 Duelo de Rendimiento: Pandas vs Dask (Big Data)")
        st.write("""
        Para demostrar la escalabilidad de nuestra solución, comparamos el procesamiento de 
        **3 millones de registros** usando un método tradicional (Pandas) frente a computación paralela (Dask).
        """)

        ruta_cal_zip = os.path.join(PATH_DATA, 'calendar.csv.zip')
        
        if os.path.exists(ruta_cal_zip):
            if st.button('🔥 Iniciar Comparativa de Velocidad'):
                col_pan, col_das = st.columns(2)
                
                # --- TEST PANDAS ---
                with col_pan:
                    st.write("🐢 **Procesando con Pandas...**")
                    with st.spinner('Pandas está sufriendo...'):
                        tiempo_pandas = ut.benchmark_pandas(ruta_cal_zip)
                    st.success(f"Tiempo Pandas: {tiempo_pandas:.2f} segundos")
                    st.caption("Carga todo el archivo en un solo núcleo de la CPU.")

                # --- TEST DASK ---
                with col_das:
                    st.write("🚀 **Procesando con Dask...**")
                    with st.spinner('Dask está repartiendo el trabajo...'):
                        # Medimos el tiempo que tarda tu función de utils
                        import time
                        start_d = time.time()
                        _ = ut.demostrar_procesamiento_big_data(ruta_cal_zip)
                        tiempo_dask = time.time() - start_d
                    st.error(f"Tiempo Dask: {tiempo_dask:.2f} segundos")
                    st.caption("Divide el archivo y usa todos los núcleos en paralelo.")

                # --- CONCLUSIÓN ---
                mejora = (tiempo_pandas / tiempo_dask)
                st.info(f"💡 **Conclusión:** Pandas ha sido **{mejora:.1f} veces más rápido** que Dask al paralelizar la tarea.")
                
                # Gráfico comparativo simple
                fig_comp, ax_comp = plt.subplots(figsize=(8, 3))
                sns.barplot(x=['Pandas (Secuencial)', 'Dask (Paralelo)'], 
                            y=[tiempo_pandas, tiempo_dask], 
                            palette=['green', 'red'], ax=ax_comp)
                ax_comp.set_ylabel("Segundos (menos es mejor)")
                st.pyplot(fig_comp)
        else:
            st.warning("No se encontró el archivo 'calendar.csv.zip' para la demostración de Big Data.")