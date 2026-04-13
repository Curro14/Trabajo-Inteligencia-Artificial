import streamlit as st
import numpy as np
import joblib
import os
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
except FileNotFoundError:
    st.error(f"Error: No se encuentran los archivos del modelo en {PATH_DATA}")
    st.stop()

# -----------------------------------------------------------------------------
# SECCION 3: ENCABEZADO Y ESTRUCTURA DE PESTAÑAS
# -----------------------------------------------------------------------------
st.title("Tasador de Mercado Inteligente - Sevilla")
st.markdown("Algoritmo de K-Nearest Neighbors entrenado con datos reales de Airbnb.")

tab1, tab2 = st.tabs(["Calculadora de Precios (IA)", "Explorador del Mercado (Mapa)"])

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
        bathrooms = c3.number_input("Banos", 0.5, 10.0, 1.0, step=0.5)

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
# PESTAÑA 2: EXPLORADOR DEL MERCADO (MAPA R HTML)
# =============================================================================
with tab2:
    st.subheader("Filtra y explora los datos geolocalizados")
    st.markdown("Este mapa interactivo ha sido generado con R y Crosstalk.")
    
    ruta_mapa = os.path.join(PATH_DATA, "mapa_interactivo_filtros.html")
    
    if os.path.exists(ruta_mapa):
        with open(ruta_mapa, 'r', encoding='utf-8') as f:
            html_mapa = f.read()
            
        components.html(html_mapa, height=800, scrolling=True)
    else:
        st.warning("No se encontro el mapa. Ejecuta tu script de R primero para generarlo en la carpeta 'data'.")