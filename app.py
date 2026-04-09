import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.set_page_config(page_title="Tasador Airbnb Sevilla", page_icon="🏠")

# Buscador inteligente de archivos para Streamlit
if os.path.exists('data/knn_model.pkl'):
    PATH_DATA = 'data'
elif os.path.exists('../data/knn_model.pkl'):
    PATH_DATA = '../data'
else:
    PATH_DATA = '.'

try:
    knn_model = joblib.load(os.path.join(PATH_DATA, 'knn_model.pkl'))
    scaler = joblib.load(os.path.join(PATH_DATA, 'scaler.pkl'))
except FileNotFoundError:
    st.error("❌ Faltan los archivos del modelo. Ejecuta primero tu notebook de KNN.")
    st.stop()

st.title("🏠 Tasador Inteligente Airbnb - Sevilla")
st.markdown("Esta herramienta utiliza Inteligencia Artificial (**K-Nearest Neighbors**) para analizar el mercado en tiempo real y sugerir el precio óptimo por noche.")

# Interfaz gráfica
col1, col2 = st.columns(2)

with col1:
    st.subheader("Características del Inmueble")
    accommodates = st.number_input("Capacidad (Personas)", 1, 16, 2)
    bedrooms = st.number_input("Habitaciones", 1, 10, 1)
    bathrooms = st.number_input("Baños", 0.5, 10.0, 1.0, step=0.5)

with col2:
    st.subheader("Contexto de la Zona")
    renta_media = st.number_input("Renta media del barrio (€)", 5000, 40000, 15000)
    score_sentimiento = st.slider("Calidad de reseñas esperada (0-1)", 0.0, 1.0, 0.8)
    lat = st.number_input("Latitud (Ej: 37.388)", value=37.388, format="%.5f")
    lon = st.number_input("Longitud (Ej: -5.999)", value=-5.999, format="%.5f")

# Botón mágico
if st.button("💰 Calcular Precio Óptimo", type="primary"):
    # Orden estricto: lat, lon, acc, bed, bath, renta, score
    input_data = np.array([[lat, lon, accommodates, bedrooms, bathrooms, renta_media, score_sentimiento]])
    
    input_scaled = scaler.transform(input_data)
    prediccion = knn_model.predict(input_scaled)[0]
    margen = prediccion * 0.15 # Asumimos 15% de elasticidad de precio
    
    st.success(f"### 💶 Precio Sugerido: {prediccion:.2f} € / noche")
    st.info(f"**Rango recomendado:** {prediccion-margen:.2f} € - {prediccion+margen:.2f} €")