# ==========================================
# CONFIGURACIÓN GLOBAL DEL PROYECTO
# ==========================================

def librerias():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import re
    import sys
    import subprocess
    import importlib
    import warnings
    import geopandas as gpd
    from transformers import pipeline
    import dask.dataframe as dd
    import time

    from tqdm.auto import tqdm

    # Preprocesamiento (Scikit-Learn)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    # Deep Learning (TensorFlow & Keras)
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    
    
    return pd, np, plt, sns, os, re, sys, subprocess, importlib, warnings, gpd, pipeline, dd, time, tqdm, train_test_split, StandardScaler, OneHotEncoder, ColumnTransformer, Sequential, Dense, Dropout, EarlyStopping

def pipeline_completo_preparacion():
    
    # 1. FUNCIÓN INTERNA: INSTALL_IF_MISSING (Lo que tenías al inicio)
    def install_if_missing(package):
        try:
            __import__(package)
        except ImportError:
            print(f"📦 Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # Aseguramos que las librerías críticas estén presentes
    for lib in ['pandas', 'numpy', 'matplotlib', 'seaborn']:
        install_if_missing(lib)
    
  

    # 2. CONFIGURACIÓN DE DIRECTORIOS (Contexto del proyecto)
    # Buscamos la carpeta raíz del proyecto para que las rutas siempre funcionen
    directorio_actual = os.getcwd()
    print(f"🏠 Directorio de trabajo: {directorio_actual}")

    # 3. RUTAS DE ARCHIVOS (Ajustadas a tu estructura)
    # Nota: He usado rutas relativas para que a Cristo también le funcione
    PATH_LISTINGS = 'data/listings_limpio.csv'
    PATH_CALENDAR = 'data/calendar_limpio.csv.zip'
    PATH_RENTA = 'data/renta_sevilla_capital_limpio.csv'

    print("🚀 Iniciando limpieza de datasets...")

    # --- Lógica de Limpieza (El corazón de tu cuaderno) ---
    
    # Carga y limpieza de Listings
    df_listings = pd.read_csv(PATH_LISTINGS)
    if 'price' in df_listings.columns and df_listings['price'].dtype == 'object':
        df_listings['price'] = df_listings['price'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Carga y limpieza de Calendar
    df_calendar = pd.read_csv(PATH_CALENDAR, compression='zip', parse_dates=['date'])
    df_calendar['available'] = df_calendar['available'].map({'t': True, 'f': False})
    # Creamos la columna 'ocupado' que necesitábamos para el análisis
    df_calendar['ocupado'] = df_calendar['available'].apply(lambda x: 0 if x else 1)

    # Carga y limpieza de Renta con el mapeo de distritos
    df_renta = pd.read_csv(PATH_RENTA)
    
    def mapear_distrito(texto):
        t = str(texto).lower()
        if 'distrito 01' in t: return 'Casco Antiguo'
        if 'distrito 02' in t: return 'Macarena'
        if 'distrito 03' in t: return 'Nervión'
        if 'distrito 04' in t: return 'San Pablo - Santa Justa'
        if 'distrito 05' in t: return 'Sur'
        if 'distrito 06' in t: return 'Palmera - Bellavista'
        if 'distrito 07' in t: return 'Los Remedios'
        if 'distrito 08' in t: return 'Triana'
        if 'distrito 09' in t: return 'Este - Alcosa - Torreblanca'
        if 'distrito 10' in t: return 'Cerro - Amate'
        if 'distrito 11' in t: return 'Norte'
        return 'Otros'

    df_renta['distrito_limpio'] = df_renta['Distritos'].apply(mapear_distrito)
    if df_renta['Total'].dtype == 'object':
        df_renta['Total'] = df_renta['Total'].str.replace('.', '').astype(float)

    print("✅ Todo listo. Los 3 datasets han sido procesados.")
    return df_listings, df_calendar, df_renta
