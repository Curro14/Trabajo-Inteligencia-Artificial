# ==========================================
# CONFIGURACIÓN GLOBAL DEL PROYECTO
# ==========================================
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
import time
import joblib
import shutil  # <--- Soluciona el error de "shutil is not defined"
from tqdm.auto import tqdm

# RUTAS GLOBALES (Soluciona el error de "PATH_DATA is not defined")
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PATH_DATA = os.path.join(BASE_DIR, 'data')

# Ocultar warnings molestos de TensorFlow en la terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Machine Learning & Deep Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Importaciones condicionales para Big Data
try:
    import geopandas as gpd
except ImportError:
    pass
try:
    import dask.dataframe as dd
except ImportError:
    pass

# Machine Learning & Deep Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def pipeline_completo_preparacion():
    """
    Pipeline integral final:
    Hardware + Librerías + Limpieza de 4 datasets + Join Espacial + Exportación.
    """
    
    # --- 1. GESTIÓN DE LIBRERÍAS (Añadido geopandas) ---
    def install_if_missing(package):
        try:
            __import__(package)
        except ImportError:
            print(f"📦 Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # Importante añadir geopandas aquí
    for lib in ['pandas', 'numpy', 'torch', 'tqdm', 'geopandas']:
        install_if_missing(lib)
    

    # --- 2. CONFIGURACIÓN DE ENTORNO ---
    warnings.filterwarnings('ignore')
    tqdm.pandas()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    PATH_DATA = os.path.join(BASE_DIR, 'data')

    if not os.path.exists(PATH_DATA):
        print(f"❌ Error: No se encuentra la carpeta '{PATH_DATA}'")
        return None

    # --- 3. PROCESAMIENTO DE LISTINGS ---
    final_cols = [
        'id', 'host_id', 'host_is_superhost', 'neighbourhood_cleansed', 
        'neighbourhood_group_cleansed', 'latitude', 'longitude', 
        'property_type', 'room_type', 'accommodates', 'bathrooms_text', 
        'bedrooms', 'price', 'minimum_nights', 'number_of_reviews', 
        'review_scores_rating', 'license', 'instant_bookable',
        'availability_365', 'calculated_host_listings_count', 
        'reviews_per_month', 'amenities'
    ]
    ruta_listings = os.path.join(PATH_DATA, 'listings.csv')
    df_listings = pd.read_csv(ruta_listings, usecols=final_cols, low_memory=False) if os.path.exists(ruta_listings) else pd.DataFrame()
    if not df_listings.empty:
        df_listings['price'] = df_listings['price'].replace(r'[\$,]', '', regex=True).astype(float)

    # --- 4. PROCESAMIENTO DE CALENDAR ---
    ruta_calendar = os.path.join(PATH_DATA, 'calendar.csv.zip')
    if os.path.exists(ruta_calendar):
        df_calendar = pd.read_csv(ruta_calendar, compression='zip', low_memory=False)
        df_calendar['date'] = pd.to_datetime(df_calendar['date'])
        df_calendar['available'] = df_calendar['available'].map({'t': True, 'f': False})
        df_calendar.to_csv(os.path.join(PATH_DATA, 'calendar_limpio.csv.zip'), index=False, compression='zip')

    # --- 5. PROCESAMIENTO DE RENTA (INE) ---
    ruta_renta_raw = os.path.join(PATH_DATA, 'renta_provincia_sevilla.csv')
    df_renta = pd.DataFrame()
    if os.path.exists(ruta_renta_raw):
        df_renta_raw = pd.read_csv(ruta_renta_raw, sep=';', encoding='latin-1', decimal=',')
        df_renta = df_renta_raw[df_renta_raw['Secciones'].astype(str).str.contains('^41091')].copy()
        if 'Total' in df_renta.columns and df_renta['Total'].dtype == 'object':
            df_renta['Total'] = df_renta['Total'].str.replace('.', '', regex=False).astype(float)

    # --- 6. CARGA DEL MAPA ---
    ruta_mapa = os.path.join(PATH_DATA, 'secciones_sevilla.json') 
    mapa_sevilla = gpd.read_file(ruta_mapa) if os.path.exists(ruta_mapa) else None

    # --- 7. EXPORTACIÓN FINAL CON JOIN ESPACIAL ---
    if mapa_sevilla is not None and not df_listings.empty:
        print("🌍 Iniciando Join Espacial...")
        gdf_listings = gpd.GeoDataFrame(
            df_listings, 
            geometry=gpd.points_from_xy(df_listings.longitude, df_listings.latitude), 
            crs="EPSG:4326"
        ).to_crs(mapa_sevilla.crs)

        pisos_con_seccion = gpd.sjoin(gdf_listings, mapa_sevilla[['CUSEC', 'geometry']], how="left", predicate="intersects")

        if not df_renta.empty:
            df_renta['Secciones'] = df_renta['Secciones'].astype(str)
            pisos_con_seccion['CUSEC'] = pisos_con_seccion['CUSEC'].astype(str)
            col_renta = 'Total' if 'Total' in df_renta.columns else 'renta_media'
            
            dataset_final = pd.merge(pisos_con_seccion, df_renta[['Secciones', col_renta]], 
                                    left_on='CUSEC', right_on='Secciones', how='left')

            dataset_final.to_csv(os.path.join(PATH_DATA, 'dataset_final.csv'), index=False)
            print("🏁 ¡HECHO! Dataset final guardado.")
            return dataset_final # Devuelve el dataframe listo
    
    print("⚠️ El proceso terminó pero no se pudo realizar el Join Espacial.")
    return df_listings, df_calendar, df_renta, mapa_sevilla



import pandas as pd
import numpy as np
import re

def clasificar_propietario(df):
    """
    Clasifica al anfitrión como Particular (<= 5 pisos) o Empresa (> 5 pisos).
    """
    df = df.copy()
    if 'calculated_host_listings_count' in df.columns:
        df['tipo_propietario'] = df['calculated_host_listings_count'].apply(
            lambda x: 'Particular' if x <= 5 else 'Empresa'
        )
    return df

def preparar_datos_prediccion(df):
    """
    Clon exacto de la lógica de Ingeniería de Características del Notebook 05 (KNN).
    Aplica las transformaciones para que coincidan con los datos de entrenamiento.
    """
    df = df.copy()

    # 1. Baños (Extracción de decimales/enteros)
    def extraer_banos(texto):
        if pd.isna(texto): return 1.0
        numeros = re.findall(r"[-+]?\d*\.\d+|\d+", str(texto))
        return float(numeros[0]) if numeros else 1.0

    if 'bathrooms_text' in df.columns:
        df['bathrooms_num'] = df['bathrooms_text'].apply(extraer_banos)
    elif 'bathrooms_num' not in df.columns:
        df['bathrooms_num'] = 1.0

    # 2. Habitaciones y Capacidad (Relleno de nulos)
    if 'bedrooms' in df.columns:
        df['bedrooms'] = df['bedrooms'].fillna(1.0)
    if 'accommodates' in df.columns:
        df['accommodates'] = df['accommodates'].fillna(2.0)

    # 3. Tipo de alojamiento
    if 'room_type' in df.columns:
        dict_room = {'Entire home/apt': 3, 'Private room': 2, 'Shared room': 1, 'Hotel room': 2}
        df['room_type_num'] = df['room_type'].map(dict_room).fillna(3)

    # 4. Confianza y Reputación (Superhost y Reseñas)
    if 'host_is_superhost' in df.columns:
        df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0}).fillna(0)
    
    if 'number_of_reviews' in df.columns:
        df['number_of_reviews'] = df['number_of_reviews'].fillna(0)

    # 5. Extracción de Amenities Premium (Binarias 1/0)
    if 'amenities' in df.columns:
        amenities_lower = df['amenities'].str.lower().fillna('')
        df['has_pool'] = amenities_lower.str.contains('pool').astype(int)
        df['has_ac'] = amenities_lower.str.contains('air conditioning').astype(int)
        df['has_parking'] = amenities_lower.str.contains('parking').astype(int)
        df['has_elevator'] = amenities_lower.str.contains('elevator').astype(int)
        df['has_balcony'] = amenities_lower.str.contains('balcony|patio').astype(int)
        df['has_workspace'] = amenities_lower.str.contains('workspace').astype(int)

    # 6. Renta Media Oficial del INE por Distrito
    if 'neighbourhood_group_cleansed' in df.columns:
        rentas_sevilla = {
            'Los Remedios': 30000.0, 'Nervión': 28000.0, 'Casco Antiguo': 26000.0,
            'Sur': 25000.0, 'Triana': 24000.0, 'San Pablo - Santa Justa': 22000.0,
            'Macarena': 20000.0, 'Bellavista - La Palmera': 21000.0,
            'Este - Alcosa - Torreblanca': 19000.0, 'Norte': 17000.0, 'Cerro - Amate': 16000.0
        }
        df['renta_media'] = df['neighbourhood_group_cleansed'].map(rentas_sevilla).fillna(22000.0)

    # 7. Score Sentimiento (IA de Reseñas)
    if 'score_sentimiento' in df.columns:
        df['score_sentimiento'] = df['score_sentimiento'].fillna(0.5)
    else:
        df['score_sentimiento'] = 0.5

    # 8. Limpieza de Precios (Solo aplica si el df tiene la columna 'price')
    if 'price' in df.columns and df['price'].dtype == 'object':
        df['price'] = df['price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

    return df

# =============================================================================
# MÓDULO VISUAL 1: GRÁFICOS DINÁMICOS PARA USUARIO FINAL (PLOTLY - PESTAÑA 2)
# =============================================================================
import plotly.express as px

def generar_mapa_interactivo(df):
    fig = px.scatter_mapbox(
        df, lat="latitude", lon="longitude", color="price", 
        size="accommodates", hover_name="neighbourhood_cleansed", 
        hover_data={"price": ":.2f", "bedrooms": True, "accommodates": True, "latitude": False, "longitude": False},
        color_continuous_scale=px.colors.sequential.Agsunset,
        zoom=12, height=500, title="📍 Mapa de alojamientos disponibles"
    )
    fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":40,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)')
    return fig

def grafico_precio_medio_barrio(df):
    df_agrupado = df.groupby('neighbourhood_group_cleansed')['price'].mean().reset_index().sort_values('price')
    fig = px.bar(df_agrupado, x="price", y="neighbourhood_group_cleansed", orientation='h', title="1. ¿Cuánto cuesta la noche de media en cada distrito?", labels={"neighbourhood_group_cleansed": "", "price": "Precio Medio (€)"}, color="price", color_continuous_scale="Viridis")
    fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def grafico_distribucion_precios(df):
    fig = px.histogram(df, x="price", nbins=30, title="2. ¿En qué rango de precios hay más opciones?", labels={"price": "Precio por Noche (€)", "count": "Número de Alojamientos"}, color_discrete_sequence=['#3498db'])
    fig.update_layout(yaxis_title="Cantidad de Alojamientos", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def grafico_oferta_por_barrio(df):
    conteo = df['neighbourhood_group_cleansed'].value_counts().reset_index()
    conteo.columns = ['Distrito', 'Alojamientos']
    fig = px.pie(conteo, values='Alojamientos', names='Distrito', hole=0.4, title="3. ¿Dónde hay más cantidad de alojamientos?")
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def grafico_capacidad_vs_precio(df):
    df_agrupado = df.groupby('accommodates')['price'].mean().reset_index()
    fig = px.line(df_agrupado, x="accommodates", y="price", markers=True, title="4. ¿Cómo sube el precio según los huéspedes?", labels={"accommodates": "Número de Huéspedes", "price": "Precio Medio (€)"}, color_discrete_sequence=['#e74c3c'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def grafico_tipo_habitaciones(df):
    df['tipo_tamano'] = df['bedrooms'].apply(lambda x: f"{int(x)} Hab." if x <= 3 else "4+ Habs.")
    conteo = df['tipo_tamano'].value_counts().reset_index()
    conteo.columns = ['Tamaño', 'Cantidad']
    fig = px.pie(conteo, values='Cantidad', names='Tamaño', title="5. ¿De cuántas habitaciones son los pisos?", color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def grafico_top_barrios_baratos(df):
    df_agrupado = df.groupby('neighbourhood_cleansed')['price'].mean().reset_index().sort_values('price').head(10)
    fig = px.bar(df_agrupado, x="price", y="neighbourhood_cleansed", orientation='h', title="6. Top 10 Barrios más económicos", labels={"neighbourhood_cleansed": "Barrio", "price": "Precio (€)"}, color_discrete_sequence=['#2ecc71'])
    fig.update_layout(yaxis={'categoryorder':'total descending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig
# =============================================================================
# MÓDULO VISUAL 2: LECTURA DE GRÁFICOS ESTÁTICOS DE R (PESTAÑA 3)
# =============================================================================
# =============================================================================
# MÓDULO VISUAL 2: LECTURA DE GRÁFICOS ESTÁTICOS DE R (PESTAÑA 3)
# =============================================================================
def obtener_rutas_graficos_r():
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    PATH_DATA = os.path.join(BASE_DIR, 'data')
    
    return {
        "precios_cajas": os.path.join(PATH_DATA, "r_1_cajas_precios.png"),
        "precios_barras": os.path.join(PATH_DATA, "r_2_barras_precios.png"),
        "capacidad": os.path.join(PATH_DATA, "r_3_capacidad_precio.png"),
        "renta": os.path.join(PATH_DATA, "r_4_renta_vs_precio.png"),
        "ocupacion": os.path.join(PATH_DATA, "r_5_ocupacion_tipo.png"),
        "nlp": os.path.join(PATH_DATA, "r_6_nlp_vs_estrellas.png")
    }

# =============================================================================
# MÓDULO BIG DATA: PANDAS VS DASK (PESTAÑA 3)
# =============================================================================
def benchmark_pandas(path_zip):
    import pandas as pd
    import time
    inicio = time.time()
    # Carga completa en RAM
    df = pd.read_csv(path_zip, compression='zip')
    # Transformación
    df['ocupado'] = df['available'].map({'f': 1, 't': 0, False: 1, True: 0})
    # Agregación
    resumen = df.groupby('date')['ocupado'].mean()
    fin = time.time()
    return fin - inicio

def demostrar_procesamiento_big_data(path_calendar):
    import dask.dataframe as dd
    import time
    # 1. Lectura Perezosa (Lazy Loading)
    ddf_calendar = dd.read_csv(path_calendar, compression='zip', blocksize=None)
    
    # 2. Transformación de datos distribuida
    ddf_calendar['ocupado'] = ddf_calendar['available'].map(
        {'f': 1, 't': 0, False: 1, True: 0}, 
        meta=('available', 'int8')
    )
    
    # 3. Operación de Agrupación (Plan de ejecución) y Cómputo Real
    ocupacion_por_dia_pandas = ddf_calendar.groupby('date')['ocupado'].mean().compute()
    
    return ocupacion_por_dia_pandas