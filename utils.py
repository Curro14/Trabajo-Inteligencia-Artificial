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
import geopandas as gpd
import dask.dataframe as dd
import time
import joblib
from tqdm.auto import tqdm

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


def preparar_fechas_y_eventos(df):
    """
    Realiza la ingeniería de variables temporales y aplica el etiquetado 
    avanzado de eventos (Navidad, Puentes, Feria y Semana Santa 2024-2026).
    """
    
    # 1. Variables básicas de tiempo
    df['mes'] = df['date'].dt.month
    df['dia_semana'] = df['date'].dt.day_name()
    df['es_finde'] = df['date'].dt.dayofweek.isin([4, 5, 6])
    
    # 2. Inicializar columna de evento
    df['evento'] = 'Normal'

    # 3. Aplicar Máscaras de Eventos (Lógica Avanzada)

    # NAVIDAD (Fijo: 22 Dic al 6 Ene)
    mask_navidad = ((df['date'].dt.month == 12) & (df['date'].dt.day >= 22)) | \
                   ((df['date'].dt.month == 1) & (df['date'].dt.day <= 6))
    df.loc[mask_navidad, 'evento'] = 'Navidad'

    # PUENTES NACIONALES (Fechas fijas aproximadas)
    df.loc[(df['date'].dt.month == 12) & (df['date'].dt.day.between(5, 10)), 'evento'] = 'Puente Diciembre'
    df.loc[(df['date'].dt.month == 10) & (df['date'].dt.day.between(11, 15)), 'evento'] = 'Puente Hispanidad'
    df.loc[(df['date'].dt.month == 5) & (df['date'].dt.day.between(1, 3)), 'evento'] = 'Puente de Mayo'

    # FERIA DE ABRIL (Fechas 2025 y 2026)
    mask_feria = ((df['date'] >= '2025-05-05') & (df['date'] <= '2025-05-11')) | \
                 ((df['date'] >= '2026-04-20') & (df['date'] <= '2026-04-26'))
    df.loc[mask_feria, 'evento'] = 'Feria de Abril'

    # SEMANA SANTA (Fechas 2025 y 2026)
    mask_ss = ((df['date'] >= '2025-04-13') & (df['date'] <= '2025-04-20')) | \
              ((df['date'] >= '2026-03-29') & (df['date'] <= '2026-04-05'))
    df.loc[mask_ss, 'evento'] = 'Semana Santa'

    # 4. Creación de la columna 'periodo' (Consolidando Evento vs Finde/Semana)
    df['periodo'] = np.where(
        df['evento'] != 'Normal', 
        df['evento'], 
        np.where(df['es_finde'], 'Finde Normal', 'Semana Normal')
    )
    
    return df


def integrar_informacion_barrios(df_calendar, df_listings, df_renta):
    """
    Cruza el calendario con listings para obtener barrios y con el INE para rentas.
    """
    # Unimos con listings para traer barrios y tipo de habitación
    df_merged = pd.merge(df_calendar, 
                         df_listings[['id', 'neighbourhood_group_cleansed', 'room_type']], 
                         left_on='listing_id', right_on='id', how='left')
    
    # Unimos con los datos de renta del INE
    df_final = pd.merge(df_merged, 
                        df_renta[['distrito_limpio', 'Total']], 
                        left_on='neighbourhood_group_cleansed', 
                        right_on='distrito_limpio', how='left')
    
    return df_final

def calcular_metricas_rentabilidad(df):
    """
    Calcula la tasa de ocupación y estima la ganancia por anuncio.
    """
    # 1 representa ocupado, 0 disponible
    df['ocupado'] = df['available'].apply(lambda x: 0 if x else 1)
    
    # Resumen por evento
    resumen_evento = df.groupby('evento').agg({
        'price': 'mean',
        'ocupado': 'mean'
    }).reset_index()
    
    resumen_evento['ocupacion_pct'] = resumen_evento['ocupado'] * 100
    
    return resumen_evento

def visualizar_ocupacion_festividades(df):
    """
    Calcula y grafica la tasa de ocupación media para cada periodo 
    y festividad definida en el dataset.
    """

    # 1. Cálculo de la ocupación media por periodo
    # Multiplicamos por 100 para obtener el porcentaje
    resumen_ocupacion = (df.groupby('periodo')['ocupado'].mean() * 100).sort_values(ascending=False)

    # 2. Configuración del estilo y tamaño
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # 3. Creación del gráfico de barras
    grafico = sns.barplot(x=resumen_ocupacion.index, y=resumen_ocupacion.values, palette="mako")

    # 4. Añadir etiquetas de porcentaje sobre las barras
    for p in grafico.patches:
        grafico.annotate(f"{p.get_height():.1f}%", 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'center', 
                         xytext = (0, 9), 
                         textcoords = 'offset points',
                         fontweight='bold')

    # 5. Personalización de títulos y ejes
    plt.title('Tasa de Ocupación de Airbnb en Sevilla según Festividades', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Porcentaje de Pisos Ocupados (%)', fontweight='bold')
    plt.xlabel('Periodo', fontweight='bold')

    # Ajustes visuales finales
    plt.ylim(0, 100) 
    plt.xticks(rotation=15) 
    plt.tight_layout()
    
    plt.show()


def visualizar_heatmap_ocupacion(df):
    """
    Genera un mapa de calor que muestra el porcentaje de ocupación 
    cruzando los meses del año con los días de la semana.
    """

    # 1. Creamos copias locales de los nombres para no alterar el dataframe original si no es necesario
    # Extraemos nombres de mes y día directamente de la columna 'date'
    temp_df = df.copy()
    temp_df['mes_nombre'] = temp_df['date'].dt.month_name()
    temp_df['dia_semana_nombre'] = temp_df['date'].dt.day_name()

    # 2. Creamos la tabla pivote (Media de ocupación * 100)
    pivot_ocupacion = temp_df.pivot_table(
        index='mes_nombre', 
        columns='dia_semana_nombre', 
        values='ocupado', 
        aggfunc='mean'
    ) * 100

    # 3. Definimos el orden lógico cronológico
    dias_ordenados = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    meses_ordenados = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']

    # Reindexamos para que el gráfico no salga en orden alfabético (A-Z)
    pivot_ocupacion = pivot_ocupacion.reindex(index=meses_ordenados, columns=dias_ordenados)

    # 4. Configuración del Gráfico
    plt.figure(figsize=(12, 8))
    
    # Usamos 'YlOrRd' (Amarillo-Naranja-Rojo) para que el rojo indique "lleno/difícil encontrar sitio"
    sns.heatmap(pivot_ocupacion, 
                annot=True, 
                cmap='YlOrRd', 
                fmt=".1f", 
                linewidths=.5, 
                cbar_kws={'label': '% Ocupación'})

    # 5. Títulos y etiquetas
    plt.title('Heatmap: Probabilidad de Ocupación en Sevilla (Mes vs Día)', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Mes', fontweight='bold')
    plt.xlabel('Día de la Semana', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def demostrar_procesamiento_big_data(path_calendar):
    """
    Utiliza Dask para procesar el dataset de calendario de forma distribuida,
    optimizando el uso de los núcleos del procesador.
    """

    inicio = time.time()

    # 1. Lectura Perezosa (Lazy Loading)
    # blocksize=None es clave para leer archivos comprimidos en Dask
    ddf_calendar = dd.read_csv(path_calendar, compression='zip', blocksize=None)

    # 2. Transformación de datos distribuida
    # Mapeamos la disponibilidad (f -> ocupado, t -> libre)
    ddf_calendar['ocupado'] = ddf_calendar['available'].map(
        {'f': 1, 't': 0, False: 1, True: 0}, 
        meta=('available', 'int8')
    )

    # 3. Operación de Agrupación (Plan de ejecución)
    ocupacion_por_dia_dask = ddf_calendar.groupby('date')['ocupado'].mean()

    print("⏳ Plan de ejecución creado. Dask está ejecutando en paralelo...")

    # 4. Ejecución real (.compute)
    # Aquí es donde se activan todos los núcleos del Mac
    ocupacion_por_dia_pandas = ocupacion_por_dia_dask.compute()

    fin = time.time()
    tiempo_total = fin - inicio

    print(f"✅ ¡Cálculo completado en {tiempo_total:.2f} segundos!")
    print("-" * 40)
    print("Top 5 Días con mayor ocupación (Cálculo Distribuido):")
    
    # Formateamos el índice y mostramos resultados
    ocupacion_por_dia_pandas.index = pd.to_datetime(ocupacion_por_dia_pandas.index)
    top_5 = (ocupacion_por_dia_pandas * 100).sort_values(ascending=False).head(5)
    print(top_5)
    
    return ocupacion_por_dia_pandas

def benchmark_pandas(path_zip):
    """
    Realiza el mismo proceso que la función de Dask pero usando Pandas puro
    para medir el tiempo de ejecución.
    """
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

def clasificar_propietario(df):
    """
    Clasifica como Particular (1-5 pisos) o Empresa (>5 pisos)
    """
    df['tipo_propietario'] = df['calculated_host_listings_count'].apply(
        lambda x: 'Particular' if x <= 5 else 'Empresa'
    )
    return df

def preparar_datos_prediccion(df):
    """
    Consolida la lógica del Notebook 05 para que la App y los Notebooks
    usen exactamente el mismo preprocesamiento.
    """
    # Mapeo de tipos de habitación
    dict_room = {'Entire home/apt': 3, 'Private room': 2, 'Shared room': 1, 'Hotel room': 2}
    df['room_type_num'] = df['room_type'].map(dict_room).fillna(3)
    
    # Extracción de amenities
    amenities_lower = df['amenities'].str.lower().fillna('')
    df['has_pool'] = amenities_lower.str.contains('pool').astype(int)
    df['has_ac'] = amenities_lower.str.contains('air conditioning').astype(int)
    # ... (añadir el resto de amenities del Notebook 05)
    
    return df
