# 1. MONTAR GOOGLE DRIVE Y PREPARAR ENTORNO
# ==============================================================================
from google.colab import drive
import os
import sys

# Montamos tu Drive para acceder a los archivos
drive.mount('/content/drive')

# Define aquí la ruta donde tienes tu carpeta 'data' en Drive
# Ajusta 'Trabajo_IA' al nombre de tu carpeta real
PATH_DRIVE = '/content/drive/MyDrive/Trabajo_IA/data'

if not os.path.exists(PATH_DRIVE):
    print(f" Error: No se encuentra la ruta {PATH_DRIVE}. Verifica el nombre en tu Drive.")
else:
    print(f" Conectado a Drive en: {PATH_DRIVE}")

# Instalación de librerías necesarias

# ==============================================================================
# 2. IMPORTACIONES Y CONFIGURACIÓN DE IA (GPU ACTIVADA)
# ==============================================================================
import pandas as pd
import re
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings('ignore')

# Verificamos si la GPU está disponible
device = 0 if torch.cuda.is_available() else -1
 

# ==============================================================================
# 3. CARGA Y MUESTREO ESTRATIFICADO (TOP 10 POR PISO)
# ==============================================================================
 
ruta_reviews = os.path.join(PATH_DRIVE, 'reviews.csv')

# Cargamos columnas clave (ajusta si 'date' no existe en tu archivo)
try:
    df_reviews = pd.read_csv(ruta_reviews, usecols=['listing_id', 'comments', 'date']).dropna()
    # Ordenamos por fecha para tener las más recientes primero
    df_reviews = df_reviews.sort_values(by=['listing_id', 'date'], ascending=[True, False])
except:
 
    df_reviews = pd.read_csv(ruta_reviews, usecols=['listing_id', 'comments']).dropna()

# Aplicamos tu estrategia: Máximo 10 reseñas por piso (Muestreo Estratificado)
 
df_reviews = df_reviews.groupby('listing_id').head(10).copy()

def limpiar_texto(texto):
    texto = str(texto)
    texto = re.sub(r'<br\s*/?>|[\r\n]+', ' ', texto)
    return re.sub(r'\s+', ' ', texto).strip()

df_reviews['comments'] = df_reviews['comments'].apply(limpiar_texto)
df_reviews = df_reviews[df_reviews['comments'].str.len() > 10].copy()

 

# ==============================================================================
# 4. INFERENCIA CON TRANSFORMER MULTILINGÜE
# ==============================================================================
 

# El parámetro device=0 es la clave para usar la GPU
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    truncation=True,
    max_length=512,
    device=device
)

def obtener_score(texto):
    try:
        # El modelo procesa lotes de texto muy rápido en GPU
        resultado = sentiment_analyzer(texto[:512])[0]
        return (int(resultado['label'].split()[0]) - 1) / 4.0
    except:
        return 0.5

# Aplicamos el modelo
df_reviews['score_sentimiento'] = df_reviews['comments'].apply(obtener_score)

# ==============================================================================
# 5. AGRUPACIÓN Y GUARDADO FINAL EN DRIVE
# ==============================================================================
 
df_nlp_final = df_reviews.groupby('listing_id')['score_sentimiento'].mean().reset_index()

ruta_salida = os.path.join(PATH_DRIVE, 'reviews_scores_nlp.csv')
df_nlp_final.to_csv(ruta_salida, index=False)