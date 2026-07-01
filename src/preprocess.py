import os
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile
from scipy import ndimage as ndi
from skimage import exposure
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. Cargar Configuración
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

raw_data_dir = PROJECT_ROOT / config["path"]["raw_data_dir"] / "dataset_fracturas"
processed_data_dir = PROJECT_ROOT / config["path"]["processed_data_dir"] / "dataset_fracturas"

print(f"Directorio Raíz: {PROJECT_ROOT}")
print(f"Directorio de Datos Originales: {raw_data_dir}")
print(f"Directorio de Datos Procesados: {processed_data_dir}")

def get_long_path(path):
    """Soporte para rutas largas en Windows (>260 caracteres)"""
    abs_path = os.path.abspath(path)
    if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
        return '\\\\?\\' + abs_path
    return abs_path

# 2. Funciones de Filtros y Pipeline
def filtro_mediana(img, k=3):
    return ndi.median_filter(img, size=k, mode="reflect").astype(np.uint8)

def unsharp(img, sigma=1.0, k=1.0):
    img_float = img.astype(float)
    img_blur = ndi.gaussian_filter(img_float, sigma=sigma)
    detalle = img_float - img_blur
    salida = img_float + k * detalle
    return np.clip(salida, 0, 255).astype(np.uint8)

def preprocessing_pipeline(ruta_imagen, size=320):
    long_ruta = get_long_path(ruta_imagen)
    img = Image.open(long_ruta).convert("L")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    img_arr = np.array(img).astype(np.uint8)
    
    # Filtro mediana
    temp = filtro_mediana(img_arr, k=3)
    
    # CLAHE
    clahe = exposure.equalize_adapthist(temp / 255.0, clip_limit=0.03)
    clahe = np.clip(clahe * 255, 0, 255).astype(np.uint8)
    
    # Unsharp Masking
    resultado = unsharp(clahe, sigma=1.0, k=1)
    
    return resultado

# 3. Escanear y Filtrar Imágenes
if not raw_data_dir.exists():
    print(f"Error: El directorio {raw_data_dir} no existe.")
    exit(1)

extensiones_img = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
rutas_imagenes = []
for root, dirs, files in os.walk(raw_data_dir):
    for file in files:
        if file.lower().endswith(extensiones_img):
            rutas_imagenes.append(Path(root) / file)

print(f"Total de imágenes encontradas en crudo: {len(rutas_imagenes)}")

datos = []
for ruta in tqdm(rutas_imagenes, desc="Analizando dataset original"):
    try:
        categoria = ruta.parent.parent.name
        split = ruta.parent.name
        
        long_ruta = get_long_path(ruta)
        img = Image.open(long_ruta)
        ancho, alto = img.size
        
        img_gray = img.convert("L")
        gray_array = np.array(img_gray)
        intensidad_media = gray_array.mean()
        
        datos.append({
            "ruta": ruta,
            "archivo": ruta.name,
            "categoria": categoria,
            "split": split,
            "ancho": ancho,
            "alto": alto,
            "intensidad_media": intensidad_media
        })
    except Exception as e:
        print(f"Error al leer {ruta}: {e}")

df = pd.DataFrame(datos)

# Aplicar los filtros definidos en la EDA (Notebook 01)
print(f"Dataset antes de filtros: {len(df)} imágenes")

# Filtro 1: Dropear imágenes con intensidad media mayor a 165
df = df[df["intensidad_media"] < 165]
print(f"Dataset después de filtrar por intensidad media < 165: {len(df)} imágenes")

# Filtro 2: Dropear categorías no deseadas
df = df[~df["categoria"].isin(["Avulsion fracture", "Hairline Fracture"])]
print(f"Dataset después de remover Avulsion y Hairline: {len(df)} imágenes")

# Filtro 3: Dropear imágenes con dimensiones menores a 200
df = df[(df["ancho"] >= 200) & (df["alto"] >= 200)]
print(f"Dataset después de filtrar por resolución >= 200: {len(df)} imágenes")

# 4. Crear Carpetas y Procesar Imágenes
os.makedirs(get_long_path(processed_data_dir), exist_ok=True)

for categoria in df["categoria"].unique():
    for split in df["split"].unique():
        carpeta_destino = processed_data_dir / split / categoria
        os.makedirs(get_long_path(carpeta_destino), exist_ok=True)

procesadas = 0
errores = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocesando"):
    try:
        ruta_original = row["ruta"]
        categoria = row["categoria"]
        split = row["split"]
        archivo = row["archivo"]
        
        carpeta_destino = processed_data_dir / split / categoria
        ruta_destino = carpeta_destino / archivo
        
        img_procesada = preprocessing_pipeline(ruta_original)
        img_pil = Image.fromarray(img_procesada)
        img_pil.save(get_long_path(ruta_destino), quality=95)
        
        procesadas += 1
    except Exception as e:
        errores += 1
        print(f"Error en {row['ruta']}: {e}")

print("=" * 50)
print("PREPROCESAMIENTO FINALIZADO")
print(f"Imágenes procesadas con éxito: {procesadas}")
print(f"Errores en el proceso: {errores}")
print("=" * 50)
