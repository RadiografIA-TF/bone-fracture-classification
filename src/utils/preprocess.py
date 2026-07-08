import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure
import os
from tqdm import tqdm


# Crear las funciones de filtros
def filtro_mediana(img, k=3):
    salida = ndi.median_filter(img, size=k, mode="reflect")
    return salida.astype(np.uint8)


def unsharp(img, sigma=2, k=1.0):
    img_float = img.astype(float)

    img_blur = ndi.gaussian_filter(img_float, sigma=sigma)
    detalle = img_float - img_blur

    salida = img_float + k * detalle
    salida = np.clip(salida, 0, 255)

    return salida.astype(np.uint8)

# Aplicamos el redimensionamiento con relleno para mantener la relación de aspecto
def resize_with_padding(img, size=384, fill=0):
    w, h = img.size
    ratio = min(size / w, size / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    new_img = Image.new("L", (size, size), fill)
    new_img.paste(img_resized, ((size - new_w) // 2, (size - new_h) // 2))
    return new_img

def preprocessing_pipeline(ruta_imagen):
    # Cargar imagen en escala de grises
    img = Image.open(ruta_imagen).convert("L")

    # Redimensionar con relleno para mantener la relación de aspecto
    img = resize_with_padding(img)
    img_arr = np.array(img).astype(np.uint8)

    # Paso 1: Suavizar ruido impulsivo con filtro mediana
    temp = filtro_mediana(img_arr, k=3)

    # Paso 2: Mejorar contraste local con CLAHE
    clahe = exposure.equalize_adapthist(
        temp / 255.0,
        clip_limit=0.03
    )
    clahe = np.clip(clahe * 255, 0, 255).astype(np.uint8)

    # Paso 3: Resaltar bordes con unsharp masking
    resultado = unsharp(clahe, sigma=1.0, k=1)

    return resultado

# ------------------------------------------------------------------
# 1. Función genérica de filtrado + procesamiento + guardado
# ------------------------------------------------------------------
def process_and_store_fracatlas(src_path, filename, target_dir,
                           prefix="new_ds_"):
    """
    Valida una imagen contra filtros de calidad y, si pasa,
    aplica preprocessing_pipeline y la guarda en target_dir.

    Parámetros
    ----------
    src_path : str          ruta de la imagen original
    filename : str          nombre del archivo (para logs y prefijo)
    target_dir : str        carpeta destino donde se guarda el resultado
    prefix : str            prefijo que se antepone al filename al guardar
    """
    try:
        # Pipeline de preprocesamiento (Filtros + CLAHE + Unsharp)
        img_processed = preprocessing_pipeline(src_path)

        img = Image.open(src_path)

        ancho, alto = img.size

        # Filtro 1: Dimensiones mínimas
        if ancho < 200 or alto < 200:
            return f"Omitida (Tamaño): {filename} ({ancho}x{alto})"

        # Filtro 2: Intensidad media
        img_gray = img.convert("L")
        intensidad_media = np.array(img_gray).mean()
        if intensidad_media >= 165:
            return f"Omitida (Intensidad): {filename} (Media: {intensidad_media:.2f})"

        # Guardar en la carpeta destino
        dst_path = os.path.join(target_dir, f"{prefix}{filename}")
        Image.fromarray(img_processed).save(dst_path)
        return True

    except Exception as e:
        return f"Error en {filename}: {e}"


# ------------------------------------------------------------------
# 2. Función genérica para recolectar tareas (buscar imágenes)
# ------------------------------------------------------------------
def collect_tasks(raw_dir, extensions=('.png', '.jpg', '.jpeg', '.webp')):
    """
    Recorre raw_dir (y todas sus subcarpetas) y arma la lista de
    (src_path, filename) de todas las imágenes encontradas.
    """
    tasks = []
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(extensions):
                tasks.append((os.path.join(root, file), file))
    return tasks