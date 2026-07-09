# Guía de Traspaso (Handover) para Despliegue de RadiografIA

Este documento resume el contexto del proyecto, la estructura del repositorio y las especificaciones técnicas para el agente encargado de construir el **Frontend** y **Backend** local para el despliegue del modelo.

---

## 1. Contexto del Proyecto y Objetivo
**RadiografIA** es un sistema de apoyo al diagnóstico médico asistido por computadora. Su objetivo es clasificar imágenes radiográficas (Rayos X) en dos categorías:
1. **Fractura** (Clase `1`)
2. **No Fractura** (Clase `0`)

El modelo actual está optimizado para maximizar el **Recall (Sensibilidad)** clínicos, reduciendo al mínimo los falsos negativos (fracturas no detectadas) a la vez que mantiene una alta exactitud global.

---

## 2. Especificaciones del Modelo y Rendimiento
- **Arquitectura**: `EfficientNet-B3` (implementado en PyTorch mediante la librería `timm`).
- **Últimos Resultados de Validación (v3)**:
  - **Exactitud (Accuracy) Global**: **91.29%**
  - **Sensibilidad (Recall) de Fractura**: **80.82%**
  - **Pérdida (Loss)**: **0.2683**
- **Pesos del Modelo**: 
  - Los pesos entrenados se encuentran en `checkpoints/radriografia_efficientnet_b3_ft3.h5` (es un `state_dict` de PyTorch guardado con extensión `.h5`).

---

## 3. Estructura Relevante del Repositorio
El agente de desarrollo de API y UI debe conocer las siguientes rutas clave del proyecto:
- **`src/models/efficientnet.py`**: Contiene la definición de la clase del modelo `RadriografiaEfficientNetB3`.
- **`src/utils/preprocess.py`**: Contiene el pipeline oficial de preprocesamiento de imágenes (`preprocessing_pipeline`).
- **`src/utils/gradcam.py`**: Contiene la función `show_gradcam` para calcular mapas de calor de activación neural sobre el backbone `blocks` del modelo.
- **`config/config.yaml`**: Contiene los hiperparámetros globales (formas de entrada de la imagen, pesos, directorios).

---

## 4. Requerimientos del Backend (Servidor de Inferencia)
Se sugiere crear una API local en Python (usando **FastAPI** por su velocidad y documentación automática con Swagger).

### Endpoints Requeridos:
1. **`POST /predict`**
   - **Entrada**: Archivo de imagen radiográfica (`multipart/form-data`).
   - **Proceso**:
     1. Aplicar la función `preprocessing_pipeline` de `src.utils.preprocess` (redimensiona a 320x320, aplica filtro de mediana, ecualización local CLAHE mediante `skimage` y máscara de enfoque *unsharp*).
     2. Cargar el modelo `RadriografiaEfficientNetB3` con los pesos de `checkpoints/radriografia_efficientnet_b3_ft3.h5`.
     3. Ejecutar inferencia en el dispositivo disponible (CPU o CUDA).
   - **Salida (JSON)**:
     ```json
     {
       "prediction": "Fractura" o "No Fractura",
       "class_id": 1 o 0,
       "confidence": 0.9452,
       "inference_time_ms": 120.5
     }
     ```

2. **`POST /explain`**
   - **Entrada**: Archivo de imagen radiográfica y opcionalmente el bloque de la red a auditar (ej. `conv_head` o `model.blocks[6]`).
   - **Proceso**:
     1. Ejecutar Grad-CAM utilizando las funciones de `src.utils.gradcam`.
     2. Superponer el mapa de calor (rojo/amarillo para activaciones altas) sobre la imagen preprocesada.
   - **Salida**: Imagen resultante en formato PNG (`image/png`) o en formato Base64 lista para ser renderizada en el frontend.

---

## 5. Requerimientos del Frontend (Interfaz de Usuario)
Una aplicación web local sencilla y moderna (HTML/JS/CSS vainilla, o React/Streamlit si se prefiere agilidad).

### Flujo de la Interfaz:
1. **Panel de Carga**: Un área de arrastrar y soltar (Drag and Drop) para que el médico cargue la radiografía en formato JPG/PNG.
2. **Visualización de Resultados**:
   - Mostrar la predicción en letras grandes con colores semánticos (Rojo para "Fractura Detectada", Verde para "Sin Fractura").
   - Barra de confianza porcentual (ej. "Confianza: 92.4%").
3. **Panel de Explicabilidad (Grad-CAM)**:
   - Mostrar una vista comparativa lado a lado:
     - Lado izquierdo: La imagen radiográfica original preprocesada.
     - Lado derecho: El mapa de calor Grad-CAM que resalta las zonas en las que el modelo basó su decisión de fractura.

---

## 6. Código de Referencia para Inferencia Local
El backend puede realizar inferencias utilizando el siguiente bloque de código integrado del proyecto:

```python
import torch
from PIL import Image
from src.models.efficientnet import RadriografiaEfficientNetB3
from src.utils.preprocess import preprocessing_pipeline
from src.utils.transform import get_val_transform # Ajustar si es necesario

# 1. Configurar dispositivo y cargar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RadriografiaEfficientNetB3(num_classes=2, pretrained=False, freeze_backbone=False)
model.load_state_dict(torch.load("checkpoints/radriografia_efficientnet_b3_ft3.h5", map_location=device))
model.to(device)
model.eval()

# 2. Preprocesar imagen
img_path = "ruta_de_imagen_subida.jpg"
img_proc_np = preprocessing_pipeline(img_path)
img_pil = Image.fromarray(img_proc_np).convert("RGB")

# Transformación final de tensores
transform = get_val_transform()
input_tensor = transform(img_pil).unsqueeze(0).to(device)

# 3. Inferencia
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)
    class_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][class_id].item()

classes = {0: "No Fractura", 1: "Fractura"}
print(f"Predicción: {classes[class_id]} | Confianza: {confidence*100:.2f}%")
```
