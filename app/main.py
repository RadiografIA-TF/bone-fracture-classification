import time
import os
import sys
import io
import shutil
import yaml
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Configurar sys.path para importar desde la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.efficientnet import RadriografiaEfficientNetB3
from src.utils.preprocess import preprocessing_pipeline
from src.utils.transform import get_val_transform

# Importar dependencias de Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = FastAPI(
    title="RadiografIA API",
    description="API local para clasificación de fracturas en radiografías y análisis de explicabilidad.",
    version="1.0.0"
)

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorio temporal para almacenar imágenes subidas
TEMP_DIR = PROJECT_ROOT / "app" / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Variables globales para el modelo y transformaciones
model = None
device = None
transform = None
classes = {0: "No Fractura", 1: "Fractura"}

@app.on_event("startup")
def startup_event():
    global model, device, transform
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Usando dispositivo para inferencia: {device}")

        # 1. Cargar Configuración
        config_path = PROJECT_ROOT / "config" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de configuración en {config_path}")
            
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        mean = config["model"]["transforms"]["mean"]
        std = config["model"]["transforms"]["std"]
        input_shape = config["model"]["transforms"]["input_shape"]

        # 2. Inicializar Modelo
        # Se requiere pretrained=False y freeze_backbone=False para cargar pesos locales sin descargar de Internet
        model = RadriografiaEfficientNetB3(num_classes=2, pretrained=False, freeze_backbone=False)
        
        # IMPORTANTE: Para que coincida con los checkpoints, hay que descongelar a fase 0 (cambia el clasificador)
        model.unfreze_backbone(phase_num=0)

        # 3. Cargar los pesos oficiales (v3)
        checkpoint_path = PROJECT_ROOT / config["paths"]["model_path_ft3"]
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de pesos (checkpoint) en {checkpoint_path}")

        print(f"[*] Cargando pesos del modelo desde: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        print("[*] Modelo cargado exitosamente y listo para inferencia.")

        # 4. Configurar transformaciones de validación
        transform = get_val_transform(output_size=(input_shape[0], input_shape[1]), mean=mean, std=std)
        print(f"[*] Transformación configurada con tamaño de entrada: {input_shape}")

    except Exception as e:
        print(f"[!] Error crítico durante el inicio del servidor: {e}")
        # No detenemos el proceso para permitir depuración, pero los endpoints fallarán con 503
        model = None

@app.get("/health")
def health_check():
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "El modelo no se cargó correctamente en el inicio."}
        )
    return {"status": "ok", "device": str(device)}

@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = Form(0.5)):
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")

    # Guardar temporalmente la imagen
    temp_file_path = TEMP_DIR / file.filename
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        start_time = time.time()

        # 1. Aplicar preprocessing pipeline
        img_processed = preprocessing_pipeline(str(temp_file_path))
        img_pil = Image.fromarray(img_processed).convert("RGB")

        # 2. Transformación final a tensor
        input_tensor = transform(img_pil).unsqueeze(0).to(device)

        # 3. Inferencia
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            
            prob_fracture = probs[1].item()
            prob_no_fracture = probs[0].item()

            # Aplicar umbral de decisión personalizado para priorizar recall clínico
            class_id = 1 if prob_fracture >= threshold else 0
            prediction_label = classes[class_id]
            confidence = prob_fracture if class_id == 1 else prob_no_fracture

        inference_time_ms = (time.time() - start_time) * 1000

        return {
            "prediction": prediction_label,
            "class_id": class_id,
            "confidence": float(confidence),
            "inference_time_ms": round(inference_time_ms, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen para predicción: {str(e)}")
    finally:
        # Eliminar archivo temporal
        if temp_file_path.exists():
            os.remove(temp_file_path)

@app.post("/explain")
async def explain(
    file: UploadFile = File(...), 
    layer_name: str = Form("conv_head"),
    target_class: int = Form(None)
):
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")

    temp_file_path = TEMP_DIR / f"explain_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Preprocesar y transformar
        img_processed = preprocessing_pipeline(str(temp_file_path))
        img_pil = Image.fromarray(img_processed).convert("RGB")
        input_tensor = transform(img_pil).unsqueeze(0).to(device)

        # 2. Resolver la clase objetivo si es None
        if target_class is None:
            model.eval()
            with torch.no_grad():
                outputs = model(input_tensor)
                target_class = torch.argmax(outputs, dim=1).item()

        # 3. Resolver la capa objetivo
        if layer_name == "blocks[6]" or layer_name == "blocks.6":
            target_layers = [model.model.blocks[6]]
        elif layer_name == "blocks[5]" or layer_name == "blocks.5":
            target_layers = [model.model.blocks[5]]
        elif layer_name == "conv_head":
            target_layers = [model.model.conv_head]
        else:
            # Por defecto usar conv_head
            target_layers = [model.model.conv_head]

        # 4. Generar Grad-CAM
        targets = [ClassifierOutputTarget(target_class)]
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # Preparar la imagen base (valores entre 0.0 y 1.0)
        img_normalized = np.array(img_pil.resize((input_tensor.shape[-1], input_tensor.shape[-2]))) / 255.0
        
        # Generar mapa de calor
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        
        # Superponer mapa de calor en la imagen normalizada
        visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)

        # 5. Convertir el resultado a PNG en memoria
        result_img = Image.fromarray(visualization)
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando explicación Grad-CAM: {str(e)}")
    finally:
        if temp_file_path.exists():
            os.remove(temp_file_path)

# Montar frontend estático
static_dir = PROJECT_ROOT / "app" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
