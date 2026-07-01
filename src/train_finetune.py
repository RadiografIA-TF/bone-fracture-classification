import os
import yaml
import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.utils import get_transforms
from src.models import RadriografiaEfficientNetB2
from src.train import train_model

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # 1. Cargar Configuración
    CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Parámetros
    num_classes = config["model"]["parameters"]["num_classes"]
    batch_size = config["model"]["parameters"]["batch_size"]
    input_shape = config["model"]["transforms"]["input_shape"]
    mean = config["model"]["transforms"]["mean"]
    std = config["model"]["transforms"]["std"]

    # Nuevos Parámetros para Fine-tuning
    finetune_lr = 1e-5  # Tasa de aprendizaje muy baja para no destruir pesos preentrenados
    finetune_epochs = 10

    # Rutas
    train_path = PROJECT_ROOT / config["model"]["paths"]["train_path"]
    test_path = PROJECT_ROOT / config["model"]["paths"]["test_path"]
    save_phase1_model_path = PROJECT_ROOT / config["model"]["paths"]["save_model_path"]
    
    save_finetuned_model_path = PROJECT_ROOT / "checkpoints" / "radriografia_efficientnet_b2_finetuned.h5"
    save_finetuned_historial_path = PROJECT_ROOT / "checkpoints" / "historial_entrenamiento_finetuned.json"

    print("=" * 60)
    print("INICIANDO FASE 2: FINE-TUNING DEL MODELO")
    print("=" * 60)
    print(f"Cargando modelo base de Fase 1: {save_phase1_model_path}")
    print(f"Tasa de aprendizaje (Fine-tuning): {finetune_lr}")
    print(f"Épocas: {finetune_epochs}")

    # 2. Cargar Datasets
    transforms_model = get_transforms(
        output_size=(input_shape[0], input_shape[1]),
        mean=mean,
        std=std,
        output_channels=input_shape[2]
    )

    dataset_train = ImageFolder(root=train_path, transform=transforms_model)
    dataset_test = ImageFolder(root=test_path, transform=transforms_model)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    print(f"Imágenes de entrenamiento: {len(dataset_train)}")
    print(f"Imágenes de validación: {len(dataset_test)}")

    # 3. Inicializar Modelo y Cargar Pesos de Fase 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en dispositivo: {device}")

    # Inicializamos el modelo (con freeze_backbone=False para descongelar por defecto o lo descongelamos manualmente)
    model = RadriografiaEfficientNetB2(num_classes=num_classes, freeze_backbone=False)
    
    if not save_phase1_model_path.exists():
        print(f"Error: No se encontró el modelo de la Fase 1 en {save_phase1_model_path}")
        print("Por favor, asegúrese de haber completado la Fase 1.")
        return
        
    model.load_state_dict(torch.load(save_phase1_model_path, map_location=device))
    model.to(device)

    # 4. Descongelar las capas del backbone para Fine-Tuning
    # Para fine-tuning, permitimos que los gradientes se actualicen en TODO el modelo
    for param in model.parameters():
        param.requires_grad = True

    # 5. Configurar Optimizador y Criterio
    criterion = nn.CrossEntropyLoss()
    
    # Optimizador Adam con una tasa de aprendizaje muy baja
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)

    # 6. Entrenar el Modelo
    history = train_model(
        model=model,
        train_loader=loader_train,
        val_loader=loader_test,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=finetune_epochs,
        save_best_model_path=save_finetuned_model_path,
        save_historial_path=save_finetuned_historial_path
    )

    print("=" * 60)
    print("FINE-TUNING COMPLETADO EXITOSAMENTE")
    print(f"Mejor modelo guardado en: {save_finetuned_model_path}")
    print(f"Historial de entrenamiento guardado en: {save_finetuned_historial_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
