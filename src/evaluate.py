import os
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.utils import get_transforms
from src.models import RadriografiaEfficientNetB2

def evaluate_model():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # 1. Cargar Configuración
    CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Rutas
    test_path = PROJECT_ROOT / config["model"]["paths"]["test_path"]
    save_model_path = PROJECT_ROOT / config["model"]["paths"]["save_model_path"]
    checkpoint_dir = PROJECT_ROOT / "checkpoints"

    # Parámetros del modelo
    num_classes = config["model"]["parameters"]["num_classes"]
    batch_size = config["model"]["parameters"]["batch_size"]
    input_shape = config["model"]["transforms"]["input_shape"]
    mean = config["model"]["transforms"]["mean"]
    std = config["model"]["transforms"]["std"]

    print("=" * 60)
    print("INICIANDO EVALUACIÓN DEL MODELO DE FASE 1")
    print("=" * 60)
    print(f"Ruta de test: {test_path}")
    print(f"Ruta del modelo guardado: {save_model_path}")

    # 2. Cargar Dataset de Test
    transforms_model = get_transforms(
        output_size=(input_shape[0], input_shape[1]),
        mean=mean,
        std=std,
        output_channels=input_shape[2]
    )

    if not test_path.exists():
        print(f"Error: La ruta de test {test_path} no existe. Por favor ejecute la preparación de datos.")
        return

    dataset_test = ImageFolder(root=test_path, transform=transforms_model)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    classes = dataset_test.classes
    print(f"Imágenes de prueba encontradas: {len(dataset_test)}")
    print(f"Clases a evaluar: {classes}")

    # 3. Cargar el Modelo Entrenado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluando en dispositivo: {device}")

    model = RadriografiaEfficientNetB2(num_classes=num_classes)
    if not save_model_path.exists():
        print(f"Error: No se encontró el modelo guardado en {save_model_path}")
        return

    model.load_state_dict(torch.load(save_model_path, map_location=device))
    model.to(device)
    model.eval()

    # 4. Inferencia
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in loader_test:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 5. Calcular Métricas
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=classes)

    print("\n" + "=" * 60)
    print("REPORTE DE CLASIFICACIÓN:")
    print("=" * 60)
    print(report_text)
    print(f"Accuracy Total: {accuracy:.4f}")
    print("=" * 60)

    # Guardar reporte en JSON
    report_json_path = checkpoint_dir / "reporte_evaluacion_fase1.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Reporte de clasificación guardado en: {report_json_path}")

    # 6. Generar Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Matriz de Confusión - EfficientNet-B2 (Fase 1)")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()

    cm_path = checkpoint_dir / "matriz_confusion_fase1.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Matriz de confusión guardada en: {cm_path}")

if __name__ == "__main__":
    evaluate_model()
