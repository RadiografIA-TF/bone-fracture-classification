import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

from .utils.preprocess import preprocessing_pipeline


def predict_image(model, image_path, device, transform=None,
                   class_names=('No Fractura', 'Fractura'),
                   threshold=0.5):
    """
    Corre inferencia sobre una sola imagen.

    Parámetros
    ----------
    model : modelo ya cargado en memoria (con los pesos correctos ya cargados
            vía load_state_dict antes de llamar a esta función).
    image_path : str, ruta a la imagen a evaluar.
    device : torch.device.
    transform : transformación de PyTorch (torchvision.transforms) a aplicar
                DESPUÉS de preprocessing_pipeline, para convertir el array
                resultante en un tensor normalizado listo para el modelo
                (ej. ToTensor + Normalize, las mismas que usaste en val/test).
                Si es None, solo se convierte a tensor sin normalizar
                (ajusta esto a como entrenaste el modelo).
    class_names : nombres de las 2 clases, en orden (índice 0, índice 1).
    threshold : umbral de probabilidad para decidir la clase positiva
                (por defecto 0.5, ajústalo si tu modelo se calibró distinto,
                por ejemplo priorizando recall).

    Retorna
    -------
    dict con: 'class_name', 'class_idx', 'prob_positive', 'probs' (ambas clases)
    """
    img_processed = preprocessing_pipeline(image_path)

    if transform is not None:
        img_pil = Image.fromarray(img_processed).convert('RGB')
        img_tensor = transform(img_pil)
    else:
        img_tensor = torch.tensor(img_processed / 255.0, dtype=torch.float32).unsqueeze(0)

    img_tensor = img_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        prob_positive = probs[1].item()
        pred_idx = 1 if prob_positive >= threshold else 0

    return {
        'class_name': class_names[pred_idx],
        'class_idx': pred_idx,
        'prob_positive': prob_positive,
        'probs': probs.cpu().numpy(),
    }

def predict_and_show(model, image_path, device, transform=None,
                      class_names=('No Fractura', 'Fractura'),
                      threshold=0.5):
    """
    Corre inferencia sobre una imagen y la muestra junto con la predicción
    y probabilidad, usando matplotlib.

    Mismos parámetros que predict_image; internamente la reutiliza.
    """
    resultado = predict_image(
        model=model, image_path=image_path, device=device,
        transform=transform, class_names=class_names, threshold=threshold
    )

    # Cargamos la imagen original (sin preprocesar) solo para visualización
    img_original = Image.open(image_path).convert('RGB')

    plt.figure(figsize=(6, 6))
    plt.imshow(img_original)
    plt.axis('off')

    color = 'red' if resultado['class_idx'] == 1 else 'green'
    titulo = (f"Predicción: {resultado['class_name']}\n"
              f"Probabilidad de fractura: {resultado['prob_positive']:.2%}")
    plt.title(titulo, color=color, fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.show()

    return resultado