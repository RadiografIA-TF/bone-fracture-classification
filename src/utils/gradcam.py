import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .preprocess import preprocessing_pipeline

def show_gradcam(model, image_path, device, transform,
                              layer_names_and_layers,
                              class_names=('No Fractura', 'Fractura'),
                              target_class=None,
                              cols=3,
                              subplot_size=4):
    """
    Muestra un Grad-CAM por cada capa especificada, organizados en un
    grid de varias filas/columnas (en vez de una sola fila larga).

    cols : cantidad de columnas en el grid.
    subplot_size : tamaño (en pulgadas) de cada subplot individual,
                   controla qué tan grande se ve cada imagen.
    """
    model.eval()

    img_processed = preprocessing_pipeline(image_path)
    img_pil = Image.fromarray(img_processed).convert('RGB')
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        pred_idx = torch.argmax(outputs, dim=1).item()

    clase_a_explicar = target_class if target_class is not None else pred_idx
    targets = [ClassifierOutputTarget(clase_a_explicar)]

    img_para_mostrar = np.array(img_pil.resize((input_tensor.shape[-1], input_tensor.shape[-2]))) / 255.0

    total_plots = len(layer_names_and_layers) + 1  # +1 por la imagen original
    rows = math.ceil(total_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(subplot_size * cols, subplot_size * rows))
    axes = axes.flatten()  # para poder indexar linealmente sin importar filas/columnas

    # Primer subplot: imagen original
    axes[0].imshow(img_pil)
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')

    # Resto de subplots: uno por capa
    for i, (nombre_capa, capa) in enumerate(layer_names_and_layers.items()):
        cam = GradCAM(model=model, target_layers=[capa])
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        visualization = show_cam_on_image(img_para_mostrar, grayscale_cam, use_rgb=True)

        axes[i + 1].imshow(visualization)
        axes[i + 1].set_title(f'{nombre_capa}\n({class_names[clase_a_explicar]})')
        axes[i + 1].axis('off')

    # Apagar ejes sobrantes si el grid tiene más celdas que imágenes
    for j in range(total_plots, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()