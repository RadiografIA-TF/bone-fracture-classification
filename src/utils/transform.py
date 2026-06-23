from torchvision import transforms

def get_transforms(output_size=(288, 288), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], output_channels=3):
    """
    Devuelve las transformaciones de imagen para el conjunto de datos.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=output_channels),
        transforms.Resize(output_size),
        transforms.ToTensor(),
        # Aplicamos la normalización de los valores de los píxeles a un rango de [0, 1]
        # se aplica la normalizacion que indica la documentacion de pytorch
        transforms.Normalize(mean=mean, std=std)
    ])