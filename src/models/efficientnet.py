from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import torch.nn as nn

class RadriografiaEfficientNetB2(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=False):
        super(RadriografiaEfficientNetB2, self).__init__()

        # Cargamos el modelo EfficientNet-B2 preentrenado con ImageNet
        self.model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

        # Transfer-Learning: Congelamos los pesos del modelo si freeze_backbone es True
        # Esto nos permite solo entrenar la capa de clasificacion y no modificar lo que el
        # modelo ya ha aprendido de ImageNet
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Obtenemos el numero de features de salida del modelo preentrenado
        num_features = self.model.classifier[1].in_features

        # Creamos un sequential con un dropout y una capa lineal para la clasificación
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)