import torch.nn as nn
import timm

class RadriografiaEfficientNetB3(nn.Module):
    def __init__(self, num_classes=2, model_name='efficientnet_b3', pretrained=True, freeze_backbone=True):
        super(RadriografiaEfficientNetB3, self).__init__()
        # Cargamos el modelo EfficientNet-B3 preentrenado con ImageNet
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained)
        self.num_classes = num_classes

        
        # Fine-Tuning: Congelamos los pesos del modelo si freeze_backbone es True
        # Esto nos permite solo entrenar la capa de clasificacion y no modificar lo que el
        # modelo ya ha aprendido de ImageNet
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def unfreze_backbone(self, phase_num=0):
        if phase_num == 0:
            # Obtenemos el numero de features de salida del modelo preentrenado
            num_features = self.model.classifier.in_features

            # Creamos un sequential con un dropout y una capa lineal para la clasificación
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(num_features, self.num_classes)
            )
        elif phase_num == 1:
            # Fase 1: Descongelamos las últimas capas del backbone para permitir que se ajusten durante el entrenamiento
            for name, param in self.model.named_parameters():
                if any(x in name for x in ["blocks.6", "conv_head", "classifier"]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif phase_num == 2:
            # Fase 2: Descongelamos más capas del backbone para permitir un ajuste más fino
            for name, param in self.model.named_parameters():
                if any(x in name for x in ["blocks.3", "blocks.4", "blocks.5", "blocks.6",
                                            "conv_head", "classifier"]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif phase_num == 3:
            # Fase 3: Descongelamos todas las capas del backbone para permitir un ajuste completo
            for name, param in self.model.named_parameters():
                if any(x in name for x in ["blocks.1", "blocks.2", "blocks.3", "blocks.4", 
                                           "blocks.5", "blocks.6", "conv_head", "classifier"]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)