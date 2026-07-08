import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                               roc_curve, roc_auc_score)


def evaluate_model(model, test_loader, criterion, device,
                    model_path=None,
                    desc='Testing'):
    """
    Evalúa un modelo sobre un set de test/validación: calcula loss,
    reporte de clasificación, matriz de confusión y curva ROC.

    Parámetros
    ----------
    model : el modelo ya instanciado (arquitectura).
    test_loader : DataLoader del set de evaluación.
    criterion : función de pérdida.
    device : torch.device.
    model_path : str o None. Si se pasa, carga esos pesos en el modelo
                 antes de evaluar (torch.load + load_state_dict).
                 Si es None, evalúa el modelo tal como está en memoria.
    desc : texto para la barra de progreso.

    Retorna
    -------
    dict con: 'loss', 'labels', 'preds', 'probs', 'auc'
    """
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=desc):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = F.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(test_loader)
    print(f'\nLoss: {avg_loss:.4f}')
    
    auc_score = roc_auc_score(all_labels, all_probs)

    return {
        'loss': avg_loss,
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs,
        'auc': auc_score,
    }