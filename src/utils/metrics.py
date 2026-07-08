from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

def show_classification_metrics(y_true, y_pred, target_names=None):
    """
    Muestra el informe de clasificación y la matriz de confusión.

    Args:
        y_true (list): Etiquetas verdaderas.
        y_pred (list): Etiquetas predichas.
        target_names (list, optional): Nombres de las clases. Si no se proporciona, se usarán los índices de clase.
    """
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

def show_confusion_matrix(y_true, y_pred, target_names=None):
    """
    Muestra la matriz de confusión.

    Args:
        y_true (list): Etiquetas verdaderas.
        y_pred (list): Etiquetas predichas.
        target_names (list, optional): Nombres de las clases. Si no se proporciona, se usarán los índices de clase.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Matriz de Confusion - Test Set')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Prediccion')
    plt.show()

def show_roc_curve(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.show()