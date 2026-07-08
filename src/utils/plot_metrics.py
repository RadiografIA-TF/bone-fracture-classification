import matplotlib.pyplot as plt

def plot_training_history(stages, metrics=('loss', 'acc'), figsize=None,
                            colors=None, title_prefix='Historial de Entrenamiento',
                            layout='vertical'):
    """
    Grafica el historial de entrenamiento de una o más etapas de fine-tuning,
    para una o más métricas, en subplots.

    Parámetros
    ----------
    stages : list of dict
        Cada dict: {'label': str, 'historial': dict con train_X/val_X}
    metrics : tuple/list de str
        Métricas a graficar, cada una en su propio subplot. Ej: ('loss', 'acc')
    figsize : tuple o None. Si None, se ajusta automáticamente según layout y cantidad de métricas.
    colors : list de str o None. Un color por etapa (mismo color para train/val).
    title_prefix : str, prefijo del título de cada subplot.
    layout : 'vertical' (subplots apilados, uno debajo del otro) o
             'horizontal' (subplots lado a lado).
    """
    default_colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown', 'teal']
    if colors is None:
        colors = default_colors

    n_metrics = len(metrics)

    if figsize is None:
        figsize = (15, 5 * n_metrics) if layout == 'vertical' else (6 * n_metrics, 5)

    plt.figure(figsize=figsize)

    metric_display = {
        'loss': {'ylabel': 'Loss', 'title': 'Loss Curves'},
        'acc': {'ylabel': 'Accuracy', 'title': 'Accuracy Curves'},
        'recall': {'ylabel': 'Recall', 'title': 'Recall Curves'},
    }

    for i, metric in enumerate(metrics):
        if layout == 'vertical':
            plt.subplot(n_metrics, 1, i + 1)
        else:
            plt.subplot(1, n_metrics, i + 1)

        for stage_idx, stage in enumerate(stages):
            label = stage['label']
            historial = stage['historial']
            color = colors[stage_idx % len(colors)]

            train_key = f'train_{metric}'
            val_key = f'val_{metric}'

            if train_key not in historial or val_key not in historial:
                print(f"Aviso: la etapa '{label}' no tiene la métrica '{metric}', se omite.")
                continue

            # Si solo hay una etapa, mantenemos las etiquetas simples ("Train Loss")
            # igual que tu script original sin sufijo de etapa
            suffix = f' ({label})' if len(stages) > 1 else ''

            plt.plot(historial[train_key], label=f'Train {metric.capitalize()}{suffix}',
                      linestyle='--', color=color)
            plt.plot(historial[val_key], label=f'Validation {metric.capitalize()}{suffix}',
                      color=color)

        info = metric_display.get(metric, {'ylabel': metric.capitalize(), 'title': f'{metric.capitalize()} Curves'})
        title = info['title'] if len(stages) == 1 else f'{title_prefix} - {info["title"]}'
        plt.title(title)
        plt.ylabel(info['ylabel'])
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()