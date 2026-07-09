# Bone Fracture Classification

Proyecto de clasificación de fracturas óseas a partir de radiografías usando **PyTorch** y **EfficientNet-B3**. El repositorio combina scripts de entrenamiento, evaluación e inferencia con notebooks para análisis exploratorio y preparación de datos.

## Resumen

El objetivo es entrenar un clasificador binario para distinguir entre radiografías con fractura y sin fractura. El proyecto incluye:

- Preprocesado de imágenes médicas.
- Entrenamiento con transfer learning y fine-tuning.
- Evaluación con accuracy, recall, matriz de confusión y curva ROC.
- Explicación visual del modelo con Grad-CAM.
- Guardado de modelos y métricas en `checkpoints/`.
- Despliegue local con API FastAPI y frontend estático para inferencia y explicabilidad.

## Estructura Del Proyecto

```text
bone-fracture-classification/
├── README.md
├── LICENSE
├── requirements.txt
├── app/
│   ├── main.py
│   ├── temp/
│   └── static/
│       ├── index.html
│       ├── css/
│       │   └── styles.css
│       └── js/
│           └── app.js
├── checkpoints/
│   ├── historial_entrenamiento_tf.json
│   ├── historial_entrenamiento_ft1.json
│   ├── historial_entrenamiento_ft2.json
│   ├── historial_entrenamiento_ft3.json
│   ├── radriografia_efficientnet_b3_tf.h5
│   ├── radriografia_efficientnet_b3_ft1.h5
│   ├── radriografia_efficientnet_b3_ft2.h5
│   └── radriografia_efficientnet_b3_ft3.h5
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   └── dataset_fracturas/
│   │       ├── BoneBreak_raw/
│   │       └── FractAtlas_raw/
│   └── processed/
│       └── dataset_fracturas/
│           ├── fractured/
│           └── non_fractured/
├── notebooks/
│   ├── 01_eda_d1.ipynb
│   ├── 01_eda_d2.ipynb
│   ├── 01_preprocess_data.ipynb
│   ├── 02_model_v1.ipynb
│   ├── 02_model_v2.ipynb
│   └── 02_model_v3.ipynb
└── src/
    ├── evaluate.py
    ├── inference.py
    ├── train.py
    ├── models/
    │   ├── __init__.py
    │   └── efficientnet.py
    └── utils/
        ├── __init__.py
        ├── gradcam.py
        ├── metrics.py
        ├── plot_metrics.py
        ├── preprocess.py
        └── transform.py
```

## Datos

El repositorio trabaja con dos orígenes de datos en `data/raw/dataset_fracturas/`:

- `BoneBreak_raw/`, con subcarpetas por tipo de fractura y partición `Train/Test`.
- `FractAtlas_raw/`, con imágenes `Fractured` y `Non_fractured`, además de anotaciones en varios formatos.

El conjunto procesado se guarda en `data/processed/dataset_fracturas/` con clases binarias:

- `fractured/`
- `non_fractured/`

Por tamaño, `data/` está pensado para no versionarse en Git.

## Configuración

La configuración principal está en [config/config.yaml](config/config.yaml). Ahí se definen:

- Rutas de checkpoints y datasets.
- Tamaño de entrada (`384 x 384`).
- Parámetros de entrenamiento por fases.
- Hiperparámetros de augmentación y normalización.

## Componentes Principales

- `src/models/efficientnet.py`: define el modelo `RadriografiaEfficientNetB3` sobre EfficientNet-B3.
- `src/train.py`: bucle de entrenamiento con métricas, AMP, early stopping y guardado del mejor modelo.
- `src/train_finetune.py`: flujo de fine-tuning a partir de un modelo ya entrenado.
- `src/evaluate.py`: evaluación sobre validación/test con métricas y ROC AUC.
- `src/inference.py`: inferencia sobre una imagen individual y visualización del resultado.
- `src/utils/preprocess.py`: pipeline de preprocesado de imágenes.
- `src/utils/gradcam.py`: generación de mapas Grad-CAM para interpretar las predicciones del modelo.
- `src/utils/transform.py`: transformaciones para entrenamiento y validación.

## Aplicación Web

La carpeta `app/` contiene una aplicación local basada en **FastAPI** con una interfaz web estática para probar el modelo final.

La API usa `fastapi`, `uvicorn`, `python-multipart` y `pytorch-grad-cam` para inferencia y explicabilidad.

### Componentes

- `app/main.py`: crea la API, carga el modelo y monta el frontend estático.
- `app/static/index.html`: interfaz principal de la aplicación.
- `app/static/css/styles.css`: estilos de la interfaz.
- `app/static/js/app.js`: lógica del cliente para consumir la API.
- `app/test_api.py`: prueba automatizada del flujo completo de la API.

### Endpoints

- `GET /health`: verifica que el modelo quedó cargado correctamente.
- `POST /predict`: recibe una imagen y devuelve la clase predicha, la confianza y el tiempo de inferencia.
- `POST /explain`: genera una explicación visual con Grad-CAM para la predicción.

### Ejecución Local

```bash
fastapi run app/main.py
```

Luego abre [http://127.0.0.1:8000](http://127.0.0.1:8000) para usar la interfaz web.

## Instalación

### 1. Crear entorno e instalar dependencias

```bash
pip install -r requirements.txt
```

Si vas a usar GPU, instala la versión de `torch` compatible con tu entorno CUDA.

### 2. Verificar estructura de datos

Asegúrate de que existan las carpetas esperadas en `data/raw/` y, si ya procesaste el dataset, en `data/processed/`. Sino ejecutar el notebook `notebooks/01_preprocess_data.ipynb`

## Uso

El proyecto está organizado para usarse desde notebooks o desde scripts Python. Flujo recomendado:

1. Explorar datos con `notebooks/01_eda_d1.ipynb` y `notebooks/01_eda_d2.ipynb`.
2. Preparar el dataset con `notebooks/01_preprocess_data.ipynb`.
3. Entrenar modelos con `notebooks/02_model_v1.ipynb`, `02_model_v2.ipynb` o `02_model_v3.ipynb` según la versión que quieras reproducir.
4. Revisar resultados en `checkpoints/`.

## Versiones Del Modelo

Los notebooks `model_v#` representan etapas concretas del proyecto:

- `v1`: primer entrenamiento con el primer enfoque para la clasificación.
- `v2`: segundo enfoque para distinguir fractura vs no fractura.
- `v3`: versión final que combina ambos datasets para predecir entre fractura y no fractura.

## Archivos Generados

- Modelos guardados en `checkpoints/*.h5`.
- Historial de entrenamiento en `checkpoints/*.json`.
- Dataset procesado en `data/processed/dataset_fracturas/`.

## Dependencias

Las dependencias principales del proyecto son:

- `numpy`
- `pandas`
- `matplotlib`
- `pyyaml`
- `Pillow`
- `scipy`
- `scikit-image`
- `tqdm`
- `torch`
- `timm`

## Notas

- El nombre de algunos checkpoints usa el prefijo `radriografia_...` tal como aparece en el repositorio.
- Hay más de un dataset raw disponible; el pipeline puede adaptarse según la fuente utilizada en los notebooks.

