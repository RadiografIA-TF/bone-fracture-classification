# Bone Fracture Classification

Un proyecto de **clasificación de fracturas óseas** mediante modelos de aprendizaje automático utilizando radiografías del dataset [Bone Break Classification Image Dataset](https://www.kaggle.com/datasets/pkdarabi/bone-break-classification-image-dataset) de Kaggle.

## Descripción

Este proyecto implementará un sistema de clasificación de imágenes médicas (radiografías) para detectar diferentes tipos de fracturas óseas. Utiliza técnicas de procesamiento de imágenes y modelos de deep learning para automatizar el análisis de radiografías.

### Dataset
- **Fuente:** Kaggle - Bone Break Classification Image Dataset
- **Contenido:** Radiografías categorizadas por tipo de fractura
- **Licencia:** Consultar página del dataset en Kaggle

## Inicio Rápido

### 1. Clonar el Repositorio

```bash
git clone https://github.com/AlexC0dex/bone-fracture-classification.git
cd bone-fracture-classification
```

### 2. Instalar Dependencias

#### Opción A: Usando PIP

```bash
pip install -r requirements.txt
```

#### Opción B: Usando CONDA

```bash
conda install -y numpy pandas matplotlib pyyaml pillow scipy scikit-image
```

### 3. Ejecutar el Análisis Exploratorio

Abre el notebook del análisis exploratorio

## 📁 Estructura del Proyecto

```
bone-fracture-classification/
│
├── README.md                    # Este archivo
├── LICENSE                      # Licencia del proyecto
├── requirements.txt             # Dependencias del proyecto
│
├── config/
│   └── config.yaml             # Configuración del proyecto
│
├── data/
│   ├── raw/                    # Datos sin procesar
│   │   └── dataset_fracturas/  # Dataset descargado de Kaggle
│   └── processed/              # Datos procesados
│
└── notebooks/
    └── 01_eda.ipynb            # Análisis Exploratorio de Datos (EDA)
```

### Descripción de Carpetas

- **`config/`** - Archivos de configuración (rutas, parámetros, etc.)
- **`data/`** - Almacenamiento de datos
  - `raw/` - Dataset original de Kaggle
  - `processed/` - Datos preprocesados y transformados
- **`notebooks/`** - Jupyter notebooks con análisis y experimentos
  - `01_eda.ipynb` - Exploración inicial del dataset

## Dependencias

| Librería | Versión | Uso |
|----------|---------|-----|
| numpy | >=1.24.0 | Operaciones numéricas |
| pandas | >=2.0.0 | Manipulación de datos |
| matplotlib | >=3.7.0 | Visualización |
| pyyaml | >=6.0 | Lectura de configuración |
| Pillow | >=10.0.0 | Procesamiento de imágenes |
| scipy | >=1.11.0 | Funciones científicas |
| scikit-image | >=0.21.0 | Procesamiento avanzado de imágenes |