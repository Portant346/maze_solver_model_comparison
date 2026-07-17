# Repositorio del TFG — guía rápida

Evaluación del rendimiento de modelos de clasificación de imágenes en tiempo real
para toma de decisiones en robots móviles.

Este repositorio contiene **el código** del Trabajo Fin de Grado.

## Contenido del repositorio

```
.
├── README.md                    # documentación completa del proyecto
├── requirements.txt             # dependencias de Python
├── .gitignore
├── Modelos/
│   ├── src/                     # código de entrenamiento y comparación de modelos
│   └── dataset_unificado/       # conjunto de datos (10 clases: train/val/test)
├── Robot/                       # navegación en Gazebo, inferencia, mapas y tests
├── MundosPrueba/                # mundos .sdf de prueba (3)
├── MundosTest/                  # mundos .sdf de test (5)
└── memoria/scripts/             # generación de las figuras de la memoria
```

## Instalación

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate    |  Linux/Mac:  source .venv/bin/activate
pip install -r requirements.txt
```

La parte de **navegación** (`Robot/`) requiere además **Gazebo** con sus bindings de
Python (`gz.transport13`, `gz.msgs10`), que no se instalan con pip. El desarrollo se
hizo sobre **WSL2**.

## Uso rápido

Entrenar un modelo y comparar las familias (desde `Modelos/src/`):

```bash
python train.py --model cnn --data-dir ../dataset_unificado --epochs 40
python compare_models.py --data-dir ../dataset_unificado --epochs 30
```

Navegar con un modelo en Gazebo (desde `Robot/`, en WSL2 con un mundo abierto):

```bash
bash run_navigate.sh --run-dir ../Modelos/runs/cnn_unificado --show
```

## Ficheros grandes NO incluidos en el repositorio

Para mantener el repositorio ligero, **no** se versionan (ver `.gitignore`):

- **`Modelos/runs/`** — pesos y salidas de entrenamiento (`best_model.pt`, ~2 GB).
- **`logs/`** — registros de navegación.
- Datasets intermedios (`imagenes/`, `dataset/`, `dataset_nuevo/`, `dataset_arrows/`).

Los **pesos de los modelos entrenados** se publican aparte como
**GitHub Release** (o enlace externo). Para reproducir el entrenamiento desde cero
basta con `Modelos/dataset_unificado/`, que sí está incluido.
