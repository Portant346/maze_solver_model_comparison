# Clasificación de señales del laberinto — modelos

Pipeline de entrenamiento, evaluación y comparación de modelos para clasificar
las imágenes que ve la cámara del robot en Gazebo (10 clases) y, más adelante,
guiar la navegación por el laberinto.

## Clases (10)

Señales (decisión de alto nivel): `ARROW_LEFT`, `ARROW_RIGHT`, `ARROW_UP`,
`CROSS` (dar la vuelta), `GOAL` (salida).
Percepción del entorno (control reactivo): `FREE_PATH`, `WALL`, `WALL_LEFT`,
`WALL_RIGHT`, `UNKNOWN`.

## Estructura

| Fichero        | Qué hace |
|----------------|----------|
| `data.py`      | DataLoaders (ImageFolder) + pesos de clase para el desbalanceo. Sin flip horizontal (rompería las clases direccionales). |
| `models.py`    | Registro de modelos: `cnn` (SimpleCNN, referencia), `mlp`, `rnn` (GRU bidireccional) y `transformer`. Misma firma de constructor → intercambiables. |
| `engine.py`    | Bucle de entrenamiento/evaluación, historial de estadísticas, checkpoint del mejor modelo. |
| `plots.py`     | Curvas loss/accuracy, matriz de confusión, comparativas del sweep. |
| `train.py`     | Entrenar UNA configuración y volcar todas sus estadísticas. |
| `evaluate.py`  | Evaluar un checkpoint sobre test sin reentrenar. |
| `sweep.py`     | Barrido de hiperparámetros de UN modelo (`lr × dropout`) + gráficas comparativas. |
| `compare_models.py` | Comparativa entre TIPOS de modelo con receta común + rankings (`test_acc`, F1, params, ms/img). |

Las salidas se guardan en `../runs/<nombre>/`.

## Uso

Desde esta carpeta (`Modelos/src`):

```bash
# Entrenar la CNN de referencia
python train.py --epochs 40 --run-name cnn_baseline

# Entrenar otro tipo de modelo (mlp/rnn/transformer)
python train.py --model transformer --data-dir ../dataset_unificado --epochs 30

# Comparar TIPOS de modelo entre sí (misma receta) -> runs/compare_<ts>/
python compare_models.py --data-dir ../dataset_unificado --epochs 30

# Probar combinaciones de hiperparámetros de UN modelo (edita GRID en sweep.py)
python sweep.py --model transformer --data-dir ../dataset_unificado --epochs 12

# Reevaluar un modelo ya entrenado
python evaluate.py --run-dir ../runs/cnn_baseline
```

### Comparativa de tipos de modelo (objetivo del TFG)

`compare_models.py` entrena cada arquitectura del registro con la **misma receta** (datos,
épocas, lr, optimizador, dropout, scheduler) para aislar la diferencia que aporta la
*familia* de modelo. Resultado de referencia (30 épocas sobre `dataset_unificado`):

| Modelo | test_acc | F1 macro | nº params | ms/img |
|---|---|---|---|---|
| **cnn** | **0.873** | **0.833** | 242.826 | 4.16 |
| rnn (GRU bidir.) | 0.808 | 0.757 | 767.498 | 14.34 |
| transformer | 0.782 | 0.741 | 460.554 | **1.34** |
| mlp | 0.644 | 0.599 | **24.711.946** | 3.70 |

La CNN gana (sesgo inductivo espacial); el MLP es el peor pese a ~100× más parámetros
(aplana y pierde la estructura 2D). Para comparar hiperparámetros *dentro* de un modelo,
usar `sweep.py --model <tipo>` (deja las gráficas en `runs/sweep_<modelo>_<ts>/`).

Argumentos útiles de `train.py`: `--lr`, `--batch-size`, `--optimizer`
{adam,adamw,sgd}, `--dropout`, `--base-channels`, `--weight-decay`,
`--no-augment`, `--no-class-weights`, `--img-h`, `--img-w`.

## Salidas por run

`config.json`, `history.csv` (loss/acc por época), `curves.png`,
`confusion_matrix.png`, `classification_report.txt` (P/R/F1 por clase),
`metrics.json` (resumen: best_val_acc, test_acc, F1 macro, ms/imagen, nº params)
y `best_model.pt`.

## Añadir un nuevo tipo de modelo

1. Define la clase en `models.py` con la firma
   `__init__(self, num_classes, in_channels=3, base_channels=32, dropout=0.3, img_h=..., img_w=..., **kwargs)`.
   `train.py` inyecta `base_channels`, `dropout`, `img_h` e `img_w`; cada modelo usa los
   que necesite (la CNN ignora `img_h/img_w`; MLP/RNN/transformer los usan para fijar el
   tamaño de entrada). El `**kwargs` final absorbe lo que no use.
2. Regístrala en `MODEL_REGISTRY`.
3. Entrénala con `python train.py --model <nombre>` y compárala con
   `compare_models.py` / `sweep.py`.

El motor, las estadísticas, el sweep y la comparativa funcionan igual para cualquier modelo.
⚠️ Evita `nn.LazyLinear`: el optimizador se crea en `train.py` **antes** del primer
forward, así que los parámetros perezosos no se registrarían. Usa `img_h/img_w` para
dimensionar las capas explícitamente.

## Notas

- **CPU**: torch instalado es build CPU. Para usar la RTX 3060 hay que
  reinstalar la build CUDA (`pip install torch torchvision --index-url
  https://download.pytorch.org/whl/cu121`). El código detecta GPU solo.
- **Aumentos**: no se usa volteo horizontal porque cambiaría el significado
  de las clases direccionales. Revisar si los `_flip` del dataset original
  no contaminaron `ARROW_*`/`WALL_*`.
