# c21928 — Clasificador de Contaminaciones en Línea de Producción Simulada

**Estudiante:** Fabricio Phillips Castro  
**Carné:** C21928  
**Curso:** IE0435 — Machine Learning  
**Universidad de Costa Rica | I Semestre 2026**

---

## Descripción

Sistema de clasificación binaria para detectar la presencia de granos de arroz (contaminaciones) en imágenes de una línea de producción simulada. Utiliza aprendizaje automático clásico sobre imágenes binarizadas de 128×128 píxeles.

- **Clase 1 (Positivo):** imagen contiene granos de arroz
- **Clase 0 (Negativo):** imagen no contiene arroz (puede tener otros objetos)

---

## Estructura del repositorio

```
├── README.md
├── DATASET.md
├── MODEL_CARD.md
├── LICENSE
├── requirements.txt
├── entrenar_imagenes.py       # Script principal de entrenamiento
├── c21928_Fabricio_Phillips_Castro.joblib   # Mejor modelo exportado
└── reports/
    └── informe_final.md       # Informe final con análisis comparativo
```

---

## Requisitos

```bash
pip install -r requirements.txt
```

Contenido de `requirements.txt`:
```
python>=3.9
scikit-learn>=1.2
Pillow>=9.0
numpy>=1.23
pandas>=1.5
joblib>=1.2
```

---

## Cómo correr el entrenamiento

### 1. Preparar las imágenes

Organiza tus imágenes en la siguiente estructura de carpetas:

```
MICARPETA/
├── Positivas/     ← imágenes CON arroz (jpg, jpeg, png)
└── Negativas/     ← imágenes SIN arroz (jpg, jpeg, png)
```

### 2. Ejecutar el script

```bash
python3 entrenar_imagenes.py ~/ruta/a/MICARPETA
```

El script hará lo siguiente automáticamente:
1. Lee y binariza todas las imágenes (escala de grises, 128×128, umbral 180)
2. Genera un archivo `dataset_MICARPETA.csv`
3. Entrena 4 modelos: KNN (k=3), SVM lineal, Naive Bayes, Árbol de Decisión
4. Evalúa cada modelo con 80/20 split estratificado
5. Exporta el mejor modelo como `modelo_MICARPETA.joblib`

**Ejemplo de salida esperada:**
```
Positivas encontradas: 15
Negativas encontradas: 15
Total: 30
Positivas: 15
Negativas: 15

CSV guardado: dataset_MICARPETA.csv

KNN
accuracy=0.8333
...

SVM
accuracy=1.0000
...

====================
Mejor modelo: SVM
Accuracy: 1.0000
Modelo guardado: modelo_MICARPETA.joblib
====================
```

---

## Cómo correr inferencia

```python
import joblib
import numpy as np
from PIL import Image

# Cargar modelo
model = joblib.load("c21928_Fabricio_Phillips_Castro.joblib")

# Preprocesar imagen nueva
img = Image.open("nueva_imagen.jpg").convert("L").resize((128, 128))
arr = np.array(img)
x = (arr > 180).astype(int).flatten().reshape(1, -1)

# Predecir
pred = model.predict(x)
resultado = "POSITIVO — contiene arroz" if pred[0] == 1 else "NEGATIVO — no contiene arroz"
print(resultado)
```

---

## Modelo seleccionado

| Propiedad | Valor |
|---|---|
| Algoritmo | SVM (kernel lineal) |
| Accuracy en prueba | Ver MODEL_CARD.md |
| Archivo | `c21928_Fabricio_Phillips_Castro.joblib` |

Ver [`MODEL_CARD.md`](MODEL_CARD.md) para detalles completos del modelo.

---

## Documentación adicional

- [`DATASET.md`](DATASET.md) — descripción del conjunto de datos, protocolo de recolección y limitaciones
- [`MODEL_CARD.md`](MODEL_CARD.md) — ficha técnica completa del modelo
- [`reports/informe_final.md`](reports/informe_final.md) — informe final con comparación de modelos

---

## Licencia

MIT License — ver [`LICENSE`](LICENSE)
