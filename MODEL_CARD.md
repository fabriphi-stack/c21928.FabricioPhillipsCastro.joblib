# Model Card

## Model name + version

Contamination Classifier SVM v1

---

## Intended use / out-of-scope

Uso:
Detección de contaminaciones (granos de arroz) en imágenes con fondo blanco.

Fuera de alcance:

* Fondos distintos
* Iluminación extrema
* Objetos similares al arroz

---

## Data summary

Dataset de 60 imágenes:

* 30 positivas
* 30 negativas

Variaciones en iluminación y posición.

---

## Labeling process

Etiquetado manual:

* 1 → presencia de arroz
* 0 → ausencia de arroz

Alta consistencia, baja ambigüedad.

---

## Metrics

Métrica: Accuracy
Split: 80% entrenamiento / 20% prueba

Resultados:

* SVM: 1.00
* Naive Bayes: 1.00
* Decision Tree: 0.83
* KNN: 0.66

---

## Ethical / safety notes

* Sensible a cambios de iluminación
* Puede confundir objetos similares
* No robusto fuera del entorno controlado

---

## Limitations

* Dataset pequeño
* Posible sobreajuste
* Sensible a blur y ruido
* No generaliza a escenarios complejos

---

## Reproducibility

Comandos:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python pipeline.py
```

Hardware:

* CPU (laptop personal)
* Sin GPU
