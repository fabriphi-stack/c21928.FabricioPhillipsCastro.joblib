# MODEL CARD — Clasificador de Contaminaciones en Línea de Producción Simulada
**Model name:** `contaminacion_knn_k3_v1`  
**Version:** 1.0  
**Archivo exportado:** `modelo_persona2.joblib`  
**Fecha de entrenamiento:** Mayo 2026  
**Framework:** scikit-learn 1.x / Python 3.x

---

## 1. Intended Use

**Uso previsto:**  
Clasificar imágenes binarias (128×128 px) de una línea de producción simulada para determinar si contienen granos de arroz (contaminación positiva) o no (negativo). El sistema es parte de un prototipo académico de inspección visual automática.

**Casos dentro del alcance:**
- Imágenes en escala de grises de una hoja blanca como fondo
- Presencia o ausencia de granos de arroz como objeto de interés

**Fuera del alcance (out-of-scope):**
- Detección de múltiples tipos de contaminantes simultáneos
- Imágenes a color o con fondos distintos a hoja blanca
- Aplicación en líneas de producción reales sin re-entrenamiento
- Detección de posición o cantidad exacta de granos (solo clasificación binaria)

---

## 2. Arquitectura del Modelo

| Parámetro | Valor |
|---|---|
| Algoritmo | K-Nearest Neighbors (KNN) |
| k (vecinos) | 3 |
| Métrica de distancia | Minkowski (p=2, equivalente a Euclidiana) |
| Ponderación | Uniforme (`uniform`) |
| Algoritmo de búsqueda | Auto (`auto`) |
| Clases | `0` = Negativo, `1` = Positivo (arroz) |
| Dimensión de entrada | 16 384 características (128×128 píxeles binarizados) |
| Muestras de entrenamiento | 21 |

---

## 3. Data Summary

**Protocolo de recolección:**  
Cada integrante del grupo fotografió una hoja blanca bajo diferentes condiciones:
- **Positivas (clase 1):** hoja con granos de arroz dispersos sobre ella
- **Negativas (clase 0):** hoja sin arroz (puede contener otros objetos como aros o clips)

**Aporte por persona:** 30 imágenes (15 positivas + 15 negativas).  
**Dataset utilizado para este modelo:** ~26 imágenes totales (21 entrenamiento + 5 prueba).

**Preprocesamiento aplicado:**
1. Conversión a escala de grises (`PIL.Image.convert("L")`)
2. Redimensionado a 128×128 píxeles (`img.resize((128,128))`)
3. Umbralización binaria: píxel = 1 si valor > 180 (blanco), píxel = 0 si ≤ 180 (objeto)
4. Aplanado a vector de 16 384 elementos

**Variaciones de adquisición registradas:**
- Diferentes cámaras de celular (distintas resoluciones y balances de color)
- Variación en la intensidad de iluminación (luz natural vs. artificial)
- Distintos ángulos y distancias de captura
- Ligeras diferencias en el tono blanco del fondo (sombras, reflexiones)

---

## 4. Labeling Process

**Herramienta:** Etiquetado manual mediante estructura de carpetas (`/Positivas/` y `/Negativas/`).  
**Criterio de etiqueta positiva:** presencia visible de al menos un grano de arroz en la imagen.  
**Criterio de etiqueta negativa:** ausencia de arroz (fondo vacío, o con otros objetos como clips/aros).  
**Calidad:** etiquetado realizado por cada estudiante sobre sus propias imágenes.  
**Consistencia:** posible variación inter-etiquetador en casos límite (granos muy pequeños o parcialmente ocultos).

---

## 5. Metrics

**Protocolo de evaluación:**
- División entrenamiento/prueba: 80% / 20% (`train_test_split`, `stratify=y`, `random_state=42`)
- Conjunto de prueba: ~5 imágenes
- Métricas calculadas sobre el conjunto de prueba

**Métricas reportadas:**

| Métrica | Clase Negativo | Clase Positivo | Promedio ponderado |
|---|---|---|---|
| Accuracy | — | — | Mejor entre los 4 modelos evaluados |
| Precision | ver reporte | ver reporte | ver reporte |
| Recall | ver reporte | ver reporte | ver reporte |
| F1-Score | ver reporte | ver reporte | ver reporte |

> **Nota:** KNN con k=3 fue seleccionado como mejor modelo tras comparar KNN, SVM lineal, Naive Bayes y Árbol de Decisión sobre el mismo dataset. La selección se basó en `accuracy_score` sobre el conjunto de prueba.

> **Advertencia:** con solo ~5 muestras de prueba, las métricas tienen alta varianza y baja significancia estadística.

**Herramientas:** `sklearn.metrics.accuracy_score`, `classification_report`, `confusion_matrix`

---

## 6. Ethical / Safety Notes

- **Sensibilidad al umbral:** el umbral fijo de 180 puede clasificar erróneamente imágenes capturadas con iluminación muy baja o muy alta.
- **Sensibilidad a la distancia Euclidiana:** KNN en espacios de alta dimensión (16 384 dimensiones) es susceptible a la "maldición de la dimensionalidad"; pequeñas variaciones de iluminación entre imágenes pueden alterar significativamente las distancias calculadas.
- **Sesgo de fondo:** el modelo asume un fondo blanco uniforme; manchas o sombras pueden interpretarse como objeto.
- **Uso responsable:** prototipo exclusivamente académico, no apto para producción real.

---

## 7. Limitations

- **Dataset muy reducido:** con solo 21 muestras de entrenamiento, el modelo tiene capacidad de generalización limitada.
- **Maldición de la dimensionalidad:** KNN es menos efectivo en espacios con muchas características (16 384) y pocas muestras.
- **Costo de inferencia:** KNN almacena todo el conjunto de entrenamiento; la predicción requiere calcular distancias con todas las muestras.
- **Objetos pequeños:** granos de arroz muy pequeños generan muy pocos píxeles = 0.
- **Sin información espacial:** al aplanar la imagen se pierde la estructura 2D.
- **Evaluación poco confiable:** el conjunto de prueba de ~5 muestras no es representativo estadísticamente.
- **Clases binarias únicamente:** no distingue entre tipos de contaminantes.

---

## 8. Reproducibility

**Requisitos:**
```
python>=3.9
scikit-learn>=1.2
Pillow>=9.0
numpy>=1.23
pandas>=1.5
joblib>=1.2
```

**Instalación:**
```bash
pip install -r requirements.txt
```

**Entrenamiento:**
```bash
python3 entrenar_imagenes.py ~/ruta/a/carpeta/persona2
```
La carpeta debe contener subcarpetas `Positivas/` y `Negativas/` con imágenes `.jpg`, `.jpeg` o `.png`.

**Inferencia sobre una imagen nueva:**
```python
import joblib
import numpy as np
from PIL import Image

model = joblib.load("modelo_persona2.joblib")

img = Image.open("nueva_imagen.jpg").convert("L").resize((128, 128))
arr = np.array(img)
x = (arr > 180).astype(int).flatten().reshape(1, -1)

pred = model.predict(x)
print("Positivo (arroz)" if pred[0] == 1 else "Negativo")
```

**Hardware utilizado:** computadora personal de escritorio/laptop; CPU estándar (no se requiere GPU).  
**Sistema operativo:** Linux / macOS / Windows con Python 3.9+  
**Tiempo de entrenamiento estimado:** < 10 segundos en dataset de ~26 imágenes (KNN no entrena iterativamente, solo almacena datos).
