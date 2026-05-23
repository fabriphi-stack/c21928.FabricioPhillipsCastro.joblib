# MODEL CARD — Clasificador de Contaminaciones en Línea de Producción Simulada
**Model name:** `contaminacion_svm_linear_v1`  
**Version:** 1.0  
**Archivo exportado:** `c21928_Fabricio_Phillips_Castro.joblib`  
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
| Algoritmo | Support Vector Machine (SVM) |
| Kernel | Lineal (`linear`) |
| C (regularización) | 1.0 |
| `random_state` | 42 |
| Clases | `0` = Negativo, `1` = Positivo (arroz) |
| Dimensión de entrada | 16 384 características (128×128 píxeles binarizados) |
| Vectores de soporte | 41 (17 negativos, 24 positivos) |

---

## 3. Data Summary

**Protocolo de recolección:**  
Cada integrante del grupo fotografió una hoja blanca bajo diferentes condiciones:
- **Positivas (clase 1):** hoja con granos de arroz dispersos sobre ella
- **Negativas (clase 0):** hoja sin arroz (puede contener otros objetos como aros o clips)

**Aporte por persona:** 30 imágenes (15 positivas + 15 negativas).  
**Dataset utilizado para este modelo:** datos combinados del grupo (~50–60 imágenes totales).

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
**Calidad:** etiquetado realizado por cada estudiante sobre sus propias imágenes, revisado colaborativamente.  
**Consistencia:** posible variación inter-etiquetador en casos límite (granos muy pequeños, parcialmente fuera de cuadro o solapados con el borde de la hoja).

---

## 5. Metrics

**Protocolo de evaluación:**
- División entrenamiento/prueba: 80% / 20% (`train_test_split`, `stratify=y`, `random_state=42`)
- Métricas calculadas sobre el conjunto de prueba

**Métricas reportadas:**

| Métrica | Clase Negativo | Clase Positivo | Promedio ponderado |
|---|---|---|---|
| Accuracy | — | — | Mejor entre los 4 modelos evaluados |
| Precision | ver reporte | ver reporte | ver reporte |
| Recall | ver reporte | ver reporte | ver reporte |
| F1-Score | ver reporte | ver reporte | ver reporte |

> **Nota:** El SVM lineal fue seleccionado como mejor modelo tras comparar KNN (k=3), SVM lineal, Naive Bayes y Árbol de Decisión sobre el mismo dataset. La selección se basó en `accuracy_score` sobre el conjunto de prueba.

**Herramientas:** `sklearn.metrics.accuracy_score`, `classification_report`, `confusion_matrix`

---

## 6. Ethical / Safety Notes

- **Sesgo por iluminación:** el umbral fijo de 180 puede clasificar erróneamente imágenes capturadas con iluminación muy baja (píxeles oscuros en el fondo) o muy alta (objetos sobreexpuestos que parecen blancos).
- **Sesgo por cámara:** diferentes sensores producen imágenes con distintos niveles de ruido y nitidez, lo que afecta la binarización.
- **Sesgo de fondo:** el modelo asume un fondo blanco uniforme; cualquier mancha o sombra sobre la hoja puede interpretarse como objeto.
- **Uso responsable:** este prototipo es exclusivamente académico. No debe usarse para decisiones de calidad en producción real sin validación extensiva y reentrenamiento con datos industriales.

---

## 7. Limitations

- **Objetos pequeños:** granos de arroz muy pequeños o fragmentados generan muy pocos píxeles = 0, dificultando la detección.
- **Oclusión:** granos solapados entre sí o con el borde de la hoja pueden no generar suficiente contraste.
- **Blur / desenfoque:** imágenes borrosas suavizan los bordes, reduciendo la cantidad de píxeles binarizados a 0.
- **Tamaño del dataset:** el conjunto de datos es reducido (~30–60 imágenes), lo que limita la capacidad de generalización.
- **Dimensionalidad alta:** 16 384 características para pocas muestras puede inducir sobreajuste en algunos modelos.
- **Sin información espacial:** al aplanar la imagen se pierde la estructura 2D; el modelo no "sabe" dónde están los objetos.
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
python3 entrenar_imagenes.py ~/ruta/a/carpeta/PERSONAL
```
La carpeta debe contener subcarpetas `Positivas/` y `Negativas/` con imágenes `.jpg`, `.jpeg` o `.png`.

**Inferencia sobre una imagen nueva:**
```python
import joblib
import numpy as np
from PIL import Image

model = joblib.load("c21928_Fabricio_Phillips_Castro.joblib")

img = Image.open("nueva_imagen.jpg").convert("L").resize((128, 128))
arr = np.array(img)
x = (arr > 180).astype(int).flatten().reshape(1, -1)

pred = model.predict(x)
print("Positivo (arroz)" if pred[0] == 1 else "Negativo")
```

**Hardware utilizado:** computadora personal de escritorio/laptop; CPU estándar (no se requiere GPU).  
**Sistema operativo:** Linux / macOS / Windows con Python 3.9+  
**Tiempo de entrenamiento estimado:** < 60 segundos en dataset de ~60 imágenes.
