# MODEL CARD — Clasificador de Contaminaciones en Línea de Producción Simulada
**Model name:** `contaminacion_gaussiannb_v1`  
**Version:** 1.0  
**Archivo exportado:** `modelo_persona3.joblib`  
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
| Algoritmo | Gaussian Naive Bayes (GNB) |
| `var_smoothing` | 1×10⁻⁹ |
| Priors | Estimados desde los datos (`None`) |
| Clases | `0` = Negativo, `1` = Positivo (arroz) |
| Dimensión de entrada | 16 384 características (128×128 píxeles binarizados) |
| Muestras de entrenamiento | 24 (12 negativas + 12 positivas, clases balanceadas) |

---

## 3. Data Summary

**Protocolo de recolección:**  
Cada integrante del grupo fotografió una hoja blanca bajo diferentes condiciones:
- **Positivas (clase 1):** hoja con granos de arroz dispersos sobre ella
- **Negativas (clase 0):** hoja sin arroz (puede contener otros objetos como aros o clips)

**Aporte por persona:** 30 imágenes (15 positivas + 15 negativas).  
**Dataset utilizado para este modelo:** ~30 imágenes totales (24 entrenamiento + 6 prueba), perfectamente balanceado entre clases.

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
**Balance de clases:** el conjunto de entrenamiento resultó perfectamente balanceado (12 muestras por clase).  
**Consistencia:** posible variación inter-etiquetador en casos límite.

---

## 5. Metrics

**Protocolo de evaluación:**
- División entrenamiento/prueba: 80% / 20% (`train_test_split`, `stratify=y`, `random_state=42`)
- Conjunto de entrenamiento: 24 muestras (12 por clase)
- Conjunto de prueba: ~6 muestras
- Métricas calculadas sobre el conjunto de prueba

**Métricas reportadas:**

| Métrica | Clase Negativo | Clase Positivo | Promedio ponderado |
|---|---|---|---|
| Accuracy | — | — | Mejor entre los 4 modelos evaluados |
| Precision | ver reporte | ver reporte | ver reporte |
| Recall | ver reporte | ver reporte | ver reporte |
| F1-Score | ver reporte | ver reporte | ver reporte |

> **Nota:** Gaussian Naive Bayes fue seleccionado como mejor modelo tras comparar KNN, SVM lineal, Naive Bayes y Árbol de Decisión. La selección se basó en `accuracy_score` sobre el conjunto de prueba.

> **Supuesto clave de GNB:** se asume independencia condicional entre características (píxeles) dada la clase, y distribución Gaussiana por característica. Estos supuestos son aproximaciones en imágenes reales.

**Herramientas:** `sklearn.metrics.accuracy_score`, `classification_report`, `confusion_matrix`

---

## 6. Ethical / Safety Notes

- **Supuesto de independencia:** Naive Bayes asume independencia entre píxeles, lo cual es incorrecto para imágenes (píxeles vecinos están correlacionados). Sin embargo, puede funcionar bien con datos pequeños.
- **Sesgo por iluminación:** el umbral fijo de 180 es sensible a condiciones de iluminación extremas.
- **Sesgo de fondo:** asume fondo blanco uniforme; manchas o sombras se codifican como objetos.
- **Sesgo de cámara:** distintos sensores producen niveles de ruido diferentes que afectan la binarización.
- **Uso responsable:** prototipo exclusivamente académico, no apto para producción real.

---

## 7. Limitations

- **Supuesto Gaussiano:** los datos de entrada son binarios (0 o 1), no continuos; el supuesto de distribución Gaussiana es una aproximación.
- **Supuesto de independencia:** la correlación espacial entre píxeles viola el supuesto de independencia de Naive Bayes.
- **Dataset reducido:** con 24 muestras de entrenamiento, la estimación de media y varianza por característica puede ser inestable.
- **Objetos pequeños:** granos de arroz muy pequeños generan muy pocos píxeles = 0.
- **Sin información espacial:** al aplanar la imagen se pierde la estructura 2D.
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
python3 entrenar_imagenes.py ~/ruta/a/carpeta/persona3
```
La carpeta debe contener subcarpetas `Positivas/` y `Negativas/` con imágenes `.jpg`, `.jpeg` o `.png`.

**Inferencia sobre una imagen nueva:**
```python
import joblib
import numpy as np
from PIL import Image

model = joblib.load("modelo_persona3.joblib")

img = Image.open("nueva_imagen.jpg").convert("L").resize((128, 128))
arr = np.array(img)
x = (arr > 180).astype(int).flatten().reshape(1, -1)

pred = model.predict(x)
print("Positivo (arroz)" if pred[0] == 1 else "Negativo")
```

**Hardware utilizado:** computadora personal de escritorio/laptop; CPU estándar (no se requiere GPU).  
**Sistema operativo:** Linux / macOS / Windows con Python 3.9+  
**Tiempo de entrenamiento estimado:** < 5 segundos en dataset de ~30 imágenes (GNB es uno de los modelos más rápidos de entrenar).
