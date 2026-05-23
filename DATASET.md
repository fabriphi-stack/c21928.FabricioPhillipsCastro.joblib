# DATASET.md — Conjunto de Datos: Contaminaciones en Línea de Producción Simulada

**Proyecto:** IE0435 Proyecto 1 — I Semestre 2026  
**Universidad de Costa Rica**

---

## Descripción general

El conjunto de datos consiste en imágenes fotográficas de una hoja blanca sobre una superficie plana, simulando una línea de producción. El objetivo es clasificar si la imagen contiene granos de arroz (contaminación positiva) o no.

---

## Protocolo de recolección

**Escenario simulado:**
- Superficie: hoja de papel blanco tamaño carta o A4
- Contaminantes: granos de arroz (clase positiva)
- Objetos neutros: aros, clips, o ningún objeto (clase negativa)

**Proceso:**
1. El estudiante coloca la hoja blanca sobre una superficie plana
2. Para imágenes positivas: esparce granos de arroz sobre la hoja
3. Para imágenes negativas: deja la hoja vacía o coloca aros/clips
4. Captura la fotografía desde arriba con cámara de celular

**Aporte individual:** 30 imágenes por estudiante (15 positivas + 15 negativas)

---

## Composición del dataset

| Conjunto | Imágenes positivas | Imágenes negativas | Total |
|---|---|---|---|
| PERSONAL (este trabajo) | 15 | 15 | 30 |
| Grupo completo | Variable | Variable | ~90+ |

---

## Preprocesamiento

Todas las imágenes son transformadas con el siguiente pipeline antes de ser usadas como features:

```
Imagen original (cualquier resolución, color)
    ↓
Conversión a escala de grises (PIL .convert("L"))
    ↓
Redimensionado a 128×128 píxeles (PIL .resize((128,128)))
    ↓
Umbralización binaria: pixel = 1 si valor > 180, else 0
    ↓
Aplanado a vector de 16,384 elementos
    ↓
Guardado en CSV con etiqueta (última columna: 0 o 1)
```

**Justificación del umbral 180:** un píxel con valor >180 en escala de grises se considera "blanco" (fondo); cualquier valor ≤180 indica presencia de un objeto oscuro (arroz u otro).

---

## Variaciones de adquisición

Las imágenes del grupo presentan variaciones en:

| Factor | Variación observada |
|---|---|
| Cámara | Distintos modelos de celular (diferentes sensores, resoluciones) |
| Iluminación | Luz natural (ventana), luz artificial (LED, fluorescente), mixta |
| Ángulo | Mayormente cenital (desde arriba), con leve variación angular |
| Distancia | Varía entre ~20–50 cm de la hoja |
| Fondo | Siempre hoja blanca, pero con variaciones de sombra y textura |
| Cantidad de arroz | Variable en positivas (de 1 grano a varios) |

---

## Etiquetado

- **Método:** estructura de carpetas (`/Positivas/`, `/Negativas/`)
- **Criterio positivo:** al menos un grano de arroz visible en la imagen
- **Criterio negativo:** ningún grano de arroz (puede haber otros objetos)
- **Responsable:** cada estudiante etiqueta sus propias imágenes

---

## Limitaciones y sesgos conocidos

1. **Dataset pequeño:** ~30 imágenes por persona es insuficiente para entrenamiento robusto en un espacio de 16,384 dimensiones.

2. **Sesgo de iluminación:** el umbral fijo de 180 no se adapta a condiciones de iluminación extremas (muy oscuro o sobreexpuesto).

3. **Sesgo de cámara:** distintos sensores de celular producen imágenes con diferentes niveles de ruido, nitidez y balance de blancos.

4. **Variabilidad del fondo:** sombras, dobleces o manchas en la hoja blanca generan falsos negativos (píxeles oscuros en el fondo).

5. **Variabilidad de la cantidad:** una imagen con 1 grano de arroz y otra con 20 tienen la misma etiqueta (1), pero featuresmuymuy distintas.

6. **Sesgo de posición:** el arroz puede estar en cualquier parte de la imagen; al aplanar se pierde información espacial.

7. **Sin validación cruzada entre etiquetadores:** no se verificó consistencia entre distintos estudiantes en casos ambiguos.

---

## Formato del CSV generado

Cada fila corresponde a una imagen:
- **Columnas 0–16383:** valores binarios (0 o 1) del píxel en posición (i,j) donde j varía más rápido
- **Columna 16384:** etiqueta (`1` = positivo/arroz, `0` = negativo)

Sin encabezados (header=False en la exportación).

```
1,1,1,0,1,...,1,0,1,1,1,  ← 16,384 valores de píxel ← etiqueta (1 o 0)
```

---

## Reproducibilidad

Para regenerar el dataset desde las imágenes originales:

```bash
python3 entrenar_imagenes.py ~/ruta/a/MICARPETA
```

Esto genera `dataset_MICARPETA.csv` con el formato descrito arriba.
