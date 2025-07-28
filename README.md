# 🌸 Clasificación de flores con el Dataset IRIS

Este proyecto utiliza el clásico dataset **IRIS** para entrenar un modelo de clasificación supervisada usando técnicas de machine learning.  
Fue desarrollado como parte del aprendizaje de conceptos fundamentales de modelos, pipelines y despliegue básico con Streamlit.

---

## 🎯 Objetivo

Clasificar correctamente la especie de flor (Iris-setosa, Iris-versicolor o Iris-virginica) con base en las siguientes características:
- Largo y ancho del pétalo
- Largo y ancho del sépalo

---

## 📁 Contenido del repositorio

| Archivo                  | Descripción                                                  |
|--------------------------|--------------------------------------------------------------|
| `IRISmachinelearnig.ipynb` | Notebook con el análisis exploratorio, entrenamiento y evaluación |
| `IRISpipeline.sav`       | Modelo serializado listo para desplegar                      |
| `IRISstreamlit.py`       | App web básica para hacer predicciones con Streamlit         |
| `requirements.txt`       | Librerías necesarias                                          |
| `logo.png`               | Imagen decorativa o para la app (opcional)                   |

---

## 🛠 Tecnologías utilizadas

- Python
- Pandas / Scikit-learn / NumPy / Matplotlib
- Streamlit (para la interfaz de predicción)
- Joblib (serialización del pipeline)

---

## Resultados obtenidos

Precisión del modelo: 97% en validación.

Se usó un pipeline con preprocesamiento y clasificación integrada.

El modelo es capaz de predecir la especie de la flor con base en 4 características numéricas.

---
## 🚀 Cómo ejecutar el proyecto

### 1. Clonar el repositorio
```bash
git clone https://github.com/andrestorres29/IRIS-dataset.git
cd IRIS-dataset
