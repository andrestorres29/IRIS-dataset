# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 22:40:07 2024

@author: torres
"""
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import joblib

# Cargar el modelo
modeloLR = joblib.load('IRISpipeline.sav')

def main():
    st.title("Clasificación de IRIS mediante Regresión Logística")
    
    # Crear columnas para la imagen y los datos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        image = Image.open('logo.png')
        st.image(image, use_column_width=True)
        st.write('### Irvin A. Torres')
        st.write('### Matrícula: 315463')
        st.write('### Machine Learning MIC')
    
    with col2:
        sepal_length = st.number_input('Longitud del sépalo (cm)', min_value=0.0, max_value=10.0, value=5.0)
        sepal_width = st.number_input('Ancho del sépalo (cm)', min_value=0.0, max_value=10.0, value=3.0)
    
    with col3:
        petal_length = st.number_input('Longitud del pétalo (cm)', min_value=0.0, max_value=10.0, value=4.0)
        petal_width = st.number_input('Ancho del pétalo (cm)', min_value=0.0, max_value=10.0, value=1.0)
    
    if st.button("Predecir tipo de IRIS"):
        features = [[sepal_length, sepal_width, petal_length, petal_width]]
        
        # Crear DataFrame para la predicción
        df = pd.DataFrame(features, columns=["Longitud del sépalo", "Ancho del sépalo", "Longitud del pétalo", "Ancho del pétalo"])
        
        # Usar la pipeline para hacer predicción
        prediction = modeloLR.predict(df)
        
        # Obtener el nombre de la clase si el modelo proporciona etiquetas
        class_names = ['setosa', 'versicolor', 'virginica']  # Asegúrate de que estos nombres coincidan con tu modelo
        output = class_names[int(prediction[0])]
        
        st.success(f'Predicción: {output}')

if __name__ == '__main__': 
    main()
