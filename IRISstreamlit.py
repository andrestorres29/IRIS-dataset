# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 22:40:07 2024

@author: torres
"""
import numpy as np
import pandas as pd
import streamlit as st 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from PIL import Image

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household] 


modeloLR = joblib.load('IRISpipeline.sav')


def main():
    st.title("Clasificacion por medio Logistic regression de IRIS")
    col1, col2, col3 = st.columns(3)
    with col1:
        image = Image.open('logo.png')
        st.image(image, use_column_width=True)
        st.write('### Irvin A. Torres')
        st.write('### Matricula: 315463')
        st.write('### Machine Learning MIC')
    with col2:
        sepal_length = st.number_input('Longitud del sépalo (cm)', min_value=0.0, max_value=10.0, value=5.0)
        sepal_width = st.number_input('Ancho del sépalo (cm)', min_value=0.0, max_value=10.0, value=3.0)
    with col3:
        petal_length = st.number_input('Longitud del pétalo (cm)', min_value=0.0, max_value=10.0, value=4.0)
        petal_width = st.number_input('Ancho del pétalo (cm)', min_value=0.0, max_value=10.0, value=1.0)
    if st.button("Predecir tipo de IRIS"):
        features = [[sepal_length, sepal_width, petal_length, petal_width]]
        data = {"Longitud del sépalo":float(sepal_length),"Ancho del sépalo":float(sepal_width),"Longitud del pétalo":float(petal_length),
                "Ancho del pétalo ":float(petal_width)}
        df = pd.DataFrame([list(data.values())],columns=["Longitud del sépalo","Ancho del sépalo ","Longitud del pétalo","Ancho del pétalo"])
        df_prepared = pipeline.transform(df)
        prediction = modeloLR.predict(df_prepared)
        output = float(prediction[0])
        st.success('Prediction is {}'.format(output))
        
if __name__=='__main__': 
    main()
