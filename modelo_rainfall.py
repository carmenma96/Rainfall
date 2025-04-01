import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Configuración
st.set_page_config(page_title="Predicción de Lluvia", layout="wide")
st.title("🌧️ Predicción de Lluvia")
shap.initjs()

# Cargar modelo
modelo = joblib.load("modelo_rainfall.pkl") 


# Preprocesamiento de datos 
def preprocesamiento(data):
    #Creación de nuevas columnas
    data['humidity x cloud']= data['humidity']*data['cloud']
    data['dewpoint x cloud']= data['dewpoint']*data['cloud']
    
    data['temp_diff']= data['maxtemp']-data['mintemp']
    
    data['pressure x temp_diff']= data['pressure']*data['temp_diff']
    
    # Categorizamos 2 columnas: cloud y sunshine
    data["sunshine_category"] = pd.cut(data["sunshine"], bins=3, labels=["Pocas", "Moderadas", "Muchas"])
    
    labels = ["Parcialmente nublado", "Nublado", "Muy nublado"]
    data["cloud_category"] = pd.cut(data["cloud"], bins=3, labels=labels)
    
    data = pd.get_dummies(data, columns=["cloud_category"], drop_first=True)
    data = pd.get_dummies(data, columns=["sunshine_category"],drop_first=False)
    data.drop(columns=["cloud_category_Nublado"], inplace=True)
    
    # Pasamos columnas a 1 o 0
    data['sunshine_category_Pocas'] = data['sunshine_category_Pocas'].apply(lambda x: 1 if x else 0)
    data['cloud_category_Muy nublado'] = data['cloud_category_Muy nublado'].apply(lambda x: 1 if x else 0)
    
    # Creamos más columnas con las categóricas
    data['muynublado x pocosol'] = data['cloud_category_Muy nublado'] * data['sunshine_category_Pocas']

    if 'id' in data.columns:
        data = data.drop(columns=['id'])

    return data



# CREACIÓN DE PESTAÑAS

tab1, tab2,  = st.tabs([
    "🧮 Predicción",
    "📊 Exploración",
])


# Cargar y preprocesar los datos de entrenamiento para obtener columnas reales
data = pd.read_csv("train.csv")
X = data.drop(columns=["rainfall"])
X = preprocesamiento(X) 
columnas = X.columns
dtypes = X.dtypes

#vamos a poner todo este bloque esto en en la tab 1 , pestaña de predicción

# ======================== PESTAÑA DE PREDICCIÓN ========================
with tab1:
    st.header("🌧️ Predicción de Lluvia en Tiempo Real")

    # Sidebar: captura de entradas del usuario
    st.sidebar.header("Ajusta los valores de entrada")
    
    humidity = st.sidebar.slider("Humedad", 0.0, 100.0, 50.0)
    cloud = st.sidebar.slider("Nubes", 0.0, 100.0, 50.0)
    dewpoint = st.sidebar.slider("Punto de rocío", -10.0, 30.0, 10.0)
    maxtemp = st.sidebar.slider("Temperatura máxima", -10.0, 50.0, 25.0)
    mintemp = st.sidebar.slider("Temperatura mínima", -10.0, 30.0, 15.0)
    pressure = st.sidebar.slider("Presión", 900.0, 1100.0, 1013.0)
    sunshine = st.sidebar.slider("Sol", 0.0, 15.0, 7.0)

    # Crear DataFrame base
    input_data = pd.DataFrame({
        "humidity": [humidity],
        "cloud": [cloud],
        "dewpoint": [dewpoint],
        "maxtemp": [maxtemp],
        "mintemp": [mintemp],
        "pressure": [pressure],
        "sunshine": [sunshine]
    })

    # Aplicar preprocesamiento
    input_data = preprocesamiento(input_data)

    # Reconstruir esquema del modelo usando train.csv
    data_entrenamiento = pd.read_csv("train.csv")
    X_entrenamiento = preprocesamiento(data_entrenamiento.drop(columns=["rainfall"]))
    columnas = X_entrenamiento.columns
    dtypes = X_entrenamiento.dtypes

    # Alinear el input con el modelo
    input_data = input_data.reindex(columns=columnas, fill_value=0)
    input_data = input_data.astype(dtypes.to_dict())

    # Predicción
    prediction = modelo.predict(input_data)
    probabilidad = modelo.predict_proba(input_data)[:, 1]

    # Mostrar resultados
    st.subheader("Resultado de la predicción")
    if prediction[0] == 1:
        st.write("🌧️ Lluvia pronosticada!")
    else:
        st.write("☀️ No se espera lluvia.")
    st.write(f"Probabilidad de lluvia: {probabilidad[0]:.2f}")

    # SHAP (si está habilitado)
    st.subheader("Importancia de las características")
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(input_data)
    shap.initjs()
    shap.summary_plot(shap_values, input_data)

    

# ======================== PESTAÑA DE EXPLORACIÓN ========================
with tab2:
    st.header("🔍 Exploración de Datos")

    st.subheader("Información General")
    st.write(data.describe())

    # Gráficos de dispersión
    st.subheader("Gráficos de dispersión")
    feature1 = st.selectbox("Selecciona la primera característica", data.columns)
    feature2 = st.selectbox("Selecciona la segunda característica", data.columns)
    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[feature1], y=data[feature2])
    st.pyplot(fig)

    # Matriz de correlación
    st.subheader("Correlación entre características")
    fig_corr = plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(fig_corr)

    # Histograma
    st.subheader("Distribución de características")
    feature = st.selectbox("Selecciona una característica para ver su distribución", data.columns)
    fig_dist = plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True)
    st.pyplot(fig_dist)

    # Boxplot
    st.subheader("Boxplot de características")
    feature_box = st.selectbox("Selecciona una característica para ver su boxplot", data.columns)
    fig_box = plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[feature_box])
    st.pyplot(fig_box)
