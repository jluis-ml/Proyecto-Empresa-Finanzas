import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor

# Título de la aplicación
st.title("Predicción con ExtraTreesRegressor")
st.write("Esta aplicación permite realizar predicciones ingresando datos manualmente.")

# Configurar el estilo de Seaborn
sns.set(style="whitegrid")

# Cargar el modelo entrenado
with open('modelo.pkl', 'rb') as f:
    model = pickle.load(f)

# Panel lateral para métricas del modelo
st.sidebar.header("Métricas del Modelo")
st.sidebar.write("Ajusta las métricas manualmente según tu evaluación.")
mse = st.sidebar.number_input("MSE (Error Cuadrático Medio)", value=0.0, format="%.4f")
r2 = st.sidebar.number_input("R² Score", value=0.0, format="%.4f")

# Predicción de múltiples filas manualmente
st.header("Predicción Manual")
st.write("Ingresa los valores correspondientes para las características de cada fila:")

# Número de características (en tu caso, 3 características)
num_features = 3  # Actualizado a 3

# Crear inputs dinámicos para las características
inputs = []
for i in range(num_features):
    value = st.number_input(f"Feature {i + 1}", step=0.1, format="%.2f")
    inputs.append(value)

# Convertir los inputs a un formato que el modelo pueda usar
input_array = np.array([inputs])

# Realizar predicción
if st.button("Predecir"):
    prediction = model.predict(input_array)
    st.success(f"Predicción para los datos ingresados: {prediction[0]:.2f}")

# Visualización de gráficos
st.header("Análisis Gráfico")
st.write("Distribución de las predicciones manuales:")

# Registrar las predicciones en un DataFrame local
if "predicciones" not in st.session_state:
    st.session_state.predicciones = []

if st.button("Guardar Predicción"):
    st.session_state.predicciones.append(prediction[0])
    st.write("Predicción guardada correctamente.")

# Mostrar las predicciones guardadas
if st.session_state.predicciones:
    predicciones_df = pd.DataFrame(st.session_state.predicciones, columns=["Predicción"])
    st.write(predicciones_df)

    # Visualización de la distribución
    fig, ax = plt.subplots()
    sns.histplot(predicciones_df["Predicción"], kde=True, color="blue", ax=ax)
    ax.set_title("Distribución de Predicciones Guardadas")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)

# Información adicional
st.sidebar.header("Acerca de")
st.sidebar.write("""
Este proyecto usa ExtraTreesRegressor para realizar predicciones manuales.
Incluye análisis gráfico y gestión de predicciones guardadas.
""")

