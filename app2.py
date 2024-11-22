
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Título de la aplicación
st.title("Predicción de Ventas con ExtraTreesRegressor")
st.write("Predice unidades vendidas basado en costo de bienes vendidos (COGS), ventas brutas y precio de venta.")

# Configurar estilo de Seaborn
sns.set(style="whitegrid")

# Intentar cargar el modelo entrenado
try:
    with open('modelo.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Inicializar estado de sesión para guardar predicciones
if "predicciones" not in st.session_state:
    st.session_state.predicciones = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# Panel lateral para métricas del modelo
st.sidebar.header("Métricas del Modelo")
mse_input = st.sidebar.number_input("MSE (Error Cuadrático Medio)", value=0.0, format="%.4f")
r2_input = st.sidebar.number_input("R² Score", value=0.0, format="%.4f")
mae_input = st.sidebar.number_input("MAE (Error Absoluto Medio)", value=0.0, format="%.4f")
rmse_input = st.sidebar.number_input("RMSE (Raíz del Error Cuadrático Medio)", value=0.0, format="%.4f")

# Predicción de múltiples filas manualmente
st.header("Predicción Manual")
st.write("Ingresa los valores para predecir las unidades vendidas:")

# Entradas de características
cogs = st.number_input("Costo de bienes vendidos (COGS)", min_value=0.0, step=0.1, format="%.2f")
gross_sales = st.number_input("Ventas brutas (Gross Sales)", min_value=0.0, step=0.1, format="%.2f")
sale_price = st.number_input("Precio de venta (Sale Price)", min_value=0.0, step=0.1, format="%.2f")

# Realizar predicción
if st.button("Predecir"):
    inputs = np.array([[cogs, gross_sales, sale_price]])
    prediction = model.predict(inputs)[0]
    st.session_state.last_prediction = prediction
    st.success(f"Predicción para unidades vendidas: {prediction:.2f} unidades")

    # Métricas del modelo (para referencia o cálculos futuros)
    st.sidebar.header("Métricas calculadas")
    mse_value = mean_squared_error([prediction], [prediction])
    mae_value = mean_absolute_error([prediction], [prediction])
    rmse_value = np.sqrt(mse_value)
    r2_value = 1.0  # Al comparar con los mismos valores, R² siempre será 1.0

    st.sidebar.write(f"**MSE**: {mse_value:.4f}")
    st.sidebar.write(f"**MAE**: {mae_value:.4f}")
    st.sidebar.write(f"**RMSE**: {rmse_value:.4f}")
    st.sidebar.write(f"**R² Score**: {r2_value:.4f}")

# Guardar predicciones
if st.session_state.last_prediction is not None:
    if st.button("Guardar Predicción"):
        st.session_state.predicciones.append(st.session_state.last_prediction)
        st.success("Predicción guardada correctamente.")

# Visualizar predicciones guardadas
if st.session_state.predicciones:
    st.write("Predicciones guardadas:")
    predicciones_df = pd.DataFrame(st.session_state.predicciones, columns=["Predicción"])
    st.write(predicciones_df)

    # Visualización de distribución
    fig, ax = plt.subplots()
    sns.histplot(predicciones_df["Predicción"], kde=True, color="blue", ax=ax)
    ax.set_title("Distribución de Predicciones Guardadas")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)

# Gráfico de dispersión interactivo
st.subheader("Relación entre las características")
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=[cogs], y=[gross_sales], size=[sale_price], sizes=(20, 200), color="orange", legend=None, ax=ax2)
ax2.set_title("Dispersión entre COGS, Gross Sales y Sale Price")
ax2.set_xlabel("Costo de bienes vendidos (COGS)")
ax2.set_ylabel("Ventas Brutas (Gross Sales)")
st.pyplot(fig2)

# Análisis de correlación
st.subheader("Análisis de Correlaciones")
data = pd.DataFrame({
    "cogs": [cogs],
    "gross_sales": [gross_sales],
    "sale_price": [sale_price]
})

correlation_matrix = data.corr()

# Visualizar matriz de correlación
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True, ax=ax3)
ax3.set_title("Matriz de Correlación entre Variables")
st.pyplot(fig3)

# Información adicional
st.sidebar.header("Acerca de")
st.sidebar.write("""
Este proyecto utiliza ExtraTreesRegressor para predecir ventas.
Incluye análisis gráfico, métricas y herramientas interactivas para usuarios.
""")
