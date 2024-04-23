import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from regression_algorithms.LinearRegressionGD import LinearRegressionGD
st.title("Regressão Linear com Gradiente Descendente")

"""
## Dados utilizados
"""

# Criar uma função que pega dados de um txt ou csv
# Criar uma função que gera um conjunto de dados aleatório
# Forncer a opção de utilizar um conjunto de dados de exemplo
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [50, 55, 60, 65, 75, 80, 85, 90, 95, 100]

"X: ", *x
"Y: ", *y

"""
## Gráfico de Dispersão dos Dados
"""
#
scatterFig = go.Figure()
scatterFig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Pontos"))
st.plotly_chart(scatterFig, use_container_width=True)

batch = st.slider("Digite a quantidade de items: ", 1, len(x))
b0 = st.number_input("Digite o valor inicial de b0: ", value=1)
b1 = st.number_input("Digite o valor inicial de b1: ", value=1)
learning_rate = st.number_input("Digite a taxa de aprendizado: ", 0.0, 1.0, step=0.00001, format="%f")
max_steps = st.number_input("Número de passos: ", 1, 1000000)

b0, b1, steps, error = LinearRegressionGD(x, y, batch, learning_rate, max_steps, [b0, b1])

eq = f"h(x) = {round(b0, 2)} + {round(b1, 2)}x"
x_line = np.linspace(min(x), max(x), 1000)
y_line = b1 * x_line + b0

f"""
## Passos para minimizar: {steps}
## MSE: {error}
"""
fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Pontos'))
fig2.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name=eq))

st.plotly_chart(fig2, use_container_width=True)