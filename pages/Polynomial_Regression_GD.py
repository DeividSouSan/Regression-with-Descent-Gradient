import streamlit as st
import plotly.graph_objects as go
import random

st.title("Regressão Polinomial com Gradiente Descendente")

"""
A regressão polinomial segue o formato:
"""

st.latex("h(x) = b_0 + b_1 x + b_2 x^2")

x_values = [-10, -9, -8, -7, -6, -5, -4, -
            3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_values = [130, 101, 76, 55, 38, 25, 16, 11, 10, 13,
            20, 31, 46, 65, 88, 115, 146, 181, 220, 263, 310]

"""
## Gráfico de Dispersão dos Dados
"""
scatterFig = go.Figure()
scatterFig.add_trace(go.Scatter(x=x_values, y=y_values, mode="markers"))

st.plotly_chart(scatterFig)

"""
## Configurações do Algoritmo Gradiente Descendente
"""
samples_size = len(x_values)
batch = st.slider("Número de amostras: ", 1, samples_size)

b0_prev = st.number_input("Digite o valor inicial de b0: ")
b1_prev = st.number_input("Digite o valor inicial de b1: ")
b2_prev = st.number_input("Digite o valor inicial de b2: ")

learning_rate = st.number_input(
    "Digite o valor da taxa de aprendizado: ",
    min_value=0.0000,
    max_value=1.0,
    step=0.00001,
    format="%f")

max_steps = st.number_input("Número de passos", 0, 1000000, step=10)
# Hipótese


def h(x, b0=1, b1=1, b2=1):
    # Melhor hipótese seria: 20 + 9x + 2x²
    return b0 + (b1 * x) + (b2 * x**2)

# Função de Custo / Mean Squared Error


def J(b0, b1, b2):
    squared_errors = 0
    samples_index = random.sample(list(range(samples_size)), batch)

    x_sample, y_sample = [], []
    for index in samples_index:
        x_sample.append(x_values[index])
        y_sample.append(y_values[index])

    for index in range(batch):
        squared_errors += (h(x_sample[index], b0,
                           b1, b2) - y_sample[index]) ** 2

    return squared_errors / batch


def dJb0(b2, b1, b0):
    samples_index = random.sample(list(range(samples_size)), batch)

    x_sample, y_sample = [], []
    for index in samples_index:
        x_sample.append(x_values[index])
        y_sample.append(y_values[index])

    rows_sum = 0
    for i in range(batch):
        rows_sum += h(x_sample[i], b0, b1, b2) - y_sample[i]

    return rows_sum / batch


def dJb1(b2, b1, b0):
    samples_index = random.sample(list(range(samples_size)), batch)

    x_sample, y_sample = [], []
    for index in samples_index:
        x_sample.append(x_values[index])
        y_sample.append(y_values[index])

    rows_sum = 0
    for i in range(batch):
        rows_sum += (h(x_sample[i], b0, b1, b2) - y_sample[i]) * x_sample[i]

    return rows_sum / batch


def dJb2(b2, b1, b0):
    samples_index = random.sample(list(range(samples_size)), batch)

    x_sample, y_sample = [], []
    for index in samples_index:
        x_sample.append(x_values[index])
        y_sample.append(y_values[index])

    rows_sum = 0
    for i in range(batch):
        rows_sum += (h(x_sample[i], b0, b1, b2) - y_sample[i]) * (x_sample[i]**2)

    return rows_sum / batch


steps = 0
b0, b1, b2 = 0, 0, 0

for step in range(max_steps):
    b0_new = b0_prev - (learning_rate * dJb0(b2_prev, b1_prev, b0_prev))
    b1_new = b1_prev - (learning_rate * dJb1(b2_prev, b1_prev, b0_prev))
    b2_new = b2_prev - (learning_rate * dJb2(b2_prev, b1_prev, b0_prev))

    b0_minimized = round(b0_new, 10) == round(b0_prev, 10)
    b1_minimized = round(b1_new, 10) == round(b1_prev, 10)
    b2_minimized = round(b2_new, 10) == round(b2_prev, 10)

    if b0_minimized and b1_minimized and b2_minimized:
        b0, b1, b2 = b0_new, b1_new, b2_new
        steps = step + 1
        break

    elif step + 1 == max_steps:
        steps = step + 1
        b0, b1, b2 = b0_new, b1_new, b2_new

    b0_prev, b1_prev, b2_prev = b0_new, b1_new, b2_new

fit_equation = f"h(x) = {round(b0, 2)} + {round(b1, 2)}x + {round(b2, 2)}x²"
y_eq = [h(i, b0, b1, b2) for i in x_values]

f"""
## Resultado

### Passos para Miminizar: {steps}
### MSE (Erro Quadrático Médio): {J(b0, b1, b2)}
"""

regressionFig = go.Figure()
regressionFig.add_trace(go.Scatter(x=x_values, y=y_values, mode="markers"))
regressionFig.add_trace(go.Scatter(
    x=x_values, y=y_eq, mode="lines", name=fit_equation))

st.plotly_chart(regressionFig)
