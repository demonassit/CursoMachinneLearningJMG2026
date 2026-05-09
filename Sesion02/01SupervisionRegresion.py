# vamos con un flujo universal de regresión lineal
# primero una carga de datos
# segundo debemos de dividir los elementos para entrenar y los elementos para probar
# tercero debemos entrenar el modelo
# cuarto predecir y evaluar

from sklearn.datasets import fetch_california_housing # información sobre las casas en california
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split   #vamos a tomar secciones definidas para entrenamiento y prueba
from sklearn.metrics import mean_squared_error

import numpy as np
import matplotlib.pyplot as plt

#para la carga de datos
# son calas de california que son 20,640 muestras con 8 caracteristicas 
# variables objetivo y = precio mediano de la casa en 100 000 dolares

housing = fetch_california_housing()

X, y = housing.data, housing.target # x es todo el dataset y el precio

# de todo el dataset tenemos que definir cuantos datos son para entrenamiento y cuantos para probar
# 80% entrenamiento y 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42  #el valor de la semilla
)


#modelo

modelo = LinearRegression()

modelo.fit(X_train, y_train) #con lo que vamos a entrenar el modelo
y_pred = modelo.predict(X_test) #generar predcciones a partir de datos no vistos

# la regresion lineal es el calculo de minimos cuadrados, significa que tiene un coeficiente de determinación no de presicion por lo tanto r2 = 1.0 la predicción es perfecta si r2= 0.0 equivale a predecir siempre la media de y   0.1 - 0.9 

r2 = modelo.score(X_test, y_test)

# posibles errores: error en las mismas unidades , el manejo para el error promedio 50k

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R2 = {r2:.4f}")
print(f"RMSE = {rmse:.4f} en $100,000 USD")

#vamos a visualizarlo

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.3, s=10, color='steelblue')
minimo, maximo = y_test.min(), y_test.max()
plt.plot([minimo, maximo], [minimo, maximo], 'r--', linewidth=1.5,
         label='Prediccion perfecta')
plt.xlabel('Precio real ($100,000 UDS)')
plt.ylabel('Precio predicho respecto ($100,000 UDS)')
plt.title('Regresión Linelan de Casas de California')
plt.legend()
plt.tight_layout()
plt.show()

