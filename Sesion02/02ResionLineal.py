# la regresion lienal es tipo de aprendizaje supervisado cuyo objetivo es predecir un valor numerico a partir de variables de entrada, el precio, el tamaño, ubicacion, 

#queremos predecir el precio bajo 3 modelos de entrenamiento regresion lineal minimizar el error cuadratico entre cada de las predicciones
# ridge evitar que los coeficientes crezcan demasiado (overfitting), cuando hay muchas variables
# lasso maneja los valores absolutos, y esto lleva que muchos coeficientes se vayan a 0, esto hace que su seleccion sea automatica en las variables

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_predict #producto cruz
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 20640 muestras con las siguientes caracteristicas: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Lat, Long, precio de la casa 100 000
housing = fetch_california_housing()
X, y = housing.data, housing.target

#entrenamiento 80 20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#vamos a crear una funcion para visualiacion de los elementos generando dispersión del valor real vs prediccion, y distribucion de cada uno de los elementos de error, 

def grafica_modelo(y_real, y_pred, nombre, rmse, r2, ax_scatter, ax_error):
    """ 
        Generar dos subgraficas para evaluar el modelo de regresion
        1.- Dispersion de los valores reales vs predictivos, cada punto es una casa del conjunto de prueba. Eje x = precio real, Eje y = precio predicho y mi linea de regresion va a representar una prediccion perfecta, osea los puntos mas cercanos a la diagonal son los que mas vas a tener la tendencia del modelo
        2.- Distribución de los errores (residuos), dentro de cada prediccion existe un residuo, el cual es algo fuera de los parametros que estamos buscando, representando en el modelo un histograma, centrado en el 0 no tiene un sesgo sistematico, no se sobreestima ni se subestima
        parametros
        y_real valores reales de la prueba
        y_pred valores predichos del modelo
        nombre nombre del modelo
        rmse metrica del error calculada
        r2 es la metrica del error cuadratico
        ax_scatter son los ejes para la grafica de dispersion
        ax_error son los ejes de la grafica del histograma
    """
    #valores reales vs prediccion
    ax_scatter.scatter(
        y_real, y_pred,
        alpha=0.3, #transparencia de las zonas de alta densidad osea los choques
        color='steelblue',
        s=10 #tamaño pequeño para los puntos y no saturar la grafica de dispersion
    )

    #mis condiciones para prediccion perfecta 
    minimo = min(y_real.min(), y_pred.min())
    maximo = max(y_real.max(), y_pred.max())
    ax_scatter.plot(
        [minimo, maximo], [minimo, maximo],
        'r--', linewidth=1.5,
        label='Predccion Perfecta'
    )

    ax_scatter.set_xlabel('Precio Real 100K ')
    ax_scatter.set_ylabel('Precio Predicho 100K ')
    ax_scatter.set_title(f'{nombre}\n RMSE = {rmse:.4f}  R2 = {r2:.4f}')
    ax_scatter.legend(fontsize=8)

    #histograma de residuos
    residuos = y_real, y_pred

    ax_error.hist(
        residuos,
        bins=50, #tamaño de las barras
        color='salmon',
        edgecolor='white',
        linewidth=0.4

    )
    ax_error.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Error = 0')
    ax_error.set_xlabel('Error real - predicho')
    ax_error.set_ylabel('Frecuencia')
    ax_error.set_title(f'Distribucion de errores - {nombre}')
    ax_error.legend(fontsize=8)

