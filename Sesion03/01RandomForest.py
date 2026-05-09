#arbol de desicion es un modelo que aprende una serie de preguntas (condiciones sobre las caracteristicas), organizadas de forma de arbol para llegar a una prediccion

#el problema de los arboles de desicion es el overfitting (sobreajuste) memoriza los datos de entrenamiento tan bien que pierde la capacidad de generalizar a datos nuevos

#nuestro ejemplo va a tomar una muestra aletoria de un dataset y un subconjunto aletorio de caractertiscias en cada division 

#vamos a comprar la accuracy de un arbol de descicion sobre un random forest de 100 arboles usando un dataset de Breast Cancer

from sklearn.ensemble import RandomForestClassifier  #este nos va a traer los diferentes arboles de desiones
from sklearn.tree import DecisionTreeClassifier #arboles de desciones para comparar con el RF
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #proporcion de predccion correcta (calcular los errores)
import numpy as np
import matplotlib.pyplot as plt

#breast cancer 568 muestras de tumores con 30 caracteristicas a partir de imagenes de biopsias 0=maligno y 1=benigno

data = load_breast_cancer()
X, y = data.data, data.target #y es el diagnostico x son todos los datos

#entrenamiento 75 25
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

#la creacion del modelo, tiene una caractetisca sin restricciones, crezca hasta que clasifique perfectamente los datos; pero la consecuencia es un alto riesgo de overfitting
dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

#en el arbol de random forest es donde nosotros limitamos con 100 arboles, y debemos establecer el numero de iteraciones de cada arbol
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

#calcular el nivel de prediccion de cada arbol con entrenamiento
print(f'Abol de descion : {accuracy_score(y_test, dt.predict(X_test)):.4f}')
print(f'Random Forest : {accuracy_score(y_test, dt.predict(X_test)):.4f}')

#la importancia de las caracteristicas del atributo de random forest, que indica cuanto contribuyo cada caracteristica a mejorar dentro de las pruebas de cada nodo
importancia = rf.feature_importances_

#los elementos del ordenamiento de los indices de menor a mayor importancia
# [ : : -1  ] los invierte (de mayor a menor)
# [ : 10 ] tomar solo los 10 mas importantes

indices = np.argsort(importancia)[::-1][:10]

#vamos a graficar
plt.barh(
    range(10), #posciones verticales 0 al 9
    importancia[indices],
    color='steelblue', edgecolor='white'
)
# vamos a obtener los nombres reales de las caracteristicas del data set
plt.yticks(range(10), [data.feature_names[i] for i in indices])

plt.xlabel('Importancia (reduccion media del indice: )')
plt.title('Top 10 de caracteristicas mas importantes con RF')

#invertimos para colocar las caracteristicas en orden
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
