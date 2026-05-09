#en este ejercicio vamos a utilizar las tecnicas mas importantes para evaluar y optimizar un modelo de clasificación rigurosa
# GridSearchCv que es para la busqueda de hiperparametros, los cuales son configuraciones que no se aprenden durante el entranimiento, esos los elege el programador, por ejemplo numero de aboles, la profundidad del recorrido, las combines de los posibles y la seleccion para el mejor desempeño.
# validacion cruzada, en lugar de evaluar el modelo, con una solo vision train / test, la validacion cruzada divide los datos en k=5 partes, el modelo se entrena k veces, cada vez usa k-1 de las partes para entrenar el siguiente y asi validar. El desempeño final es el promedio de las evaluaciones.
# curva ROC y AUC para evaluar los umbrales, un clasificador binario porque predice la probabilidad del umbral, (0, 1) con esta curva nosotros podemos aplicar TPR verdadero positvo , FPR falso positivo 

# ocupamos la matriz de confusion para la tabla de predicciones, los elementos correctos e incorrectos
# confusionmatrizdisplay convierte la matriz en una grafica 
# roc_curve para calcular los TPR y FPR, para cada umbral en la desicion
# calculamos el area bajo la curva para medir la probabilidad un valor entre el 0 y 1
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# cross va a evaluar un modelo por cruzes simples 
# grid ese va a realizar la busqueda de los hiperparametros a partir de la validacion cruzada 
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


data = load_breast_cancer()
X, y = data.data, data.target # 0 = maligno 1 = benigno

#vamos a entrenar el modelo buscando hiperparametros, 75% y el resto para evaluar
X_train, X_test, y__train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# para la busqueda de los hiperparametros tenemos que definir sus combinaciones, 
# ocupar n_estimator, numero de arboles en el bosque 50 a 100
# debemos definir la profundidad del arbol arbol -> None (no tiene limite), 5 a 10
# tenemos que tener consideración de las k (las partes del entrenamiento) considerar las combinaciones de los elementos de validacion cruzada de cada subseccion
param_grid = {
    'n_estimators' : [50, 100],
    'max_depth' : [None, 5, 10]
}

#para los elementos de la busqueda del grid, tenemos que tener un modelo base que debe optimizarse
# estimator = modelo base a optimizar
# param_grid = combinaciones de los hiperparametros
# cv = 5 = validaciones cruzadas por los k (dividir X_train en 5 partes)
# scoring = 'criterio ' = f1 es el score que sirve para maximizar F1-Score (es un clasificador binario, de balance precision con regresión)

gs = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5, 
    scoring='f1'
)

# ahora si quiero que lo entrenes y lo evalues 5 * 6= 30 combinaciones 
gs.fit(X_train, y__train)

#al modelo ya entrenamo vamos con los mejores hiperparametros, para hacer un reentrenamiento 
best = gs.best_estimator_

#vamos a ver el diccionario de las combinaciones de los hiperparametros ganadores
print("Mejores parametros: ", gs.best_estimator_)

#primero ocupamos curvas ROC, los cuales en su elemento de prediccion nos devuelve la probabilidad de cada clase de muestra [:, 1] seleciona la probabilidad de que sea positiva (1 = benignos)
y_prob = best.predict_proba(X_test)[:, 1]

#para calcular las curvas roc, tenemos que calcular los puntos que varian del umbral de desicion VV FV  los que usan "_" son los umbrales que no usamos
fpr, tpr, _ = roc_curve(y_test, y_prob)

# auc() calcula el area bajo la curba de ROC, 
# 1.0 el modelo es perfecto (clasificar correctamente cualquier umbral)
# 0.5 el modelo es aleatorio, (diagonal, sin poder discriminar)
# < 0.5 es el peor escenario el modelo es completamente al azar

#graficamos
plt.figure(figsize=(7,5))

#vamos a graficar las curvas roc del modelo cada punto es un umbral diferente

plt.plot(fpr, tpr, color='steelblue', linewidth=2, label=f'Random Forest (AUC = {auc(fpr, tpr):.3f})')

#linea diagonal para representar los niveles del clasificador (AUC = 0.5) es aleatorio; pero si la curva del modelo esta muy por encima de la linea entonces es bueno
plt.plot([0,1], [0,1], '--', color='gray', linewidth=1, label='Clasificador aleatorio?')

plt.xlabel('FPR, Tasa de Falsos Positivos \n Proporcion de benignos mal clasificados como malignos')
plt.ylabel('TPR, Tasa de Verdaderos Positivos \n Proporcion de benignos detectados correctamente')

plt.title('Curva ROC para la Optimizacion de Random Forest en la Deteccion de Cancer con GridSearch')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

