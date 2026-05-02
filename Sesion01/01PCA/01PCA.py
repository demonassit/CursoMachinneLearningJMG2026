# vamos a crear un ejemplo donde tenemos que aplicar PCA a una matriz utilizando elementos de algebra linal directa SVD a traves de la libreria de Numpy para poder comparar resultados de un caso practico

#Vamos a ocupar un caso de la libreria sklearn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

#carga del dataset, para ello vamos a sacar todas las muestras y vamos a checar los tumores

data = load_breast_cancer() #este elemento tiene un total de muchas dimensiones
X = data.data #matriz de caracteristicas
y = data.target #etiquetar cada clase

#estandarizacion tengo que definir las variables que dominan el analisis por ejemplo vamos a utilizar media=0 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Vamos aplicar los elementos de calculo de SVD 
#Vamos a descomponer todo el dataset a una matriz de 3 variables
#V = vectores singulares izquierdos, los cuales nos van a obtener muestras * componentes
#S = valores singulares vana  obtener la magnitud de cada componente 
#Vh = los vectores singulares derechos a partir de la matriz transpuesta (componentes * variables ) -> cada fila de Vh es un componente principal

V, S, Vh = np.linalg.svd(X_scaled, full_matrices = False)

#vamos a extraer los componentes
pc1 = Vh[0] #primer componente principal del vector aqui va a obtener solo 30 variables
pc2 = Vh[1] #segundo componente ortogonal a pc1

#todo esto lo estamos calculando con base a una proyeccion de ponentes

W = Vh[:2].T #aqui vamos a formar 30x2 componentes de la variable a partir de la extracción de componentes, T es la Matriz Transpuesta

X_nueva = X_scaled.dot(W)  #es empezar a reducir las dimensiones de la matriz 

#Aplicar la varianza
#la varianza expicada relativa indica que porcentaje de informacion retiene cada compoente 
varianza_total = (S ** 2).sum()
varianza_explicada = (S[:2] **2)/varianza_total

print(f"Varianza explicada por cada componente: {varianza_explicada}")
print(f"Varianza total retenida: {varianza_explicada.sum():.2%}")

#para poderlo visualizar
plt.figure(figsize=(8,6))

plt.scatter(
    X_nueva[:, 0], #componente principal para x
    X_nueva[:, 1],  #componente principal para y 
    c=y,  #color segun clase: maligno[0] benigno[1]
    cmap = 'coolwarm', #paleta de colores azul = benigno, rojo = maligno
    alpha = 0.7 #transparencia para ver los puntos superpuestos
)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA con Numpy para Dataset de Cancer')
plt.colorbar(label='Clase')
plt.show()