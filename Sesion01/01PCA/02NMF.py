# vamos a cargar un dataset de rostros humanos, alrededor de 400 fotografias las cuales vamos a reconocer personas, especificamente 40, para ello necesitmaos aplicar NMF Matriz Factorización No Negativa, es una tecnica de reducción de variables o dimensiones a partir de una matriz X en dos matrices W y H ambas con valores no negativos

# x = W X H
#datos = pesos * componentes
# pesos son las caracteristicas que buscamos por ejemplo son como regiones, rasgos faciales, caracteristicas unicas del rostro
# componentes son los elementos de clasificación para cada rostro los rostros yo busco ojos cafes, ojos azules, ojos rasgados, nariz chata

#vamos a definir 15 componentes base
#el algoritmo de factorizacion de matrices no negativas
from sklearn.decomposition import NMF
#el dataset de las imagenes de los rostros de las personas en escala a grises
from sklearn.datasets import fetch_olivetti_faces

import matplotlib.pyplot as plt

#en el dataset tenemos 400 imagenes de rostros, 40 personas, necesitamos almenos 10 fotos por persona, cada imagen representa un vector de tamaño 4096 valores (64x64) por lo que su taño es de (400, 4096), 
#aleatorias para ello tenemos que mezclarlo 
#tenemos que definir una semilla es el elemento fijo 

faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces.data #matriz de forma (400, 4096)

#vamos a crear nuestro modelo vamos a identificar 15 componentes base
#un rostro_base 

nmf = NMF(n_components=15, random_state=42)

#vamos ajustar el modelo para obtener pesos "W" :
# primero debemos obtener los rostros base (15, 4096) 
# tenemos que darle forma (400, 15)
# cada imagen debe de ser reconstruida

X_nmf = nmf.fit_transform(X)

#vamos a pintarlo

#vamos a crear 3 figuras 3 x 5 para mostrar cada componente

fig, axes = plt.subplots(3,5, figsize=(12,8) )

#recorremos cada subgrafia y dibujamos su componente

for i, ax in enumerate(axes.ravel()):
    #ravel() conviertelo en una cuadricula en lista plana
    #de cada componente componets_[i] va a tener un vector de valor 4096
    # lo vamos a redimensionar (la parte de reconstruccion) a 64x64 
    ax.imshow(nmf.components_[i].reshape(64,64), cmap='gray')
    #ocultar los ejes para una mejor visualizacion
    ax.axis('off')

plt.suptitle('Componentes NMF de Rostros')
plt.show()