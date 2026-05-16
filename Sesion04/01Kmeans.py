# k means es un algoritmo de aprendizaje no supervisado, de agrupamiento (clustering), a diferencia de todo lo anterior aqui no hay etiquetas. como funciona
# 1 elige k puntos aleatorios como centroides iniciales
# 2 asigna cada punto al centroide mas cercano (forma k grupos)
# 3 recalculamos el centroida de cada grupo como el promedio de sus puntos
# 4 repetimos los pasos 2 y 3 hasta que los centroides no se muevan (convergencia)

# Metodo del codo grafica la inercia (suma de distancias al cuadrado de cada punto del centroide) [la distancia mas corta]a medida que k crece la inercia baja. El codo de la curva (donde la baja se vuelve lenta) indica el k optimo: añadir mas clausters no mejora la conexion

#coeficiente de la silueta, mide que tan bien definido esta cada cluster, compara la distancia media de un punto a los demas de su propio cluster, (cohesion), contra la distancia media al cluster vecino mas cercano. 


#Ejercicio: vamos a generar datos artificiales, con 4 grupos conocidos para poder aplicar K means, para valores de k entre 2 y 9 y graficarlos

from sklearn.cluster import KMeans

#definir las metricas de evaluación de calidad de KMeans (cohesión / separación)
from sklearn.metrics import silhouette_score

#vamos a generar los datos artificiales 
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

#de esos datos maje_blobs vamos a generar puntos aleatorios con 500 muestras, 4 centroidas y una semilla

# las variables que vamos a ocupar son 
# X que seran las coordenadas (x, y) para los elementos de las muestras (500) 
# _ como etiquetas reales, de cada punto 
X, _ = make_blobs(n_samples=500, centers=4, random_state=42)

#evaluar las distancias para cohesion y la separacion

inercias = [] # aplicar el metodo del codo
siluetas = [] # aplicar silueta

# probar k desde 2 hasta 9 (kmeans requiere al menos 2 clusters)

# primero creamos el modelo con sus clusters, segundo empezamos agrupar, y a partir de la semilla es como los agrupamos, y esto se tienq eue ejecutar un numero de veces tal que kmeans se eejctia con los centroides inciales y debemos de guardar el resultado con menor inercia

k_range = range(2,9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)

#entrenar el modelo Y devuelve la etiqueta de cada cluster
# a cada punto (0, 1, ... , k-1)
labels = km.fit_predict(X)

inercias.append(km.inertia_)

siluetas.append(silhouette_score(X, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

#graficamos primero los codos (inercia)

ax1.plot(k_range, inercias, 'bo_', #puntos azules unidos por la linea
         linewidth=2, markersize=7)
ax1.set_title('Metodo del codo')
ax1.set_xlabel('Numero de clusters (k)')
ax1.set_ylabel('Inercia (suma de las distancias)')

#graficamos la silueta

ax2.plot(k_range, siluetas, 'rs-', #cuadrados rojos unidpor por una linea
         linewidth=2, markersize=7)
ax2.set_title('Coeficientes de Silueta')
ax2.set_xlabel('Numero de clusters (k)')
ax2.set_ylabel('Silueta promedio')

plt.suptitle('Seleccion del numero optimo de clusters', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()