# t-SNE es una tecnica de reduccion de dimensionalidad especialmente diseñada para VISUALIZAR datos de alta dimension 2d o 3d
# a diferencia de PCA (busca direccion de maxima varianza) se enfoca en preservar las relaciones de vecinidad local, los puntos mas cercanos al espacio original
#  calculo de probabilidades de que dos vecinos en el espacio origianal se correlacionen con x
# crear una distribución de similitudes 
# ajustar iterativamente las posiciones 2d para que ambas distribuciones sean lo mas parecido posible 
#para diferenciar entre PCA y NMF y SNE
#PCA y NMF son lineales 
#t-SNE es no lineal, multiples dimensiones para ello se hace captura o calculo de curvas 

from sklearn.manifold import TSNE
#vamos a cargar un dataset de 1797 imagenes de digitos (0-9) de 8x8
from sklearn.datasets import load_digits

import matplotlib.pyplot as plt

#un dataset de 1797 imagenes de digitos escritos a mano, cada imagen tiene un tamaño de 8x8 en un vector de dimension 64
#x tiene una forma de (1797, 64) : 1797 muestras con 64 caracteres cada una
#y contiene las etiquetas 0 al 9

digits = load_digits() #dataset

X, y = digits.data, digits.target

#crear los modelos de acuerdo al siguiente parametro:
# tenemos que reducir las dimensiones para poderlo graficar en 2d, para ello lo reducimos en 2 dimensiones 
#tenemos que controlar cuantos vecinos se debe de considerar para cada punto que se va a calcular, entre 5 y 50 
#fija la semilla 
#definir el numero de iteraciones para que sea lo mas optimo o preciso 

tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)

#vamos a proyectar el resultado
X_tsne = tsne.fit_transform(X) #todo el dataset  matriz (1797, 2)

#vamos a visualizarlo
plt.figure(figsize=(10,8))

#vamos a graficar cada elemento en 2d 
scatter = plt.scatter(
    X_tsne[:, 0],  #coordenadas 1 en el esapcio tsne
    X_tsne[:, 1],  #coordenadas 2 en el espacio tsne
    c=y, # colores segun
    cmap='tab10', #10 colores distintos para 10 digitos
    alpha=0.8 #vamos agregar un nivel de transparencia sirve para ver los elementos superpuestos 
)

plt.colorbar(scatter, label='Digito')
plt.title('TSNE Dataset de digitos')
plt.show()
