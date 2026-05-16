#vamos a buscar una imagen florecita, y lo que vamos hacer es agrupar sus pixeles para que identifique el algoritmo RGB 


from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
import numpy as np
import matplotlib.pyplot as plt

# una imagen de color es una muestra en 3D, recordemos que serian 3 matrices una R, G, B 
# K means agrupar los pixeles y debe de identificarlos por la similitud de su color, para eso cada cluster representa ya un color promedio, cada pixel debe de ser reemplazada por su centroide k = 2 si la imagen es b/n, k = 16 

#vamos a cargar la imagen

img = load_sample_image('flower.jpg')

alto, ancho, canales = img.shape

print(f"Fomra original de la imagen {img.shape}")
print(f"Pixeles totales :        {alto, ancho}")

#preprocesamiento normalización de los Kmeans [0, 1] dentro de un rango de 3 matrices, que son las combinaciones de 0 a 255 
img_normalizada = img / 255.0

#aplanar altoXanchoX3 con esto formamos R, G, B
pixeles = img_normalizada.reshape(-1, 3)
print(f"Formar tras aplanar     {pixeles.shape}")

#segmentar con kmeans, para ello probamos los 3 valores de k para compararlos
valores_k = [2, 6, 16]
imagenes_segmentadas = []

for k in valores_k:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(pixeles)

    #a partir del centroide de su cluster tenemos que reempazar cada pixel, para que encuetre los valores [0,1] 
    colores_centroide = km.cluster_centers_
    pixeles_segmentados = colores_centroide[km.labels_]

    #restaurar a su forma original la imagen
    img_seg = (pixeles_segmentados.reshape(alto, ancho, 3)*255).astype(np.uint8)
    imagenes_segmentadas.append(img_seg)
    print(f"k = {k:>2} segmentacion completa")

#tenemos que comprar las imagenes
fig, axes = plt.subplots(1, 4, figsize=(18,5))
fig.suptitle('Segmentación de imagenes con KMeans\n, cada pixel se reemaplaza por el color de su centroide', fontsize=13, fontweight='bold')

#imagen original
axes[0].imshow(img)
axes[0].set_title('Img Original')
axes[0].axis('off')

#imagenes segmentadas
titulos = [f'k = {k}\n{k} colores unicos ' for k in valores_k]

for ax, img_seg, titulo in zip(axes[1:], imagenes_segmentadas, titulos):
    ax.imshow(img_seg)
    ax.set_title(titulos)
    ax.axis('off')

plt.tight_layout()
plt.show()

