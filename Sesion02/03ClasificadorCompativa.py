#los clasificadores por KNN y SVM miden distancias euclidianas entre cada uno de los puntos  
#este tipo de clasificador domina el calculo y sesga las predicciones 0 y 1 media es el valor 0 y 1 cercania

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  #es el manejo de escalas de 0 y 1
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#vamos a tener un archivo para comparar el data set de iris de petalos y sepalo de flores, para identificar 

#vamos a cargar los datos 150 flores, 3 especies diferentes con 4 caracteristicas de sepalo y petalo
#clases 0 = setosa, 1= versicolor , 2=virginica

iris = load_iris()
X, y = iris.data, iris.target

#dividir los elementos de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y #garatizar que las clases queden representadas
)

#escalar osea ajustar los elementos para evitar un sobreentrenamiento
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train) #ajusta t transforma el entranamiento
X_test_s = scaler.transform(X_test) #transformar los elementos de prueba sin reajustarlos

#definir los modelos
#una regresion logistica es un clasificador lineal, rapido e interpretable
#KNN clasificador segun los k vecinos mas cercanos 
#SVM encuentra el hiperplano maximo del margen 

modelos = {
    'Reg. Logística': LogisticRegression(max_iter=200),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf')
}

for nombre, m in modelos.items():
    m.fit(X_train_s, y_train)
    print(f"\n == {nombre} ==")
    print(classification_report(
        y_test, m.predict(X_test_s), 
        target_names=iris.target_names
    ))

#necesitamos una matriz de confucion para cuantas muestras necesitamos de cada elemento real
fig, axes = plt.subplots(1, 3, figsize=(15,4))
fig.suptitle('Matrices de Confusion por Clasificador Iris', fontsize=14, fontweight='bold')

for ax, (nombre, m) in zip(axes, modelos.items()):
    cm = confusion_matrix(y_test, m.predict(X_test_s))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(nombre)
    ax.tick_params(axis='x', labelrotation=15)

plt.tight_layout()
plt.show()