#GMM es ideal para detectar anomalias, zonas densas = log-densidad alta comportamiento normal, zonas escasas log-densidades bajas una posible anomalia

# vamos a manejar un umbral, perceptible de densidades, p = 4% de los puntos con menor densidad se considera anomalia, significa que dependiendo del modelo solo debemos ajustar la sensibilidad, pero ojo p alto mas puntos marcados como anomalias (mfp) p bajo solo los mas extremos se marcan (mfn)

from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# vamos a generar un ejemplo de datos normales a partir de make_blobs, y vamos a crear 3 clusters, para inyectar anomalias sinteticas, osea puntos aleatorios. para ello tenemos ajustar el GMM y calcular sus densidades, marcamos las anomalias para evaluar

X_normal, _ = make_blobs(
    n_samples=300, 
    centers=3,
    cluster_std=0.6,
    random_state=42
)

#debemos estandarizar gmm ya que mide densidades en el espacio
scaler = StandardScaler()
X_normal = scaler.fit_transform(X_normal)

#vamos a inyectar las anomalias, por ejemplo generamos 20 puntos aletorios uniformes en un rango [-4, 4] 
np.random.seed(0)
n_anomalias_reales = 20
X_anomalias = np.random.uniform(low=-4, high=4, size=(n_anomalias_reales, 2))

#las combinamos
X_todo = np.vstack([X_normal, X_anomalias])

#vamos a etiquetar 0 = normal, 1 = anomalia
y_verdad = np.array([0]*len(X_normal) + [1]*n_anomalias_reales)

print(f'Total de puntos : {len(X_todo)} ' )
print(f'Puntos Normales : {len(X_normal)} ' )
print(f'Anomalias inyectadas : {(n_anomalias_reales)} ' )

#ajustamos el modelo de gmm
gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X_todo)

#calculamos densidades
densidad = gm.score_samples(X_todo)
# cuando mas negativo menos probable es el punto de ser sospechoso de anomalia

#sensibilidad 
percentil = 4
umbral_densidad = np.percentile(densidad, percentil)
mascara_anomalia = densidad < umbral_densidad

print(f'Umbral de percentil : {umbral_densidad:.4f}')
print(f'Puntos destacados como anomalias : {mascara_anomalia.sum()}')

#evaluamos
detectadas_reales = ((mascara_anomalia == 1) & (y_verdad == 1)).sum()
falsos_positivos = ((mascara_anomalia == 1) & (y_verdad == 0)).sum()
no_dectados = ((mascara_anomalia == 0) & (y_verdad == 1)).sum()

print("------Evaluación----------")
print(f"Anomalias reales detectadas: {detectadas_reales}/{n_anomalias_reales}")
print(f"falsos positovos: {falsos_positivos}")
print(f"anomalias no detectadas : {no_dectados}")


#a graficar
fig, axes = plt.subplots(1,2, figsize=(14,6))

fig.suptitle('GMM Detecciión de Anomalias por Densidades', fontsize=13, fontweight='bold')

#grafica1
ax1 = axes[0]
# puntos normales por el modelo
ax1.scatter(
    X_todo[~mascara_anomalia, 0], X_todo[~mascara_anomalia, 1], c='steelblue', s=20, alpha=0.6, label='Normal '
)
#puntos detectados como anomalias
ax1.scatter(
    X_todo[~mascara_anomalia, 0], X_todo[~mascara_anomalia, 1], c='red', s=80, marker='x', linewidth=1.5, label='Anomalia detectada ({mascara_anomalia.sum()})  '
)
# medias de cada gausiana
ax1.scatter(
    gm.means_[:, 0], gm.means_[:, 1], c='black', marker='*', s=200, zorder=5, label='Medias de GMM'
)

ax1.set_title(f'Anomalias detectas \n (umbral = percentil: {percentil})')
ax1.set_xlabel(f'caracteristica 1')
ax1.set_ylabel(f'caracteristica 2')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

#grafica 2 densidad
ax2= axes[1]

x_min, x_max = X_todo[:, 0].min() - 0.5,  X_todo[:, 0].max() + 0.5 
y_min, y_max = X_todo[:, 1].min() - 0.5,  X_todo[:, 1].max() + 0.5 

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

Z = gm.score_samples(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#coloreamos las zonas oscuras
cf = ax2.contourf(xx, yy, Z, levels=30, cmap='RdYlGn')
plt.colorbar(cf, ax=ax2, label='log-densidad')

#superponemos los puntos
ax2.scatter(X_normal[:, 0], X_normal[:, 1], c='white', s=10, alpha=0.4, label='Normal')
ax2.scatter(X_anomalias[:, 0], X_anomalias[:, 1], c='black', s=60, marker='x', linewidths=1.5, label='Anomialia Real')

#umbral de detección
ax2.contour(xx, yy, Z, levels=[umbral_densidad], colors='red', linestyles='--', linewidths=1.5)

ax2.set_title('Mapa de log de Densidad GMM')
ax2.set_xlabel('Caracteristica 1 (estandarizada)')
ax2.set_ylabel('Caracteristica 2 (estandarizada)')
ax2.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.show()