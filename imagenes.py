import numpy as np
import matplotlib.pyplot as plt

# Definición de vectores
vector_a = np.array([2, 3])
vector_b = np.array([1, 1])

# Producto interno
product_ab = np.dot(vector_a, vector_b)
print(f"Producto interno (a · b): {product_ab}")

# Cálculo del ángulo
norm_a = np.linalg.norm(vector_a)
norm_b = np.linalg.norm(vector_b)
cos_theta = product_ab / (norm_a * norm_b)
angle = np.arccos(cos_theta) * (180 / np.pi)
print(f"Ángulo entre a y b: {angle:.2f} grados")

# Visualización gráfica
plt.quiver(0, 0, vector_a[0], vector_a[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector a')
plt.quiver(0, 0, vector_b[0], vector_b[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector b')
plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.title('Visualización de los Vectores y su Ángulo')
plt.show()

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np

# Generar datos de ejemplo
X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=6)
model = SVC(kernel='linear')
model.fit(X, y)

# Visualizar el hiperplano
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Crear una cuadrícula para el hiperplano
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

# Dibujar el hiperplano y las márgenes
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.title('Hiperplano de Separación en SVM')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# Crear datos de ejemplo
X, y = make_blobs(n_samples=100, centers=2, random_state=42)
model = SVC(kernel='linear')
model.fit(X, y)

# Visualizar datos y hiperplano
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Cuadrícula para el hiperplano
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

# Dibujar el hiperplano y márgenes
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.title('Ejemplo de Hiperplano y Producto Interno en SVM')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles

# Crear datos que no son linealmente separables
X, y = make_circles(n_samples=100, factor=0.5, noise=0.1, random_state=42)

# Visualizar los datos originales
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.title('Datos No Linealmente Separables')
plt.xlabel('X1')
plt.ylabel('X2')

# Entrenar un modelo SVM con kernel RBF
model_rbf = SVC(kernel='rbf', C=1)
model_rbf.fit(X, y)

# Visualizar la frontera de decisión
ax = plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model_rbf.decision_function(xy).reshape(XX.shape)

# Dibujar la frontera de decisión
ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
plt.title('Separación con Kernel RBF')
plt.xlabel('X1')
plt.ylabel('X2')

plt.tight_layout()
plt.show()


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Generar datos de ejemplo
X, y = datasets.make_moons(n_samples=100, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelos con diferentes kernels
kernels = ['linear', 'poly', 'rbf']
for kernel in kernels:
    model = SVC(kernel=kernel, C=1)
    model.fit(X_train, y_train)
    print(f"Precisión con kernel {kernel}: {model.score(X_test, y_test):.2f}")

    # Gráfico de la frontera de decisión
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
    plt.title(f'Frontera de decisión con kernel {kernel}')
    plt.show()
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

# Crear datos de ejemplo no linealmente separables
X, y = make_circles(n_samples=100, factor=0.5, noise=0.1, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modelo SVM con kernel RBF
model_rbf = SVC(kernel='rbf', C=1)
model_rbf.fit(X_scaled, y)

# Visualización en 2D
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.title('Datos en Espacio Original')

# Visualización en 3D de la transformación
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
z = np.exp(-np.linalg.norm(X_scaled, axis=1)**2)  # Ejemplo de transformación RBF
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], z, c=y, cmap='coolwarm')
ax.set_title('Transformación a Espacio de Mayor Dimensión')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('RBF(X)')
plt.show()
