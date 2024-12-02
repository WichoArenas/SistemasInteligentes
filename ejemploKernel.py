# Autor: Luis Eduardo Arenas Deseano
# Titulo: Prueba de kernel
# Materia: Sistemas Inteligentes
# Descripción: Ejemplo con modelo y con un kernel dado para un algoritmo
# Fecha: 01/12/2024


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# Generar un conjunto de datos de ejemplo
X, y = datasets.make_moons(n_samples=400, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelos SVM con diferentes kernels
kernels = ['linear', 'rbf']
for kernel in kernels:
    model = SVC(kernel=kernel, C=1)
    model.fit(X_train, y_train)

    # Visualizar la frontera de decisión
    plt.figure(figsize=(6, 4))
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
    plt.title(f'Frontera de Decisión con Kernel {kernel}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    # Cálculo del producto interno entre vectores de ejemplo y visualización
    vector_a = X_train[0]
    vector_b = X_train[1]
    product_ab = np.dot(vector_a, vector_b)
    print(f"Producto interno entre vectores de entrenamiento (a · b): {product_ab}")

    # Cálculo del ángulo entre vectores
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    cos_theta = product_ab / (norm_a * norm_b)
    angle = np.arccos(cos_theta) * (180 / np.pi)
    print(f"Ángulo entre los vectores de entrenamiento: {angle:.2f} grados")

    # Visualización de los vectores en 2D
    plt.figure()
    plt.quiver(0, 0, vector_a[0], vector_a[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector a')
    plt.quiver(0, 0, vector_b[0], vector_b[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector b')
    plt.xlim(-2, 3)
    plt.ylim(-2, 3)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.title('Visualización de Vectores de Entrenamiento y su Ángulo')
    plt.show()


#error cuadratico promedio, accuracy
