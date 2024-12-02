# Autor: Luis Eduardo Arenas Deseano
# Titulo: Funcion Rosenbrock
# Materia: Sistemas Inteligentes
# Descripción: Prueba de benchmark de funcion Rosenbrock para obtener imagen del plano
# Fecha: 01/12/2024


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title("Función de Rosenbrock")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("f(X, Y)")
plt.show()
