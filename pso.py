# Autor: Luis Eduardo Arenas Deseano
# Titulo: Algoritmo PSO
# Materia: Sistemas Inteligentes
# Descripción: Prueba de benchmark de funcion Rosenbrock con algoritmo PS
# Fecha: 01/12/2024


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Función objetivo: Rosenbrock
def rosenbrock(x):
    """
    Calcula el valor de la función de Rosenbrock.
    Entrada:
        x: vector con las coordenadas [x1, x2, ...].
    Salida:
        valor de la función en el punto x.
    """
    return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Parámetros comunes
x_min, x_max = -5, 5  # Límites del espacio de búsqueda
num_dimensions = 2    # Dimensiones del problema
max_iterations = 100   # Máximo número de iteraciones

# Crear la malla para la función (para graficar la superficie de Rosenbrock)
x = np.linspace(x_min, x_max, 100)  # Puntos en el eje X
y = np.linspace(x_min, x_max, 100)  # Puntos en el eje Y
X, Y = np.meshgrid(x, y)  # Malla bidimensional
Z = np.array([rosenbrock(np.array([xi, yi])) for xi, yi in zip(X.ravel(), Y.ravel())]).reshape(X.shape)  # Calcular Z

# --- PSO: Inicialización ---
num_particles = 30  # Número de partículas en el enjambre
positions = np.random.uniform(x_min, x_max, (num_particles, num_dimensions))  # Posiciones iniciales aleatorias
velocities = np.zeros((num_particles, num_dimensions))  # Velocidades iniciales (empiezan en 0)
p_best = positions.copy()  # Mejor posición personal inicial de cada partícula
p_best_scores = np.array([rosenbrock(pos) for pos in positions])  # Evaluación inicial de cada partícula
g_best = positions[np.argmin(p_best_scores)]  # Mejor posición global basada en el mejor puntaje
g_best_score = min(p_best_scores)  # Mejor puntaje global inicial
pso_best_score_history = []  # Historial de los mejores puntajes para PSO

# --- GA: Inicialización ---
num_population = 30  # Tamaño de la población inicial
population = np.random.uniform(x_min, x_max, (num_population, num_dimensions))  # Población inicial aleatoria
mutation_rate = 0.1  # Tasa de mutación
ga_best_score_history = []  # Historial de los mejores puntajes para GA

# --- Métricas para medir rendimiento ---
pso_convergence_time = None  # Tiempo de convergencia de PSO
ga_convergence_time = None  # Tiempo de convergencia de GA
tolerance = 1e-3  # Tolerancia para considerar que se alcanzó el mínimo

# Guardar tiempos iniciales para calcular tiempos de convergencia
start_time_pso = time.time()
start_time_ga = time.time()

# --- Configuración de la gráfica ---
fig = plt.figure(figsize=(14, 7))
ax_pso = fig.add_subplot(121, projection='3d')  # Subplot para PSO (gráfica 3D)
ax_ga = fig.add_subplot(122, projection='3d')   # Subplot para GA (gráfica 3D)
plt.ion()  # Habilitar modo interactivo para actualizar las gráficas en tiempo real

# --- Bucle Principal ---
for iteration in range(max_iterations):
    # --- PSO: Actualización de Partículas ---
    r1, r2 = np.random.rand(num_particles, num_dimensions), np.random.rand(num_particles, num_dimensions)  # Factores aleatorios
    velocities = 0.7 * velocities + 1.5 * r1 * (p_best - positions) + 1.5 * r2 * (g_best - positions)  # Actualizar velocidad
    positions += velocities  # Actualizar posiciones basadas en la nueva velocidad
    positions = np.clip(positions, x_min, x_max)  # Restringir posiciones dentro del espacio de búsqueda

    # Evaluar nuevas posiciones
    scores = np.array([rosenbrock(pos) for pos in positions])
    for i in range(num_particles):
        if scores[i] < p_best_scores[i]:  # Actualizar el mejor personal si se encuentra un puntaje menor
            p_best_scores[i] = scores[i]
            p_best[i] = positions[i]
    g_best = p_best[np.argmin(p_best_scores)]  # Actualizar el mejor global
    g_best_score = min(p_best_scores)  # Mejor puntaje global
    pso_best_score_history.append(g_best_score)  # Guardar puntaje global en el historial

    # Verificar convergencia de PSO
    if pso_convergence_time is None and g_best_score < tolerance:
        pso_convergence_time = time.time() - start_time_pso

    # --- GA: Evolución de la Población ---
    fitness = np.array([rosenbrock(ind) for ind in population])  # Evaluar la aptitud de la población
    sorted_idx = np.argsort(fitness)  # Ordenar por aptitud
    best_individuals = population[sorted_idx[:num_population // 2]]  # Seleccionar los mejores individuos
    offspring = best_individuals + mutation_rate * np.random.randn(*best_individuals.shape)  # Generar descendientes
    population = np.vstack((best_individuals, offspring))  # Actualizar población
    population = np.clip(population, x_min, x_max)  # Limitar al espacio de búsqueda
    ga_best_score = min(fitness)  # Mejor puntaje en la población actual
    ga_best_score_history.append(ga_best_score)  # Guardar puntaje en el historial

    # Verificar convergencia de GA
    if ga_convergence_time is None and ga_best_score < tolerance:
        ga_convergence_time = time.time() - start_time_ga

    # --- Actualizar Gráficas ---
    ax_pso.cla()
    ax_ga.cla()

    # Graficar la superficie de la función objetivo
    ax_pso.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    ax_ga.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    # Graficar partículas (PSO)
    ax_pso.scatter(positions[:, 0], positions[:, 1], 
                   np.array([rosenbrock(pos) for pos in positions]), 
                   color='r', label='Partículas (PSO)', s=50)
    ax_pso.set_title(f"PSO - Iteración {iteration+1}")
    ax_pso.set_xlabel("X")
    ax_pso.set_ylabel("Y")
    ax_pso.set_zlabel("f(X, Y)")

    # Graficar individuos (GA)
    ax_ga.scatter(population[:, 0], population[:, 1], 
                  np.array([rosenbrock(ind) for ind in population]), 
                  color='b', label='Individuos (GA)', s=50)
    ax_ga.set_title(f"GA - Iteración {iteration+1}")
    ax_ga.set_xlabel("X")
    ax_ga.set_ylabel("Y")
    ax_ga.set_zlabel("f(X, Y)")

    plt.pause(0.1)  # Pausa para actualizar las gráficas

plt.ioff()  # Deshabilitar modo interactivo

# --- Mostrar Métricas ---
if pso_convergence_time is not None:
    print(f"Tiempo de convergencia PSO: {pso_convergence_time:.2f} segundos")
else:
    print("PSO no alcanzó la tolerancia definida.")

print(f"Mejor valor alcanzado por PSO: {g_best_score:.6f}")

if ga_convergence_time is not None:
    print(f"Tiempo de convergencia GA: {ga_convergence_time:.2f} segundos")
else:
    print("GA no alcanzó la tolerancia definida.")

print(f"Mejor valor alcanzado por GA: {ga_best_score:.6f}")

# --- Graficar Convergencia ---
plt.figure(figsize=(8, 5))
plt.plot(pso_best_score_history, label='PSO', color='r')
plt.plot(ga_best_score_history, label='GA', color='b')
plt.xlabel('Iteraciones')
plt.ylabel('Mejor Valor de la Función Objetivo')
plt.title('Convergencia de PSO vs GA')
plt.legend()
plt.grid()
plt.show()
