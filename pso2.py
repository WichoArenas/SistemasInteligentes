# Autor: Luis Eduardo Arenas Deseano
# Titulo: Algoritmo PSO 2
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
max_iterations = 50   # Máximo número de iteraciones
num_particles = 30    # Número de partículas

# Parámetros PSO y GA para probar combinaciones
w_values = [0.5, 0.7, 0.9]  # Diferentes factores de inercia para PSO
c1_values = [1.0, 1.5, 2.0]  # Diferentes pesos cognitivos para PSO
c2_values = [1.0, 1.5, 2.0]  # Diferentes pesos sociales para PSO
mutation_rates = [0.05, 0.1, 0.2]  # Diferentes tasas de mutación para GA

# Resultados para la gráfica final
pso_results = []  # Guardar la convergencia de PSO para cada combinación
ga_results = []   # Guardar la convergencia de GA para cada combinación

# Bucle para probar combinaciones de parámetros
for w in w_values:
    for c1 in c1_values:
        for c2 in c2_values:
            # Inicialización de PSO
            positions = np.random.uniform(x_min, x_max, (num_particles, num_dimensions))
            velocities = np.zeros((num_particles, num_dimensions))
            p_best = positions.copy()
            p_best_scores = np.array([rosenbrock(pos) for pos in positions])
            g_best = positions[np.argmin(p_best_scores)]
            g_best_score = min(p_best_scores)
            pso_best_score_history = []

            # Ejecutar PSO
            for iteration in range(max_iterations):
                # Actualizar velocidades y posiciones
                r1, r2 = np.random.rand(num_particles, num_dimensions), np.random.rand(num_particles, num_dimensions)
                velocities = w * velocities + c1 * r1 * (p_best - positions) + c2 * r2 * (g_best - positions)
                positions += velocities
                positions = np.clip(positions, x_min, x_max)

                # Evaluar las nuevas posiciones
                scores = np.array([rosenbrock(pos) for pos in positions])
                for i in range(num_particles):
                    if scores[i] < p_best_scores[i]:
                        p_best_scores[i] = scores[i]
                        p_best[i] = positions[i]
                g_best = p_best[np.argmin(p_best_scores)]
                g_best_score = min(p_best_scores)
                pso_best_score_history.append(g_best_score)

            # Guardar los resultados de esta combinación
            pso_results.append((w, c1, c2, pso_best_score_history))

# Ejecutar GA con diferentes tasas de mutación
for mutation_rate in mutation_rates:
    # Inicialización de GA
    num_population = 30
    population = np.random.uniform(x_min, x_max, (num_population, num_dimensions))
    ga_best_score_history = []

    # Ejecutar GA
    for iteration in range(max_iterations):
        # Evaluar la aptitud de la población
        fitness = np.array([rosenbrock(ind) for ind in population])
        sorted_idx = np.argsort(fitness)
        best_individuals = population[sorted_idx[:num_population // 2]]
        offspring = best_individuals + mutation_rate * np.random.randn(*best_individuals.shape)
        population = np.vstack((best_individuals, offspring))
        population = np.clip(population, x_min, x_max)

        # Guardar el mejor puntaje de esta iteración
        ga_best_score_history.append(min(fitness))

    # Guardar los resultados de esta combinación
    ga_results.append((mutation_rate, ga_best_score_history))

# --- Gráfica Final Comparativa ---
plt.figure(figsize=(14, 7))

# Comparación PSO
plt.subplot(1, 2, 1)
for w, c1, c2, history in pso_results:
    plt.plot(history, label=f'w={w}, c1={c1}, c2={c2}')
plt.title('Convergencia de PSO para Diferentes Parámetros')
plt.xlabel('Iteraciones')
plt.ylabel('Mejor Valor de la Función Objetivo')
plt.legend()
plt.grid()

# Comparación GA
plt.subplot(1, 2, 2)
for mutation_rate, history in ga_results:
    plt.plot(history, label=f'Mutación={mutation_rate}')
plt.title('Convergencia de GA para Diferentes Tasa de Mutación')
plt.xlabel('Iteraciones')
plt.ylabel('Mejor Valor de la Función Objetivo')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
