import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generar nombres para las ciudades
def generar_nombres_ciudades(num_cities):
    return [f"Ciudad {chr(65 + i)}" for i in range(num_cities)]

# Número de ciudades
num_cities = 10
np.random.seed(42)

# Coordenadas aleatorias de las ciudades
cities = np.random.rand(num_cities, 2) * 100
city_names = generar_nombres_ciudades(num_cities)

# Matriz de distancias
def calcular_matriz_distancias(cities):
    num_cities = len(cities)
    matriz_distancias = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            matriz_distancias[i, j] = np.linalg.norm(cities[i] - cities[j])
    return matriz_distancias

matriz_distancias = calcular_matriz_distancias(cities)

# Calcular distancia total de una ruta
def calcular_distancia_ruta(ruta, matriz_distancias):
    return sum(matriz_distancias[ruta[i], ruta[i + 1]] for i in range(len(ruta) - 1)) + \
           matriz_distancias[ruta[-1], ruta[0]]

# Parámetros del algoritmo genético
tam_poblacion = 200
num_generaciones = 500
tasa_mutacion_base = 0.2
tasa_cruce = 0.8
mejoras_consecutivas = 15
umbral_mejora = 1e-3
elitismo = 2

# Inicializar población
def inicializar_poblacion(tam_poblacion, num_cities, ciudad_inicial):
    poblacion = [np.random.permutation(num_cities) for _ in range(tam_poblacion)]
    for individuo in poblacion:
        idx_inicial = np.where(individuo == ciudad_inicial)[0][0]
        individuo[0], individuo[idx_inicial] = individuo[idx_inicial], individuo[0]
    return poblacion

# Evaluar población
def evaluar_poblacion(poblacion, matriz_distancias):
    return np.array([1 / calcular_distancia_ruta(ind, matriz_distancias) for ind in poblacion])

# Selección
def seleccionar_poblacion(poblacion, fitness):
    seleccionados = []
    for _ in range(len(poblacion)):
        candidatos = np.random.choice(len(poblacion), 3, replace=False)
        mejor = candidatos[np.argmax(fitness[candidatos])]
        seleccionados.append(poblacion[mejor])
    return seleccionados

# Cruce ordenado
def cruce_ordenado(padre1, padre2):
    if np.random.rand() > tasa_cruce:
        return padre1.copy()
    inicio, fin = sorted(np.random.choice(len(padre1), 2, replace=False))
    hijo = [-1] * len(padre1)
    hijo[inicio:fin + 1] = padre1[inicio:fin + 1]
    puntero = 0
    for ciudad in padre2:
        if ciudad not in hijo:
            while hijo[puntero] != -1:
                puntero += 1
            hijo[puntero] = ciudad
    return np.array(hijo)

# Mutación
def mutar(individuo, tasa_mutacion):
    if np.random.rand() < tasa_mutacion:
        i, j = np.random.choice(len(individuo), 2, replace=False)
        individuo[i], individuo[j] = individuo[j], individuo[i]
    return individuo

# Algoritmo genético
def algoritmo_genetico_tsp(ciudad_inicial):
    poblacion = inicializar_poblacion(tam_poblacion, num_cities, ciudad_inicial)
    ruta_inicial = poblacion[0].copy()
    mejor_distancia = float('inf')
    mejoras_sin_cambio = 0
    mejor_rutas_generacion = []
    mejor_generacion = 0  # Definir mejor_generacion

    for generacion in range(num_generaciones):
        fitness = evaluar_poblacion(poblacion, matriz_distancias)
        mejor_ruta_actual = poblacion[np.argmax(fitness)]
        distancia_actual = calcular_distancia_ruta(mejor_ruta_actual, matriz_distancias)

        # Actualizar mejor solución
        if distancia_actual < mejor_distancia:
            mejor_distancia = distancia_actual
            mejor_ruta_final = mejor_ruta_actual
            mejoras_sin_cambio = 0
            mejor_generacion = generacion  # Actualizar mejor_generacion
        else:
            mejoras_sin_cambio += 1

        mejor_rutas_generacion.append(mejor_ruta_actual)

        # Detener si no hay mejoras
        if mejoras_sin_cambio >= mejoras_consecutivas:
            print(f"Convergencia alcanzada tras {mejoras_consecutivas} generaciones sin mejora.")
            break

        # Selección, cruce y mutación
        indices_elitismo = np.argsort(fitness)[-elitismo:]  # Índices de los mejores
        nueva_poblacion = [poblacion[i] for i in indices_elitismo]  # Elitismo
        seleccionados = seleccionar_poblacion(poblacion, fitness)
        for i in range(0, len(seleccionados), 2):
            padre1 = seleccionados[i]
            padre2 = seleccionados[i + 1] if i + 1 < len(seleccionados) else seleccionados[0]
            hijo1 = mutar(cruce_ordenado(padre1, padre2), tasa_mutacion_base)
            hijo2 = mutar(cruce_ordenado(padre2, padre1), tasa_mutacion_base)
            nueva_poblacion.extend([hijo1, hijo2])
        poblacion = nueva_poblacion

    print(f"Mejor distancia: {mejor_distancia:.2f}, encontrada en generación {mejor_generacion}.")
    return mejor_rutas_generacion, mejor_ruta_final, mejor_distancia, mejor_generacion, ruta_inicial

# Ejecutar algoritmo genético
ciudad_inicial = 0  # Ciudad A
mejor_rutas_generacion, mejor_ruta_final, mejor_distancia, mejor_generacion, ruta_inicial = algoritmo_genetico_tsp(ciudad_inicial)

# Crear animación
fig, ax = plt.subplots(figsize=(8, 8))

def actualizar(frame):
    ax.clear()
    ruta_actual = mejor_rutas_generacion[frame]
    ruta_coords = [cities[i] for i in ruta_actual] + [cities[ruta_actual[0]]]
    inicial_coords = [cities[i] for i in ruta_inicial] + [cities[ruta_inicial[0]]]

    # Dibujar ruta inicial
    ax.plot([coord[0] for coord in inicial_coords], [coord[1] for coord in inicial_coords], 'r--', label='Ruta Inicial')
    # Dibujar ruta actual
    ax.plot([coord[0] for coord in ruta_coords], [coord[1] for coord in ruta_coords], 'g-', label='Ruta Actual')

    # Dibujar puntos de inicio y fin destacados
    inicio = cities[ruta_actual[0]]
    fin = cities[ruta_actual[-1]]
    ax.scatter(inicio[0], inicio[1], s=150, c='blue', label='Inicio')
    ax.scatter(fin[0], fin[1], s=150, c='orange', label='Fin')

    # Dibujar flechas con pesos
    for i in range(len(ruta_actual)):
        start = cities[ruta_actual[i]]
        end = cities[ruta_actual[(i + 1) % len(ruta_actual)]]
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color='blue'))
        peso = matriz_distancias[ruta_actual[i], ruta_actual[(i + 1) % len(ruta_actual)]]
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, f"{peso:.1f}", color='red')

    for i, city in enumerate(cities):
        ax.text(city[0], city[1], city_names[i], fontsize=10, color='black')

    ax.set_title(f'Generación {frame + 1} - Distancia: {calcular_distancia_ruta(ruta_actual, matriz_distancias):.2f}')
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.grid(True)
    ax.legend()

ani = animation.FuncAnimation(fig, actualizar, frames=len(mejor_rutas_generacion), repeat=False)

# Guardar GIF
ani.save("TSP_Optimization_Enhanced.gif", fps=2, writer='pillow')
print("Animación guardada como TSP_Optimization_Enhanced.gif")
