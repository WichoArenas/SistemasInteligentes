# Autor: Luis Eduardo Arenas Deseano
# Titulo: Series de tiempo con Ciudad y APi de Open Weather
# Materia: Sistemas Inteligentes
# Descripción: Prueba de algoritmo de series de tiempo con conexion de APi a OpenWeather
# Fecha: 01/12/2024

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import requests
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


# API de OpenWeatherMap
API_KEY = '341d91b86abd97e3931935e2ca301021'
CIUDAD = 'Puebla'

# Función para obtener datos de temperatura del pronóstico (forecast a 5 días)
def obtener_datos_climaticos(ciudad):
    url = f'http://api.openweathermap.org/data/2.5/forecast?q={ciudad}&appid={API_KEY}&units=metric'
    respuesta = requests.get(url)
    datos = respuesta.json()

    # Datos actuales y pronóstico para las próximas 24 horas
    temp_actual = datos['list'][0]['main']['temp']
    temp_min = min([periodo['main']['temp_min'] for periodo in datos['list'][:8]])  # Mínimo en las próximas 24 horas
    temp_max = max([periodo['main']['temp_max'] for periodo in datos['list'][:8]])  # Máximo en las próximas 24 horas

    return temp_actual, temp_min, temp_max

# Obtener los datos de temperatura
temp_actual, temp_min, temp_max = obtener_datos_climaticos(CIUDAD)

# Simulamos la obtención de datos históricos para formar una serie de tiempo
# En este caso, se simulan los últimos tres días
temp_dia1 = temp_actual - 3  # Hace 2 días más frío
temp_dia2 = temp_actual - 2  # Hace un día más frío
temp_dia3 = temp_actual - 1  # Hoy un poco más cálido

# Datos históricos de temperatura (simulación)
historico_temps = [temp_dia1, temp_dia2, temp_dia3]
dias = ['Hace 2 días', 'Ayer', 'Hoy', 'Mañana']

# Calcular el cambio de temperatura (diferencia entre hoy y ayer)
cambio_temp = temp_dia3 - temp_dia2

# Calcular la tendencia de los últimos días (diferencia entre el día 3 y el día 1)
tendencia = temp_dia3 - temp_dia1

# ----------- Definición del sistema difuso ------------ #
# Definición de las variables difusas
cambio_temp_fuzzy = ctrl.Antecedent(np.arange(-10, 11, 1), 'cambio_temp')
tendencia_fuzzy = ctrl.Antecedent(np.arange(-10, 11, 1), 'tendencia')
prediccion_temp_fuzzy = ctrl.Consequent(np.arange(-10, 41, 1), 'prediccion_temp')

# Definir las funciones de membresía para las variables de entrada y salida
cambio_temp_fuzzy['bajo'] = fuzz.trimf(cambio_temp_fuzzy.universe, [-10, -5, 0])
cambio_temp_fuzzy['moderado'] = fuzz.trimf(cambio_temp_fuzzy.universe, [-5, 0, 5])
cambio_temp_fuzzy['alto'] = fuzz.trimf(cambio_temp_fuzzy.universe, [0, 5, 10])

tendencia_fuzzy['negativa'] = fuzz.trimf(tendencia_fuzzy.universe, [-10, -5, 0])
tendencia_fuzzy['neutra'] = fuzz.trimf(tendencia_fuzzy.universe, [-5, 0, 5])
tendencia_fuzzy['positiva'] = fuzz.trimf(tendencia_fuzzy.universe, [0, 5, 10])

prediccion_temp_fuzzy['fria'] = fuzz.trimf(prediccion_temp_fuzzy.universe, [-10, 0, 10])
prediccion_temp_fuzzy['templada'] = fuzz.trimf(prediccion_temp_fuzzy.universe, [10, 20, 30])
prediccion_temp_fuzzy['caliente'] = fuzz.trimf(prediccion_temp_fuzzy.universe, [20, 30, 40])

# Definir las reglas difusas
rule1 = ctrl.Rule(cambio_temp_fuzzy['bajo'] & tendencia_fuzzy['negativa'], prediccion_temp_fuzzy['fria'])
rule2 = ctrl.Rule(cambio_temp_fuzzy['moderado'] & tendencia_fuzzy['neutra'], prediccion_temp_fuzzy['templada'])
rule3 = ctrl.Rule(cambio_temp_fuzzy['alto'] & tendencia_fuzzy['positiva'], prediccion_temp_fuzzy['caliente'])

# Crear el sistema de control difuso
sistema_control = ctrl.ControlSystem([rule1, rule2, rule3])
simulacion = ctrl.ControlSystemSimulation(sistema_control)

# Ingresar los valores obtenidos de los últimos días
simulacion.input['cambio_temp'] = cambio_temp
simulacion.input['tendencia'] = tendencia

# Ejecutar la simulación difusa
simulacion.compute()

# Obtener la predicción para la temperatura del próximo día
prediccion_futura = simulacion.output['prediccion_temp']

# --------- Validación del modelo difuso --------- #
# Simulamos valores reales históricos de los últimos días (simulación)
historico_real = [temp_dia1, temp_dia2, temp_dia3, temp_actual + 2]  # Ejemplo: temperaturas reales
predicciones_difusas = historico_temps + [prediccion_futura]

# Cálculo de MAE y RMSE
mae = mean_absolute_error(historico_real, predicciones_difusas)
rmse = np.sqrt(mean_squared_error(historico_real, predicciones_difusas))

# Imprimir las temperaturas actuales, mínimas y máximas de mañana en consola
print(f"Temperatura actual: {temp_actual:.2f} °C")
print(f"Temperatura mínima prevista para mañana: {temp_min:.2f} °C")
print(f"Temperatura máxima prevista para mañana: {temp_max:.2f} °C")
print(f"Predicción difusa de la temperatura para mañana: {prediccion_futura:.2f} °C")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")

# --------- Visualización de la serie de tiempo y predicción ---------- #
# Agregamos la predicción a los datos históricos
historico_temps.append(prediccion_futura)

# Graficar la serie de tiempo y la predicción
plt.plot(dias, historico_temps, marker='o', linestyle='-', color='b', label='Predicciones difusas')
plt.plot(dias, historico_real, marker='x', linestyle='--', color='r', label='Valores reales')
plt.axhline(y=temp_min, color='green', linestyle='--', label='Mínima prevista')
plt.axhline(y=temp_max, color='orange', linestyle='--', label='Máxima prevista')
plt.title('Aproximación de la serie de tiempo (Temperaturas)')
plt.xlabel('Días')
plt.ylabel('Temperatura (°C)')
plt.grid(True)
plt.legend()
plt.show()

# Mostrar la salida final en consola
print(f"Temperaturas de los últimos días: {historico_temps[:-1]}")
