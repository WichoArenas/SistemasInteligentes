# Autor: Luis Eduardo Arenas Deseano
# Titulo: Algoritmo difuso para calefaccion
# Materia: Sistemas Inteligentes
# Descripción: Prueba de algoritmo difuso para calcular la calefaccion
# Fecha: 01/12/2024


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Paso 1: Definir las variables de entrada y salida
# Temperatura de 0 a 40 grados Celsius
temperatura = ctrl.Antecedent(np.arange(0, 41, 1), 'temperatura')
# Humedad de 0 a 100 por ciento
humedad = ctrl.Antecedent(np.arange(0, 101, 1), 'humedad')
# Nivel de calefacción de 0 a 100 por ciento
calefaccion = ctrl.Consequent(np.arange(0, 101, 1), 'calefaccion')

# Paso 2: Definir las funciones de pertenencia
# Para temperatura: frío, cómodo, caliente
temperatura['frio'] = fuzz.trimf(temperatura.universe, [0, 0, 15])
temperatura['comodo'] = fuzz.trimf(temperatura.universe, [10, 20, 30])
temperatura['caliente'] = fuzz.trimf(temperatura.universe, [25, 40, 40])

# Para humedad: baja, media, alta
humedad['baja'] = fuzz.trimf(humedad.universe, [0, 0, 50])
humedad['media'] = fuzz.trimf(humedad.universe, [30, 50, 70])
humedad['alta'] = fuzz.trimf(humedad.universe, [60, 100, 100])

# Para calefacción: baja, media, alta
calefaccion['baja'] = fuzz.trimf(calefaccion.universe, [0, 0, 50])
calefaccion['media'] = fuzz.trimf(calefaccion.universe, [25, 50, 75])
calefaccion['alta'] = fuzz.trimf(calefaccion.universe, [50, 100, 100])

# Paso 3: Visualizar las funciones de pertenencia
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Visualización de las funciones de pertenencia de 'temperatura'
for label in temperatura.terms:
    axs[0, 0].plot(temperatura.universe, temperatura[label].mf, label=label)
axs[0, 0].set_title('Funciones de Pertenencia de la Temperatura')
axs[0, 0].legend()

# Visualización de las funciones de pertenencia de 'humedad'
for label in humedad.terms:
    axs[0, 1].plot(humedad.universe, humedad[label].mf, label=label)
axs[0, 1].set_title('Funciones de Pertenencia de la Humedad')
axs[0, 1].legend()

# Visualización de las funciones de pertenencia de 'calefaccion'
for label in calefaccion.terms:
    axs[1, 0].plot(calefaccion.universe, calefaccion[label].mf, label=label)
axs[1, 0].set_title('Funciones de Pertenencia de la Calefacción')
axs[1, 0].legend()

# Paso 4: Definir las reglas difusas
# Regla 1: Si la temperatura es fría O la humedad es alta, entonces la calefacción es alta
rule1 = ctrl.Rule(temperatura['frio'] | humedad['alta'], calefaccion['alta'])
# Regla 2: Si la temperatura es cómoda, entonces la calefacción es media
rule2 = ctrl.Rule(temperatura['comodo'], calefaccion['media'])
# Regla 3: Si la temperatura es caliente Y la humedad es baja, entonces la calefacción es baja
rule3 = ctrl.Rule(temperatura['caliente'] & humedad['baja'], calefaccion['baja'])

# Paso 5: Crear el sistema de control difuso
calefaccion_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
calefaccion_simulador = ctrl.ControlSystemSimulation(calefaccion_ctrl)

# Paso 6: Probar el sistema con valores específicos
# Ajustar los valores de entrada
calefaccion_simulador.input['temperatura'] = 35  # Grados Celsius
calefaccion_simulador.input['humedad'] = 30     # Porcentaje de humedad

# Computar la salida difusa
calefaccion_simulador.compute()

# Mostrar la cantidad de calefacción sugerida
print(f"El nivel de calefacción sugerido es: {calefaccion_simulador.output['calefaccion']:.2f}%")

# Mostrar la inferencia difusa en el subplot adecuado
plt.sca(axs[1, 1])
calefaccion.view(sim=calefaccion_simulador)
axs[1, 1].set_title('Resultado de la Inferencia Difusa para la Calefacción')

# Ajustar el layout para evitar que los subplots se sobrepongan
plt.tight_layout()
plt.show()
