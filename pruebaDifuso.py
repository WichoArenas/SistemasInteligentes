import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Definición de las variables de entrada
servicio = ctrl.Antecedent(np.arange(0, 11, 1), 'servicio')  # Calidad del servicio de 0 a 10
comida = ctrl.Antecedent(np.arange(0, 11, 1), 'comida')      # Calidad de la comida de 0 a 10

# Definición de la variable de salida
propina = ctrl.Consequent(np.arange(0, 26, 1), 'propina')    # Propina sugerida de 0% a 25%

# Definición de las funciones de memebresia para servicio
servicio['malo'] = fuzz.trimf(servicio.universe, [0, 0, 5])
servicio['aceptable'] = fuzz.trimf(servicio.universe, [0, 5, 10])
servicio['excelente'] = fuzz.trimf(servicio.universe, [5, 10, 10])

# Definición de las funciones de membresia para comida
comida['mala'] = fuzz.trimf(comida.universe, [0, 0, 5])
comida['aceptable'] = fuzz.trimf(comida.universe, [0, 5, 10])
comida['deliciosa'] = fuzz.trimf(comida.universe, [5, 10, 10])

# Definición de las funciones de membresia para propina
propina['baja'] = fuzz.trimf(propina.universe, [0, 0, 13])
propina['media'] = fuzz.trimf(propina.universe, [0, 13, 25])
propina['alta'] = fuzz.trimf(propina.universe, [13, 25, 25])

# Crear una figura con subplots para mostrar todas las funciones de membresia y el resultado
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Visualización de las funciones de pertenencia de 'servicio'
for label in servicio.terms:
    axs[0, 0].plot(servicio.universe, servicio[label].mf, label=label)
axs[0, 0].set_title('Funciones de membresia del Servicio')
axs[0, 0].legend()

# Visualización de las funciones de pertenencia de 'comida'
for label in comida.terms:
    axs[0, 1].plot(comida.universe, comida[label].mf, label=label)
axs[0, 1].set_title('Funciones de membresia de la Comida')
axs[0, 1].legend()

# Visualización de las funciones de pertenencia de 'propina'
for label in propina.terms:
    axs[1, 0].plot(propina.universe, propina[label].mf, label=label)
axs[1, 0].set_title('Funciones de membresia de la Propina')
axs[1, 0].legend()

# Definición de las reglas difusas
rule1 = ctrl.Rule(servicio['malo'] | comida['mala'], propina['baja'])
rule2 = ctrl.Rule(servicio['aceptable'], propina['media'])
rule3 = ctrl.Rule(servicio['excelente'] | comida['deliciosa'], propina['alta'])

# Creación del sistema de control difuso con las reglas definidas
tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

# Simulación del sistema con entradas específicas
tipping.input['servicio'] = 8.5  # calidad del servicio
tipping.input['comida'] = 7.0    # calidad de la comida

# Computar la salida difusa
tipping.compute()

# Mostrar la cantidad de propina sugerida
print(f"La propina sugerida es: {tipping.output['propina']:.2f}")

# Mostrar la inferencia difusa en el subplot adecuado
# Guardar el estado original de la figura activa
original_fig = plt.gcf()

# Redirigir la gráfica al subplot (1, 1)
plt.sca(axs[1, 1])
propina.view(sim=tipping)
axs[1, 1].set_title('Resultado de la Inferencia Difusa para la Propina')

# Restaurar la figura activa
plt.figure(original_fig.number)
plt.title('Resultado de la Inferencia Difusa para la Propina')

# Ajustar el layout para evitar que los subplots se sobrepongan
plt.tight_layout()
plt.show()
