# Autor: Luis Eduardo Arenas Deseano
# Titulo: Prueba de algoritmo ANFIS
# Materia: Sistemas Inteligentes
# Descripción: llamdo de libreria de algoritmo ANFIS
# Fecha: 01/12/2024


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ANFIS import EVOLUTIONARY_ANFIS
from scipy.stats import pearsonr

# Generación de datos sintéticos
np.random.seed(0)
temperaturas = np.random.uniform(15, 35, 500)
humedades = np.random.uniform(30, 90, 500)
indice_confort = temperaturas - 0.55 * (1 - humedades / 100) * (temperaturas - 58)

# Crear DataFrame
data = pd.DataFrame({
    'temperatura': temperaturas,
    'humedad': humedades,
    'indice_confort': indice_confort
})

# Discretizar índice de confort en categorías
bins = [0, 75, 85, 100]  # Ejemplo de rangos
labels = ['Confortable', 'Desconfortable', 'Peligroso']
data['categoria_confort'] = pd.cut(indice_confort, bins=bins, labels=labels)

# Separar en datos de entrenamiento y prueba
X = data[['temperatura', 'humedad']].values
y = data['indice_confort'].values  # O usar 'categoria_confort' si es clasificador
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Escalado
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test = scaler_y.transform(y_test.reshape(-1, 1))

# Configuración y entrenamiento de ANFIS
E_Anfis = EVOLUTIONARY_ANFIS(functions=3, generations=500, offsprings=10, mutationRate=0.2, learningRate=0.2, chance=0.7, ruleComb="simple")
bestParam, bestModel = E_Anfis.fit(X_train, y_train, optimize_test_data=False)
bestParam, bestModel = E_Anfis.fit(X_train, y_train, X_test, y_test, optimize_test_data=True)

# Predicciones y evaluación
pred_train = E_Anfis.predict(X_train, bestParam, bestModel)
pred_test = E_Anfis.predict(X_test, bestParam, bestModel)
pearson_corr_train, _ = pearsonr(y_train.ravel(), pred_train.ravel())
pearson_corr_test, _ = pearsonr(y_test.ravel(), pred_test.ravel())

print("Pearson correlation for training data:", pearson_corr_train)
print("Pearson correlation for testing data:", pearson_corr_test)
