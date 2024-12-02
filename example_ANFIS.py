# Autor: Luis Eduardo Arenas Deseano
# Titulo: Ejemplo ANFIS
# Materia: Sistemas Inteligentes
# Descripción: Prueba de algoritmo ANFIS
# Fecha: 01/12/2024


from sklearn.model_selection import train_test_split 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from ANFIS import EVOLUTIONARY_ANFIS

# Cargar datos
data = pd.read_excel('walkertrain.xlsx')
test = pd.read_excel('walkertest.xlsx')

Data = data.loc[:,['x','y','v']]
test = test.loc[:,['x','y','v']]

# Digitizar variable continua
aa = Data['v']
minima = aa.min()
maxima = aa.max()
bins = np.linspace(minima-1, maxima+1, 3)
binned = np.digitize(aa, bins)
plt.hist(binned, bins=50)
data_train, data_test = train_test_split(Data, test_size=0.2, random_state=101, stratify=binned)

# Separar características y etiquetas
X_train = data_train.drop("v", axis=1).values
y_train = data_train["v"].values
X_test = data_test.drop("v", axis=1).values
y_test = data_test["v"].values
X_val = test.drop("v", axis=1).values
y_val = test["v"].values

# Escalar datos
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(X_train)
X_train = scaler_x.transform(X_train)
X_test = scaler_x.transform(X_test)
#X_val = scaler_x.transform(X_val)
scaler_y.fit(y_train.reshape(-1,1))
y_train = scaler_y.transform(y_train.reshape(-1,1))
y_test = scaler_y.transform(y_test.reshape(-1,1))
#y_val = scaler_y.transform(y_val.reshape(-1,1))

# Configuración y entrenamiento de ANFIS
E_Anfis = EVOLUTIONARY_ANFIS(functions=3, generations=500, offsprings=10,
                             mutationRate=0.2, learningRate=0.2, chance=0.7, ruleComb="simple")

bestParam, bestModel = E_Anfis.fit(X_train, y_train, optimize_test_data=False)
bestParam, bestModel = E_Anfis.fit(X_train, y_train, X_test, y_test, optimize_test_data=True)

# Predicciones y cálculo de Pearson
pred_train = E_Anfis.predict(X_train, bestParam, bestModel)
pearson_corr_train, _ = pearsonr(y_train.ravel(), pred_train.ravel())
print("Pearson correlation for training data:", pearson_corr_train)

pred_test = E_Anfis.predict(X_test, bestParam, bestModel)
pearson_corr_test, _ = pearsonr(y_test.ravel(), pred_test.ravel())
print("Pearson correlation for testing data:", pearson_corr_test)
