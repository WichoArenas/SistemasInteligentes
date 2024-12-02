from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

# Cargar los datos y dividirlos
X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SVM lineal
model_linear = SVC(kernel='linear')
model_linear.fit(X_train, y_train)

# Para el caso no lineal (con RBF)
model_rbf = SVC(kernel='rbf')
model_rbf.fit(X_train, y_train)

# Hiperplano del modelo lineal
w = model_linear.coef_[0]
b = model_linear.intercept_[0]
print(f"Hiperplano lineal: w={w}, b={b}")

# Puntuaciones de precisión
print("Precisión SVM Lineal:", model_linear.score(X_test, y_test))
print("Precisión SVM RBF:", model_rbf.score(X_test, y_test))
