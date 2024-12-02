import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

# Función auxiliar para graficar el hiperplano y los vectores de soporte
def plot_hyperplane(svm_model, X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], 
                s=100, facecolors='none', edgecolors='k', label='Vectores de Soporte')
    
    # Obtener coeficientes y calcular el hiperplano
    for i, coef in enumerate(svm_model.dual_coef_):
        intercept = svm_model.intercept_[i]
        slope = -coef[0] / coef[1] if coef[1] != 0 else 0
        intercept_line = intercept / coef[1] if coef[1] != 0 else 0
        
        # Calcular las líneas de margen y el hiperplano
        x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        y_line = slope * x_line + intercept_line
        plt.plot(x_line, y_line, 'k--', label=f'Hiperplano Clase {i+1}')
    
    plt.legend()
    plt.title('Hiperplano y Vectores de Soporte')
    plt.xlabel('Eje X1')
    plt.ylabel('Eje X2')
    plt.show()

# Función auxiliar para graficar la matriz de confusión
def display_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title('Matriz de Confusión del Modelo SVM')
    plt.show()

# Función auxiliar para graficar las curvas ROC
def display_roc_curves(model, X_train, y_train, X_test, y_test, num_classes):
    y_bin = label_binarize(y_test, classes=np.unique(y_test))
    classifier = OneVsRestClassifier(model)
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
    
    plt.figure(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_classes), colors):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'Clase {i+1} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curvas ROC para Modelo SVM')
    plt.legend(loc="lower right")
    plt.show()

# Escenario 1: Clases Separadas
def svm_separated_classes(num_classes=3, points_per_class=50):
    X, y = datasets.make_blobs(n_samples=num_classes * points_per_class, centers=num_classes, cluster_std=1.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Graficar los datos generados
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.title(f'{num_classes} Clases Separadas')
    plt.xlabel('Eje X1')
    plt.ylabel('Eje X2')
    plt.colorbar()
    plt.show()
    
    # Entrenamiento del modelo
    svm_model = SVC(kernel='linear', decision_function_shape='ovo', probability=True)
    svm_model.fit(X_train, y_train)
    
    # Graficar la separación y el hiperplano
    plot_hyperplane(svm_model, X, y)
    display_confusion_matrix(svm_model, X_test, y_test)
    display_roc_curves(svm_model, X_train, y_train, X_test, y_test, num_classes)

# Escenario 2: Clases con Superposición Parcial
def svm_partial_overlap(num_classes=3, points_per_class=50, overlap_classes=1, cluster_std=2.5):
    cluster_std_array = [cluster_std if i < overlap_classes else 1.5 for i in range(num_classes)]
    centers = np.random.uniform(-5, 5, (num_classes, 2))
    X, y = datasets.make_blobs(n_samples=num_classes * points_per_class, centers=centers, cluster_std=cluster_std_array, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Graficar los datos generados
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.title(f'{num_classes} Clases (Con {overlap_classes} Encimadas)')
    plt.xlabel('Eje X1')
    plt.ylabel('Eje X2')
    plt.colorbar()
    plt.show()
    
    # Entrenamiento del modelo
    svm_model = SVC(kernel='linear', decision_function_shape='ovo', probability=True)
    svm_model.fit(X_train, y_train)
    
    # Graficar la separación y el hiperplano
    plot_hyperplane(svm_model, X, y)
    display_confusion_matrix(svm_model, X_test, y_test)
    display_roc_curves(svm_model, X_train, y_train, X_test, y_test, num_classes)

# Escenario 3: Alta Superposición
def svm_high_overlap(num_classes=3, points_per_class=50, cluster_std=3.5):
    X, y = datasets.make_blobs(n_samples=num_classes * points_per_class, centers=num_classes, cluster_std=cluster_std, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Graficar los datos generados
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.title(f'{num_classes} Clases (Alta Superposición)')
    plt.xlabel('Eje X1')
    plt.ylabel('Eje X2')
    plt.colorbar()
    plt.show()
    
    # Entrenamiento del modelo
    svm_model = SVC(kernel='linear', decision_function_shape='ovo', probability=True)
    svm_model.fit(X_train, y_train)
    
    # Graficar la separación y el hiperplano
    plot_hyperplane(svm_model, X, y)
    display_confusion_matrix(svm_model, X_test, y_test)
    display_roc_curves(svm_model, X_train, y_train, X_test, y_test, num_classes)

# Ejecutar las funciones de prueba para cada escenario
# Puedes descomentar las funciones siguientes para probar cada caso:

svm_separated_classes(num_classes=4, points_per_class=100)
svm_partial_overlap(num_classes=4, points_per_class=100, overlap_classes=1)
svm_high_overlap(num_classes=4, points_per_class=100)
