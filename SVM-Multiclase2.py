# Autor: Luis Eduardo Arenas Deseano
# Titulo: Multiclase SVM 2
# Materia: Sistemas Inteligentes
# Descripción: Calculo de hyper plano en SVM multiclase
# Fecha: 01/12/2024


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

def svm_separated_classes(num_classes=3, points_per_class=50):
    # Generación de datos bien separados
    X, y = datasets.make_blobs(n_samples=num_classes * points_per_class, centers=num_classes, cluster_std=1.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Graficar datos originales
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.title(f'{num_classes} Clases Separadas')
    plt.xlabel('Eje X1')
    plt.ylabel('Eje X2')
    plt.colorbar()
    plt.show()
    
    # Entrenamiento y visualización
    svm_model = SVC(decision_function_shape='ovo', probability=True)
    svm_model.fit(X_train, y_train)
    
    # Visualización de decisión y matriz de confusión
    display_decision_boundary(svm_model, X, y)
    display_confusion_matrix(svm_model, X_test, y_test)
    display_roc_curves(svm_model, X_train, y_train, X_test, y_test, num_classes)

def display_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.title('Separación de Clases usando SVM')
    plt.show()

def display_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title('Matriz de Confusión del Modelo SVM')
    plt.show()

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



def svm_partial_overlap(num_classes=3, points_per_class=50, overlap_classes=1, cluster_std=2.5):
    cluster_std_array = [cluster_std if i < overlap_classes else 1.5 for i in range(num_classes)]
    centers = np.random.uniform(-5, 5, (num_classes, 2))
    X, y = datasets.make_blobs(n_samples=num_classes * points_per_class, centers=centers, cluster_std=cluster_std_array, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.title(f'{num_classes} Clases (Con {overlap_classes} Encimadas)')
    plt.show()
    
    svm_model = SVC(decision_function_shape='ovo', probability=True)
    svm_model.fit(X_train, y_train)
    
    display_decision_boundary(svm_model, X, y)
    display_confusion_matrix(svm_model, X_test, y_test)
    display_roc_curves(svm_model, X_train, y_train, X_test, y_test, num_classes)




def svm_high_overlap(num_classes=3, points_per_class=50, cluster_std=3.5):
    X, y = datasets.make_blobs(n_samples=num_classes * points_per_class, centers=num_classes, cluster_std=cluster_std, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.title(f'{num_classes} Clases (Alta Superposición)')
    plt.show()
    
    svm_model = SVC(decision_function_shape='ovo', probability=True)
    svm_model.fit(X_train, y_train)
    
    display_decision_boundary(svm_model, X, y)
    display_confusion_matrix(svm_model, X_test, y_test)
    display_roc_curves(svm_model, X_train, y_train, X_test, y_test, num_classes)



# Ejecutar función opcion 1
svm_separated_classes(num_classes=8, points_per_class=100)
# Ejecutar función opcion 2
svm_partial_overlap(num_classes=8, points_per_class=100, overlap_classes=1)
# Ejecutar función opcion 3
svm_high_overlap(num_classes=8, points_per_class=100)
