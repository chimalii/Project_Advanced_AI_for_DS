import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_performance(
        train_losses, val_losses, test_losses, train_accuracies, 
        val_accuracies, test_accuracies, overfit_epochs=None):
    """
    Curvas de loss y accuracy para train, validación y test. 
    Marca sobreajuste si se indica.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.plot(val_losses, label="Validation Loss", color='green')
    plt.plot(test_losses, label="Test Loss", color='red')
    if overfit_epochs:
        for epoch in overfit_epochs:
            plt.axvline(
                x=epoch-1, 
                color='orange', 
                linestyle='--', 
                alpha=0.7, 
                label='Overfitting Warning' if epoch == overfit_epochs[0] else None)
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Curva de error (Loss)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Train Accuracy", color='blue')
    plt.plot(val_accuracies, label="Validation Accuracy", color='green')
    plt.plot(test_accuracies, label="Test Accuracy", color='red')
    if overfit_epochs:
        for epoch in overfit_epochs:
            plt.axvline(
                x=epoch-1, 
                color='orange', 
                linestyle='--', 
                alpha=0.7, 
                label='Overfitting Warning' if epoch == overfit_epochs[0] else None)
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title("Curva de accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Matriz de confusión para predicciones binarias."""
    # Convertir probabilidades a etiquetas binarias
    y_pred_labels = (y_pred >= 0.5).astype(int)

    # Calcular los valores de la matriz de confusión
    tn = np.sum((y_true == 0) & (y_pred_labels == 0))  # Verdaderos negativos
    fp = np.sum((y_true == 0) & (y_pred_labels == 1))  # Falsos positivos
    fn = np.sum((y_true == 1) & (y_pred_labels == 0))  # Falsos negativos
    tp = np.sum((y_true == 1) & (y_pred_labels == 1))  # Verdaderos positivos
    
    confusion_matrix = np.array([
        [tn, fp],
        [fn, tp]
    ])

    plt.figure(figsize=(4, 4))
    plt.imshow(confusion_matrix, cmap="Blues")
    plt.title("Matriz de confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(confusion_matrix[i, j]), 
                    ha="center", va="center", color="black", fontsize=16)
    plt.colorbar()
    plt.show()
    
def plot_coefficients(coefficients, feature_names):
    """
    Grafica los coeficientes del modelo para interpretar la importancia de cada 
    variable.
    """
    # Ordenar coeficientes por su valor absoluto
    sorted_indices = np.argsort(np.abs(coefficients))
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(coefficients)), coefficients[sorted_indices], 
            align='center')
    plt.yticks(range(len(coefficients)), 
              [feature_names[i] for i in sorted_indices])
    plt.xlabel('Coeficiente')
    plt.title('Impacto de cada variable en la predicción de prediabetes/diabetes')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_proba):
    """
    Grafica la curva ROC y muestra el valor AUC.
    Args:
        y_true: Etiquetas verdaderas (0 o 1)
        y_proba: Probabilidades predichas para la clase positiva
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()