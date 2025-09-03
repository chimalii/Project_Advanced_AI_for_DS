import pandas as pd
import numpy as np
from algorithms import hypothesis, standardize, train_log_regression
from graphs import plot_confusion_matrix, plot_coefficients, plot_performance

# 1. Cargar dataset
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# 2. Separar variables a estandarizar y variables binarias
binary_features = ['HighBP', 'HighChol', 'DiffWalk', 'HeartDiseaseorAttack', 'Sex']
non_binary_features = ['GenHlth', 'BMI', 'Age', 'Income', 'PhysHlth']

# 3. Extraer datos según su tipo
X_binary = df[binary_features].values
X_non_binary = df[non_binary_features].values
y = df["Diabetes_binary"].values

# 4. Estandarizar las variables numéricas y ordinales
X_stand = standardize(X_non_binary)

# 5. Combinar variables binarias y estandarizadas
X = np.hstack((X_binary, X_stand))

# 6. Dividir en train y test (80% train, 20% test) manteniendo el balance 50-50
np.random.seed(42)

# Separar índices de cada clase
idx_class_0 = np.where(y == 0)[0]
idx_class_1 = np.where(y == 1)[0]

# Mezclar aleatoriamente cada conjunto de índices
np.random.shuffle(idx_class_0)
np.random.shuffle(idx_class_1)

# Calcular cantidad para entrenamiento (80% de cada clase)
n_train_0 = int(0.8 * len(idx_class_0))
n_train_1 = int(0.8 * len(idx_class_1))

# Crear índices para train y test
train_idx = np.concatenate([idx_class_0[:n_train_0], idx_class_1[:n_train_1]])
test_idx = np.concatenate([idx_class_0[n_train_0:], idx_class_1[n_train_1:]])

# Mezclar los índices de entrenamiento y prueba para evitar patrones
np.random.shuffle(train_idx)
np.random.shuffle(test_idx)

# Crear conjuntos de entrenamiento y prueba
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 7. Entrenar el modelo y mostrar resultados
weights, bias, history = train_log_regression(X_train, y_train, X_test, y_test, learning_rate=0.1, epochs=1500)

plot_performance(
    history['train_losses'],
    history['test_losses'],
    history['train_accuracies'],
    history['test_accuracies']
)

y_test_pred = hypothesis(X_test, weights, bias)
plot_confusion_matrix(y_test, y_test_pred)

features = binary_features + non_binary_features
plot_coefficients(weights, features)
