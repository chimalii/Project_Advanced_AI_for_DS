import pandas as pd
import numpy as np
from algorithms import hypothesis, standardize, train_log_regression, one_hot_encode
from graphs import plot_confusion_matrix, plot_coefficients, plot_performance

# 1. Cargar dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

# 2. Eliminar instancias con categorías no relevantes
df = df[df['gender'] != 'Other']
df = df[df['smoking_history'] != 'No Info']

# 3. Realizar el encoding de las variables categóricas
categorical = ['gender', 'smoking_history']
categorical_encoded_features = []
one_hot_encoded_columns = []
for col_name in categorical:
    one_hot_matrix, categories = one_hot_encode(df[col_name].values)
    one_hot_encoded_columns.append(one_hot_matrix)
    categorical_encoded_features.extend([f"{col_name}_{category}" for category in categories])

# Combinar todas las columnas codificadas y añadirlas al dataframe
df_categorical_encoded = np.hstack(one_hot_encoded_columns)
df_categorical_encoded = pd.DataFrame(df_categorical_encoded, columns=categorical_encoded_features)
df = pd.concat([df.reset_index(drop=True), df_categorical_encoded.reset_index(drop=True)], axis=1)
df.drop(columns=categorical, inplace=True)

# 5. Aplicar la estandarización a las variables numéricas
numerical = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
X_numerical = df[numerical].values
X_stand = standardize(X_numerical)

# Reemplazar las columnas numéricas estandarizadas en el dataframe
df[numerical] = X_stand

# 6. Dividir en grupos de train(60%), validation(20%) y test(20%)
np.random.seed(42)

# Separar características y etiquetas
X = df.drop(columns=['diabetes']).values
y = df['diabetes'].values

# Calcular índices para cada grupo
n = len(y)
n_train = int(0.6 * n)
n_val = int(0.2 * n)

# Mezclar los índices
indices = np.arange(n)
np.random.shuffle(indices)

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

# Crear conjuntos de entrenamiento, validación y prueba
X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

# 7. Entrenar el modelo usando train, validación y test (para seguimiento en cada época)
print('Inicio de entrenamiento...\n')
weights, bias, history = train_log_regression(
    X_train, y_train, 
    X_val, y_val,
    X_test, y_test,
    learning_rate=0.1,
    epochs=3000
)

# Visualizar el proceso de entrenamiento, validación y test
plot_performance(
    history['train_losses'],
    history['val_losses'],
    history['test_losses'],
    history['train_accuracies'],
    history['val_accuracies'],
    history['test_accuracies'],
    history.get('overfit_epochs', None)
)

# 8. Obtener los resultados finales del conjunto de prueba
y_test_pred = hypothesis(X_test, weights, bias)

# Visualizar matriz de confusión en el conjunto de prueba
plot_confusion_matrix(y_test, y_test_pred)

# Visualizar coeficientes del modelo
features = list(df.drop(columns=['diabetes']).columns)
plot_coefficients(weights, features)
