"""
Framework mejorado para análisis y predicción de diabetes
Convertido desde framework_mejorado.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from graphs import plot_confusion_matrix, plot_coefficients

def main():
    # 1. Cargar dataset
    print("1. Cargando el dataset...")
    # Nota: Modificar la ruta al dataset según corresponda
    df = pd.read_csv("diabetes_prediction_dataset.csv")
    
    # Verificar valores nulos
    print("\nValores nulos en el dataset:")
    print(df.isnull().sum())
    
    # 2. Eliminar instancias con categorías no relevantes
    print("\n2. Eliminando instancias con categorías no relevantes...")
    df = df[df['gender'] != 'Other']
    df = df[df['smoking_history'] != 'No Info']
    
    # 3. Realizar el encoding de las variables categóricas
    print("\n3. Aplicando one-hot encoding a variables categóricas...")
    categorical_cols = df.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 4. Eliminar duplicados y balancear las categorías al 50%-50%
    print("\n4. Eliminando duplicados y balanceando categorías...")
    print(f"Número de duplicados iniciales: {df_encoded.duplicated().sum()}")
    df_encoded = df_encoded.drop_duplicates()
    print(f"Número de duplicados después de eliminarlos: {df_encoded.duplicated().sum()}")
    
    # Separar instancias por clase
    df_class_0 = df_encoded[df_encoded['diabetes'] == 0]
    df_class_1 = df_encoded[df_encoded['diabetes'] == 1]
    
    # Determinar el número de instancias en la clase minoritaria (clase 1)
    num_minority_class = len(df_class_1)
    
    # Tomar muestras aleatorias de la clase mayoritaria (clase 0)
    df_class_0_sampled = df_class_0.sample(num_minority_class, random_state=42)
    
    # Concatenar las clases para crear un DataFrame balanceado
    df_balanced = pd.concat([df_class_0_sampled, df_class_1])
    
    # Mezclar el DataFrame balanceado para aleatorizar las instancias
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Forma del DataFrame balanceado: {df_balanced.shape}")
    print("\nDistribución de 'diabetes' en el DataFrame balanceado:")
    print(df_balanced['diabetes'].value_counts())
    
    # 5. Preparar los datos para el modelo
    print("\n5. Preparando los datos para el modelo...")
    features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 
                'heart_disease', 'gender_Male', 'smoking_history_ever', 'smoking_history_former', 
                'smoking_history_never', 'smoking_history_not current']
    
    X = df_balanced[features].values
    y = df_balanced["diabetes"].values
    
    # 6. Dividir en train(60%), validación(20%) y test(20%), manteniendo el balance
    print("\n6. Dividiendo los datos en conjuntos de entrenamiento, validación y prueba...")
    np.random.seed(42)
    
    # Separar índices por clase
    idx_class_0 = np.where(y == 0)[0]
    idx_class_1 = np.where(y == 1)[0]
    
    # Mezclar aleatoriamente los índices
    np.random.shuffle(idx_class_0)
    np.random.shuffle(idx_class_1)
    
    # Calcular cantidad para cada grupo
    n_0 = len(idx_class_0)
    n_1 = len(idx_class_1)
    
    n_train_0 = int(0.6 * n_0)
    n_val_0 = int(0.2 * n_0)
    n_test_0 = int(0.2 * n_0)
    
    n_train_1 = int(0.6 * n_1)
    n_val_1 = int(0.2 * n_1)
    n_test_1 = int(0.2 * n_1)
    
    # Crear índices para cada grupo
    train_idx = np.concatenate([idx_class_0[:n_train_0], idx_class_1[:n_train_1]])
    val_idx = np.concatenate([idx_class_0[n_train_0:n_train_0+n_val_0], idx_class_1[n_train_1:n_train_1+n_val_1]])
    test_idx = np.concatenate([idx_class_0[n_train_0+n_val_0:], idx_class_1[n_train_1+n_val_1:]])
    
    # Mezclar los índices de cada grupo para evitar patrones
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    # Crear conjuntos de entrenamiento, validación y prueba
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    
    # 7. Crear y entrenar el modelo Random Forest
    print("\n7. Creando y entrenando el modelo Random Forest...")
    rf = RandomForestClassifier(
        max_depth=10,          # Niveles
        n_estimators=200,      # Número de árboles en el bosque
        max_leaf_nodes=4,      # Número máximo de nodos por árbol
        min_samples_split=5,   # Mínimo de muestras para dividir un nodo
        min_samples_leaf=5,    # Mínimo de muestras en una hoja
        n_jobs=-1,             # Usar todos los procesadores disponibles
        random_state=42  
    )
    
    # Entrenar el modelo
    rf.fit(X_train, y_train)
    
    # 8. Evaluar el modelo
    print("\n8. Evaluando el modelo...")
    y_pred_train = rf.predict(X_train)
    y_pred_val = rf.predict(X_val)
    y_pred_test = rf.predict(X_test)
    
    # Calcular precisión
    train_accuracy = accuracy_score(y_train, y_pred_train)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\nRandom Forest - Train Accuracy: {train_accuracy:.4f}")
    print(f"Random Forest - Validation Accuracy: {val_accuracy:.4f}")
    print(f"Random Forest - Test Accuracy: {test_accuracy:.4f}")
    
    # Calcular cross-entropy error
    y_pred_train_probs = rf.predict_proba(X_train)[:, 1]
    y_pred_val_probs = rf.predict_proba(X_val)[:, 1]
    y_pred_test_probs = rf.predict_proba(X_test)[:, 1]
    
    train_log_loss = log_loss(y_train, y_pred_train_probs)
    val_log_loss = log_loss(y_val, y_pred_val_probs)
    test_log_loss = log_loss(y_test, y_pred_test_probs)
    
    print(f"\nRandom Forest - Train Log Loss: {train_log_loss:.4f}")
    print(f"Random Forest - Validation Log Loss: {val_log_loss:.4f}")
    print(f"Random Forest - Test Log Loss: {test_log_loss:.4f}")
    
    # 9. Métricas adicionales
    print("\n9. Calculando métricas adicionales...")

    # Obtener y visualizar las importancias de características usando la función plot_coefficients
    importances = rf.feature_importances_
    feat_imp = pd.DataFrame({"feature": features, "importance": importances})
    print("\nSignificancia de las características (ordenadas):")
    print(feat_imp.sort_values("importance", ascending=False))
    
    # Visualizar las importancias de características gráficamente
    print("\nVisualizando la importancia de las características...")
    plot_coefficients(rf.feature_importances_, features)
    
    # Mostrar el reporte de clasificación en la consola
    print("\nReporte de clasificación:")
    print(classification_report(y_test, rf.predict(X_test)))
    
    # Visualizar la matriz de confusión gráficamente
    print("\nVisualizando matriz de confusión...")
    # Probabilidades para plot_confusion_matrix
    y_pred_probs = rf.predict_proba(X_test)[:, 1]
    plot_confusion_matrix(y_test, y_pred_probs)
    
    # Probabilidades
    y_proba = rf.predict_proba(X_test)[:, 1]
    
    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # PR AUC (average precision)
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"PR AUC: {pr_auc:.4f}")

    return

# Ejecutar
main()