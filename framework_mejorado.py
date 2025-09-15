import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, classification_report, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from graphs import plot_confusion_matrix, plot_coefficients, plot_roc_curve

def main():
    # 1. Cargar dataset
    print("1. Cargando el dataset...")
    df = pd.read_csv("diabetes_prediction_dataset.csv")
    
    # 2. Eliminar instancias con categorías no relevantes
    print("\n2. Eliminando instancias con categorías no relevantes...")
    df = df[df['gender'] != 'Other']
    df = df[df['smoking_history'] != 'No Info']
    
    # 3. Realizar el encoding de las variables categóricas
    print("\n3. Aplicando one-hot encoding a variables categóricas...")
    categorical_cols = df.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 4. Preparar los datos para el modelo
    print("\n5. Preparando los datos para el modelo...")
    features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 
                'heart_disease']
    
    X = df_encoded[features].values
    y = df_encoded["diabetes"].values
    
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
    #n_test_0 = int(0.2 * n_0)
    
    n_train_1 = int(0.6 * n_1)
    n_val_1 = int(0.2 * n_1)
    #n_test_1 = int(0.2 * n_1)
    
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
        class_weight='balanced',
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=6,
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
    
    # Calcular F1-score
    train_f1 = f1_score(y_train, y_pred_train)
    val_f1 = f1_score(y_val, y_pred_val)
    test_f1 = f1_score(y_test, y_pred_test)
    
    print(f"\nTrain Accuracy: {train_accuracy:.4f} | F1: {train_f1:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f} | F1: {val_f1:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} | F1: {test_f1:.4f}")
    
    # Calcular cross-entropy
    y_pred_train_probs = rf.predict_proba(X_train)[:, 1]
    y_pred_val_probs = rf.predict_proba(X_val)[:, 1]
    y_pred_test_probs = rf.predict_proba(X_test)[:, 1]
    
    train_log_loss = log_loss(y_train, y_pred_train_probs)
    val_log_loss = log_loss(y_val, y_pred_val_probs)
    test_log_loss = log_loss(y_test, y_pred_test_probs)
    
    print(f"\nTrain Loss: {train_log_loss:.4f}")
    print(f"Validation Loss: {val_log_loss:.4f}")
    print(f"Test Loss: {test_log_loss:.4f}")
    
    # 8.1 Métricas adicionales
    
    # Obtener y visualizar valor de los coeficientes
    importances = rf.feature_importances_
    feat_imp = pd.DataFrame({"feature": features, "importance": importances})
    print("\nSignificancia de las características (ordenadas):")
    print(feat_imp.sort_values("importance", ascending=False))
    plot_coefficients(rf.feature_importances_, features)
    
    # Mostrar el reporte de clasificación
    print("\nReporte de clasificación:")
    print(classification_report(y_test, rf.predict(X_test)))
    
    # Visualizar la matriz de confusión
    print("\nVisualizando matriz de confusión...")
    y_pred_probs = rf.predict_proba(X_test)[:, 1]
    plot_confusion_matrix(y_test, y_pred_probs)
    
    # Probabilidades
    y_proba = rf.predict_proba(X_test)[:, 1]
    
    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC AUC: {roc_auc:.4f}")
    plot_roc_curve(y_test, y_proba)
    
    return

# Ejecutar
main()