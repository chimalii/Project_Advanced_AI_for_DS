import numpy as np

# ======= MÉTODOS DE PREPROCESAMIENTO =======

def standardize(X):
    """
    Estandariza columnas de X con Z-score: (x - media) / desviación estándar
    """
    X = np.asarray(X, dtype=float)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1
    z_scores = (X - means) / stds
    return z_scores

def one_hot_encode(column):
    """
    Aplica one-hot encoding a una columna de valores categóricos.
    Siempre elimina la primera categoría para evitar multicolinealidad (n-1 categorías).

    Parámetros:
        column (np.array): arreglo de valores categóricos.

    Retorna:
        np.array: matriz 2D con one-hot encoding (n-1 categorías).
        list: lista de categorías codificadas.
    """
    categories = np.unique(column)
    encoding_categories = categories[1:]
    one_hot = np.zeros((len(column), len(categories) - 1), dtype=int)
    for idx, category in enumerate(encoding_categories):
        one_hot[:, idx] = (column == category).astype(int)

    return one_hot, encoding_categories

# ======= MÉTRICAS DE EVALUACIÓN =======

def cross_entropy(y_true, y_pred):
	"""
	Función de pérdida.
	
	Args:
		y_true: Valores objetivo reales
		y_pred: Probabilidades predichas
		
	Returns:
		Pérdida promedio
	"""
	# Evitar log(0) con un valor pequeño
	epsilon = 1e-8
	n_samples = len(y_true)
	total_loss = 0.0
    
	for i in range(n_samples):
		pred = y_pred[i]
		if pred < epsilon:
			pred = epsilon
		elif pred > 1 - epsilon:
			pred = 1 - epsilon
        
        # Calcular pérdida
		if y_true[i] == 1:
			total_loss -= np.log(pred)
		else:
			total_loss -= np.log(1 - pred)

    # Calcular pérdida promedio
	loss = total_loss / n_samples
	return loss

def accuracy(y_true, y_pred):
	"""
	Calcula la precisión de las predicciones.
	
	Args:
		y_true: Valores reales
		y_pred: Valores predichos

	Returns:
		Precisión de las predicciones
	"""
	accurate = 0
	for i in range(len(y_true)):
		pred_label = 1 if y_pred[i] >= 0.5 else 0
		if pred_label == y_true[i]:
			accurate += 1
	accuracy = accurate / len(y_true)
	return accuracy

# ======= FUNCIONES DE REGRESIÓN LOGÍSTICA =======

def hypothesis(X, weights, bias):
	"""
	Implementación de la combinación lineal
	seguido de la activación sigmoide.
	
	Args:
		X: Matriz de características, forma (n_muestras, n_características)
		weights: Vector de pesos, forma (n_características,)
		bias: Término de sesgo (θ₀)
		
	Returns:
		Probabilidades 
	"""
	linear_combination = np.sum(X * weights, axis=1) + bias
	predictions = sigmoid(linear_combination)
	return predictions

def sigmoid(z):
	"""	Función de activación sigmoide.	"""
	sigmoid = 1 / (1 + np.exp(-z))
	return sigmoid

def gradient_descent(X, y, weights, bias, learning_rate):
    """
    Calcula el gradiente y actualiza los parámetros.
    
    Args:
        X: Matriz de características
        y: Valores objetivo
        weights: Vector de pesos
        bias: Sesgo
        learning_rate: Tasa de aprendizaje
        
    Returns:
        Pesos y bias actualizados
    """
    num_samples = X.shape[0]
    y_pred = hypothesis(X, weights, bias)
    error = y_pred - y  # Vector de errores

    # Gradiente para cada característica
    weight_gradients = np.zeros_like(weights)
    for j in range(X.shape[1]):
        weight_gradients[j] = np.sum(X[:, j] * error) / num_samples

    # Gradiente para el bias
    bias_gradient = np.sum(error) / num_samples

    # Actualizar parámetros
    weights -= learning_rate * weight_gradients
    bias -= learning_rate * bias_gradient
    return weights, bias

# ======= ENTRENAMIENTO DE REGRESIÓN LOGÍSTICA =======

def early_stopping(val_loss, best_val_loss, epochs_no_improve, patience=20, min_change=1e-6):
    """
    Verifica si detener entrenamiento por early stopping.

    Args:
        val_loss: Pérdida de validación en la época actual
        best_val_loss: Mejor pérdida de validación hasta ahora
        epochs_no_improve: Cuántas épocas consecutivas sin mejora llevamos
        patience: Máximo número de épocas sin mejora
        min_change: Mejora mínima para resetear el contador

    Returns:
        stop (bool): True si se debe detener, False si no
		best_val_loss: Mejor pérdida de validación actualizada
		epochs_no_improve: Número de épocas consecutivas sin mejora
    """
    if val_loss < best_val_loss - min_change:
        best_val_loss = val_loss
        epochs_no_improve = 0
        stop = False
    else:
        epochs_no_improve += 1
        stop = epochs_no_improve >= patience

    return stop, best_val_loss, epochs_no_improve

def train_log_regression(
    X_train, y_train, X_val, y_val, X_test, y_test,
    learning_rate=0.01, epochs=1000, patience=20
):
    num_features = X_train.shape[1]
    weights = np.zeros(num_features)
    bias = 0.0
    
    # Historial
    history = {
        'train_losses': [], 
        'val_losses': [], 
        'test_losses': [],
        'train_accuracies': [], 
        'val_accuracies': [], 
        'test_accuracies': [],
        'overfit_epochs': []
    }

    # Variables para early stopping
    best_val_loss = float('inf')
    best_weights = weights.copy()
    best_bias = bias
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # Forward pass
        y_train_pred = hypothesis(X_train, weights, bias)
        y_val_pred = hypothesis(X_val, weights, bias)
        y_test_pred = hypothesis(X_test, weights, bias)

        # Calcular métricas
        train_loss = cross_entropy(y_train, y_train_pred)
        val_loss = cross_entropy(y_val, y_val_pred)
        test_loss = cross_entropy(y_test, y_test_pred)

        train_acc = accuracy(y_train, y_train_pred)
        val_acc = accuracy(y_val, y_val_pred)
        test_acc = accuracy(y_test, y_test_pred)

        # Guardar métricas
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['test_losses'].append(test_loss)
        history['train_accuracies'].append(train_acc)
        history['val_accuracies'].append(val_acc)
        history['test_accuracies'].append(test_acc)

        # Detección de sobreajuste
        if epoch > 1 and val_loss > history['val_losses'][-2] and train_loss < history['train_losses'][-2]:
            history['overfit_epochs'].append(epoch)

        # Early stopping
        stop, best_val_loss, epochs_no_improve = early_stopping(
            val_loss, best_val_loss, epochs_no_improve, patience=patience
        )
        if val_loss <= best_val_loss:
            best_weights = weights.copy()
            best_bias = bias
        if stop:
            print(f"Early stopping en época {epoch}. Mejor val_loss={best_val_loss:.4f}")
            weights, bias = best_weights, best_bias
            break

        # Optimización
        weights, bias = gradient_descent(X_train, y_train, weights, bias, learning_rate)

        # Imprimir métricas cada 50 épocas
        if epoch % 50 == 0 or epoch == epochs:
            print(f"Época {epoch}: Train L={train_loss:.4f}, Val L={val_loss:.4f}, Test L={test_loss:.4f}, "
                  f"Train Ac={train_acc:.4f}, Val Ac={val_acc:.4f}, Test Ac={test_acc:.4f}")
        
    print("\nValores finales del modelo:")
    print(f"* Coeficientes: {weights} \n* Bias: {bias}")

    return weights, bias, history
