import numpy as np
import math as m

def standardize(X):
	"""
	Estandariza columnas de X con Z-score: (x - media) / desviación estándar
	
	Args:
		X: Datos a estandarizar

	Returns:
		Matriz estandarizada
	"""
	X_std = np.copy(X).astype(float)
	n_samples = len(X)
	n_features = len(X[0])
    
    # Calcular media para cada columna
	column_means = np.zeros(n_features)
	for j in range(n_features):
		total = 0.0
		for i in range(n_samples):
			total += X[i][j]
		column_means[j] = total / n_samples
    
    # Calcular desviación estándar para cada columna
	column_stds = np.zeros(n_features)
	for j in range(n_features):
		sum_squared_diff = 0.0
		for i in range(n_samples):
			sum_squared_diff += (X[i][j] - column_means[j]) ** 2
			column_stds[j] = m.sqrt(sum_squared_diff / n_samples)
    
    # Estandarizar cada valor
	for i in range(n_samples):
		for j in range(n_features):
			if column_stds[j] > 0:
				X_std[i][j] = (X[i][j] - column_means[j]) / column_stds[j]
    
	return X_std

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
	# Inicializar array para almacenar resultados
	num_samples = X.shape[0]
	predictions = np.zeros(num_samples)
	
	for sample_idx in range(num_samples):
		linear_combination = bias

		for feature_idx in range(len(weights)):
			linear_combination += X[sample_idx, feature_idx] * weights[feature_idx]
		
		# Aplicar función de activación sigmoide
		predictions[sample_idx] = sigmoid(linear_combination)
	
	return predictions

def sigmoid(z):
	"""	Función de activación sigmoide.	"""
	sigmoid = 1 / (1 + m.exp(-z))
	return sigmoid

def cross_entropy(y_true, y_pred):
	"""
	Función de error: cross entropy.
	
	Args:
		y_true: Valores objetivo reales
		y_pred: Probabilidades predichas
		
	Returns:
		Pérdida promedio
	"""
	# Evitar log(0) limitando los valores entre epsilon y 1-epsilon
	epsilon = 1e-8
	n_samples = len(y_true)
	total_error = 0.0
    
	for i in range(n_samples):
        # Evitar log(0) limitando los valores
		pred = y_pred[i]
		if pred < epsilon:
			pred = epsilon
		elif pred > 1 - epsilon:
			pred = 1 - epsilon
        
        # Calcular error
		if y_true[i] == 1:
			total_error -= m.log(pred)
		else:
			total_error -= m.log(1 - pred)
    
    # Calcular error promedio
	error = total_error / n_samples
	return error

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
	
	# Calcular el gradiente para los pesos
	weight_gradients = np.zeros(len(weights))
	for feature_idx in range(len(weights)):
		for sample_idx in range(num_samples):
			weight_gradients[feature_idx] += X[sample_idx, feature_idx] * (y_pred[sample_idx] - y[sample_idx])
	weight_gradients /= num_samples
	
	# Calcular gradiente para el bias
	bias_gradient = np.sum(y_pred - y) / num_samples
	
	# Actualizar parámetros
	weights = weights - learning_rate * weight_gradients
	bias = bias - learning_rate * bias_gradient
	
	return weights, bias

def train_log_regression(X_train, y_train, X_test, y_test, learning_rate, epochs):
	"""
	Entrena el modelo de regresión logística simple.
	
	Args:
		X_train, y_train: Variables de entrenamiento
		X_test, y_test: Variables de prueba
		learning_rate: Tasa de aprendizaje
		epochs: Número de épocas
		
	Returns:
		weights: Pesos optimizados
		bias: Sesgo optimizado
		history: Diccionario con métricas de entrenamiento (losses y accuracies)
	"""
	num_features = X_train.shape[1]
	weights = np.zeros(num_features)
	bias = 0.0
	train_losses = []
	test_losses = []
	train_accuracies = []
	test_accuracies = []
	
	for epoch in range(1, epochs + 1):
		# Forward pass
		y_train_pred = hypothesis(X_train, weights, bias)
		y_test_pred = hypothesis(X_test, weights, bias)
		# Calcular error (loss)
		train_loss = cross_entropy(y_train, y_train_pred)
		test_loss = cross_entropy(y_test, y_test_pred)
		# Calcular precisión (accuracy)
		train_acc = accuracy(y_train, y_train_pred)
		test_acc = accuracy(y_test, y_test_pred)
		# Guardar métricas
		train_losses.append(train_loss)
		test_losses.append(test_loss)
		train_accuracies.append(train_acc)
		test_accuracies.append(test_acc)
		# Imprimir métricas cada época
		print(f"Época {epoch}: Train Loss={train_loss:.8f}, Test Loss={test_loss:.8f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
		# Parar si el error es 0
		if train_loss == 0:
			print(f"Entrenamiento detenido en la época {epoch} porque el error de entrenamiento es 0.")
			break
		# Aplicar función de optimización
		weights, bias = gradient_descent(X_train, y_train, weights, bias, learning_rate)
	
	print("\nCoeficientes finales:")
	print("* pesos:", weights)
	print("* bias:", bias)

	# Crear historial para graficar
	history = {
		'train_losses': train_losses,
		'test_losses': test_losses,
		'train_accuracies': train_accuracies,
		'test_accuracies': test_accuracies
	}
	
	return weights, bias, history
