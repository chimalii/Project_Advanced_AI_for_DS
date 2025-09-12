import pandas as pd

df = pd.read_csv("diabetes_prediction_dataset.csv")
correlations = df.corr()["diabetes"].abs().sort_values(ascending=False)

# Mostrar las 15 variables más correlacionadas
print("Correlación de las variables con la variable objetivo:")
print(correlations)

# Realizar heatmap para visualizar correlaciones
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Heatmap de Correlaciones")
plt.show()

# Analizar variables numéricas
numerical_vars = ['BMI', 'MentHlth', 'PhysHlth']
fig, axes = plt.subplots(1, len(numerical_vars), figsize=(20, 6))
for i, var in enumerate(numerical_vars):
    sns.boxplot(x='Diabetes_binary', y=var, data=df, ax=axes[i])
    axes[i].set_title(f"{var}")
    axes[i].set_xlabel("Diabetes_binary")
    axes[i].set_ylabel(var)
plt.tight_layout()
plt.suptitle("Boxplots de Variables Numéricas", y=1.02)
plt.show()

# Analizar variables categóricas
categorical_vars = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
                    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
                    'NoDocbcCost', 'DiffWalk', 'Sex', 'GenHlth', 'Age', 'Education', 'Income']
fig, axes = plt.subplots(6, 3, figsize=(20, 25))
axes = axes.flatten()
for i, var in enumerate(categorical_vars):
    sns.countplot(x=var, hue='Diabetes_binary', data=df, ax=axes[i])
    axes[i].set_title(f"{var}")
    axes[i].set_xlabel(var)
    axes[i].set_ylabel("Frecuencia")
    axes[i].legend(title="Diabetes_binary", loc="upper right")
# Eliminar subplots vacíos
for j in range(len(categorical_vars), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.suptitle("Gráficos de Barras de Variables Categóricas", y=1.02)
plt.show()
