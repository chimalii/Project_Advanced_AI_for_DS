import pandas as pd

df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
correlations = df.corr()["Diabetes_binary"].abs().sort_values(ascending=False)

# Mostrar las 15 variables más correlacionadas
print("Variables más correlacionadas con la variable objetivo:")
print(correlations[1:16])

# Realizar heatmap para visualizar correlaciones
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Heatmap de Correlaciones")
plt.show()
