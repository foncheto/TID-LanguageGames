import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Datos
data = {
    "Task": ["Strategy QA", "Commonsense QA", "Race-Middle", "Race-High", "ARC-Easy", "ARC-Challenge"],
    "Gemma2-9B": [68.8, 82.0, 83.6, 84.0, 90.9, 0],
    "Llama3.1-8B": [72.8, 78.4, 81.6, 83.6, 87.2, 0],
    "Mistral-7B-Instruct": [66.8, 72.8, 74.4, 75.2, 84.0, 0]
}

# Convertimos a DataFrame
df = pd.DataFrame(data)

# Transformamos el DataFrame para Seaborn
df_melted = df.melt(id_vars="Task", var_name="Model", value_name="Score")

# Configuración de estilo
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# custom_palette = ["#FF5733", "#33FF57", "#5733FF"]
custom_palette = ["#bc272d", "#e9c716", "#50ad9f"]

# Gráfico de barras agrupadas
sns.barplot(data=df_melted, x="Task", y="Score", hue="Model", palette=custom_palette)

# Configuración del gráfico
plt.title("Performance Comparison Across Models", fontsize=16)
plt.xlabel("Task", fontsize=12)
plt.ylabel("Score (%)", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.legend(title="Model")
plt.tight_layout()

# Guardar gráfico
plt.savefig("graphs/evaluations.png")

# Mostrar gráfico
plt.show()