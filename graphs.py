import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Datos
data = {
    "Task": ["Strategy QA", "Commonsense QA", "Race-Middle", "Race-High", "ARC-Easy", "ARC-Challenge"],
    "Gemma2-9B": [68.8, 82.0, 83.6, 84.0, 90.9, 88.8],
    "Llama3.1-8B": [72.8, 78.4, 81.6, 83.6, 87.2, 83.2],
    "Mistral-7B-Instruct": [66.8, 72.8, 74.4, 75.2, 84.0, 78.0]
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

# Calcular el promedio para cada tarea
df2 = df

df2['Average'] = df.drop(columns=['Task']).mean(axis=1)

# Configuración del gráfico
plt.figure(figsize=(10, 6))

# Gráfico de barras del promedio
sns.barplot(data=df2.reset_index(), x="Task", y="Average", palette="viridis")

# Configuración del gráfico
plt.title("Average Score per Task", fontsize=16)
plt.xlabel("Task", fontsize=12)
plt.ylabel("Average Score (%)", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Guardar gráfico
plt.savefig("graphs/average_scores.png")

# Mostrar gráfico
plt.show()

# Calcular el promedio para cada modelo
model_averages = df.drop(columns=['Task', 'Average']).mean()

# Configuración del gráfico
plt.figure(figsize=(8, 6))

# Gráfico de barras del promedio por modelo
sns.barplot(x=model_averages.index, y=model_averages.values, palette=custom_palette)

# Configuración del gráfico
plt.title("Average Performance by Model", fontsize=16)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Average Score (%)", fontsize=12)
plt.ylim(0, 100)
plt.tight_layout()

# Guardar gráfico
plt.savefig("graphs/average_performance_by_model.png")

# Mostrar gráfico
plt.show()

# Convertimos a DataFrame
df = pd.DataFrame(data).set_index("Task")

# Configuración del gráfico
plt.figure(figsize=(10, 6))
sns.heatmap(
    df,
    annot=True,           # Mostrar valores
    fmt="f",              # Formato de los números
    cmap="YlGnBu",        # Paleta de colores
    linewidths=0.5,       # Líneas entre celdas
    cbar_kws={"label": "Score (%)"}  # Etiqueta de la barra de color
)

# Personalización
plt.title("Performance Heatmap of Models", fontsize=16, pad=20)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Task", fontsize=12)
plt.tight_layout()

# Guardar el gráfico
plt.savefig("graphs/heatmap.png")

# Mostrar el gráfico
plt.show()

juegos = ["Strategy QA", "Commonsense QA", "Race-Middle", "Race-High", "ARC-Easy", "ARC-Challenge"]
porcentajes = [70.4, 80.8, 88.8, 85.2, 90.0, 84.4]
promedio = [69.5, 77.7, 79.9, 80.9, 87.3, 83.3]

# Configuración del gráfico
plt.figure(figsize=(10, 6))

# Graficar los rendimientos y los promedios como líneas
plt.plot(juegos, porcentajes, marker='o', linestyle='-', color='skyblue', label='Rendimiento')
plt.plot(juegos, promedio, marker='x', linestyle='--', color='orange', label='Promedio')

# Añadir los porcentajes sobre los puntos
for i, value in enumerate(porcentajes):
    plt.text(i, value + 0.5, f'{value}%', ha='center', fontsize=12)

for i, value in enumerate(promedio):
    plt.text(i, value + 0.5, f'{value}%', ha='center', fontsize=12)

# Personalización del gráfico
plt.title("Rendimiento por Juego con Promedio", fontsize=16)
plt.xlabel("Juego", fontsize=12)
plt.ylabel("Porcentaje (%)", fontsize=12)
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.tight_layout()

# Mostrar leyenda
plt.legend()

# Mostrar gráfico
plt.show()

# Datos para cada tarea
commonsense_qa = [167/250, (167+63)/250, (167+63+10)/250, (167+63+10+5)/250]
race_middle = [187/250, (187+43)/250, (187+43+13)/250, (187+43+13+3)/250]
race_high = [163/250, (163+65)/250, (163+65+6)/250, (163+65+6+1)/250]
arc_easy = [212/250, (212+31)/250, (212+31+2)/250, (212+31+2+1)/250]
arc_challenge = [188/250, (188+42)/250, (188+42+10)/250, (188+42+10+3)/250]

# Iteraciones
iteraciones = [1, 2, 3, 4]

# Configuración del gráfico
plt.figure(figsize=(10, 6))

# Graficamos cada tarea
plt.plot(iteraciones, commonsense_qa, marker='o', linestyle='-', label="Commonsense QA", markersize=8)
plt.plot(iteraciones, race_middle, marker='s', linestyle='-', label="Race Middle", markersize=8)
plt.plot(iteraciones, race_high, marker='^', linestyle='-', label="Race High", markersize=8)
plt.plot(iteraciones, arc_easy, marker='d', linestyle='-', label="ARC Easy", markersize=8)
plt.plot(iteraciones, arc_challenge, marker='p', linestyle='-', label="ARC Challenge", markersize=8)

# Personalización del gráfico
plt.title("Consensus Evolution by Iteration", fontsize=16)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Consensus (%)", fontsize=12)
plt.xticks(iteraciones)  # Asegura que las iteraciones se muestren correctamente
plt.yticks([i/10 for i in range(11)])  # Para mostrar porcentajes de 0 a 1
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.ylim(0.55, 1)  # Limitar el eje y de 0.5 a 1

# Mostrar leyenda
plt.legend(title="Tasks")

# Guardar el gráfico
plt.savefig("graphs/consensus.png")

# Mostrar gráfico
plt.show()

