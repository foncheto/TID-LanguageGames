from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def similitud_1(respuesta, lista_respuestas):
    if len(lista_respuestas) == 1:
        return cosine_similarity(respuesta, lista_respuestas[0])
    else:
        similitudes = []
        for emb in lista_respuestas:
            similitudes.append(cosine_similarity(respuesta, emb))
        return max(similitudes)
    
def similitud_2(respuesta, lista_respuestas):
    if len(lista_respuestas) == 1:
        return cosine_similarity(respuesta, lista_respuestas[0])
    else:
        similitudes = []
        for emb in lista_respuestas:
            similitudes.append(cosine_similarity(respuesta, emb))
        return np.mean(similitudes)
    
#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # 42.3% y 41.5%
#model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2') # 42.84% y 42.84%
#model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') # 40.29% y 40.29%
#model = SentenceTransformer('sentence-transformers/all-distilroberta-v1') # 40.83% y 41.37%
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens') # 25.97% y 24.10%

ds = load_dataset("truthfulqa/truthful_qa", "generation")
ds = ds['validation']

cont_comparaciones = 0
cont_fallas_1 = 0
cont_fallas_2 = 0

for row in ds:
    mejor_respuesta = row['best_answer']
    respuestas_correctas = row['correct_answers']
    respuestas_incorrectas = row['incorrect_answers']
    
    # Eliminar la mejor respuesta de las respuestas correctas e incorrectas
    if mejor_respuesta in respuestas_correctas:
        respuestas_correctas.remove(mejor_respuesta)
    if mejor_respuesta in respuestas_incorrectas:
        respuestas_incorrectas.remove(mejor_respuesta)

    # Verificar que existan respuestas correctas e incorrectas
    if not respuestas_correctas or not respuestas_incorrectas:
        continue

    cont_comparaciones += 1

    # Emmbeding de la mejor respuesta
    mejor_respuesta_embedding = model.encode(mejor_respuesta)

    # Embeddings de las respuestas correctas e incorrectas
    respuestas_correctas_embeddings = model.encode(respuestas_correctas)
    respuestas_incorrectas_embeddings = model.encode(respuestas_incorrectas)

    similitud_correctas_1 = similitud_1(mejor_respuesta_embedding, respuestas_correctas_embeddings)
    similitud_incorrectas_1 = similitud_1(mejor_respuesta_embedding, respuestas_incorrectas_embeddings)

    similitud_correctas_2 = similitud_2(mejor_respuesta_embedding, respuestas_correctas_embeddings)
    similitud_incorrectas_2 = similitud_2(mejor_respuesta_embedding, respuestas_incorrectas_embeddings)

    # Verificar si falla
    if similitud_incorrectas_1 > similitud_correctas_1:
        cont_fallas_1 += 1

    if similitud_incorrectas_2 > similitud_correctas_2:
        cont_fallas_2 += 1

print(f"Total de comparaciones realizadas: {cont_comparaciones}")
print(f"Total de fallas detectadas: {cont_fallas_1}")
print(f"Porcentaje de fallas: {cont_fallas_1 / cont_comparaciones * 100:.2f}%")
print()
print(f"Total de fallas detectadas: {cont_fallas_2}")
print(f"Porcentaje de fallas: {cont_fallas_2 / cont_comparaciones * 100:.2f}%")