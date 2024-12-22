from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from collections import Counter
import math

def tokenize(text):
    """Convierte un texto en una lista de tokens."""
    return text.split()

def ngrams(tokens, n):
    """Genera n-gramas a partir de una lista de tokens."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def modified_precision(candidate, references, n):
    """Calcula la precisión modificada para n-gramas."""
    candidate_ngrams = Counter(ngrams(candidate, n))
    max_ref_ngrams = Counter()

    for reference in references:
        reference_ngrams = Counter(ngrams(reference, n))
        for ngram in candidate_ngrams:
            max_ref_ngrams[ngram] = max(max_ref_ngrams[ngram], reference_ngrams[ngram])

    overlap = {ngram: min(count, max_ref_ngrams[ngram]) for ngram, count in candidate_ngrams.items()}
    return sum(overlap.values()), sum(candidate_ngrams.values())

def calcular_bleu_score_1(respuesta, lista_referencias):
    bleu_scores = []
    respuesta_tokens = respuesta.split()
    smoothie = SmoothingFunction().method1  # Para evitar problemas con puntuaciones bajas
    for referencia in lista_referencias:
        referencia_tokens = referencia.split()
        score = sentence_bleu([referencia_tokens], respuesta_tokens, smoothing_function=smoothie)
        bleu_scores.append(score)
    # Retornar el promedio de los BLEU scores
    if bleu_scores:
        return sum(bleu_scores) / len(bleu_scores)
    else:
        return 0
    
def calcular_bleu_score_2(respuesta, lista_referencias):
    candidate_tokens = word_tokenize(respuesta)
    references_tokens = [word_tokenize(ref) for ref in lista_referencias]
    smooth_fn = SmoothingFunction().method1
    bleu_score = sentence_bleu(references_tokens, candidate_tokens, smoothing_function=smooth_fn)
    return bleu_score

def calcular_bleu_score_3(respuesta, lista_referencias, max_n=4):
    """Calcula el BLEU score para una respuesta candidata y referencias."""
    candidate_tokens = tokenize(respuesta)
    reference_tokens = [tokenize(ref) for ref in lista_referencias]

    precisions = []
    for n in range(1, max_n + 1):
        overlap, total = modified_precision(candidate_tokens, reference_tokens, n)
        if total == 0:
            precisions.append(0)
        else:
            precisions.append(overlap / total)

    # Penalización por longitud
    candidate_length = len(candidate_tokens)
    reference_lengths = [len(ref) for ref in reference_tokens]
    closest_ref_length = min(reference_lengths, key=lambda ref_len: (abs(ref_len - candidate_length), ref_len))

    if candidate_length > closest_ref_length:
        brevity_penalty = 1
    else:
        brevity_penalty = math.exp(1 - closest_ref_length / candidate_length) if candidate_length > 0 else 0

    # Calcular el score final
    if all(p == 0 for p in precisions):
        return 0

    score = brevity_penalty * math.exp(sum(math.log(p) for p in precisions if p > 0) / max_n)
    return score

# Cargar el dataset TruthfulQA
ds = load_dataset("truthfulqa/truthful_qa", "generation")
ds = ds['validation']

cont_comparaciones = 0

cont_fallas_1 = 0
cont_fallas_2 = 0
cont_fallas_3 = 0
cont_fallas_4 = 0

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

    # Calcular el BLEU score
    bleu_score_correctas_1 = calcular_bleu_score_1(mejor_respuesta, respuestas_correctas)
    bleu_score_correctas_2 = calcular_bleu_score_2(mejor_respuesta, respuestas_correctas)
    bleu_score_correctas_3 = calcular_bleu_score_3(mejor_respuesta, respuestas_correctas)

    bleu_score_incorrectas_1 = calcular_bleu_score_1(mejor_respuesta, respuestas_incorrectas)
    bleu_score_incorrectas_2 = calcular_bleu_score_2(mejor_respuesta, respuestas_incorrectas)
    bleu_score_incorrectas_3 = calcular_bleu_score_3(mejor_respuesta, respuestas_incorrectas)

    # Verificar si falla
    if bleu_score_incorrectas_1 > bleu_score_correctas_1:
        cont_fallas_1 += 1
    
    if bleu_score_incorrectas_2 > bleu_score_correctas_2:
        cont_fallas_2 += 1

    if bleu_score_incorrectas_3 > bleu_score_correctas_3:
        cont_fallas_3 += 1

print(f"Total de comparaciones realizadas: {cont_comparaciones}")
print(f"Total de fallas detectadas: {cont_fallas_1}")
print(f"Porcentaje de fallas: {cont_fallas_1 / cont_comparaciones * 100:.2f}%")

print(f"Total de fallas detectadas: {cont_fallas_2}")
print(f"Porcentaje de fallas: {cont_fallas_2 / cont_comparaciones * 100:.2f}%")

print(f"Total de fallas detectadas: {cont_fallas_3}")
print(f"Porcentaje de fallas: {cont_fallas_3 / cont_comparaciones * 100:.2f}%")

