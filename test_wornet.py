# Solo para la primera vez
#import nltk
#nltk.download('wordnet')

from nltk.corpus import wordnet as wn

# Función para obtener la similitud
def word_similarity(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    if not synsets1 and not synsets2:
        return f"'{word1}' y '{word2}' no se encuentran en WordNet"
    elif not synsets1:  # Verifica si las palabras existen en WordNet
        return f"'{word1}' no se encuentra en WordNet"
    elif not synsets2:
        return f"'{word2}' no se encuentra en WordNet"

    # Obtiene la máxima similitud entre los synsets
    max_similarity = max((synset1.wup_similarity(synset2) or 0)
                        for synset1 in synsets1 for synset2 in synsets2)
    return max_similarity

# Ejemplo de uso
word1 = "not"
word2 = "yes"
similarity = word_similarity(word1, word2)

if isinstance(similarity, float):
    print(f"La similitud entre '{word1}' y '{word2}' es: {similarity:.2f}")
else:
    print(similarity)