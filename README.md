- Agentes nuevos
- Evaluador de respuestas correctas

### Interacciones con nuevo agente

- Si un nuevo agente se incorpora luego de comenzado el juego, existen dos comportamientos:
  - Si entra como emisor "confunde" al que ya estaba jugando y puede alterar su vocabulario. (¿Es bueno para evitar optimos locales?)
  - Si entra como receptor aprende del que ya estaba jugando.

Usar una taxonomía de agentes para poder compararlos. (¿Qué es una taxonomía de agentes?)

Como puedo usar una taxonomía en vez de cosine similarity para comparar agentes:

- Usar una taxonomía para comparar agentes y ver si se parecen o no.
- UMLS: https://www.nlm.nih.gov/research/umls/
- https://www.nltk.org/howto/wordnet.html
- https://pythonprogramming.net/wordnet-nltk-tutorial/
- https://www.educative.io/answers/how-to-use-wordnet-in-python
- https://github.com/sklarman/wordnet-distance
- https://en-word.net/lemma/demo


## Evaluaciones con datasets sin "juego"

Los errores son los outputs en los cuales el LLM no cumplio con el formato esperado, un error puede sumar al Score, ya que se selecciona de forma aleatoria una respuesta.

## gemma2-9b

- Commonsense_QA: 
    - Score: 71/100 -> 71.0%
    - Errors: 5/100 -> 5.0%
- ECQA:
    - Score: 73/100 -> 73.0%
    - Errors: 9/100 -> 9.0%
- StrategyQA:
    - Score: 32/100 -> 32.0%
    - Errors: 6/100 -> 6.0%

## llama3.1-8b

- Commonsense_QA:
    - Score: 79/100 -> 79.0%
    - Errors: 0/100 -> 0.0%
- ECQA:
    - Score: 67/100 -> 67.0%
    - Errors: 0/100 -> 0.0%
- StrategyQA:
    - Score: 41/100 -> 41.0%
    - Errors: 4/100 -> 4.0%
## mistral-7b-instruct

- Commonsense_QA:
    - Score: 74/100 -> 74.0%
    - Errors: 3/100 -> 3.0%
- ECQA:
    - Score: 67/100 -> 67.0%
    - Errors: 3/100 -> 3.0%
- StrategyQA:
    - Score: 29/100 -> 29.0%
    - Errors: 8/100 -> 8.0%