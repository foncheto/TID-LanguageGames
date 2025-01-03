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

| Dataset         | Iteration | Score |Errors|
|-----------------|-----------|-------|------|
| Commonsense_QA  | 1         | 71.0% | 5.0% |
| Commonsense_QA  | 2         | 73.0% | 4.0% |
| Commonsense_QA  | 3         | 69.0% | 0.0% |
| ECQA            | 1         | 73.0% | 9.0% |
| ECQA            | 2         | 73.0% | 9.0% |
| ECQA            | 3         | 79.0% | 2.0% |
| ECQA            | 4         | 75.0% | 0.0% | # 0% de error con nuevo método
| StrategyQA      | 1         | 32.0% | 6.0% |
| StrategyQA      | 2         | 36.0% | 2.0% |
| StrategyQA      | 3         | 33.0% | 3.0% |
| StrategyQA      | 4         | 31.0% | 0.0% |

## llama3.1-8b

| Dataset         | Iteration | Score |Errors|
|-----------------|-----------|-------|------|
| Commonsense_QA  | 1         | 79.0% | 0.0% |
| Commonsense_QA  | 2         | 79.0% | 0.0% |
| Commonsense_QA  | 3         | 79.0% | 0.0% |
| ECQA            | 1         | 67.0% | 0.0% |
| ECQA            | 2         | 72.0% | 2.0% |
| ECQA            | 3         | 68.0% | 0.0% |
| ECQA            | 4         | 68.0% | 0.0% | # 0% de error con nuevo método
| StrategyQA      | 1         | 41.0% | 4.0% |
| StrategyQA      | 2         | 37.0% | 8.0% |
| StrategyQA      | 3         | 37.0% | 5.0% |
| StrategyQA      | 4         | 33.0% | 3.0% |

## mistral-7b-instruct

| Dataset         | Iteration | Score | Errors |
|-----------------|-----------|-------|--------|
| Commonsense_QA  | 1         | 74.0% | 3.0%   |
| Commonsense_QA  | 2         | 71.0% | 4.0%   |
| Commonsense_QA  | 3         | 72.0% | 0.0%   |
| ECQA            | 1         | 67.0% | 3.0%   |
| ECQA            | 2         | 68.0% | 1.0%   |
| ECQA            | 3         | 69.0% | 0.0%   |
| ECQA            | 4         | 70.0% | 0.0%   | # 0% de error con nuevo método
| StrategyQA      | 1         | 29.0% | 8.0%   |
| StrategyQA      | 2         | 30.0% | 5.0%   |
| StrategyQA      | 3         | 27.0% | 11.0%  |
| StrategyQA      | 4         | 27.0% | 0.0%  |

* Iteración 3 de ECQA se realizó preguntando más veces si existía error de formato.
* Iteración 4 de ECQA, 3 de Commonsense y 4 de Strategy se daba el feedback de error de formato + la forma correcta de contestar.

## Con juego

- ECQA:
  - Score: 76.0%
  - Score: 74.0%
  - Score: 74.0%
  - Score: 75.0%
- StrategyQA:
  - Score: 73.0%

Quizas agregar un debate "falso", si en la primera iteración llegan a un error.

Clustering
| Dataset | Iteration | Score |Errors|
|-----------------|-----------|-------|------|
| Commonsense_QA | 1 | 71.0% | 5.0% |
| Commonsense_QA | 2 | 73.0% | 4.0% |
| ECQA | 1 | 73.0% | 9.0% |
| ECQA | 2 | 73.0% | 9.0% |
| StrategyQA | 1 | 32.0% | 6.0% |
| StrategyQA | 2 | 36.0% | 2.0% |
| StrategyQA | 3 | 33.0% | 3.0% |

## llama3.1-8b

| Dataset        | Iteration | Score | Errors |
| -------------- | --------- | ----- | ------ |
| Commonsense_QA | 1         | 79.0% | 0.0%   |
| Commonsense_QA | 2         | 79.0% | 0.0%   |
| ECQA           | 1         | 67.0% | 0.0%   |
| ECQA           | 2         | 72.0% | 2.0%   |
| StrategyQA     | 1         | 41.0% | 4.0%   |
| StrategyQA     | 2         | 37.0% | 8.0%   |
| StrategyQA     | 3         | 37.0% | 5.0%   |

## mistral-7b-instruct

| Dataset        | Iteration | Score | Errors |
| -------------- | --------- | ----- | ------ |
| Commonsense_QA | 1         | 74.0% | 3.0%   |
| Commonsense_QA | 2         | 71.0% | 4.0%   |
| ECQA           | 1         | 67.0% | 3.0%   |
| ECQA           | 2         | 68.0% | 1.0%   |
| StrategyQA     | 1         | 29.0% | 8.0%   |
| StrategyQA     | 2         | 30.0% | 5.0%   |
| StrategyQA     | 3         | 27.0% | 11.0%  |

## Con juego

| Dataset        | Iteration | Score |
| -------------- | --------- | ----- |
| ECQA           | 1         | 76.0% |
| ECQA           | 2         | 74.0% |
| ECQA           | 3         | 74.0% |
| ECQA           | 4         | 75.0% |
| StrategyQA     | 1         | 73.0% |
| StrategyQA     | 2         | 74.0% |
| StrategyQA     | 3         | 75.0% |
| Commonsense_QA | 1         | 77.0% |
| Commonsense_QA | 2         | 75.0% |

- Quizas agregar un debate "falso", si en la primera iteracion llegan todos a consenso
- Se puede agregar un While cuando no responden en el formato esperado, (En seleccion multiple algunos responden fuera de las letras (A-E), tambien entrar en este caso)
