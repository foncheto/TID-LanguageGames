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

## gemma2-9b

- StrategyQA: 30/100 -> 30%

- ECQA: 75/100 -> 75%

- Commonsense_QA: 69/100 -> 69%

## llama3.1-8b

- StrategyQA: 34/100 -> 34%

- ECQA: 68/100 -> 68%

- Commonsense_QA: 79/100 -> 79%

## mistral-7b-instruct

- StrategyQA: 25/100 -> 25%

- ECQA: 68/100 -> 68%

- Commonsense_QA: 74/100 -> 74%

El dataset que más errores con formato produce es Strategy QA, con 17 en total, el LLM con más errores de formato es Gemma2-9b
Number of 'X' in dataframe gemma2-9b_commonsense_qa_01-12-2024_17-48-51.csv: 6
Number of 'X' in dataframe gemma2-9b_ecqa_01-12-2024_17-52-10.csv: 5
Number of 'X' in dataframe gemma2-9b_strategy_qa_01-12-2024_17-55-19.csv: 8
Number of 'X' in dataframe llama3.1-8b_commonsense_qa_01-12-2024_17-57-31.csv: 0
Number of 'X' in dataframe llama3.1-8b_ecqa_01-12-2024_18-00-00.csv: 0
Number of 'X' in dataframe llama3.1-8b_strategy_qa_01-12-2024_18-04-07.csv: 5
Number of 'X' in dataframe mistral-7b-instruct_commonsense_qa_01-12-2024_18-07-08.csv: 2
Number of 'X' in dataframe mistral-7b-instruct_ecqa_01-12-2024_18-09-52.csv: 1
Number of 'X' in dataframe mistral-7b-instruct_strategy_qa_01-12-2024_18-12-49.csv: 4
