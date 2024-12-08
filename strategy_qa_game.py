from collections import Counter
from openai import OpenAI
from datasets import load_dataset
from utils import *
import datetime as time

def strategy_qa_game(n, model_1, model_2, model_3, api_key):
    # Inicializar modelos solo una vez
    llms = [OpenAI(api_key=api_key, base_url="https://api.llama-api.com") for _ in range(3)]
    models = [model_1, model_2, model_3]

    # Cargar el dataset
    ds = load_dataset("wics/strategy-qa")

    # Inicializar puntajes
    score = []

    # Inicializar outputs
    outputs = []

    # Función para manejar conflictos y consenso
    def resolve_conflict(main_index, other_indices, answers, explanations, correct_answer, times): # Se añade times, que es el número de veces que se intenta resolver el conflicto
        outputs.append(f"Resolving conflict for LLM {main_index + 1}")
        prompt = feedback_prompt_2(
            answers[other_indices[0]], explanations[other_indices[0]], explanations[other_indices[1]]
        )
        new_response = get_llm_response(llms[main_index], prompt, models[main_index], 3) # Se añade times, que es el número de veces que intenta conseguir una respuesta válida
        new_answer, new_explanation = clean_response_boolean(new_response)
        outputs.append(f"{new_answer}: {new_explanation}")

        if new_answer == answers[other_indices[0]]:
            outputs.append(f"LLM {main_index + 1} changed its mind, choosing that")
            score.append(new_answer.lower() == correct_answer)
            return
        elif times > 0:
            outputs.append(f"LLM {main_index + 1} didn't change its mind, trying again")
            return resolve_conflict(main_index, other_indices, answers, explanations, correct_answer, times - 1)
        else:
            outputs.append("No consensus, choosing most common answer")
            all_answers = answers + [new_answer]
            all_explanations = explanations + [new_explanation]
            
            # Eliminar respuestas aleatorias
            filtered_answers = [
                ans for ans, exp in zip(all_answers, all_explanations) if exp != "Random answer"
            ]
            outputs.append(f"Filtered answers: {filtered_answers}")
            try:
                most_common_answer = Counter(filtered_answers).most_common(1)[0][0]
                score.append(most_common_answer.lower() == correct_answer)
                return
            except:
                score.append(False)
                return
            return

    # Procesar las preguntas
    for i in range(n):
        outputs.append(f"QUESTION N°{i}")
        prompt, answer = get_data_strategy_qa_and_prompt(ds["test"][i])

        # Obtener respuestas iniciales
        llm_answers = [get_llm_response(llm, prompt, model, 3) for llm, model in zip(llms, models)]
        answers, explanations = zip(*(clean_response_boolean(resp) for resp in llm_answers))
        
        correct_answer = "yes" if answer else "no"

        outputs.append(f"{ds['test'][i]['question']}")
        outputs.append(f"The correct answer is: {correct_answer}")

        outputs.append("Initial answers:")
        for ans, exp in zip(answers, explanations):
            outputs.append(f"{ans}: {exp}")

        # Decisiones basadas en las respuestas
        if len(set(answers)) == 1:
            outputs.append(f"Consensus reached on iteration N°1")
            score.append(answers[0].lower() == correct_answer)
        else:
            # Identificar conflictos y resolverlos
            if answers[0] == answers[1] != answers[2]:
                outputs.append("Conflict detected: 1 and 2 are equal, 3 is different")
                resolve_conflict(2, [0, 1], list(answers), list(explanations), correct_answer, 3)
            elif answers[1] == answers[2] != answers[0]:
                outputs.append("Conflict detected: 2 and 3 are equal, 1 is different")
                resolve_conflict(0, [1, 2], list(answers), list(explanations), correct_answer, 3)
            elif answers[2] == answers[0] != answers[1]:
                outputs.append("Conflict detected: 3 and 1 are equal, 2 is different")
                resolve_conflict(1, [2, 0], list(answers), list(explanations), correct_answer, 3)
        outputs.append("------------------------------------------------------------")

    # Registrar el puntaje
    outputs.append(score)
    outputs.append(f"Score: {sum(score)}/{n} ({sum(score)/n*100}%)")

    date = time.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    with open(f"outputs_games/strategy_qa_game_{date}.txt", "w", encoding="utf-8") as f:
        for output in outputs:
            if isinstance(output, tuple):
                f.write(" ".join(map(str, output)) + "\n")
            else:
                f.write(str(output) + "\n")

    return score
