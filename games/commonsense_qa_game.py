from collections import Counter
from openai import OpenAI
from datasets import load_dataset
from utils import *
import datetime as time
import logging

def commonsense_qa_game(n, model_1, model_2, model_3, api_key):
    # Set up logging
    log_filename = setup_logging("cqa_game", "outputs_games")
    logging.info(f"Starting CQA Game with {n} questions")
    logging.info(f"Models used: {model_1}, {model_2}, {model_3}")

    try:
        # Inicializar modelos solo una vez
        llms = [OpenAI(api_key=api_key, base_url="https://api.llama-api.com") for _ in range(3)]
        models = [model_1, model_2, model_3]

        # Cargar el dataset
        logging.info("Loading CQA dataset")
        ds = load_dataset("tau/commonsense_qa")

        # Inicializar puntajes
        score = []

        # Inicializar outputs
        outputs = []

        # Función para manejar conflictos y consenso
        def resolve_conflict(answers, explanations, correct_answer, times):
            # Si todas las respuestas son iguales, se llega a un consenso
            if len(set(answers)) == 1:
                logging.info(f"Consensus reached on iteration N°{4 - times}")
                outputs.append(f"Consensus reached on iteration N°{4 - times}")
                score.append(answers[0] == correct_answer)
                return
            
            # Si no se llega a un consenso después de 3 iteraciones, se elige la respuesta más común
            elif times <= 0:
                logging.info("No consensus, choosing most common answer after 3 iterations")
                outputs.append("No consensus, choosing most common answer after 3 iterations")
                
                # Eliminar respuestas aleatorias
                filtered_answers = [
                    ans for ans, exp in zip(answers, explanations) if exp != "Random answer"
                ]
                outputs.append(f"Filtered answers: {filtered_answers}")
                logging.info(f"Filtered answers: {filtered_answers}")
                
                try:
                    most_common_answer = Counter(filtered_answers).most_common(1)[0][0]
                    score.append(most_common_answer == correct_answer)
                    logging.info(f"Most common answer: {most_common_answer}")
                    return
                except Exception as e:
                    logging.error(f"Error in choosing most common answer: {e}")
                    score.append(False)
                    return
            else:
                # Detectar y resolver diferentes tipos de conflictos
                try:
                    # 0 y 1 son iguales, 2 es diferente
                    if answers[0] == answers[1] != answers[2]:
                        logging.info("Conflict detected: 1 and 2 are equal, 3 is different")
                        outputs.append("Conflict detected: 1 and 2 are equal, 3 is different")
                        return resolve_conflict_one_different(2, [0, 1], list(answers), list(explanations), correct_answer, times)
                    
                    # 0 y 2 son iguales, 1 es diferente
                    elif answers[0] == answers[2] != answers[1]:
                        logging.info("Conflict detected: 1 and 3 are equal, 2 is different")
                        outputs.append("Conflict detected: 1 and 3 are equal, 2 is different")
                        return resolve_conflict_one_different(1, [0, 2], list(answers), list(explanations), correct_answer, times)
                    
                    # 1 y 2 son iguales, 0 es diferente
                    elif answers[1] == answers[2] != answers[0]:
                        logging.info("Conflict detected: 2 and 3 are equal, 1 is different")
                        outputs.append("Conflict detected: 2 and 3 are equal, 1 is different")
                        return resolve_conflict_one_different(0, [1, 2], list(answers), list(explanations), correct_answer, times)
                    
                    # Todas las respuestas son diferentes
                    elif answers[0] != answers[1] != answers[2]:
                        logging.info("Conflict detected: all answers are different")
                        outputs.append("Conflict detected: all answers are different")
                        return resolve_conflict_all_different(list(answers), list(explanations), correct_answer, times)
                
                except Exception as e:
                    logging.error(f"Error in conflict resolution: {e}")
                    score.append(False)
                    return
        
        def generate_fake_conflict(correct_answer, times):
            logging.info("Generating fake conflict")
            outputs.append("Generating fake conflict")
            outputs.append("New answers:")
            logging.info("Generating new answers")

            llm_answers = [get_llm_response(llm, fake_conflict_prompt(), model, 3) for llm, model in zip(llms, models)]

            new_answers, new_explanations = zip(*(clean_response_multiple(resp) for resp in llm_answers))

            for ans, exp in zip(new_answers, new_explanations):
                outputs.append(f"{ans}: {exp}")
                logging.info(f"New answer: {ans}, Explanation: {exp}")

            return resolve_conflict(new_answers, new_explanations, correct_answer, times - 1)

        def resolve_conflict_one_different(main_index, other_indices, answers, explanations, correct_answer, times):
            logging.info(f"Resolving conflict for LLM {main_index + 1}")
            outputs.append(f"Resolving conflict for LLM {main_index + 1}")
            outputs.append("New answer:")

            prompt = feedback_prompt_2(
                answers[other_indices[0]], explanations[other_indices[0]], explanations[other_indices[1]]
            )
            
            try:
                new_response = get_llm_response(llms[main_index], prompt, models[main_index], 3)
                new_answer, new_explanation = clean_response_multiple(new_response)
                
                outputs.append(f"{new_answer}: {new_explanation}")
                logging.info(f"New answer for LLM {main_index + 1}: {new_answer}")

                # Se actualiza la respuesta y la explicación
                answers[main_index] = new_answer
                explanations[main_index] = new_explanation

                return resolve_conflict(answers, explanations, correct_answer, times - 1)
            
            except Exception as e:
                logging.error(f"Error in resolving conflict for LLM {main_index + 1}: {e}")
                score.append(False)
                return

        def resolve_conflict_all_different(answers, explanations, correct_answer, times):
            logging.info("Resolving conflict for all different answers")
            outputs.append("Resolving conflict for all different answers")
            outputs.append("New answers:")

            try:
                combinations = [(1, 2), (0, 2), (0, 1)]
                prompts = [
                    feedback_prompt_1(answers[i1], explanations[i1], answers[i2], explanations[i2]) 
                    for i1, i2 in combinations
                ]
                llm_answers = [
                    get_llm_response(llm, prompt, model, 3) 
                    for llm, model, prompt in zip(llms, models, prompts)
                ]
                new_answers, new_explanations = zip(*(clean_response_multiple(resp) for resp in llm_answers))

                for ans, exp in zip(new_answers, new_explanations):
                    outputs.append(f"{ans}: {exp}")
                    logging.info(f"New answer: {ans}, Explanation: {exp}")

                return resolve_conflict(new_answers, new_explanations, correct_answer, times - 1)
            
            except Exception as e:
                logging.error(f"Error in resolving conflict for all different answers: {e}")
                score.append(False)
                return

        # Procesar las preguntas
        for i in range(n):
            logging.info(f"Processing Question N°{i}")
            outputs.append((f"QUESTION N°{i}"))

            try:
                row = ds["validation"][i] # Se prueba con validation

                prompt, answer = get_data_commonsense_qa_and_prompt(row)

                outputs.append(row['question'])
                outputs.append(f"The correct answer is: {answer}")
                
                logging.info(f"Question: {row['question']}")
                logging.info(f"Correct Answer: {answer}")

                llm_answers = [get_llm_response(llm, prompt, model, 3) for llm, model in zip(llms, models)]
                answers, explanations = zip(*(clean_response_multiple(resp) for resp in llm_answers))

                outputs.append("Initial answers:")
                for ans, exp in zip(answers, explanations):
                    outputs.append(f"{ans}: {exp}")
                    logging.info(f"Initial Answer: {ans}, Explanation: {exp}")
                
                resolve_conflict(answers, explanations, answer, 3)
                outputs.append("------------------------------------------------------------")

            except Exception as e:
                logging.error(f"Error processing question {i}: {e}")
                score.append(False)

        # Registrar el puntaje
        outputs.append(score)
        outputs.append(f"Score: {sum(score)}/{n} ({sum(score)/n*100}%)")
        logging.info(f"Final Score: {sum(score)}/{n} ({sum(score)/n*100}%)")

        date = time.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        with open(f"outputs_games/cqa_game_{date}.txt", "w", encoding="utf-8") as f:
            for output in outputs:
                if isinstance(output, tuple):
                    f.write(" ".join(map(str, output)) + "\n")
                else:
                    f.write(str(output) + "\n")

        logging.info(f"Results saved to outputs_games/cqa_game_{date}.txt")
        logging.info(f"Log file saved to {log_filename}")

        return score
    except Exception as e:
        logging.error(f"Unhandled exception in cqa_game: {e}")
        raise
