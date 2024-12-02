from openai import OpenAI
from datasets import load_dataset
from collections import Counter
from utils import *
from dotenv import load_dotenv
import os

load_dotenv()

def ecqa_game(n, model_1, model_2, model_3, api_key):

    # Se inicializan los modelos
    llm_1 = OpenAI(
        api_key = api_key,
        base_url = "https://api.llama-api.com"
    )

    llm_2 = OpenAI(
        api_key = api_key,
        base_url = "https://api.llama-api.com"
    )

    llm_3 = OpenAI(
        api_key = api_key,
        base_url = "https://api.llama-api.com"
    )

    # Se carga el dataset
    ds = load_dataset("yangdong/ecqa")

    # Se inicializa el score
    score = []

    # Se itera sobre el dataset de preguntas
    for i in range (0, n):
        print("Question: ", i)

        # Contener la fila de la pregunta
        row = ds["train"][i]

        # Armar el prompt, la respuesta correcta y el dict de respuestas
        prompt, q_ans = get_data_ecqa_and_prompt(row)

        options = {
            "A": row['q_op1'],
            "B": row['q_op2'],
            "C": row['q_op3'],
            "D": row['q_op4'],
            "E": row['q_op5']
        }

        # Conseguir respuestas
        llm_answer_1 = get_llm_response(llm_1, prompt, model_1)
        llm_answer_2 = get_llm_response(llm_2, prompt, model_2)
        llm_answer_3 = get_llm_response(llm_3, prompt, model_3)
        
        # Dejar solo el contenido JSON
        answer_1, explanation_1 = clean_response_multiple(llm_answer_1)
        print(answer_1, explanation_1)
        answer_2, explanation_2 = clean_response_multiple(llm_answer_2)
        print(answer_2, explanation_2)
        answer_3, explanation_3 = clean_response_multiple(llm_answer_3)
        print(answer_3, explanation_3)

        # Si todas las respuestas son iguales, nos quedamos con esa
        if answer_1 == answer_2 and answer_2 == answer_3:
            if answer_1 in options:
                answer = options[answer_1]
                is_correct = answer == q_ans
            else:
                is_correct = False

        # Si todas las respuestas son distintas, se les pregunta a todos si no creen que es la respuesta de los otros + explicación
        elif answer_1 != answer_2 and answer_2 != answer_3 and answer_1 != answer_3:
            prompt_1 = feedback_prompt_1(answer_2, explanation_2, answer_3, explanation_3)
            prompt_2 = feedback_prompt_1(answer_1, explanation_1, answer_3, explanation_3)
            prompt_3 = feedback_prompt_1(answer_1, explanation_1, answer_2, explanation_2)

            llm_answer_1_1 = get_llm_response(llm_1, prompt_1, model_1)
            llm_answer_2_1 = get_llm_response(llm_2, prompt_2, model_2)
            llm_answer_3_1 = get_llm_response(llm_3, prompt_3, model_3)

            answer_1_1, explanation_1_1 = clean_response_multiple(llm_answer_1_1)
            answer_2_1, explanation_2_1 = clean_response_multiple(llm_answer_2_1)
            answer_3_1, explanation_3_1 = clean_response_multiple(llm_answer_3_1)

            # Si hay consenso, nos quedamos con la respuesta
            if answer_1_1 == answer_2_1 == answer_3_1:
                if answer_1 in options:
                    answer = options[answer_1]
                    is_correct = answer == q_ans
                else:
                    is_correct = False
            # Si no hay consenso, nueva iteracion
            else:
                # Respuesta de 1 y 2 iguales, le volvemos a preguntar a 3
                if answer_1_1 == answer_2_1 and answer_2_1 != answer_3_1:
                    prompt_3 = feedback_prompt_2(answer_1_1, explanation_1_1, explanation_2_1)
                    llm_answer_3_2 = get_llm_response(llm_3, prompt_3, model_3)
                    answer_3_2, explanation_3_2 = clean_response_multiple(llm_answer_3_2)
                    # Si ahora hay consenso nos quedamos con esa respuesta
                    if answer_1_1 == answer_3_2:
                        if answer_1_1 in options:
                            answer = options[answer_1_1]
                            is_correct = answer == q_ans
                        else:
                            is_correct = False
                    # Si no hay consenso, se elige la respuesta con más apariciones (Pueden agregarse iteraciones u otra forma)
                    else:
                        explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1, explanation_3_2]
                        answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1, answer_3_2]

                        while explanations.count("Random answer") > 0:
                            index = explanations.index("Random answer")
                            answers.pop(index)
                            explanations.pop(index)

                        most_common_answer = Counter(answers).most_common(1)[0][0]

                        if most_common_answer in options:
                            answer = options[most_common_answer]
                            is_correct = answer == q_ans
                        else:
                            is_correct = False
                
                # Respuesta de 1 y 3 iguales, le volvemos a preguntar a 2
                elif answer_1_1 == answer_3_1 and answer_2_1 != answer_3_1:
                    prompt_2 = feedback_prompt_2(answer_1_1, explanation_1_1, explanation_3_1)
                    llm_answer_2_2 = get_llm_response(llm_2, prompt_2, model_2)
                    answer_2_2, explanation_2_2 = clean_response_multiple(llm_answer_2_2)
                    # Si ahora hay consenso nos quedamos con esa respuesta
                    if answer_1_1 == answer_2_2:
                        if answer_1_1 in options:
                            answer = options[answer_1_1]
                            is_correct = answer == q_ans
                        else:
                            is_correct = False
                    # Si no hay consenso, se elige la respuesta con más apariciones (Pueden agregarse iteraciones u otra forma)
                    else:
                        explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1, explanation_2_2]
                        answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1, answer_2_2]

                        while explanations.count("Random answer") > 0:
                            index = explanations.index("Random answer")
                            answers.pop(index)
                            explanations.pop(index)

                        most_common_answer = Counter(answers).most_common(1)[0][0]

                        if most_common_answer in options:
                            answer = options[most_common_answer]
                            is_correct = answer == q_ans
                        else:
                            is_correct = False

                # Respuesta de 2 y 3 iguales, le volvemos a preguntar a 1
                elif answer_2_1 == answer_3_1 and answer_2_1 != answer_1_1:
                    prompt_1 = feedback_prompt_2(answer_2_1, explanation_2_1, explanation_3_1)
                    llm_answer_1_2 = get_llm_response(llm_1, prompt_1, model_1)
                    answer_1_2, explanation_1_2 = clean_response_multiple(llm_answer_1_2)
                    # Si ahora hay consenso nos quedamos con esa respuesta
                    if answer_2_1 == answer_1_2:
                        if answer_2_1 in options:
                            answer = options[answer_1_1]
                            is_correct = answer == q_ans
                        else:
                            is_correct = False
                    # Si no hay consenso, se elige la respuesta con más apariciones (Pueden agregarse iteraciones u otra forma)
                    else:
                        explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1, explanation_1_2]
                        answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1, answer_1_2]

                        while explanations.count("Random answer") > 0:
                            index = explanations.index("Random answer")
                            answers.pop(index)
                            explanations.pop(index)

                        most_common_answer = Counter(answers).most_common(1)[0][0]

                        if most_common_answer in options:
                            answer = options[most_common_answer]
                            is_correct = answer == q_ans
                        else:
                            is_correct = False

                # Todas las respuestas distintas
                elif answer_1_1 != answer_2_1 != answer_3_1:
                    # Se le pregunta a cada uno si no creen que es la respuesta de los otros + explicación
                    prompt_1 = feedback_prompt_1(answer_2, explanation_2, answer_3, explanation_3)
                    prompt_2 = feedback_prompt_1(answer_1, explanation_1, answer_3, explanation_3)
                    prompt_3 = feedback_prompt_1(answer_1, explanation_1, answer_2, explanation_2)

                    llm_answer_1_2 = get_llm_response(llm_1, prompt_1, model_1)
                    llm_answer_2_2 = get_llm_response(llm_2, prompt_2, model_2)
                    llm_answer_3_2 = get_llm_response(llm_3, prompt_3, model_3)

                    answer_1_2, explanation_1_2 = clean_response_multiple(llm_answer_1_2)
                    answer_2_2, explanation_2_2 = clean_response_multiple(llm_answer_2_2)
                    answer_3_2, explanation_3_2 = clean_response_multiple(llm_answer_3_2)

                    # Si se llega a un consenso, se elige esa respuesta
                    if answer_1_2 == answer_2_2 == answer_3_2:
                        if answer_1_2 in options:
                            answer = options[answer_1_2]
                            is_correct = answer == q_ans
                        else:
                            is_correct = False
                    # Si no se llega a un consenso, se elige la respuesta con más apariciones (Pueden agregarse iteraciones u otra forma)
                    else:
                        explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1, explanation_1_2, explanation_2_2, explanation_3_2]
                        answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1, answer_1_2, answer_2_2, answer_3_2]

                        while explanations.count("Random answer") > 0:
                            index = explanations.index("Random answer")
                            answers.pop(index)
                            explanations.pop(index)

                        most_common_answer = Counter(answers).most_common(1)[0][0]

                        if most_common_answer in options:
                            answer = options[most_common_answer]
                            is_correct = answer == q_ans
                        else:
                            is_correct = False

        # Dos respuesta iguales y una distinta
        elif answer_1 == answer_2 and answer_2 != answer_3:
            prompt_3 = feedback_prompt_2(answer_1, explanation_1, explanation_2)

            llm_answer_3_1 = get_llm_response(llm_3, prompt_3, model_3)
            answer_3_1, explanation_3_1 = clean_response_multiple(llm_answer_3_1)

            # Si cambia de opinion el que pensaba distinto
            if answer_1 == answer_3_1:
                if answer_1 in options:
                    answer = options[answer_1]
                    is_correct = answer == q_ans
                else:
                    is_correct = False
            # Si no cambia de opinion, todos vuelven a contestar y se elige la respuesta con más apariciones (Pueden agregarse iteraciones u otra forma)
            else:
                llm_answer_1_1 = get_llm_response(llm_1, prompt, model_1)
                llm_answer_2_1 = get_llm_response(llm_2, prompt, model_2)

                answer_1_1, explanation_1_1 = clean_response_multiple(llm_answer_1_1)
                answer_2_1, explanation_2_1 = clean_response_multiple(llm_answer_2_1)

                answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1]
                explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1]

                while explanations.count("Random answer") > 0:
                    index = explanations.index("Random answer")
                    answers.pop(index)
                    explanations.pop(index)

                most_common_answer = Counter(answers).most_common(1)[0][0]

                if most_common_answer in options:
                    answer = options[most_common_answer]
                    is_correct = answer == q_ans
                else:
                    is_correct = False

        # Misma logica elif anterior, solo que cambian los roles
        elif answer_1 == answer_3 and answer_3 != answer_2:
            prompt_2 = feedback_prompt_2(answer_1, explanation_1, explanation_3)

            llm_answer_2_1 = get_llm_response(llm_2, prompt_2, model_2)
            answer_2_1, explanation_2_1 = clean_response_multiple(llm_answer_2_1)

            if answer_1 == answer_2_1:
                if answer_1 in options:
                    answer = options[answer_1]
                    is_correct = answer == q_ans
                else:
                    is_correct = False
            else:
                llm_answer_1_1 = get_llm_response(llm_1, prompt, model_1)
                llm_answer_3_1 = get_llm_response(llm_3, prompt, model_3)

                answer_1_1, explanation_1_1 = clean_response_multiple(llm_answer_1_1)
                answer_3_1, explanation_3_1 = clean_response_multiple(llm_answer_3_1)

                answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1]
                explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1]

                while explanations.count("Random answer") > 0:
                    index = explanations.index("Random answer")
                    answers.pop(index)
                    explanations.pop(index)

                most_common_answer = Counter(answers).most_common(1)[0][0]

                if most_common_answer in options:
                    answer = options[most_common_answer]
                    is_correct = answer == q_ans
                else:
                    is_correct = False

        # Misma logica elif anterior, solo que cambian los roles
        elif answer_2 == answer_3 and answer_3 != answer_1:
            prompt_1 = feedback_prompt_2(answer_2, explanation_2, explanation_3)

            llm_answer_1_1 = get_llm_response(llm_1, prompt_1, model_1)
            answer_1_1, explanation_1_1 = clean_response_multiple(llm_answer_1_1)

            if answer_2 == answer_1_1:
                if answer_2 in options:
                    answer = options[answer_2]
                    is_correct = answer == q_ans
                else:
                    is_correct = False
            else:
                llm_answer_2_1 = get_llm_response(llm_2, prompt, model_2)
                llm_answer_3_1 = get_llm_response(llm_3, prompt, model_3)

                answer_2_1, explanation_2_1 = clean_response_multiple(llm_answer_2_1)
                answer_3_1, explanation_3_1 = clean_response_multiple(llm_answer_3_1)

                answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1]
                explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1]

                while explanations.count("Random answer") > 0:
                    index = explanations.index("Random answer")
                    answers.pop(index)
                    explanations.pop(index)

                most_common_answer = Counter(answers).most_common(1)[0][0]

                if most_common_answer in options:
                    answer = options[most_common_answer]
                    is_correct = answer == q_ans
                else:
                    is_correct = False

        score.append(is_correct)

    return score

n = 100
model_1 = "gemma2-9b"
model_2 = "llama3.1-8b"
model_3 = "mistral-7b-instruct"
api_key = os.getenv("API_KEY")

score = ecqa_game(n, model_1, model_2, model_3, api_key)
print(f"Score: {sum(score)}/{n} ({sum(score)/n*100}%)")
