from openai import OpenAI
from datasets import load_dataset
from collections import Counter
from utils import *
from dotenv import load_dotenv
import os

load_dotenv()

def strategy_qa_game(n, model_1, model_2, model_3, api_key):

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
    ds = load_dataset("wics/strategy-qa")

    # Se inicializa el score
    score = []

    # Se itera sobre el dataset de preguntas
    for i in range (0, n):
        print("Question: ", i)

        # Contener la fila de la pregunta
        prompt, answer = get_data_strategy_qa_and_prompt(ds["test"][i])

        # Conseguir respuestas
        llm_answer_1 = get_llm_response(llm_1, prompt, model_1)
        llm_answer_2 = get_llm_response(llm_2, prompt, model_2)
        llm_answer_3 = get_llm_response(llm_3, prompt, model_3)

        print("INITIAL ANSWERS")

        # Dejar solo el contenido JSON
        answer_1, explanation_1 = clean_response_boolean(llm_answer_1)
        print(answer_1, explanation_1)
        answer_2, explanation_2 = clean_response_boolean(llm_answer_2)
        print(answer_2, explanation_2)
        answer_3, explanation_3 = clean_response_boolean(llm_answer_3)
        print(answer_3, explanation_3)

        correct_answer = "yes" if answer else "no"

        # Si todas las respuestas son iguales, nos quedamos con esa
        if answer_1 == answer_2 and answer_2 == answer_3:
            print("ALL ANSWERS ARE EQUAL, CHOOSING THAT")
            is_correct = answer_1.lower() == correct_answer
        # Una respuesta distinta
        elif answer_1 != answer_2 and answer_2 == answer_3:
            print("ANSWER 2 AND 3 ARE EQUAL, ASKING 1")
            prompt_1 = feedback_prompt_2(answer_2, explanation_2, explanation_3)
            llm_answer_1_1 = get_llm_response(llm_1, prompt_1, model_1)
            answer_1_1, explanation_1_1 = clean_response_boolean(llm_answer_1_1)
            print(answer_1_1, explanation_1_1)

            if answer_1_1 == answer_2:
                print("1 CHANGED HIS MIND, CHOOSING THAT")
                is_correct = answer_1_1.lower() == correct_answer
            else:
                print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")
                explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1]
                answers = [answer_1, answer_2, answer_3, answer_1_1]

                while explanations.count("Random answer") > 0:
                    index = explanations.index("Random answer")
                    answers.pop(index)
                    explanations.pop(index)

                print(f"ANSWERS: {answers}")

                most_common_answer = Counter(answers).most_common(1)[0][0]

                is_correct = most_common_answer.lower() == correct_answer
        # Misma logica elif anterior, solo que cambian los roles
        elif answer_2 != answer_3 and answer_3 == answer_1:
            print("ANSWER 3 AND 1 ARE EQUAL, ASKING 2")
            prompt_2 = feedback_prompt_2(answer_3, explanation_3, explanation_1)
            llm_answer_2_1 = get_llm_response(llm_2, prompt_2, model_2)
            answer_2_1, explanation_2_1 = clean_response_boolean(llm_answer_2_1)
            print(answer_2_1, explanation_2_1)

            if answer_2_1 == answer_3:
                print("2 CHANGED HIS MIND, CHOOSING THAT")
                is_correct = answer_2_1.lower() == correct_answer
            else:
                print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")
                explanations = [explanation_1, explanation_2, explanation_3, explanation_2_1]
                answers = [answer_1, answer_2, answer_3, answer_2_1]

                while explanations.count("Random answer") > 0:
                    index = explanations.index("Random answer")
                    answers.pop(index)
                    explanations.pop(index)

                print(f"ANSWERS: {answers}")

                most_common_answer = Counter(answers).most_common(1)[0][0]

                is_correct = most_common_answer.lower() == correct_answer
        # Misma logica elif anterior, solo que cambian los roles
        elif answer_3 != answer_1 and answer_1 == answer_2:
            print("ANSWER 1 AND 2 ARE EQUAL, ASKING 3")
            prompt_3 = feedback_prompt_2(answer_1, explanation_1, explanation_2)
            llm_answer_3_1 = get_llm_response(llm_3, prompt_3, model_3)
            answer_3_1, explanation_3_1 = clean_response_boolean(llm_answer_3_1)
            print(answer_3_1, explanation_3_1)

            if answer_3_1 == answer_1:
                print("3 CHANGED HIS MIND, CHOOSING THAT")
                is_correct = answer_3_1.lower() == correct_answer
            else:
                print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")
                explanations = [explanation_1, explanation_2, explanation_3, explanation_3_1]
                answers = [answer_1, answer_2, answer_3, answer_3_1]

                while explanations.count("Random answer") > 0:
                    index = explanations.index("Random answer")
                    answers.pop(index)
                    explanations.pop(index)

                print(f"ANSWERS: {answers}")

                most_common_answer = Counter(answers).most_common(1)[0][0]

                is_correct = most_common_answer.lower() == correct_answer
        score.append(1 if is_correct else 0)
    return score

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

        print("INITIAL ANSWERS")
        
        # Dejar solo el contenido JSON
        answer_1, explanation_1 = clean_response_multiple(llm_answer_1)
        print(answer_1, explanation_1)
        answer_2, explanation_2 = clean_response_multiple(llm_answer_2)
        print(answer_2, explanation_2)
        answer_3, explanation_3 = clean_response_multiple(llm_answer_3)
        print(answer_3, explanation_3)

        # Si todas las respuestas son iguales, nos quedamos con esa
        if answer_1 == answer_2 and answer_2 == answer_3:
            print("ALL ANSWERS ARE EQUAL, CHOOSING THAT")
            if answer_1 in options:
                answer = options[answer_1]
                is_correct = answer == q_ans
            else:
                is_correct = False

        # Si todas las respuestas son distintas, se les pregunta a todos si no creen que es la respuesta de los otros + explicación
        elif answer_1 != answer_2 and answer_2 != answer_3 and answer_1 != answer_3:
            print("ALL ANSWERS ARE DIFFERENT, STARTING THE GAME")
            prompt_1 = feedback_prompt_1(answer_2, explanation_2, answer_3, explanation_3)
            prompt_2 = feedback_prompt_1(answer_1, explanation_1, answer_3, explanation_3)
            prompt_3 = feedback_prompt_1(answer_1, explanation_1, answer_2, explanation_2)

            llm_answer_1_1 = get_llm_response(llm_1, prompt_1, model_1)
            llm_answer_2_1 = get_llm_response(llm_2, prompt_2, model_2)
            llm_answer_3_1 = get_llm_response(llm_3, prompt_3, model_3)

            answer_1_1, explanation_1_1 = clean_response_multiple(llm_answer_1_1)
            print(answer_1_1, explanation_1_1)
            answer_2_1, explanation_2_1 = clean_response_multiple(llm_answer_2_1)
            print(answer_2_1, explanation_2_1)
            answer_3_1, explanation_3_1 = clean_response_multiple(llm_answer_3_1)
            print(answer_3_1, explanation_3_1)

            # Si hay consenso, nos quedamos con la respuesta
            if answer_1_1 == answer_2_1 == answer_3_1:
                print("CONSENSUS REACHED")
                if answer_1 in options:
                    answer = options[answer_1]
                    is_correct = answer == q_ans
                else:
                    is_correct = False
            # Si no hay consenso, nueva iteracion
            else:
                # Respuesta de 1 y 2 iguales, le volvemos a preguntar a 3
                if answer_1_1 == answer_2_1 and answer_2_1 != answer_3_1:
                    print("ANSWER 1 AND 2 ARE EQUAL, ASKING 3")
                    prompt_3 = feedback_prompt_2(answer_1_1, explanation_1_1, explanation_2_1)
                    llm_answer_3_2 = get_llm_response(llm_3, prompt_3, model_3)
                    answer_3_2, explanation_3_2 = clean_response_multiple(llm_answer_3_2)
                    print(answer_3_2, explanation_3_2)
                    # Si ahora hay consenso nos quedamos con esa respuesta
                    if answer_1_1 == answer_3_2:
                        print("CONSENSUS REACHED")
                        if answer_1_1 in options:
                            answer = options[answer_1_1]
                            is_correct = answer == q_ans
                        else:
                            is_correct = False
                    # Si no hay consenso, se elige la respuesta con más apariciones (Pueden agregarse iteraciones u otra forma)
                    else:
                        print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")

                        explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1, explanation_3_2]
                        answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1, answer_3_2]

                        while explanations.count("Random answer") > 0:
                            index = explanations.index("Random answer")
                            answers.pop(index)
                            explanations.pop(index)

                        print(f"ANSWERS: {answers}")
                        most_common_answer = Counter(answers).most_common(1)[0][0]
                        if most_common_answer in options:
                            answer = options[most_common_answer]
                            is_correct = answer == q_ans
                        else:
                            is_correct = False
                
                # Respuesta de 1 y 3 iguales, le volvemos a preguntar a 2
                elif answer_1_1 == answer_3_1 and answer_2_1 != answer_3_1:
                    print("ANSWER 1 AND 3 ARE EQUAL, ASKING 2")
                    prompt_2 = feedback_prompt_2(answer_1_1, explanation_1_1, explanation_3_1)
                    llm_answer_2_2 = get_llm_response(llm_2, prompt_2, model_2)
                    answer_2_2, explanation_2_2 = clean_response_multiple(llm_answer_2_2)
                    print(answer_2_2, explanation_2_2)
                    # Si ahora hay consenso nos quedamos con esa respuesta
                    if answer_1_1 == answer_2_2:
                        print("CONSENSUS REACHED")
                        if answer_1_1 in options:
                            answer = options[answer_1_1]
                            is_correct = answer == q_ans
                        else:
                            is_correct = False
                    # Si no hay consenso, se elige la respuesta con más apariciones (Pueden agregarse iteraciones u otra forma)
                    else:
                        print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")
                        explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1, explanation_2_2]
                        answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1, answer_2_2]

                        while explanations.count("Random answer") > 0:
                            index = explanations.index("Random answer")
                            answers.pop(index)
                            explanations.pop(index)
                        print(f"ANSWERS: {answers}")
                        most_common_answer = Counter(answers).most_common(1)[0][0]

                        if most_common_answer in options:
                            answer = options[most_common_answer]
                            is_correct = answer == q_ans
                        else:
                            is_correct = False

                # Respuesta de 2 y 3 iguales, le volvemos a preguntar a 1
                elif answer_2_1 == answer_3_1 and answer_2_1 != answer_1_1:
                    print("ANSWER 2 AND 3 ARE EQUAL, ASKING 1")
                    prompt_1 = feedback_prompt_2(answer_2_1, explanation_2_1, explanation_3_1)
                    llm_answer_1_2 = get_llm_response(llm_1, prompt_1, model_1)
                    answer_1_2, explanation_1_2 = clean_response_multiple(llm_answer_1_2)
                    print(answer_1_2, explanation_1_2)
                    # Si ahora hay consenso nos quedamos con esa respuesta
                    if answer_2_1 == answer_1_2:
                        print("CONSENSUS REACHED")
                        if answer_2_1 in options:
                            answer = options[answer_1_1]
                            is_correct = answer == q_ans
                        else:
                            is_correct = False
                    # Si no hay consenso, se elige la respuesta con más apariciones (Pueden agregarse iteraciones u otra forma)
                    else:
                        print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")
                        explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1, explanation_1_2]
                        answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1, answer_1_2]

                        while explanations.count("Random answer") > 0:
                            index = explanations.index("Random answer")
                            answers.pop(index)
                            explanations.pop(index)
                        print(f"ANSWERS: {answers}")
                        most_common_answer = Counter(answers).most_common(1)[0][0]

                        if most_common_answer in options:
                            answer = options[most_common_answer]
                            is_correct = answer == q_ans
                        else:
                            is_correct = False

                # Todas las respuestas distintas
                elif answer_1_1 != answer_2_1 != answer_3_1:
                    print("ALL ANSWERS ARE DIFFERENT, CONTINUING THE GAME")
                    # Se le pregunta a cada uno si no creen que es la respuesta de los otros + explicación
                    prompt_1 = feedback_prompt_1(answer_2, explanation_2, answer_3, explanation_3)
                    prompt_2 = feedback_prompt_1(answer_1, explanation_1, answer_3, explanation_3)
                    prompt_3 = feedback_prompt_1(answer_1, explanation_1, answer_2, explanation_2)

                    llm_answer_1_2 = get_llm_response(llm_1, prompt_1, model_1)
                    llm_answer_2_2 = get_llm_response(llm_2, prompt_2, model_2)
                    llm_answer_3_2 = get_llm_response(llm_3, prompt_3, model_3)

                    answer_1_2, explanation_1_2 = clean_response_multiple(llm_answer_1_2)
                    print(answer_1_2, explanation_1_2)
                    answer_2_2, explanation_2_2 = clean_response_multiple(llm_answer_2_2)
                    print(answer_2_2, explanation_2_2)
                    answer_3_2, explanation_3_2 = clean_response_multiple(llm_answer_3_2)
                    print(answer_3_2, explanation_3_2)

                    # Si se llega a un consenso, se elige esa respuesta
                    if answer_1_2 == answer_2_2 == answer_3_2:
                        print("CONSENSUS REACHED")
                        if answer_1_2 in options:
                            answer = options[answer_1_2]
                            is_correct = answer == q_ans
                        else:
                            is_correct = False
                    # Si no se llega a un consenso, se elige la respuesta con más apariciones (Pueden agregarse iteraciones u otra forma)
                    else:
                        print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")
                        explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1, explanation_1_2, explanation_2_2, explanation_3_2]
                        answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1, answer_1_2, answer_2_2, answer_3_2]

                        while explanations.count("Random answer") > 0:
                            index = explanations.index("Random answer")
                            answers.pop(index)
                            explanations.pop(index)
                        print(f"ANSWERS: {answers}")

                        most_common_answer = Counter(answers).most_common(1)[0][0]

                        if most_common_answer in options:
                            answer = options[most_common_answer]
                            is_correct = answer == q_ans
                        else:
                            is_correct = False

        # Dos respuesta iguales y una distinta
        elif answer_1 == answer_2 and answer_2 != answer_3:
            print("ANSWER 1 AND 2 ARE EQUAL, ASKING 3")
            prompt_3 = feedback_prompt_2(answer_1, explanation_1, explanation_2)

            llm_answer_3_1 = get_llm_response(llm_3, prompt_3, model_3)
            answer_3_1, explanation_3_1 = clean_response_multiple(llm_answer_3_1)
            print(answer_3_1, explanation_3_1)

            # Si cambia de opinion el que pensaba distinto
            if answer_1 == answer_3_1:
                print("3 CHANGED HIS MIND, CHOOSING THAT")
                if answer_1 in options:
                    answer = options[answer_1]
                    is_correct = answer == q_ans
                else:
                    is_correct = False
            # Si no cambia de opinion, todos vuelven a contestar y se elige la respuesta con más apariciones (Pueden agregarse iteraciones u otra forma)
            else:
                print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")
                llm_answer_1_1 = get_llm_response(llm_1, prompt, model_1)
                llm_answer_2_1 = get_llm_response(llm_2, prompt, model_2)

                answer_1_1, explanation_1_1 = clean_response_multiple(llm_answer_1_1)
                print(answer_1_1, explanation_1_1)
                answer_2_1, explanation_2_1 = clean_response_multiple(llm_answer_2_1)
                print(answer_2_1, explanation_2_1)

                answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1]
                explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1]

                while explanations.count("Random answer") > 0:
                    index = explanations.index("Random answer")
                    answers.pop(index)
                    explanations.pop(index)

                print(f"ANSWERS: {answers}")

                most_common_answer = Counter(answers).most_common(1)[0][0]

                if most_common_answer in options:
                    answer = options[most_common_answer]
                    is_correct = answer == q_ans
                else:
                    is_correct = False

        # Misma logica elif anterior, solo que cambian los roles
        elif answer_1 == answer_3 and answer_3 != answer_2:
            print("ANSWER 1 AND 3 ARE EQUAL, ASKING 2")
            prompt_2 = feedback_prompt_2(answer_1, explanation_1, explanation_3)

            llm_answer_2_1 = get_llm_response(llm_2, prompt_2, model_2)
            answer_2_1, explanation_2_1 = clean_response_multiple(llm_answer_2_1)
            print(answer_2_1, explanation_2_1)

            if answer_1 == answer_2_1:
                print("2 CHANGED HIS MIND, CHOOSING THAT")
                if answer_1 in options:
                    answer = options[answer_1]
                    is_correct = answer == q_ans
                else:
                    is_correct = False
            else:
                print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")
                llm_answer_1_1 = get_llm_response(llm_1, prompt, model_1)
                llm_answer_3_1 = get_llm_response(llm_3, prompt, model_3)

                answer_1_1, explanation_1_1 = clean_response_multiple(llm_answer_1_1)
                print(answer_1_1, explanation_1_1)
                answer_3_1, explanation_3_1 = clean_response_multiple(llm_answer_3_1)
                print(answer_3_1, explanation_3_1)

                answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1]
                explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1]

                while explanations.count("Random answer") > 0:
                    index = explanations.index("Random answer")
                    answers.pop(index)
                    explanations.pop(index)

                print(f"ANSWERS: {answers}")

                most_common_answer = Counter(answers).most_common(1)[0][0]

                if most_common_answer in options:
                    answer = options[most_common_answer]
                    is_correct = answer == q_ans
                else:
                    is_correct = False

        # Misma logica elif anterior, solo que cambian los roles
        elif answer_2 == answer_3 and answer_3 != answer_1:
            print("ANSWER 2 AND 3 ARE EQUAL, ASKING 1")
            prompt_1 = feedback_prompt_2(answer_2, explanation_2, explanation_3)

            llm_answer_1_1 = get_llm_response(llm_1, prompt_1, model_1)
            answer_1_1, explanation_1_1 = clean_response_multiple(llm_answer_1_1)
            print(answer_1_1, explanation_1_1)

            if answer_2 == answer_1_1:
                print("1 CHANGED HIS MIND, CHOOSING THAT")
                if answer_2 in options:
                    answer = options[answer_2]
                    is_correct = answer == q_ans
                else:
                    is_correct = False
            else:
                print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")
                llm_answer_2_1 = get_llm_response(llm_2, prompt, model_2)
                llm_answer_3_1 = get_llm_response(llm_3, prompt, model_3)

                answer_2_1, explanation_2_1 = clean_response_multiple(llm_answer_2_1)
                print(answer_2_1, explanation_2_1)
                answer_3_1, explanation_3_1 = clean_response_multiple(llm_answer_3_1)
                print(answer_3_1, explanation_3_1)

                answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1]
                explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1]

                while explanations.count("Random answer") > 0:
                    index = explanations.index("Random answer")
                    answers.pop(index)
                    explanations.pop(index)

                print(f"ANSWERS: {answers}")

                most_common_answer = Counter(answers).most_common(1)[0][0]

                if most_common_answer in options:
                    answer = options[most_common_answer]
                    is_correct = answer == q_ans
                else:
                    is_correct = False

        score.append(is_correct)

    return score

def common_sense_game(n, model_1, model_2, model_3, api_key):

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
    ds = load_dataset("tau/commonsense_qa")

    # Se inicializa el score
    score = []

    # Se itera sobre el dataset de preguntas
    for i in range (0, n):
        print("Question: ", i)

        # Contener la fila de la pregunta
        prompt, q_ans = get_data_commonsense_qa_and_prompt(ds["train"][i])

        # Conseguir respuestas
        llm_answer_1 = get_llm_response(llm_1, prompt, model_1)
        llm_answer_2 = get_llm_response(llm_2, prompt, model_2)
        llm_answer_3 = get_llm_response(llm_3, prompt, model_3)

        print("INITIAL ANSWERS")

        # Dejar solo el contenido JSON
        answer_1, explanation_1 = clean_response_multiple(llm_answer_1)
        print(answer_1, explanation_1)
        answer_2, explanation_2 = clean_response_multiple(llm_answer_2)
        print(answer_2, explanation_2)
        answer_3, explanation_3 = clean_response_multiple(llm_answer_3)
        print(answer_3, explanation_3)

        # Si todas las respuestas son iguales, nos quedamos con esa
        if answer_1 == answer_2 and answer_2 == answer_3:
            print("ALL ANSWERS ARE EQUAL, CHOOSING THAT")
            is_correct = answer_1 == q_ans

        # Si todas las respuestas son distintas, se les pregunta a todos si no creen que es la respuesta de los otros + explicación
        elif answer_1 != answer_2 and answer_2 != answer_3 and answer_1 != answer_3:
            print("ALL ANSWERS ARE DIFFERENT, STARTING THE GAME")
            prompt_1 = feedback_prompt_1(answer_2, explanation_2, answer_3, explanation_3)
            prompt_2 = feedback_prompt_1(answer_1, explanation_1, answer_3, explanation_3)
            prompt_3 = feedback_prompt_1(answer_1, explanation_1, answer_2, explanation_2)

            llm_answer_1_1 = get_llm_response(llm_1, prompt_1, model_1)
            llm_answer_2_1 = get_llm_response(llm_2, prompt_2, model_2)
            llm_answer_3_1 = get_llm_response(llm_3, prompt_3, model_3)

            answer_1_1, explanation_1_1 = clean_response_multiple(llm_answer_1_1)
            print(answer_1_1, explanation_1_1)
            answer_2_1, explanation_2_1 = clean_response_multiple(llm_answer_2_1)
            print(answer_2_1, explanation_2_1)
            answer_3_1, explanation_3_1 = clean_response_multiple(llm_answer_3_1)
            print(answer_3_1, explanation_3_1)

            # Si hay consenso, nos quedamos con la respuesta
            if answer_1_1 == answer_2_1 == answer_3_1:
                print("CONSENSUS REACHED")
                is_correct = answer_2_1 == q_ans
            # Si no hay consenso, nueva iteracion
            else:
                # Respuesta de 1 y 2 iguales, le volvemos a preguntar a 3
                if answer_1_1 == answer_2_1 and answer_2_1 != answer_3_1:
                    print("ANSWER 1 AND 2 ARE EQUAL, ASKING 3")
                    prompt_3 = feedback_prompt_2(answer_1_1, explanation_1_1, explanation_2_1)
                    llm_answer_3_2 = get_llm_response(llm_3, prompt_3, model_3)
                    answer_3_2, explanation_3_2 = clean_response_multiple(llm_answer_3_2)
                    print(answer_3_2, explanation_3_2)
                    # Si ahora hay consenso nos quedamos con esa respuesta
                    if answer_1_1 == answer_3_2:
                        print("CONSENSUS REACHED")
                        is_correct = answer_1_1 == q_ans
                    # Si no hay consenso, se elige la respuesta con más apariciones (Pueden agregarse iteraciones u otra forma)
                    else:
                        print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")

                        explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1, explanation_3_2]
                        answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1, answer_3_2]

                        while explanations.count("Random answer") > 0:
                            index = explanations.index("Random answer")
                            answers.pop(index)
                            explanations.pop(index)

                        print(f"ANSWERS: {answers}")
                        most_common_answer = Counter(answers).most_common(1)[0][0]
                        is_correct = most_common_answer == q_ans
                
                # Respuesta de 1 y 3 iguales, le volvemos a preguntar a 2
                elif answer_1_1 == answer_3_1 and answer_2_1 != answer_3_1:
                    print("ANSWER 1 AND 3 ARE EQUAL, ASKING 2")
                    prompt_2 = feedback_prompt_2(answer_1_1, explanation_1_1, explanation_3_1)
                    llm_answer_2_2 = get_llm_response(llm_2, prompt_2, model_2)
                    answer_2_2, explanation_2_2 = clean_response_multiple(llm_answer_2_2)
                    print(answer_2_2, explanation_2_2)
                    # Si ahora hay consenso nos quedamos con esa respuesta
                    if answer_1_1 == answer_2_2:
                        print("CONSENSUS REACHED")
                        is_correct = answer_1_1 == q_ans
                    # Si no hay consenso, se elige la respuesta con más apariciones (Pueden agregarse iteraciones u otra forma)
                    else:
                        print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")
                        explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1, explanation_2_2]
                        answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1, answer_2_2]

                        while explanations.count("Random answer") > 0:
                            index = explanations.index("Random answer")
                            answers.pop(index)
                            explanations.pop(index)
                        print(f"ANSWERS: {answers}")
                        most_common_answer = Counter(answers).most_common(1)[0][0]

                        is_correct = most_common_answer == q_ans

                # Respuesta de 2 y 3 iguales, le volvemos a preguntar a 1
                elif answer_2_1 == answer_3_1 and answer_2_1 != answer_1_1:
                    print("ANSWER 2 AND 3 ARE EQUAL, ASKING 1")
                    prompt_1 = feedback_prompt_2(answer_2_1, explanation_2_1, explanation_3_1)
                    llm_answer_1_2 = get_llm_response(llm_1, prompt_1, model_1)
                    answer_1_2, explanation_1_2 = clean_response_multiple(llm_answer_1_2)
                    print(answer_1_2, explanation_1_2)
                    # Si ahora hay consenso nos quedamos con esa respuesta
                    if answer_2_1 == answer_1_2:
                        print("CONSENSUS REACHED")
                        is_correct = answer_2_1 == q_ans
                    # Si no hay consenso, se elige la respuesta con más apariciones (Pueden agregarse iteraciones u otra forma)
                    else:
                        print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")
                        explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1, explanation_1_2]
                        answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1, answer_1_2]

                        while explanations.count("Random answer") > 0:
                            index = explanations.index("Random answer")
                            answers.pop(index)
                            explanations.pop(index)
                        print(f"ANSWERS: {answers}")
                        most_common_answer = Counter(answers).most_common(1)[0][0]

                        is_correct = most_common_answer == q_ans

                # Todas las respuestas distintas
                elif answer_1_1 != answer_2_1 != answer_3_1:
                    print("ALL ANSWERS ARE DIFFERENT, CONTINUING THE GAME")
                    # Se le pregunta a cada uno si no creen que es la respuesta de los otros + explicación
                    prompt_1 = feedback_prompt_1(answer_2, explanation_2, answer_3, explanation_3)
                    prompt_2 = feedback_prompt_1(answer_1, explanation_1, answer_3, explanation_3)
                    prompt_3 = feedback_prompt_1(answer_1, explanation_1, answer_2, explanation_2)

                    llm_answer_1_2 = get_llm_response(llm_1, prompt_1, model_1)
                    llm_answer_2_2 = get_llm_response(llm_2, prompt_2, model_2)
                    llm_answer_3_2 = get_llm_response(llm_3, prompt_3, model_3)

                    answer_1_2, explanation_1_2 = clean_response_multiple(llm_answer_1_2)
                    print(answer_1_2, explanation_1_2)
                    answer_2_2, explanation_2_2 = clean_response_multiple(llm_answer_2_2)
                    print(answer_2_2, explanation_2_2)
                    answer_3_2, explanation_3_2 = clean_response_multiple(llm_answer_3_2)
                    print(answer_3_2, explanation_3_2)

                    # Si se llega a un consenso, se elige esa respuesta
                    if answer_1_2 == answer_2_2 == answer_3_2:
                        print("CONSENSUS REACHED")
                        is_correct = answer_1_2 == q_ans
                    # Si no se llega a un consenso, se elige la respuesta con más apariciones (Pueden agregarse iteraciones u otra forma)
                    else:
                        print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")
                        explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1, explanation_1_2, explanation_2_2, explanation_3_2]
                        answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1, answer_1_2, answer_2_2, answer_3_2]

                        while explanations.count("Random answer") > 0:
                            index = explanations.index("Random answer")
                            answers.pop(index)
                            explanations.pop(index)
                        print(f"ANSWERS: {answers}")

                        most_common_answer = Counter(answers).most_common(1)[0][0]
                        is_correct = most_common_answer == q_ans

        # Dos respuesta iguales y una distinta
        elif answer_1 == answer_2 and answer_2 != answer_3:
            print("ANSWER 1 AND 2 ARE EQUAL, ASKING 3")
            prompt_3 = feedback_prompt_2(answer_1, explanation_1, explanation_2)

            llm_answer_3_1 = get_llm_response(llm_3, prompt_3, model_3)
            answer_3_1, explanation_3_1 = clean_response_multiple(llm_answer_3_1)
            print(answer_3_1, explanation_3_1)

            # Si cambia de opinion el que pensaba distinto
            if answer_1 == answer_3_1:
                print("3 CHANGED HIS MIND, CHOOSING THAT")
                is_correct = answer_1 == q_ans
            # Si no cambia de opinion, todos vuelven a contestar y se elige la respuesta con más apariciones (Pueden agregarse iteraciones u otra forma)
            else:
                print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")
                llm_answer_1_1 = get_llm_response(llm_1, prompt, model_1)
                llm_answer_2_1 = get_llm_response(llm_2, prompt, model_2)

                answer_1_1, explanation_1_1 = clean_response_multiple(llm_answer_1_1)
                print(answer_1_1, explanation_1_1)
                answer_2_1, explanation_2_1 = clean_response_multiple(llm_answer_2_1)
                print(answer_2_1, explanation_2_1)

                answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1]
                explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1]

                while explanations.count("Random answer") > 0:
                    index = explanations.index("Random answer")
                    answers.pop(index)
                    explanations.pop(index)

                print(f"ANSWERS: {answers}")

                most_common_answer = Counter(answers).most_common(1)[0][0]
                is_correct = most_common_answer == q_ans

        # Misma logica elif anterior, solo que cambian los roles
        elif answer_1 == answer_3 and answer_3 != answer_2:
            print("ANSWER 1 AND 3 ARE EQUAL, ASKING 2")
            prompt_2 = feedback_prompt_2(answer_1, explanation_1, explanation_3)

            llm_answer_2_1 = get_llm_response(llm_2, prompt_2, model_2)
            answer_2_1, explanation_2_1 = clean_response_multiple(llm_answer_2_1)
            print(answer_2_1, explanation_2_1)

            if answer_1 == answer_2_1:
                print("2 CHANGED HIS MIND, CHOOSING THAT")
                is_correct = answer_1 == q_ans
            else:
                print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")
                llm_answer_1_1 = get_llm_response(llm_1, prompt, model_1)
                llm_answer_3_1 = get_llm_response(llm_3, prompt, model_3)

                answer_1_1, explanation_1_1 = clean_response_multiple(llm_answer_1_1)
                print(answer_1_1, explanation_1_1)
                answer_3_1, explanation_3_1 = clean_response_multiple(llm_answer_3_1)
                print(answer_3_1, explanation_3_1)

                answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1]
                explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1]

                while explanations.count("Random answer") > 0:
                    index = explanations.index("Random answer")
                    answers.pop(index)
                    explanations.pop(index)

                print(f"ANSWERS: {answers}")

                most_common_answer = Counter(answers).most_common(1)[0][0]
                is_correct = most_common_answer == q_ans

        # Misma logica elif anterior, solo que cambian los roles
        elif answer_2 == answer_3 and answer_3 != answer_1:
            print("ANSWER 2 AND 3 ARE EQUAL, ASKING 1")
            prompt_1 = feedback_prompt_2(answer_2, explanation_2, explanation_3)

            llm_answer_1_1 = get_llm_response(llm_1, prompt_1, model_1)
            answer_1_1, explanation_1_1 = clean_response_multiple(llm_answer_1_1)
            print(answer_1_1, explanation_1_1)

            if answer_2 == answer_1_1:
                print("1 CHANGED HIS MIND, CHOOSING THAT")
                is_correct = answer_2 == q_ans
            else:
                print("NO CONSENSUS, CHOOSING MOST COMMON ANSWER")
                llm_answer_2_1 = get_llm_response(llm_2, prompt, model_2)
                llm_answer_3_1 = get_llm_response(llm_3, prompt, model_3)

                answer_2_1, explanation_2_1 = clean_response_multiple(llm_answer_2_1)
                print(answer_2_1, explanation_2_1)
                answer_3_1, explanation_3_1 = clean_response_multiple(llm_answer_3_1)
                print(answer_3_1, explanation_3_1)

                answers = [answer_1, answer_2, answer_3, answer_1_1, answer_2_1, answer_3_1]
                explanations = [explanation_1, explanation_2, explanation_3, explanation_1_1, explanation_2_1, explanation_3_1]

                while explanations.count("Random answer") > 0:
                    index = explanations.index("Random answer")
                    answers.pop(index)
                    explanations.pop(index)

                print(f"ANSWERS: {answers}")

                most_common_answer = Counter(answers).most_common(1)[0][0]

                is_correct = most_common_answer == q_ans

        score.append(is_correct)

    return score

n = 100
model_1 = "gemma2-9b"
model_2 = "llama3.1-8b"
model_3 = "mistral-7b-instruct"
api_key = os.getenv("API_KEY")

score = strategy_qa_game(n, model_1, model_2, model_3, api_key)
print(f"Score: {sum(score)}/{n} ({sum(score)/n*100}%)")

