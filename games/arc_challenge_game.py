import logging
from collections import Counter
from openai import OpenAI
from datasets import load_dataset
from utils import *

def arc_challenge_game(n, model_1, model_2, model_3, api_key):
    # Initialize models only once
    llms = [OpenAI(api_key=api_key, base_url="https://api.llama-api.com") for _ in range(3)]
    models = [model_1, model_2, model_3]

    # Load the dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")

    # Initialize scores
    score = []

    # Conflict resolution functions
    def resolve_conflict(answers, explanations, correct_answer, times):
        if len(set(answers)) == 1:
            logging.info(f"Consensus reached on iteration N°{4 - times}")
            score.append(answers[0] == correct_answer)
            return
        elif times <= 0:
            logging.info("No consensus, choosing most common answer after 3 iterations")
            filtered_answers = [ans for ans, exp in zip(answers, explanations) if exp != "Random answer"]
            logging.info(f"Filtered answers: {filtered_answers}")
            try:
                most_common_answer = Counter(filtered_answers).most_common(1)[0][0]
                score.append(most_common_answer == correct_answer)
            except Exception:
                score.append(False)
            return
        else:
            if answers[0] == answers[1] != answers[2]:
                logging.info("Conflict detected: 1 and 2 are equal, 3 is different")
                resolve_conflict_one_different(2, [0, 1], answers, explanations, correct_answer, times)
            elif answers[0] == answers[2] != answers[1]:
                logging.info("Conflict detected: 1 and 3 are equal, 2 is different")
                resolve_conflict_one_different(1, [0, 2], answers, explanations, correct_answer, times)
            elif answers[1] == answers[2] != answers[0]:
                logging.info("Conflict detected: 2 and 3 are equal, 1 is different")
                resolve_conflict_one_different(0, [1, 2], answers, explanations, correct_answer, times)
            else:
                logging.info("Conflict detected: all answers are different")
                resolve_conflict_all_different(answers, explanations, correct_answer, times)

    def resolve_conflict_one_different(main_index, other_indices, answers, explanations, correct_answer, times):
        logging.info(f"Resolving conflict for LLM {main_index + 1}")
        prompt = feedback_prompt_2(
            answers[other_indices[0]], explanations[other_indices[0]], explanations[other_indices[1]]
        )
        new_response = get_llm_response(llms[main_index], prompt, models[main_index], 3)
        new_answer, new_explanation = clean_response_multiple(new_response)
        logging.info(f"New answer: {new_answer}, Explanation: {new_explanation}")
        answers = list(answers)
        answers[main_index] = new_answer
        answers = tuple(answers)
        explanations = list(explanations)
        explanations[main_index] = new_explanation
        explanations = tuple(explanations)
        resolve_conflict(answers, explanations, correct_answer, times - 1)

    def resolve_conflict_all_different(answers, explanations, correct_answer, times):
        logging.info("Resolving conflict for all different answers")
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
            logging.info(f"New answer: {ans}, Explanation: {exp}")
        resolve_conflict(new_answers, new_explanations, correct_answer, times - 1)

    # Process questions
    for i in range(n):
        logging.info(f"Processing QUESTION N°{i}")
        row = ds["validation"][i]
        prompt, answer = get_data_arc_and_prompt(row)
        logging.info(f"Question: {row['question']}")
        logging.info(f"The correct answer is: {answer}")

        llm_answers = [get_llm_response(llm, prompt, model, 3) for llm, model in zip(llms, models)]
        answers, explanations = zip(*(clean_response_multiple(resp) for resp in llm_answers))
        logging.info("Initial answers and explanations:")
        for ans, exp in zip(answers, explanations):
            logging.info(f"Answer: {ans}, Explanation: {exp}")
        resolve_conflict(answers, explanations, answer, 3)
        logging.info("------------------------------------------------------------")

    total_score = sum(score)
    percentage = (total_score / n) * 100
    logging.info(f"Final Score: {total_score}/{n} ({percentage:.2f}%)")
    return score
