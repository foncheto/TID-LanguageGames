from collections import Counter
from openai import OpenAI
import json
import datetime as time
import logging
from utils import *

def truthful_qa_game(n, model_1, model_2, model_3, api_key):
    # Set up logging
    log_filename = setup_logging("truthful_qa_game", "outputs_games")
    logging.info(f"Starting Truthful QA Game with {n} questions")
    logging.info(f"Models used: {model_1}, {model_2}, {model_3}")

    try:
        # Initialize models with Llama API
        llms = [OpenAI(api_key=api_key, base_url="https://api.llama-api.com") for _ in range(3)]
        models = [model_1, model_2, model_3]

        # Load the dataset
        logging.info("Loading TruthfulQA dataset")
        with open("TruthfulQA.json", "r", encoding="utf-8") as f:
            ds = json.load(f)

        # Initialize scores and outputs
        score = []
        outputs = []

        # Function to handle conflicts and consensus
        def resolve_conflict(main_index, other_indices, answers, explanations, correct_answers, times):
            logging.info(f"Resolving conflict for LLM {main_index + 1}")
            outputs.append(f"Resolving conflict for LLM {main_index + 1}")

            try:
                prompt = feedback_prompt_2(
                    answers[other_indices[0]], explanations[other_indices[0]], explanations[other_indices[1]]
                )
                new_response = get_llm_response(llms[main_index], prompt, models[main_index], 3)
                new_answer, new_explanation = clean_response_boolean(new_response)
                outputs.append(f"{new_answer}: {new_explanation}")
                logging.info(f"Conflict resolution response: {new_answer}")

                if is_answer_correct_with_transformer(new_answer, correct_answers):
                    logging.info(f"LLM {main_index + 1} resolved conflict successfully")
                    outputs.append(f"LLM {main_index + 1} resolved conflict successfully")
                    score.append(True)
                    return
                elif times > 0:
                    logging.info(f"LLM {main_index + 1} didn't change its mind, trying again")
                    outputs.append("Retrying conflict resolution")
                    return resolve_conflict(main_index, other_indices, answers, explanations, correct_answers, times - 1)
                else:
                    logging.info("No consensus, choosing most common answer")
                    outputs.append("No consensus, choosing most common answer")

                    all_answers = answers + [new_answer]
                    filtered_answers = [
                        ans for ans, exp in zip(all_answers, explanations + [new_explanation]) if exp != "Random answer"
                    ]
                    most_common_answer = Counter(filtered_answers).most_common(1)[0][0]
                    score.append(is_answer_correct_with_transformer(most_common_answer, correct_answers))
                    return
            except Exception as e:
                logging.error(f"Error in conflict resolution: {e}")
                score.append(False)
                return

        # Process each question
        for i in range(n):
            logging.info(f"Processing Question N°{i}")
            outputs.append(f"QUESTION N°{i}")

            try:
                question_data = ds[i]
                prompt = question_data["Question"]
                correct_answers = question_data["Correct Answers"]

                # Get initial responses
                llm_answers = [get_llm_response(llm, prompt, model, 3) for llm, model in zip(llms, models)]
                answers, explanations = zip(*(clean_response_boolean(resp) for resp in llm_answers))

                outputs.append(f"Question: {prompt}")
                outputs.append(f"Correct Answers: {correct_answers}")
                logging.info(f"Correct Answers: {correct_answers}")

                outputs.append("Initial answers:")
                for ans, exp in zip(answers, explanations):
                    outputs.append(f"{ans}: {exp}")
                    logging.info(f"Answer: {ans}, Explanation: {exp}")

                # Consensus logic
                if len(set(answers)) == 1:
                    logging.info("Consensus reached on initial responses")
                    outputs.append("Consensus reached on initial responses")
                    score.append(is_answer_correct_with_transformer(answers[0], correct_answers))
                else:
                    if answers[0] == answers[1] != answers[2]:
                        resolve_conflict(2, [0, 1], list(answers), list(explanations), correct_answers, 3)
                    elif answers[1] == answers[2] != answers[0]:
                        resolve_conflict(0, [1, 2], list(answers), list(explanations), correct_answers, 3)
                    elif answers[2] == answers[0] != answers[1]:
                        resolve_conflict(1, [2, 0], list(answers), list(explanations), correct_answers, 3)

                outputs.append("------------------------------------------------------------")

            except Exception as e:
                logging.error(f"Error processing question {i}: {e}")
                score.append(False)

        # Log final results
        outputs.append(score)
        outputs.append(f"Score: {sum(score)}/{n} ({sum(score)/n*100}%)")
        logging.info(f"Final Score: {sum(score)}/{n} ({sum(score)/n*100}%)")

        # Save results
        date = time.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        with open(f"outputs_games/truthful_qa_game_{date}.txt", "w", encoding="utf-8") as f:
            for output in outputs:
                if isinstance(output, tuple):
                    f.write(" ".join(map(str, output)) + "\n")
                else:
                    f.write(str(output) + "\n")

        logging.info(f"Results saved to outputs_games/truthful_qa_game_{date}.txt")
        logging.info(f"Log file saved to {log_filename}")

        return score

    except Exception as e:
        logging.error(f"Unhandled exception in truthful_qa_game: {e}")
        raise
