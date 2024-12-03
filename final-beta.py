import logging
from openai import OpenAI
from datasets import load_dataset
from collections import Counter
from utils2 import *
from dotenv import load_dotenv
import os

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for verbose output, INFO for less verbosity
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("evaluation.log"),  # Log to file
        logging.StreamHandler()  # Also log to console
    ]
)

def initialize_models(api_key, model_names):
    logging.info(f"Initializing models: {model_names}")
    return [OpenAI(api_key=api_key, base_url="https://api.llama-api.com") for _ in model_names]

def get_consensus_or_feedback(models, answers, explanations, correct_answer, dataset_prompt, model_names):
    logging.debug(f"Evaluating consensus or feedback.")
    logging.debug(f"Initial answers: {answers}")
    logging.debug(f"Initial explanations: {explanations}")

    unique_answers = set(answers)
    if len(unique_answers) == 1:
        logging.info(f"Consensus reached without feedback: {answers[0]}")
        return answers[0].lower() == correct_answer, answers[0]

    most_common = Counter(answers).most_common(1)[0][0]
    if answers.count(most_common) > 1:
        logging.info(f"Majority agreement on answer: {most_common}")
        return most_common.lower() == correct_answer, most_common

    logging.info("No initial consensus. Generating feedback prompts for further evaluation.")
    feedback_prompts = [
        feedback_prompt_2(most_common, explanations[i], explanations[j])
        for i in range(len(answers))
        for j in range(len(answers))
        if i != j
    ]
    logging.debug(f"Generated feedback prompts: {feedback_prompts}")

    new_responses = [
        clean_response_boolean(get_llm_response(models[i], feedback_prompts[i], model_names[i]))
        for i in range(len(models))
    ]

    new_answers = [response[0] for response in new_responses]
    logging.debug(f"New answers after feedback: {new_answers}")

    most_common_feedback_answer = Counter(new_answers).most_common(1)[0][0]
    logging.info(f"Most common answer after feedback: {most_common_feedback_answer}")

    return most_common_feedback_answer.lower() == correct_answer, most_common_feedback_answer

def evaluate_dataset(n, models, model_names, api_key, dataset_name, process_data_fn, correct_answer_fn):
    logging.info(f"Evaluating dataset: {dataset_name} for {n} samples.")
    ds = load_dataset(dataset_name)
    scores = []

    for i in range(n):
        logging.info(f"Processing sample {i+1}/{n}")
        prompt, correct_answer = process_data_fn(ds["test"][i])
        logging.debug(f"Generated prompt: {prompt}")
        logging.debug(f"Correct answer: {correct_answer}")

        try:
            responses = [
                clean_response_boolean(get_llm_response(models[j], prompt, model_names[j]))
                for j in range(len(models))
            ]
        except Exception as e:
            logging.error(f"Error generating responses for sample {i+1}: {e}")
            continue

        answers = [response[0] for response in responses]
        explanations = [response[1] for response in responses]
        logging.debug(f"Model answers: {answers}")
        logging.debug(f"Model explanations: {explanations}")

        try:
            is_correct, chosen_answer = get_consensus_or_feedback(
                models, answers, explanations, correct_answer, prompt, model_names
            )
            logging.info(f"Final chosen answer: {chosen_answer}. Correct: {is_correct}")
            scores.append(1 if is_correct else 0)
        except Exception as e:
            logging.error(f"Error during consensus/feedback evaluation for sample {i+1}: {e}")
            scores.append(0)

    return scores

if __name__ == "__main__":
    # Parameters
    n = 10
    api_key = os.getenv("API_KEY")
    model_names = ["gemma2-9b", "llama3.1-8b", "mistral-7b-instruct"]
    models = initialize_models(api_key, model_names)

    logging.info("Starting Strategy QA evaluation.")
    strategy_qa_scores = evaluate_dataset(
        n,
        models,
        model_names,
        api_key,
        "wics/strategy-qa",
        get_data_strategy_qa_and_prompt,
        lambda x: "yes" if x else "no",
    )

    score_percentage = sum(strategy_qa_scores) / n * 100
    logging.info(f"Strategy QA Score: {sum(strategy_qa_scores)}/{n} ({score_percentage:.2f}%)")
    print(f"Strategy QA Score: {sum(strategy_qa_scores)}/{n} ({score_percentage:.2f}%)")
