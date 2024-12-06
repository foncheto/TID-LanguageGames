from datasets import load_dataset
from openai import OpenAI
from utils import *
from dotenv import load_dotenv
import os

load_dotenv()

def commonsense_qa_eval(n, model, api_key):
    llm = OpenAI(
        api_key = api_key,
        base_url = "https://api.llama-api.com"
    )

    ds = load_dataset("tau/commonsense_qa")

    score = []
    error_count = 0

    for i in range (0, n):
        print("Question: ", i)

        max_error = 5

        prompt, q_ans = get_data_commonsense_qa_and_prompt(ds["train"][i])

        llm_answer = get_llm_response(llm, prompt, model)
        
        answer, explanation = clean_response_multiple(llm_answer)

        while explanation == "Random answer" and max_error > 0:
            prompt = f"Wrong format, please provide the answer in the following format: {{\"correct_option\": \"X\", \"explanation\": \"X\"}}:"
            llm_answer = get_llm_response(llm, prompt, model)
            answer, explanation = clean_response_multiple(llm_answer)
            max_error -= 1

        if explanation == "Random answer":
            error_count += 1

        is_correct = answer == q_ans
        score.append(1 if is_correct else 0)

    return score, error_count

def ecqa_eval(n, model, api_key):
    llm = OpenAI(
        api_key = api_key,
        base_url = "https://api.llama-api.com"
    )

    ds = load_dataset("yangdong/ecqa")

    score = []
    error_count = 0

    for i in range (0, n):
        print("Question: ", i)

        row = ds["train"][i]

        max_error = 5

        prompt, q_ans = get_data_ecqa_and_prompt(row)

        options = {
            "A": row['q_op1'],
            "B": row['q_op2'],
            "C": row['q_op3'],
            "D": row['q_op4'],
            "E": row['q_op5']
        }
    
        llm_answer = get_llm_response(llm, prompt, model)
        
        answer, explanation = clean_response_multiple(llm_answer)

        while explanation == "Random answer" and max_error > 0:
            prompt = f"Wrong format, please provide the answer in the following format: {{\"correct_option\": \"X\", \"explanation\": \"X\"}}:"
            llm_answer = get_llm_response(llm, prompt, model)
            answer, explanation = clean_response_multiple(llm_answer)
            max_error -= 1
        
        if explanation == "Random answer":
            error_count += 1

        # Check if the LLM answer is in the options
        if answer in options:
            answer = options[answer]
            # Check if the LLM answer is correct
            is_correct = answer == q_ans
        # If the LLM answer is not in the options, it is incorrect
        else:
            is_correct = False

        score.append(1 if is_correct else 0)

    return score, error_count


def strategy_qa_eval(n, model, api_key):
    llm = OpenAI(
        api_key = api_key,
        base_url = "https://api.llama-api.com"
    )

    ds = load_dataset("wics/strategy-qa")

    score = []
    error_count = 0

    for i in range (0, n):
        print("Question: ", i)

        max_error = 5

        # Get the question and answer
        prompt, answer = get_data_strategy_qa_and_prompt(ds["test"][i])

        # Get the LLM answer
        llm_answer = get_llm_response(llm, prompt, model)
        
        answer, explanation = clean_response_boolean(llm_answer)
        
        while explanation == "Random answer" and max_error > 0:
            prompt = f"Wrong format, please provide the answer in the following format: {{\"correct_option\": \"X\", \"explanation\": \"X\"}}:"
            llm_answer = get_llm_response(llm, prompt, model)
            answer, explanation = clean_response_boolean(llm_answer)
            max_error -= 1
        
        if explanation == "Random answer":
            error_count += 1

        # Check if the LLM answer is correct
        correct_answer = "yes" if answer else "no"
        llm_answer_filtered = ''.join(filter(str.isalpha, answer.lower()))

        score.append(1 if llm_answer_filtered == correct_answer else 0)

    return score, error_count

n = 100
models = ["gemma2-9b", "llama3.1-8b", "mistral-7b-instruct"]
api_key = os.getenv("API_KEY")

for model in models:
    print(f"Model: {model}")
    score, error = commonsense_qa_eval(n, model, api_key)
    print(f"Score: {sum(score)}/{n} ({sum(score)/n*100}%)")
    print(f"Errors: {error} ({error/n*100}%)")
    #score, error = ecqa_eval(n, model, api_key)
    #print(f"Score: {sum(score)}/{n} ({sum(score)/n*100}%)")
    #print(f"Errors: {error} ({error/n*100}%)")
    score, error = strategy_qa_eval(n, model, api_key)
    print(f"Score: {sum(score)}/{n} ({sum(score)/n*100}%)")
    print(f"Errors: {error} ({error/n*100}%)")