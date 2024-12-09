from openai import OpenAI
from datasets import load_dataset
from utils import *

def arc_challenge_evaluation(n, model, api_key):
    llm = OpenAI(
        api_key = api_key,
        base_url = "https://api.llama-api.com"
    )

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")

    score = []

    error_count = 0

    for i in range (0, n):

        row = ds["validation"][i]

        prompt, correct_answer = get_data_race_and_prompt(row)

        llm_answer = get_llm_response(llm, prompt, model, 5)
        answer, explanation = clean_response_multiple(llm_answer)

        if explanation == "Random answer":
            error_count += 1
        
        score.append(answer == correct_answer)

    return score, error_count
