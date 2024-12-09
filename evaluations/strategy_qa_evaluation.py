from datasets import load_dataset
from openai import OpenAI
from utils import *
import datetime as time

def strategy_qa_eval(n, model, api_key):
    llm = OpenAI(
        api_key = api_key,
        base_url = "https://api.llama-api.com"
    )

    ds = load_dataset("wics/strategy-qa")

    score = []
    error_count = 0
    outputs = []

    for i in range (0, n):
        outputs.append(f"QUESTION N°{i}")

        # Get the question and answer
        prompt, answer = get_data_strategy_qa_and_prompt(ds["test"][i])

        correct_answer = "yes" if answer else "no"

        outputs.append(f"{ds['test'][i]['question']}")
        outputs.append(f"The correct answer is: {correct_answer}")

        # Get the LLM answer
        llm_answer = get_llm_response(llm, prompt, model, 5)
        
        answer, explanation = clean_response_boolean(llm_answer)

        outputs.append(f"{answer}: {explanation}")

        
        if explanation == "Random answer":
            error_count += 1

        # Check if the LLM answer is correct
        correct_answer = "yes" if answer else "no"
        llm_answer_filtered = ''.join(filter(str.isalpha, answer.lower()))

        score.append(llm_answer_filtered == correct_answer)
        print(llm_answer_filtered == correct_answer)
        outputs.append(f"LLM answer: {llm_answer_filtered}")
        outputs.append("------------------------------------------------------------")
    outputs.append(f"Score: {sum(score)}/{n} ({sum(score)/n*100}%)")
    outputs.append(f"Errors: {error_count}")
    outputs.append("------------------------------------------------------------")

    date = time.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    with open(f"outputs_evals/strategy_qa_eval_{model}_{date}.txt", "w", encoding="utf-8") as f:
        for output in outputs:
            if isinstance(output, tuple):
                f.write(" ".join(map(str, output)) + "\n")
            else:
                f.write(str(output) + "\n")
    return score, error_count