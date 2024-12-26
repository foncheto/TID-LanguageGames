import json
from openai import OpenAI
from utils import *
import datetime as time

def truthful_qa_eval(n, model, api_key):
    llm = OpenAI(
        api_key=api_key,
        base_url="https://api.llama-api.com"
    )

    # Load the TruthfulQA dataset
    with open("TruthfulQA.json", "r", encoding="utf-8") as f:
        ds = json.load(f)

    score = []
    error_count = 0
    outputs = []

    for i in range(n):
        outputs.append(f"QUESTION N°{i}")

        # Extract question and correct answers
        question_data = ds[i]
        prompt = question_data["Question"]
        correct_answers = question_data["Correct Answers"]

        outputs.append(f"{prompt}")
        outputs.append(f"Correct Answers: {correct_answers}")

        # Get the LLM answer
        llm_answer = get_llm_response(llm, prompt, model, 5)
        answer, explanation = clean_response_boolean(llm_answer)

        outputs.append(f"Model's Answer: {answer}")
        outputs.append(f"Explanation: {explanation}")

        if explanation == "Random answer":
            error_count += 1

        # Check if the LLM answer is correct using the transformer-based function
        is_correct = is_answer_correct_with_transformer(answer, correct_answers)
        score.append(is_correct)

        outputs.append(f"Is Correct: {is_correct}")
        outputs.append("------------------------------------------------------------")

    outputs.append(f"Score: {sum(score)}/{n} ({sum(score)/n*100}%)")
    outputs.append(f"Errors: {error_count}")
    outputs.append("------------------------------------------------------------")

    date = time.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    with open(f"outputs_evals/truthful_qa_eval_{model}_{date}.txt", "w", encoding="utf-8") as f:
        for output in outputs:
            if isinstance(output, tuple):
                f.write(" ".join(map(str, output)) + "\n")
            else:
                f.write(str(output) + "\n")

    return score, error_count
