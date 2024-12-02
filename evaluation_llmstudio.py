from datasets import load_dataset
from openai import OpenAI
import csv
import datetime as time

def commonsense_qa_eval(n, model):
    llm = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    ds = load_dataset("tau/commonsense_qa")

    score = []
    rows = []

    for i in range (0, n):
        q_text = ds["train"][i]["question"]
        q_op1 = ds["train"][i]["choices"]["text"][0]
        q_op2 = ds["train"][i]["choices"]["text"][1]
        q_op3 = ds["train"][i]["choices"]["text"][2]
        q_op4 = ds["train"][i]["choices"]["text"][3]
        q_op5 = ds["train"][i]["choices"]["text"][4]
        q_ans = ds["train"][i]["answerKey"]

        prompt = (
                f"I will provide a question and five possible answers. Your task is to select the most correct answer based on the information provided.\n"
                f"Question: {q_text}\n"
                f"Options:\n"
                f"A) {q_op1}\n"
                f"B) {q_op2}\n"
                f"C) {q_op3}\n"
                f"D) {q_op4}\n"
                f"E) {q_op5}\n"
                f"Answer (please respond with only with A, B, C, D, or E, no explanations):"
            )
        
        llm_answer = llm.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=model).choices[0].message.content

        is_correct = llm_answer == q_ans
        score.append(1 if is_correct else 0)

        rows.append({
            "question": q_text,
            "option_A": q_op1,
            "option_B": q_op2,
            "option_C": q_op3,
            "option_D": q_op4,
            "option_E": q_op5,
            "correct_answer": q_ans,
            "is_correct": is_correct
        })

    date = time.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    csv_file = f"csv/commonsense_qa_{date}.csv"
    fields = ["question", "option_A", "option_B", "option_C", "option_D", "option_E", "correct_answer", "is_correct"]

    # Guardar resultados en un archivo CSV
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    
    return score

def ecqa_eval(n, model):
    llm = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    ds = load_dataset("yangdong/ecqa")

    score = []
    rows = []

    for i in range (0, n):
        q_text = ds["train"][i]["q_text"]
        q_op1 = ds["train"][i]["q_op1"]
        q_op2 = ds["train"][i]["q_op2"]
        q_op3 = ds["train"][i]["q_op3"]
        q_op4 = ds["train"][i]["q_op4"]
        q_op5 = ds["train"][i]["q_op5"]
        q_ans = ds["train"][i]["q_ans"]
    
        # Make the prompt
        prompt = (
            f"I will provide a question and five possible answers. Your task is to select the most correct answer based on the information provided.\n"
            f"Question: {q_text}\n"
            f"Options:\n"
            f"A) {q_op1}\n"
            f"B) {q_op2}\n"
            f"C) {q_op3}\n"
            f"D) {q_op4}\n"
            f"E) {q_op5}\n"
            f"Answer (please respond with only with A, B, C, D, or E, no explanations):"
        )

        # Get the LLM answer
        llm_answer = llm.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=model).choices[0].message.content

        # Create a dictionary of options
        options = {
        "A": q_op1,
        "B": q_op2,
        "C": q_op3,
        "D": q_op4,
        "E": q_op5,
        }

        # Check if the LLM answer is in the options
        if llm_answer in options:
            llm_answer = options[llm_answer]
            # Check if the LLM answer is correct
            is_correct = llm_answer == q_ans
        # If the LLM answer is not in the options, it is incorrect
        else:
            is_correct = False

        score.append(1 if is_correct else 0)

        rows.append({
                "question": q_text,
                "option_A": q_op1,
                "option_B": q_op2,
                "option_C": q_op3,
                "option_D": q_op4,
                "option_E": q_op5,
                "correct_answer": q_ans,
                "is_correct": is_correct
            })
    
    date = time.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    csv_file = f"csv/ecqa_{date}.csv"
    fields = ["question", "option_A", "option_B", "option_C", "option_D", "option_E", "correct_answer", "is_correct"]

    # Guardar resultados en un archivo CSV
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return score


def strategy_qa_eval(n, model):
    llm = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    ds = load_dataset("wics/strategy-qa")

    score = []
    rows = []

    for i in range (0, n):
        # Get the question and answer
        question = ds["test"][i]["question"]
        answer = ds["test"][i]["answer"]

        # Make the prompt
        prompt = f"I will provide a question, and you must respond with 'Yes' or 'No' only.\nQuestion: {question}\nAnswer:"
        llm_answer = llm.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=model).choices[0].message.content

        # Check if the LLM answer is correct
        correct_answer = "yes" if answer else "no"
        llm_answer_filtered = ''.join(filter(str.isalpha, llm_answer.lower()))

        score.append(1 if llm_answer_filtered == correct_answer else 0)

        rows.append({
            "question": question,
            "correct_answer": correct_answer,
            "llm_answer": llm_answer,
            "is_correct": llm_answer_filtered == correct_answer
        })

    date = time.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    csv_file = f"csv/strategy_qa_{date}.csv"
    fields = ["question", "correct_answer", "llm_answer", "is_correct"]

    # Guardar resultados en un archivo CSV
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return score

n = 1000
model = "llama-3.2-3b-instruct"

score = commonsense_qa_eval(n, model)
#score = ecqa_eval(n, model)
#score = strategy_qa_eval(n, model)

print(f"Score: {sum(score)}/{n} ({sum(score)/n*100}%)")