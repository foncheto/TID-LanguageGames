from datasets import load_dataset
from openai import OpenAI

ds = load_dataset("yangdong/ecqa")

llm = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model = "llama-3.2-3b-instruct"

n=1000

score = []
llm_answers = []

for i in range (0, n):

    # Get the question, options, and answer
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
    llm_answers.append(llm_answer)

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
    
    print(is_correct)
    if is_correct:
        score.append(1)
    else:
        score.append(0)
    
print(len(score))
print(f"Score: {sum(score)}/{n}")
print(f"LLM Answers: {llm_answers}")

