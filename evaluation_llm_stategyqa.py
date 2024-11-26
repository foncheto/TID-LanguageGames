from datasets import load_dataset
from openai import OpenAI

ds = load_dataset("wics/strategy-qa")

llm = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model = "llama-3.2-3b-instruct"

n=1000

score = []
llm_answers = []

for i in range (0, n):

    # Get the question and answer
    question = ds["test"][i]["question"]
    answer = ds["test"][i]["answer"]

    # Make the prompt
    prompt = f"I will provide a question, and you must respond with 'Yes' or 'No' only.\nQuestion: {question}\nAnswer:"
    llm_answer = llm.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=model).choices[0].message.content

    # Check if the LLM answer is correct
    correct_answer = "yes" if answer else "no"
    llm_answer = ''.join(filter(str.isalpha, llm_answer.lower()))

    llm_answers.append(llm_answer)

    if llm_answer == correct_answer:
        score.append(1)
    else:
        score.append(0)

print(len(score))
print(f"Score: {sum(score)}/{n}")
print(f"LLM Answers: {llm_answers}")