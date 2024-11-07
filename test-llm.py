from openai import OpenAI
import datetime as time

def read_qa_file(file_path):
    questions = []
    correct_answers = []
    incorrect_answers = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("Question:"):
                question = line.split("Question:")[1].strip()
                questions.append(question)
            elif line.startswith("Correct answers:"):
                correct = line.split("Correct answers:")[1].strip()
                correct_answers.append(parse_answers(correct))
            elif line.startswith("Incorrect answers:"):
                incorrect = line.split("Incorrect answers:")[1].strip()
                incorrect_answers.append(parse_answers(incorrect))

    return questions, correct_answers, incorrect_answers

def parse_answers(answers_str):
    return [answer.strip() for answer in answers_str.split(';')]

file_path = "QA.txt"
questions, correct_answers, incorrect_answers = read_qa_file(file_path)


llm_1 = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
llm_2 = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

model_1 = "llama-3.2-1b-instruct"
model_2 = "mistral-7b-instruct-v0.3"

outputs = []

for i in range(len(questions)):
    date = time.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    outputs.append(date)
    
    question = questions[i]
    correct_answer = correct_answers[i]
    outputs.append(f"Question: {question}")
    outputs.append(f"Answers: {correct_answer}")

    question_prompt = f"Answer the following question in a single brief but complete sentence.\nQuestion: {question}\nAnswer:"
    llm_2_answer = llm_2.chat.completions.create(messages=[{"role": "user", "content": question_prompt}], model=model_2).choices[0].message
    outputs.append(f"Answer LLM_2: {llm_2_answer.content}")

    feedback_prompt = f"""
    We are assessing the quality of answers to the following question: {question}
    A list of expected answers is: {correct_answer}
    The proposed answer is: {llm_2_answer.content}
    Within the context of the question, does the proposed answer mean the same as the expected answer?.
    """

    outputs.append(f"Feedback prompt: {feedback_prompt}")

    llm_1_feedback = llm_1.chat.completions.create(messages=[{"role": "user", "content": feedback_prompt}], model=model_1).choices[0].message
    outputs.append(f"Feedback response: {llm_1_feedback.content}")

    llm_2_answer_feedback = llm_2.chat.completions.create(messages=[{"role": "user", "content": llm_1_feedback.content}], model=model_2).choices[0].message
    outputs.append(f"Feedback response LLM_2: {llm_2_answer_feedback.content}")

    outputs.append("-------------------------------------------------")

with open("outputs.txt", "w", encoding="utf-8") as file:
    for output in outputs:
        file.write(str(output) + "\n")