import datetime as time
from langchain_ollama.llms import OllamaLLM

# Define file reading functions
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

# Load questions and answers
file_path = "QA.txt"
questions, correct_answers, incorrect_answers = read_qa_file(file_path)

# Initialize Ollama models
llm_1 = OllamaLLM(model='llama3.1')
llm_2 = OllamaLLM(model='llama3.1')

# Prepare output storage
outputs = []

# Process each question
for i, question in enumerate(questions):
    date = time.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    outputs.append(date)
    outputs.append(f"Question: {question}")
    outputs.append(f"Expected Answers: {correct_answers[i]}")

    # Generate an answer from llm_2
    question_prompt = f"Answer the following question in a brief, complete sentence.\nQuestion: {question}\nAnswer:"
    llm_2_response = llm_2.invoke(question_prompt)
    outputs.append(f"Answer LLM_2: {llm_2_response}")

    # Generate feedback from llm_1
    feedback_prompt = f"""
    We are assessing the quality of answers to the following question: {question}
    Expected answers: {correct_answers[i]}
    Proposed answer: {llm_2_response}
    Does the proposed answer match the expected answer in meaning?
    """
    feedback_response = llm_1.invoke(feedback_prompt)
    outputs.append(f"Feedback response: {feedback_response}")

    # llm_2 provides feedback based on llm_1's response
    llm_2_feedback = llm_2.invoke(feedback_response)
    outputs.append(f"LLM_2 Feedback response: {llm_2_feedback}")

    outputs.append("-------------------------------------------------")

# Write outputs to a file
with open("outputs.txt", "w", encoding="utf-8") as file:
    for output in outputs:
        file.write(str(output) + "\n")
