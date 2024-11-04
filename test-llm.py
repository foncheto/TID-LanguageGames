from openai import OpenAI

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

for i in range(len(questions)):
    question = questions[i]
    correct_answer = correct_answers[i]

    question_prompt = f"Answer the following question in a single brief but complete sentence.\nQuestion: {question}\nAnswer:"
    llm_2_answer = llm_2.Completion.create(prompt=question_prompt, model="llama-3.2-3b-instruct").choices[0].message
    print("Answer LLM_2:", llm_2_answer)

    feedback_prompt = f"""
    We are assessing the quality of answers to the following question: {question}

    The expected answers is: {correct_answer}

    The proposed answer is: {llm_2_answer}

    Within the context of the question, does the proposed answer mean the same as the expected answer? Respond only with Is correct, or No is correct, and a short feedback.
    """
    llm_1_feedback = llm_1.Completion.create(prompt=feedback_prompt, model="llama-3.2-1b-instruct").choices[0].message
    print("Feedback LLM_1:", llm_1_feedback)

    llm_2.Completion.create(prompt=llm_1_feedback, model="llama-3.2-3b-instruct")
