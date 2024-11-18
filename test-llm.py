from openai import OpenAI
import datetime as time
from utils import *

# Tabu memory class to track repeated incorrect answers
class TabuMemory:
    def __init__(self, max_size=5):
        self.memory = []
        self.max_size = max_size

    def add(self, answer):
        if answer not in self.memory:
            self.memory.append(answer)
            if len(self.memory) > self.max_size:
                self.memory.pop(0)  # Limit size by removing oldest entry

    def contains(self, answer):
        return answer in self.memory

file_path = "QA.txt"
questions, correct_answers, incorrect_answers = read_qa_file(file_path)


llm_1 = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
llm_2 = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
tabu_memory = TabuMemory()

model_1 = "llama-3.2-1b-instruct"
model_2 = "llama-3.2-3b-instruct"

outputs = []

# Process each question with iterative language games until correct answer is reached
max_iterations = 5  # Set a maximum iteration count to avoid infinite loops

for i, question in enumerate(questions):
    date = time.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    outputs.append(date)
    
    outputs.append(f"Question: {question}")
    outputs.append(f"Answers: {correct_answers[i]}")

    # Start with LLM2 answering, alternating with each iteration
    current_llm = llm_2
    current_model = model_2

    feedback_llm = llm_1
    feedback_model = model_1

    correct = False
    iteration = 0

    while not correct and iteration < max_iterations:

        question_prompt = f"Answer the following question in a single brief but complete sentence.\nQuestion: {question}\nAnswer:"
        current_llm_answer = current_llm.chat.completions.create(messages=[{"role": "user", "content": question_prompt}], model=current_model).choices[0].message
        outputs.append(f"Iteration {iteration + 1}, {current_model}: {current_llm_answer.content}")

        # Check if the response is in Tabu memory to avoid known incorrect answers
        if tabu_memory.contains(current_llm_answer.content):
            outputs.append("Answer is in Tabu memory, skipping repeated incorrect response.")
            break

        # Apply heuristic similarity
        similarity_score = compare_responses_1(current_llm_answer.content, correct_answers[i])
        outputs.append(f"Similarity Score: {similarity_score}")

        # Check if answer is acceptable
        threshold = 0.5
        if similarity_score >= threshold:
            outputs.append("Answer accepted as correct.")
            correct = True
        else:
            outputs.append("Answer potentially incorrect. Adding to Tabu memory.")
            tabu_memory.add(current_llm_answer.content)

        feedback_prompt = f"""
        We are assessing the quality of answers to the following question: {question}
        A list of expected answers is: {correct_answers[i]}
        The proposed answer is: {current_llm_answer.content}
        Within the context of the question, does the proposed answer mean the same as the expected answer?.
        """


        feedback_response = feedback_llm.chat.completions.create(messages=[{"role": "user", "content": feedback_prompt}], model=feedback_model).choices[0].message
        outputs.append(f"Feedback from {feedback_model}: {feedback_response.content}")

        # Current LLM refines answer based on feedback from feedback LLM
        refinement_prompt = f"Refine the answer based on the following feedback:\nFeedback: {feedback_response.content}\nRefined Answer:"

        refined_response = current_llm.chat.completions.create(messages=[{"role": "user", "content": refinement_prompt}], model=current_model).choices[0].message
        outputs.append(f"Refined Response from {current_model}: {refined_response.content}")

        # Switch roles for the next iteration
        current_llm, feedback_llm = feedback_llm, current_llm
        current_model, feedback_model = feedback_model, current_model
        current_llm_answer = refined_response
        iteration += 1

    outputs.append("-------------------------------------------------")

with open("outputs.txt", "w", encoding="utf-8") as file:
    for output in outputs:
        file.write(str(output) + "\n")