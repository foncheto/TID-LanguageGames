import datetime as time
import psutil
from langchain_ollama.llms import OllamaLLM
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
                self.memory.pop(0)  # Limit size by removing the oldest entry

    def contains(self, answer):
        return answer in self.memory

# Function to log resource usage
def log_resource_usage(outputs, prefix=""):
    memory_info = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    outputs.append(f"{prefix}RAM Usage: {memory_info.percent}% ({memory_info.used / (1024**3):.2f} GB used of {memory_info.total / (1024**3):.2f} GB total)")
    outputs.append(f"{prefix}CPU Usage: {cpu_percent}%")

# Load questions and answers
file_path = "QA.txt"
questions, correct_answers, incorrect_answers = read_qa_file(file_path)

# Initialize Ollama models with different parameters
llm_1 = OllamaLLM(model="llama3.2")
llm_2 = OllamaLLM(model="llama3.2:1b")
tabu_memory = TabuMemory()

# Output storage
outputs = []

# Maximum number of iterations per question
max_iterations = 5

# Process each question
for i, question in enumerate(questions):
    date = time.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    outputs.append(date)
    outputs.append(f"Question: {question}")
    outputs.append(f"Expected Answers: {correct_answers[i]}")

    # Log initial resource usage for the question
    log_resource_usage(outputs, prefix="Initial ")

    current_llm = llm_2
    feedback_llm = llm_1
    correct = False
    iteration = 0

    while not correct and iteration < max_iterations:
        # Log resource usage before each iteration
        log_resource_usage(outputs, prefix=f"Iteration {iteration + 1} ")

        # Generate an answer from the current LLM
        question_prompt = f"Answer the following question in a brief, complete sentence.\nQuestion: {question}\nAnswer:"
        response = current_llm.invoke(question_prompt)
        outputs.append(f"Iteration {iteration + 1}, {current_llm.model} Response: {response}")

        # Check if the response is in Tabu memory
        if tabu_memory.contains(response):
            outputs.append("Answer is in Tabu memory, skipping repeated incorrect response.")
            break

        # Check semantic similarity
        similarity_score = compare_responses_1(response, correct_answers[i])
        outputs.append(f"Similarity Score: {similarity_score}")

        # Determine if the answer is correct
        threshold = 0.7
        if similarity_score >= threshold:
            outputs.append("Answer accepted as correct.")
            correct = True
        else:
            outputs.append("Answer potentially incorrect. Adding to Tabu memory.")
            tabu_memory.add(response)

            # Generate feedback using the feedback LLM
            feedback_prompt = f"""
            We are assessing the quality of answers to the following question: {question}
            Expected answers: {correct_answers[i]}
            Proposed answer: {response}
            Does the proposed answer match the expected answer in meaning?
            """
            feedback_response = feedback_llm.invoke(feedback_prompt)
            outputs.append(f"Feedback from {feedback_llm.model}: {feedback_response}")

            # Refine the answer based on feedback
            refinement_prompt = f"Refine the answer based on the following feedback:\nFeedback: {feedback_response}\nRefined Answer:"
            refined_response = current_llm.invoke(refinement_prompt)
            outputs.append(f"Refined Response from {current_llm.model}: {refined_response}")

            # Switch LLMs for the next iteration
            current_llm, feedback_llm = feedback_llm, current_llm
            response = refined_response
            iteration += 1

    # Separator for each question
    outputs.append("-------------------------------------------------")
    log_resource_usage(outputs, prefix="Final ")

# Save outputs to a file
with open("outputs.txt", "w", encoding="utf-8") as file:
    for output in outputs:
        file.write(str(output) + "\n")
