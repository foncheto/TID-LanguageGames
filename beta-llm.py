import datetime as time
from langchain_ollama.llms import OllamaLLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Heuristic function to evaluate semantic similarity
def heuristic_similarity(answer, correct_answers):
    vectorizer = TfidfVectorizer().fit_transform([answer] + correct_answers)
    vectors = vectorizer.toarray()
    similarity_scores = cosine_similarity([vectors[0]], vectors[1:])[0]
    max_similarity = max(similarity_scores)
    return max_similarity

# Load questions and answers
file_path = "QA.txt"
questions, correct_answers, incorrect_answers = read_qa_file(file_path)

# Initialize Ollama models
llm_1 = OllamaLLM(model='llama3.1')
llm_2 = OllamaLLM(model='gemma2')
tabu_memory = TabuMemory()

# Prepare output storage
outputs = []

# Process each question with iterative language games until correct answer is reached
max_iterations = 5  # Set a maximum iteration count to avoid infinite loops

for i, question in enumerate(questions):
    date = time.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    outputs.append(date)
    outputs.append(f"Question: {question}")
    outputs.append(f"Expected Answers: {correct_answers[i]}")

    # Start with LLM2 answering, alternating with each iteration
    current_llm = llm_2
    feedback_llm = llm_1
    correct = False
    iteration = 0

    while not correct and iteration < max_iterations:
        # Generate an answer from the current LLM
        question_prompt = f"Answer the following question in a brief, complete sentence.\nQuestion: {question}\nAnswer:"
        response = current_llm.invoke(question_prompt)
        outputs.append(f"Iteration {iteration + 1}, {current_llm.model} Response: {response}")

        # Check if the response is in Tabu memory to avoid known incorrect answers
        if tabu_memory.contains(response):
            outputs.append("Answer is in Tabu memory, skipping repeated incorrect response.")
            break

        # Apply heuristic similarity
        similarity_score = heuristic_similarity(response, correct_answers[i])
        outputs.append(f"Similarity Score: {similarity_score}")

        # Check if answer is acceptable
        threshold = 0.5
        if similarity_score >= threshold:
            outputs.append("Answer accepted as correct.")
            correct = True
        else:
            outputs.append("Answer potentially incorrect. Adding to Tabu memory.")
            tabu_memory.add(response)

            # Generate feedback from the feedback LLM
            feedback_prompt = f"""
            We are assessing the quality of answers to the following question: {question}
            Expected answers: {correct_answers[i]}
            Proposed answer: {response}
            Does the proposed answer match the expected answer in meaning?
            """
            feedback_response = feedback_llm.invoke(feedback_prompt)
            outputs.append(f"Feedback from {feedback_llm.model}: {feedback_response}")

            # Current LLM refines answer based on feedback from feedback LLM
            refinement_prompt = f"Refine the answer based on the following feedback:\nFeedback: {feedback_response}\nRefined Answer:"
            refined_response = current_llm.invoke(refinement_prompt)
            outputs.append(f"Refined Response from {current_llm.model}: {refined_response}")

            # Switch roles for the next iteration
            current_llm, feedback_llm = feedback_llm, current_llm
            response = refined_response
            iteration += 1

    # Final output separator for each question
    outputs.append("-------------------------------------------------")

# Write outputs to a file
with open("outputs.txt", "w", encoding="utf-8") as file:
    for output in outputs:
        file.write(str(output) + "\n")
