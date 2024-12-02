import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random
import json

# Load the spaCy language model
nlp = spacy.load('en_core_web_lg') # 400 MB

def compare_responses_1(response1, response2):
    if isinstance(response1, list):
        response1 = list_to_string(response1)
    if isinstance(response2, list):
        response2 = list_to_string(response2)

    doc1 = nlp(response1)
    doc2 = nlp(response2)
    
    similarity_matrix = [[token1.similarity(token2) for token2 in doc2] for token1 in doc1]

    similarities = [max(row) for row in similarity_matrix]
    weighted_similarity = sum(similarities) / len(doc1)

    return weighted_similarity

def compare_responses_2(response1, response2):
    if isinstance(response1, list):
        response1 = list_to_string(response1)
    if isinstance(response2, list):
        response2 = list_to_string(response2)
        
    doc1 = nlp(response1)
    doc2 = nlp(response2)
    
    return doc1.similarity(doc2)

def list_to_string(lst):
    for i in range(len(lst)):
        if i == 0:
            string = lst[i]
        else:
            string += ';' + lst[i]
    return string

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

# Heuristic function to evaluate semantic similarity
def heuristic_similarity(answer, correct_answers):
    vectorizer = TfidfVectorizer().fit_transform([answer] + correct_answers)
    vectors = vectorizer.toarray()
    similarity_scores = cosine_similarity([vectors[0]], vectors[1:])[0]
    max_similarity = max(similarity_scores)
    return max_similarity

def clean_response(response):
    try:
        json_content = re.search(r'\{.*\}', response, re.DOTALL).group(0)
        answer_dict = json.loads(json_content)
        answer = answer_dict["correct_option"][0]
        explanation = answer_dict["explanation"]
    except:
        answer = random.choice(["A", "B", "C", "D", "E"])
        explanation = "Random answer"
    
    return answer, explanation

def get_llm_response(llm, prompt, model):
    llm_answer = llm.chat.completions.create(
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ], 
            model=model).choices[0].message.content
    return llm_answer

def feedback_prompt_1(answer_1, explanation_1, answer_2, explanation_2):
    prompt = (
            f"Don't you think it's {answer_1}, because {explanation_1}, or {answer_2}, since {explanation_2}?"
            f' Provide your final answer in this format: {{"correct_option": "X", "explanation": "X"}}:'
        )
    return prompt

def feedback_prompt_2(answer_1, explanation_1, explanation_2):
    prompt = (
            f"Don't you think it's {answer_1}, because {explanation_1} and {explanation_2}?"
            f' Provide your final answer in this format: {{"correct_option": "X", "explanation": "X"}}:'
        )
    return prompt

def get_data_ecqa_and_prompt(row):
    prompt = (
            f"I will provide a question and five possible answers. Your task is to select the most correct answer based on the information provided.\n"
            f"Question: {row['q_text']}\n"
            f"Options:\n"
            f"A) {row['q_op1']}\n"
            f"B) {row['q_op2']}\n"
            f"C) {row['q_op3']}\n"
            f"D) {row['q_op4']}\n"
            f"E) {row['q_op5']}\n"
            f'Answer only with the correct letter and explanation in this format {{"correct_option": "X", "explanation": "X"}}:'
        )
    return prompt, row['q_ans']