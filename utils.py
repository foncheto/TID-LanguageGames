import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random
import json
import logging
import os
import datetime as time

# Load the spaCy language model
nlp = spacy.load('en_core_web_lg') # 400 MB
#nlp2 = spacy.load('en_trf_bertbaseuncased_lg')

from sentence_transformers import SentenceTransformer
import numpy as np

# Load the sentence transformer model
model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

def get_max_transformer_similarity(answer: str, correct_answers: list[str]) -> float:
    sims = [compare_responses_with_transformer(answer, c) for c in correct_answers]
    return max(sims) if sims else 0.0



def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vec1 (numpy.ndarray): First vector.
        vec2 (numpy.ndarray): Second vector.

    Returns:
        float: Cosine similarity score.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def compare_responses_with_transformer(response1, response2):
    """
    Compare two responses using SentenceTransformer embeddings and cosine similarity.
    
    Args:
        response1 (str): First response.
        response2 (str): Second response.

    Returns:
        float: Cosine similarity score.
    """
    embedding1 = model.encode(response1)
    embedding2 = model.encode(response2)
    return cosine_similarity(embedding1, embedding2)

def is_answer_correct_with_transformer(answer, correct_answers, threshold=0.6):
    """
    Check if the given answer is correct based on similarity to the list of correct answers.
    
    Args:
        answer (str): The model's answer.
        correct_answers (list of str): List of correct answers.
        threshold (float): Minimum similarity score to consider the answer correct.

    Returns:
        bool: True if the answer matches any correct answer above the threshold.
    """
    # Compute similarity with each correct answer
    similarities = [compare_responses_with_transformer(answer, correct) for correct in correct_answers]
    max_similarity = max(similarities)

    # Log details for debugging
    logging.info(f"Answer: {answer}")
    logging.info(f"Similarities: {similarities}")
    logging.info(f"Max Similarity: {max_similarity}")

    # Return True if similarity exceeds threshold
    return max_similarity >= threshold

def is_answer_correct(answer, correct_answers, threshold=0.75):
    """
    Check if the given answer is correct based on similarity to the list of correct answers.
    
    Args:
        answer (str): The model's answer.
        correct_answers (list of str): List of correct answers.
        threshold (float): Minimum similarity score to consider the answer correct.
    
    Returns:
        bool: True if the answer matches any correct answer above the threshold.
    """
    # Compute similarity with each correct answer
    similarities = [compare_responses_2(answer, correct) for correct in correct_answers]
    max_similarity = max(similarities)
    
    # Log details for debugging
    logging.info(f"Answer: {answer}")
    logging.info(f"Similarities: {similarities}")
    logging.info(f"Max Similarity: {max_similarity}")

    # Return True if similarity exceeds threshold
    return max_similarity >= threshold


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

def clean_response_multiple(response):
    try:
        json_content = re.search(r'\{.*\}', response, re.DOTALL).group(0)
        answer_dict = json.loads(json_content)
        answer = answer_dict["correct_option"][0]
        explanation = answer_dict["explanation"]
    except:
        answer = random.choice(["A", "B", "C", "D", "E"])
        explanation = "Random answer"
    
    return answer, explanation

def clean_response_boolean(response):
    try:
        json_content = re.search(r'\{.*\}', response, re.DOTALL).group(0)
        answer_dict = json.loads(json_content)
        answer = answer_dict["correct_option"]
        explanation = answer_dict["explanation"]
    except:
        answer = random.choice(["Yes", "No"])
        explanation = "Random answer"
    
    return answer, explanation

def get_llm_response(llm, prompt, model, times):
    llm_answer = llm.chat.completions.create(
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ], 
            model=model).choices[0].message.content
    if times > 0:
        try:
            json_content = re.search(r'\{.*\}', llm_answer, re.DOTALL).group(0)
            answer_dict = json.loads(json_content)
            answer = answer_dict["correct_option"]
            explanation = answer_dict["explanation"]
            return llm_answer
        except:
            return get_llm_response(llm, prompt, model, times - 1)
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

def get_data_commonsense_qa_and_prompt(row):
    prompt = (
            f"I will provide a question and five possible answers. Your task is to select the most correct answer based on the information provided.\n"
            f"Question: {row['question']}\n"
            f"Options:\n"
            f"A) {row['choices']['text'][0]}\n"
            f"B) {row['choices']['text'][1]}\n"
            f"C) {row['choices']['text'][2]}\n"
            f"D) {row['choices']['text'][3]}\n"
            f"E) {row['choices']['text'][4]}\n"
            f'Answer only with the correct letter and explanation in this format {{"correct_option": "X", "explanation": "X"}}:'
        )
    return prompt, row['answerKey']

def get_data_race_and_prompt(row):
    prompt = (
            f"I will provide a context, a question and four possible answers. Your task is to select the most correct answer based on the information provided.\n"
            f"Context: {row['article']}\n"
            f"Question: {row['question']}\n"
            f"Options:\n"
            f"A) {row['options'][0]}\n"
            f"B) {row['options'][1]}\n"
            f"C) {row['options'][2]}\n"
            f"D) {row['options'][3]}\n"
            f'Answer only with the correct letter and explanation in this format {{"correct_option": "X", "explanation": "X"}}:'
        )
    return prompt, row['answer']

def get_data_arc_and_prompt(row):
    if len(row['choices']['text']) == 3:
        prompt = (
            f"I will provide a context, a question and three possible answers. Your task is to select the most correct answer based on the information provided.\n"
            f"Question: {row['question']}\n"
            f"Options:\n"
            f"A) {row['choices']['text'][0]}\n"
            f"B) {row['choices']['text'][1]}\n"
            f"C) {row['choices']['text'][2]}\n"
            f'Answer only with the correct letter and explanation in this format {{"correct_option": "X", "explanation": "X"}}:'
        )
    elif len(row['choices']['text']) == 4:
        prompt = (
            f"I will provide a context, a question and four possible answers. Your task is to select the most correct answer based on the information provided.\n"
            f"Question: {row['question']}\n"
            f"Options:\n"
            f"A) {row['choices']['text'][0]}\n"
            f"B) {row['choices']['text'][1]}\n"
            f"C) {row['choices']['text'][2]}\n"
            f"D) {row['choices']['text'][3]}\n"
            f'Answer only with the correct letter and explanation in this format {{"correct_option": "X", "explanation": "X"}}:'
        )
    elif len(row['choices']['text']) == 5:
        prompt = (
            f"I will provide a context, a question and five possible answers. Your task is to select the most correct answer based on the information provided.\n"
            f"Question: {row['question']}\n"
            f"Options:\n"
            f"A) {row['choices']['text'][0]}\n"
            f"B) {row['choices']['text'][1]}\n"
            f"C) {row['choices']['text'][2]}\n"
            f"D) {row['choices']['text'][3]}\n"
            f"E) {row['choices']['text'][4]}\n"
            f'Answer only with the correct letter and explanation in this format {{"correct_option": "X", "explanation": "X"}}:'
        )
    return prompt, row['answerKey']

def get_data_strategy_qa_and_prompt(row):
    prompt = (
            f"I will provide a question, and you must respond with 'Yes' or 'No' and explanation.\n"
            f"Question: {row['question']}\n"
            f'Answer only with "Yes" or "No" and explanation in this format {{"correct_option": "X", "explanation": "X"}}:'
        )
    return prompt, row['answer']

def get_key_ecqa(row, answer):
    options = {
            "A": row['q_op1'],
            "B": row['q_op2'],
            "C": row['q_op3'],
            "D": row['q_op4'],
            "E": row['q_op5']
        }

    for option, text in options.items():
        if text == answer:
            return option

def fake_conflict_prompt():
    prompt = (
        f"I’m not entirely convinced that your answer is correct. Could there be a possibility that one of the other options is more appropriate? Please reconsider your answer and provide your final choice in this format: {{'correct_option': 'X', 'explanation': 'X'}}:"
    )
    return prompt

def setup_logging(name, output_dir):
    """
    Set up logging configuration.
    
    Args:
        output_dir (str): Directory to store log files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    log_filename = os.path.join(
        output_dir, 
        f"{name}_{time.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"
    )
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    return log_filename

def setup_main_logging(output_dir):
    """
    Set up logging configuration for the main experiment script.
    
    Args:
        output_dir (str): Directory to store log files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    log_filename = os.path.join(
        output_dir, 
        f"experiment_main_{time.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"
    )
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    return log_filename
