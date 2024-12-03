import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random
import json

# Load the spaCy language model
nlp = spacy.load("en_core_web_lg")  # Large language model for similarity

### Text Processing Utilities ###
def list_to_string(lst):
    """Convert a list of strings to a single semicolon-separated string."""
    return ";".join(lst)

def parse_answers(answers_str):
    """Parse semicolon-separated answers into a list."""
    return [answer.strip() for answer in answers_str.split(";")]

def clean_response_generic(response, options=["A", "B", "C", "D", "E"], default_explanation="Random answer"):
    """
    Generic response cleaner for structured LLM output.
    """
    try:
        json_content = re.search(r'\{.*\}', response, re.DOTALL).group(0)
        answer_dict = json.loads(json_content)
        answer = answer_dict.get("correct_option", random.choice(options))
        explanation = answer_dict.get("explanation", default_explanation)
    except (json.JSONDecodeError, AttributeError):
        answer = random.choice(options)
        explanation = default_explanation
    return answer, explanation

def clean_response_multiple(response):
    """Clean multiple-choice responses."""
    return clean_response_generic(response)

def clean_response_boolean(response):
    """Clean boolean (Yes/No) responses."""
    return clean_response_generic(response, options=["Yes", "No"])

### Heuristic Similarity Utilities ###
def heuristic_similarity(answer, correct_answers):
    """
    Compute the semantic similarity between an answer and a list of correct answers.
    """
    vectorizer = TfidfVectorizer().fit_transform([answer] + correct_answers)
    vectors = vectorizer.toarray()
    similarity_scores = cosine_similarity([vectors[0]], vectors[1:])[0]
    return max(similarity_scores)

def compare_responses_1(response1, response2):
    """
    Token-by-token similarity between two responses using spaCy.
    """
    doc1, doc2 = nlp(response1), nlp(response2)
    similarity_matrix = [[token1.similarity(token2) for token2 in doc2] for token1 in doc1]
    return sum(max(row) for row in similarity_matrix) / len(doc1)

def compare_responses_2(response1, response2):
    """
    Document-level similarity between two responses using spaCy.
    """
    doc1, doc2 = nlp(response1), nlp(response2)
    return doc1.similarity(doc2)

### Prompt Utilities ###
def feedback_prompt_1(answer_1, explanation_1, answer_2, explanation_2):
    """
    Generate feedback prompt comparing two conflicting answers with explanations.
    """
    return (
        f"Don't you think it's {answer_1}, because {explanation_1}, or {answer_2}, since {explanation_2}? "
        f'Provide your final answer in this format: {{"correct_option": "X", "explanation": "X"}}:'
    )

def feedback_prompt_2(answer, explanation_1, explanation_2):
    """
    Generate feedback prompt asking to reconsider one answer with two explanations.
    """
    return (
        f"Don't you think it's {answer}, because {explanation_1} and {explanation_2}? "
        f'Provide your final answer in this format: {{"correct_option": "X", "explanation": "X"}}:'
    )

### Dataset-Specific Data Extractors ###
def get_data_ecqa_and_prompt(row):
    """
    Extract prompt and correct answer from ECQA dataset row.
    """
    prompt = (
        f"I will provide a question and five possible answers. Your task is to select the most correct answer based on the information provided.\n"
        f"Question: {row['q_text']}\n"
        f"Options:\n"
        f"A) {row['q_op1']}\nB) {row['q_op2']}\nC) {row['q_op3']}\nD) {row['q_op4']}\nE) {row['q_op5']}\n"
        f'Answer only with the correct letter and explanation in this format {{"correct_option": "X", "explanation": "X"}}:'
    )
    return prompt, row["q_ans"]

def get_data_commonsense_qa_and_prompt(row):
    """
    Extract prompt and correct answer from Commonsense QA dataset row.
    """
    prompt = (
        f"I will provide a question and five possible answers. Your task is to select the most correct answer based on the information provided.\n"
        f"Question: {row['question']}\n"
        f"Options:\n"
        f"A) {row['choices']['text'][0]}\nB) {row['choices']['text'][1]}\nC) {row['choices']['text'][2]}\nD) {row['choices']['text'][3]}\nE) {row['choices']['text'][4]}\n"
        f'Answer only with the correct letter and explanation in this format {{"correct_option": "X", "explanation": "X"}}:'
    )
    return prompt, row["answerKey"]

def get_data_strategy_qa_and_prompt(row):
    """
    Extract prompt and correct answer from Strategy QA dataset row.
    """
    prompt = (
        f"I will provide a question, and you must respond with 'Yes' or 'No' and explanation.\n"
        f"Question: {row['question']}\n"
        f'Answer only with "Yes" or "No" and explanation in this format {{"correct_option": "X", "explanation": "X"}}:'
    )
    return prompt, row["answer"]

### LLM Interaction Utility ###
def get_llm_response(llm, prompt, model):
    """
    Get a response from the LLM using the provided model and prompt.
    """
    try:
        llm_answer = llm.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        ).choices[0].message.content
        return llm_answer
    except Exception as e:
        print(f"Error fetching LLM response: {e}")
        return '{"correct_option": "Random", "explanation": "Random answer"}'
