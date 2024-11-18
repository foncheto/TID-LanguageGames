import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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