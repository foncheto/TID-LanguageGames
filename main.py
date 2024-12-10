# Import Games
from games.strategy_qa_game import strategy_qa_game as sqa_game
from games.commonsense_qa_game import commonsense_qa_game as cqa_game
from games.race_middle_game import race_middle_game as rmiddle_game
from games.race_high_game import race_high_game as rhigh_game
from games.arc_easy_game import arc_easy_game as arc_easy_game
from games.arc_challenge_game import arc_challenge_game as arc_chall_game

# Import Evaluations
from evaluations.strategy_qa_evaluation import strategy_qa_eval as sqa_eval
from evaluations.commonsense_qa_evaluation import commonsense_qa_eval as cqa_eval
from evaluations.race_middle_evaluation import race_middle_evaluation as rmiddle_eval
from evaluations.race_high_evaluation import race_high_evaluation as rhigh_eval
from evaluations.arc_easy_evaluation import arc_easy_evaluation as arc_easy_eval
from evaluations.arc_challenge_evaluation import arc_challenge_evaluation as arc_chall_eval

# Import Generals
from utils import setup_main_logging
from dotenv import load_dotenv
import os
import logging

def game(log_filename):
    n = 5
    model_1 = "gemma2-9b"
    model_2 = "llama3.1-8b"
    model_3 = "mistral-7b-instruct"
    
    try:
        # Retrieve API key
        api_key = os.getenv("API_KEY")
        if not api_key:
            logging.error("API_KEY not found in environment variables")
            raise ValueError("API_KEY is required")
        
        logging.info(f"Experiment Parameters:")
        logging.info(f"Number of questions: {n}")
        logging.info(f"Models: {model_1}, {model_2}, {model_3}")

        # Run Strategy QA Game
        logging.info("Starting Strategy QA Game")
        sqa_score = sqa_game(n, model_1, model_2, model_3, api_key)
        sqa_result = sum(sqa_score)/n*100
        logging.info(f"Strategy QA Score: {sum(sqa_score)}/{n} ({sqa_result:.2f}%)")
        print(f"Score Strategy QA: {sum(sqa_score)}/{n} ({sqa_result:.2f}%)")

        # Run CQA Game
        logging.info("Starting Commonsense QA Game")
        cqa_score = cqa_game(n, model_1, model_2, model_3, api_key)
        cqa_result = sum(cqa_score)/n*100
        logging.info(f"Commonsense QA Score: {sum(cqa_score)}/{n} ({cqa_result:.2f}%)")
        print(f"Score Commonsense QA: {sum(cqa_score)}/{n} ({cqa_result:.2f}%)")

        # Run Race Middle Game
        logging.info("Starting Race-Middle Game")
        rmiddle_score = rmiddle_game(n, model_1, model_2, model_3, api_key)
        rmiddle_result = sum(rmiddle_score)/n*100
        logging.info(f"Race-Middle Score: {sum(rmiddle_score)}/{n} ({rmiddle_result:.2f}%)")
        print(f"Score Race-Middle: {sum(rmiddle_score)}/{n} ({rmiddle_result:.2f}%)")

        # Run Race High Game
        logging.info("Starting Race-High Game")
        rhigh_score = rhigh_game(n, model_1, model_2, model_3, api_key)
        rhigh_result = sum(rhigh_score)/n*100
        logging.info(f"Race-High Score: {sum(rhigh_score)}/{n} ({rhigh_result:.2f}%)")
        print(f"Score Race-High: {sum(rhigh_score)}/{n} ({rhigh_result:.2f}%)")

        # Run ARC Easy Game
        logging.info("Starting ARC-Easy Game")
        arc_easy_score = arc_easy_game(n, model_1, model_2, model_3, api_key)
        arc_easy_result = sum(arc_easy_score)/n*100
        logging.info(f"ARC-Easy Score: {sum(arc_easy_score)}/{n} ({arc_easy_result:.2f}%)")
        print(f"Score ARC-Easy: {sum(arc_easy_score)}/{n} ({arc_easy_result:.2f}%)")

        # Run ARC Challenge Game
        logging.info("Starting ARC-Challenge Game")
        arc_chall_score = arc_chall_game(n, model_1, model_2, model_3, api_key)
        arc_chall_result = sum(arc_chall_score)/n*100
        logging.info(f"ARC-Challenge Score: {sum(arc_chall_score)}/{n} ({arc_chall_result:.2f}%)")
        print(f"Score ARC-Challenge: {sum(arc_chall_score)}/{n} ({arc_chall_result:.2f}%)")

    except Exception as e:
        logging.error(f"An error occurred during the experiment: {e}", exc_info=True)
    
    finally:
        logging.info(f"Experiment completed. Log file saved to {log_filename}")

def evaluation():
    n = 10
    models = ["gemma2-9b", "llama3.1-8b", "mistral-7b-instruct"]
    api_key = os.getenv("API_KEY")

    for model in models:
        logging.info(f"Starting evaluation for model: {model}")

        # Run Strategy QA Evaluation
        score, error = sqa_eval(n, model, api_key)
        logging.info(f"Strategy QA Evaluation: Score={sum(score)}/{n} ({sum(score)/n*100}%), Errors={error} ({error/n*100}%)")

        # Run Commonsense QA Evaluation
        score, error = cqa_eval(n, model, api_key)
        logging.info(f"Commonsense QA Evaluation: Score={sum(score)}/{n} ({sum(score)/n*100}%), Errors={error} ({error/n*100}%)")

        # Run Race Middle Evaluation
        score, error = rmiddle_eval(n, model, api_key)
        logging.info(f"Race-Middle Evaluation: Score={sum(score)}/{n} ({sum(score)/n*100}%), Errors={error} ({error/n*100}%)")
        
        # Run Race High Evaluation
        score, error = rhigh_eval(n, model, api_key)
        logging.info(f"Race-High Evaluation: Score={sum(score)}/{n} ({sum(score)/n*100}%), Errors={error} ({error/n*100}%)")

        # Run ARC Easy Evaluation
        score, error = arc_easy_eval(n, model, api_key)
        logging.info(f"ARC-Easy Evaluation: Score={sum(score)}/{n} ({sum(score)/n*100}%), Errors={error} ({error/n*100}%)")

        # Run ARC Challenge Evaluation
        score, error = arc_chall_eval(n, model, api_key)
        logging.info(f"ARC-Challenge Evaluation: Score={sum(score)}/{n} ({sum(score)/n*100}%), Errors={error} ({error/n*100}%)")

def main():
    log_filename = setup_main_logging("outputs_games")
    logging.info("Starting LLM Experiment")

    game(log_filename)
    #evaluation()

if __name__ == "__main__":
    load_dotenv()
    main()
