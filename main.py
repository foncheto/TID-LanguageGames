# Import Games
from games.strategy_qa_game import strategy_qa_game as sqa_game
from games.commonsense_qa_game import commonsense_qa_game as cqa_game
from games.race_middle_game import race_middle_game as rmiddle_game
from games.race_high_game import race_high_game as rhigh_game
# Import Evaluations
from evaluations.strategy_qa_evaluation import strategy_qa_eval as sqa_eval
from dotenv import load_dotenv
import os
import logging
import datetime

def setup_main_logging(output_dir='outputs_games'):
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
        f"experiment_main_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"
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

def game(log_filename):
    n = 10
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
        """logging.info("Starting Strategy QA Game")
        sqa_score = sqa_game(n, model_1, model_2, model_3, api_key)
        sqa_result = sum(sqa_score)/n*100
        logging.info(f"Strategy QA Score: {sum(sqa_score)}/{n} ({sqa_result:.2f}%)")
        print(f"Score Strategy QA: {sum(sqa_score)}/{n} ({sqa_result:.2f}%)")"""

        # Run CQA Game
        logging.info("Starting CQA Game")
        cqa_score = cqa_game(n, model_1, model_2, model_3, api_key)
        cqa_result = sum(cqa_score)/n*100
        logging.info(f"CQA Score: {sum(cqa_score)}/{n} ({cqa_result:.2f}%)")
        print(f"Score CQA: {sum(cqa_score)}/{n} ({cqa_result:.2f}%)")

        # Run Race Middle Game
        """logging.info("Starting Race-Middle Game")
        rmiddle_score = rmiddle_game(n, model_1, model_2, model_3, api_key)
        rmiddle_result = sum(rmiddle_score)/n*100
        logging.info(f"Race-Middle Score: {sum(rmiddle_score)}/{n} ({rmiddle_result:.2f}%)")
        print(f"Score Race-Middle: {sum(rmiddle_score)}/{n} ({rmiddle_result:.2f}%)")"""

        # Run Race Middle Game
        """logging.info("Starting Race-High Game")
        rhigh_score = rhigh_game(n, model_1, model_2, model_3, api_key)
        rhigh_result= sum(rhigh_score)/n*100
        logging.info(f"Race-High Score: {sum(rhigh_score)}/{n} ({rhigh_result:.2f}%)")
        print(f"Score Race-High: {sum(rhigh_score)}/{n} ({rhigh_result:.2f}%)")"""

    except Exception as e:
        logging.error(f"An error occurred during the experiment: {e}", exc_info=True)
    
    finally:
        logging.info(f"Experiment completed. Log file saved to {log_filename}")

def evaluation():
    n = 10
    model = "gemma2-9b"
    api_key = os.getenv("API_KEY")

    score, error = sqa_eval(n, model, api_key)
    print(f"Score Strategy QA: {sum(score)}/{n} ({sum(score)/n*100}%)")
    print(f"Errors: {error} ({error/n*100}%)")

def main():
    log_filename = setup_main_logging()
    logging.info("Starting LLM Experiment")

    game(log_filename)
    #evaluation()

if __name__ == "__main__":
    load_dotenv()
    main()