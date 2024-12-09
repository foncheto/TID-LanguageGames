from strategy_qa_game import strategy_qa_game as sqa_game
from ecqa_game import ecqa_game
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

def main():
    # Set up logging
    log_filename = setup_main_logging()
    logging.info("Starting LLM Experiment")

    # Experiment parameters
    n = 100
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

        # Run ECQA Game
        logging.info("Starting ECQA Game")
        ecqa_score = ecqa_game(n, model_1, model_2, model_3, api_key)
        ecqa_result = sum(ecqa_score)/n*100
        logging.info(f"ECQA Score: {sum(ecqa_score)}/{n} ({ecqa_result:.2f}%)")
        print(f"Score ECQA: {sum(ecqa_score)}/{n} ({ecqa_result:.2f}%)")

    except Exception as e:
        logging.error(f"An error occurred during the experiment: {e}", exc_info=True)
    
    finally:
        logging.info(f"Experiment completed. Log file saved to {log_filename}")

if __name__ == "__main__":
    load_dotenv()
    main()