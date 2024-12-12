import os
import logging
import json
from datetime import datetime
from typing import List, Tuple

from games.strategy_qa_game import strategy_qa_game as sqa_game
from games.commonsense_qa_game import commonsense_qa_game as cqa_game
from games.race_middle_game import race_middle_game as rmiddle_game
from games.race_high_game import race_high_game as rhigh_game
from games.arc_easy_game import arc_easy_game as arc_easy_game
from games.arc_challenge_game import arc_challenge_game as arc_chall_game

from evaluations.strategy_qa_evaluation import strategy_qa_eval as sqa_eval
from evaluations.commonsense_qa_evaluation import commonsense_qa_eval as cqa_eval
from evaluations.race_middle_evaluation import race_middle_evaluation as rmiddle_eval
from evaluations.race_high_evaluation import race_high_evaluation as rhigh_eval
from evaluations.arc_easy_evaluation import arc_easy_evaluation as arc_easy_eval
from evaluations.arc_challenge_evaluation import arc_challenge_evaluation as arc_chall_eval

from dotenv import load_dotenv

class ExperimentLogger:
    def __init__(self, output_dir: str = "experiment_logs"):
        """
        Initialize the experiment logger with structured logging.
        
        :param output_dir: Directory to save experiment logs
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(output_dir, f"experiment_{timestamp}.log")
        self.results_filename = os.path.join(output_dir, f"results_{timestamp}.json")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_filename),
                logging.StreamHandler()
            ]
        )
        
        # Initialize results dictionary
        self.results = {
            "timestamp": timestamp,
            "games": {},
            "evaluations": {}
        }
    
    def log_game_results(self, game_name: str, models: List[str], n: int, scores: List[float]) -> None:
        """
        Log game results with detailed information.
        
        :param game_name: Name of the game/task
        :param models: List of models used
        :param n: Number of questions
        :param scores: Scores for each model
        """
        result = sum(scores) / n * 100
        
        logging.info(f"Game: {game_name}")
        logging.info(f"Models: {', '.join(models)}")
        logging.info(f"Number of questions: {n}")
        logging.info(f"Score: {sum(scores)}/{n} ({result:.2f}%)")
        
        # Store results in dictionary
        self.results["games"][game_name] = {
            "models": models,
            "n_questions": n,
            "raw_scores": scores,
            "percentage": result
        }
    
    def log_evaluation_results(self, eval_name: str, model: str, n: int, score: List[float], errors: int) -> None:
        """
        Log evaluation results with detailed metrics.
        
        :param eval_name: Name of the evaluation
        :param model: Model being evaluated
        :param n: Number of questions
        :param score: Scores 
        :param errors: Number of errors
        """
        score_percentage = sum(score) / n * 100
        error_percentage = errors / n * 100
        
        logging.info(f"Evaluation: {eval_name}")
        logging.info(f"Model: {model}")
        logging.info(f"Number of questions: {n}")
        logging.info(f"Score: {sum(score)}/{n} ({score_percentage:.2f}%)")
        logging.info(f"Errors: {errors} ({error_percentage:.2f}%)")
        
        # Store results in dictionary
        self.results["evaluations"][f"{eval_name}_{model}"] = {
            "model": model,
            "n_questions": n,
            "raw_scores": score,
            "score_percentage": score_percentage,
            "errors": errors,
            "error_percentage": error_percentage
        }
    
    def save_results(self) -> None:
        """
        Save experiment results to a JSON file.
        """
        try:
            with open(self.results_filename, 'w') as f:
                json.dump(self.results, f, indent=4)
            logging.info(f"Results saved to {self.results_filename}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")

def run_games(logger: ExperimentLogger, n: int = 5):
    """
    Run game experiments with multiple models.
    
    :param logger: ExperimentLogger instance
    :param n: Number of questions per game
    """
    try:
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("API_KEY is required")
        
        models = ["gemma2-9b", "llama3.1-8b", "mistral-7b-instruct"]
        
        games = [
            ("Strategy QA", sqa_game),
            ("Commonsense QA", cqa_game),
            ("Race-Middle", rmiddle_game),
            ("Race-High", rhigh_game),
            ("ARC-Easy", arc_easy_game),
            ("ARC-Challenge", arc_chall_game)
        ]
        
        for game_name, game_func in games:
            logging.info(f"Running {game_name} Game")
            scores = game_func(n, *models, api_key)
            logger.log_game_results(game_name, models, n, scores)
    
    except Exception as e:
        logging.error(f"Error in game experiments: {e}", exc_info=True)

def run_evaluations(logger: ExperimentLogger, n: int = 10):
    """
    Run model evaluations.
    
    :param logger: ExperimentLogger instance
    :param n: Number of questions per evaluation
    """
    try:
        api_key = os.getenv("API_KEY")
        models = ["gemma2-9b", "llama3.1-8b", "mistral-7b-instruct"]
        
        evaluations = [
            ("Strategy QA", sqa_eval),
            ("Commonsense QA", cqa_eval),
            ("Race-Middle", rmiddle_eval),
            ("Race-High", rhigh_eval),
            ("ARC-Easy", arc_easy_eval),
            ("ARC-Challenge", arc_chall_eval)
        ]
        
        for eval_name, eval_func in evaluations:
            for model in models:
                logging.info(f"Running {eval_name} Evaluation for {model}")
                scores, errors = eval_func(n, model, api_key)
                logger.log_evaluation_results(eval_name, model, n, scores, errors)
    
    except Exception as e:
        logging.error(f"Error in model evaluations: {e}", exc_info=True)

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize experiment logger
    logger = ExperimentLogger()
    
    logging.info("Starting LLM Experiment")
    
    # Run games
    run_games(logger)
    
    # Run evaluations (optional)
    # run_evaluations(logger)
    
    # Save final results
    logger.save_results()
    
    logging.info("Experiment Completed")

if __name__ == "__main__":
    main()