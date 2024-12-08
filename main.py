from strategy_qa_game import strategy_qa_game as sqa_game
from ecqa_game import ecqa_game
from dotenv import load_dotenv
import os

def main():

    n = 100
    model_1 = "gemma2-9b"
    model_2 = "llama3.1-8b"
    model_3 = "mistral-7b-instruct"
    api_key = os.getenv("API_KEY")

    score = sqa_game(n, model_1, model_2, model_3, api_key)
    print(f"Score Strategy QA: {sum(score)}/{n} ({sum(score)/n*100}%)")

    score = ecqa_game(n, model_1, model_2, model_3, api_key)
    print(f"Score ECQA: {sum(score)}/{n} ({sum(score)/n*100}%)")



if __name__ == "__main__":
    load_dotenv()
    main()