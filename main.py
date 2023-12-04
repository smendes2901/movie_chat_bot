from tqdm import tqdm
from llm_inference import LLM
from retreiver import Retreiver

tqdm.pandas()


class MovieChatBot:
    def __init__(self) -> None:
        print("loading models")
        self.llm = LLM("meta-llama/Llama-2-7b-hf", "model_dump/llama-7b-v1")
        self.retreiver = Retreiver()

    def query(self, query):
        category = self.retreiver.query_classifier(user_query)

        if category == "recommendation query":
            prompt = self.retreiver.fetch_reco_prompt(query)
        else:
            prompt = self.retreiver.fetch_info_prompt(query)
        output = self.llm.generate(prompt)
        print(output)


if __name__ == "__main__":
    movie_chatbot = MovieChatBot()

    user_query = "Who is Harry Potter in Harry Potter and the Goblet of Fire 2005?"

    movie_chatbot.query(user_query)
