from tqdm import tqdm
from llm_inference import LLM
from retreiver import Retreiver

try:
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception:
    print("Running on local")

tqdm.pandas()


class MovieChatBot:
    def __init__(self) -> None:
        print("loading models")
        self.llm = LLM("meta-llama/Llama-2-7b-chat-hf", "model_dump/llama-7b-chat-v1")
        self.retreiver = Retreiver()

    def query(self, query):
        category = self.retreiver.query_classifier(user_query)

        if category == 1:
            prompt = self.retreiver.fetch_reco_prompt(query)
        else:
            prompt = self.retreiver.fetch_info_prompt(query)
        output = self.llm.generate(prompt)
        # print(output)


if __name__ == "__main__":
    movie_chatbot = MovieChatBot()
    while True:
        user_query = input("Enter your query: ")

        movie_chatbot.query(user_query.strip())
