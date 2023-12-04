from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from imdb import Cinemagoer


class Prompt:
    def __init__(self) -> None:
        self.qa_prompt = """
Below is a question regarding movies and shows paired with an input that provides further context. Write a response that appropriately completes the request.
###Instruction: {question}
###Input: {description}
###Response:
""".strip()

        self.reco_prompt = """
Below is a question regarding movies and shows paired with an input that provides further context. Write a response that appropriately completes the request.
###Instruction: Given the movie in {movie_title}, recommend 3 similar movies for the input
###Input: {input_titles}
###Response:
"""


class Retreiver:
    def __init__(self):
        self.ia = Cinemagoer()
        self.query_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L12-v2"
        )
        self.movie = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.recommendation = SentenceTransformer(
            "reco_output/sentence-transformers-all-MiniLM-L6-v2"
        )
        self.movies = pd.read_csv(
            "./datasets/movie_datasets/imdb/movie_complete_clean.csv"
        )
        self.tconst_movies = self.movies.set_index("tconst")
        self.movie_name_embedding = np.load(
            "./datasets/movie_datasets/imdb/movie_name_embeddings.npy"
        )
        self.movie_reco_embedding = np.load(
            "./datasets/movie_datasets/imdb/movie_reco_embeddings.npy"
        )
        self.category = [
            "recommendation new movies",
            "fetch information regarding a movie",
        ]
        self.catgory_embeddings = self.query_model.encode(self.category)
        self.prompt = Prompt()

    def query_classifier(self, query):
        return 1 if "recommend" in query.lower() or "similar" in query.lower() else 0

        # query_embeddings = self.query_model.encode([query])
        # score = util.cos_sim(query_embeddings, self.catgory_embeddings)
        # return np.argmax(score)

    def get_movie_record(self, query):
        query_embeddings = self.movie.encode(query)
        output = util.semantic_search(
            query_embeddings, self.movie_name_embedding, top_k=1
        )[0][0]
        return self.movies.iloc[output["corpus_id"]]

    def fetch_reco_prompt(self, query):
        movie_row = self.get_movie_record(query)
        movie_title = f"{movie_row['originalTitle']} ({movie_row['startYear']})"
        movie_details = f'The movie {movie_row["originalTitle"]} was released in the year {movie_row["startYear"]} is a {movie_row["genres"]} with a  runtime of {movie_row["runtimeMinutes"]} minutes'
        movie_detail_embeddings = self.recommendation.encode(movie_details)
        top_k_list = [
            score["corpus_id"]
            for score in util.semantic_search(
                movie_detail_embeddings, self.movie_reco_embedding, top_k=20
            )[0]
        ]
        top_k_movies = []
        for corpus_id in top_k_list:
            row = self.movies.iloc[corpus_id]
            movie_desc = f"{row['originalTitle']} ({movie_row['startYear']})"
            top_k_movies.append(movie_desc)
        return self.prompt.reco_prompt.format(
            movie_title=movie_title, input_titles="\n".join(top_k_movies)
        )

    def fetch_info_prompt(self, query):
        movie_record = self.get_movie_record(query)
        tconst = movie_record["tconst"]
        the_matrix = self.ia.get_movie(tconst[2:])
        plot_summary = the_matrix.get("plot outline", "")
        directors = ", ".join([str(director) for director in the_matrix["director"]])
        release_year = movie_record["startYear"]
        rating = movie_record["averageRating"]
        runtime = movie_record["runtimeMinutes"]
        description = f"""
Description: {plot_summary},
Director(s): {directors}
Release Year: {release_year}
Rating: {rating}
Runtime(in minutes): {runtime}
"""
        return self.prompt.qa_prompt.format(question=query, description=description)
