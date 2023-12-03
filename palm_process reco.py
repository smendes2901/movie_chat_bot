import google.generativeai as palm
from random import choice

palm.configure(api_key="AIzaSyBivjCS01RgodlxI99xKL1ZpCTIjjI721o")


def generate_recommendations_palm2(prompt):
    try:
        completion = palm.generate_text(
            model="models/text-bison-001",
            prompt=prompt,
            temperature=0.8,
            # The maximum length of the response
            max_output_tokens=1000,
        )
        return completion.result
    except Exception:
        return ""


def generate_palm2_prompt(title, startYear):
    movie_count = choice(range(2, 11))
    return f"""
    Given that I like the movie `{title} ({startYear})` recommend me {movie_count} movies.
    The output should be in a list. For eg. [movie1, movie2,...]
    Each movie should have the release year attached to it. For eg. {title} ({startYear})
    Each movie should be in order of their relevance
    """.strip()


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    import os

    tqdm.pandas()

    skip_samples = 0
    df_complete = pd.read_csv("datasets/movie_datasets/imdb/reco_subset.csv")
    df_complete = df_complete.sample(frac=1, ignore_index=True)
    title = []
    startYear = []
    recommended_movies = []
    for idx, row in tqdm(df_complete.iterrows(), total=df_complete.shape[0]):
        prompt = generate_palm2_prompt(row["originalTitle"], row["startYear"])
        palm2_op = generate_recommendations_palm2(prompt)

        title.append(row["originalTitle"])
        startYear.append(row["startYear"])
        recommended_movies.append(palm2_op)
        if idx % 500 == 0 and idx > 0:
            pd.DataFrame(
                {
                    "originalTitle": title,
                    "startYear": startYear,
                    "recommended_movies": recommended_movies,
                }
            ).to_csv(
                f"./datasets/movie_datasets/imdb/prompt_recom_subset_{skip_samples}.csv",
                index=False,
            )
            title = []
            startYear = []
            recommended_movies = []
            skip_samples += 1
