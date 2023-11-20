import cohere
from cohere.error import CohereAPIError
import pandas as pd
from tqdm import tqdm
import os

tqdm.pandas()


co = cohere.Client(
    "nmkRdEBefO4hDlTTkNao7xpy6udZhyiIYZ3WgFLd"
)  # This is your prod API key


def generate_questions_cohere(prompt):
    try:
        response = co.generate(
            model="command",
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            k=0,
            stop_sequences=[],
            return_likelihoods="NONE",
        )
        return response.generations[0].text
    except:
        return ""


def generate_cohere_prompt(title, data):
    return f"""
### Instruction
You are provided with details regarding the movie `{title.strip()}`. By only referring to the data in ###Context perform the following:
1. Generate 10 question answer pairs
2. The movie name `{title.strip()}` should be mentioned in each question surrounded by double quotes
3. Output should be in a JSON list like [{{"question":"", "answer"}}, {{"question":"", "answer":""}},...].
###Context
{data}
""".strip()


if __name__ == "__main__":
    input_path = "datasets/movie datasets/imdb"
    filename = "movie_sample_desc.csv"
    skip = 1

    df_complete = pd.read_csv(os.path.join(input_path, filename), chunksize=500)
    counter = 0
    for df in df_complete:
        if counter < skip:
            print(f"Skipping {counter}")
            counter += 1
            continue
        print(f"Processing partition {counter}")

        print("Generating prompt")
        df["cohere_prompt"] = df.apply(
            lambda row: generate_cohere_prompt(row["originalTitle"], row["data"]),
            axis=1,
        )

        print("Generating questions")
        df["cohere_op"] = df["cohere_prompt"].progress_apply(generate_questions_cohere)

        df[
            ["tconst", "originalTitle", "data", "cohere_prompt", "cohere_op"]
        ].to_feather(os.path.join(input_path, f"movie_sample_desc_{counter}.feather"))
        counter += 1
