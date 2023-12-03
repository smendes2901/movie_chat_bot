import google.generativeai as palm

palm.configure(api_key="AIzaSyBivjCS01RgodlxI99xKL1ZpCTIjjI721o")


def generate_questions_palm2(prompt):
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


def generate_palm2_prompt(title, data):
    return f"""
You are a teacher who is creating a quiz about the movie `{title.strip()}`. Generate 10 question answer pairs from the given text in triple quotes 
1. The movie name `{title.strip()}` should be mentioned in each question surrounded by double quotes
2. Output should be in a JSON list like [{{"question":"", "answer"}}, {{"question":"", "answer":""}},...].
```{data}```""".strip()


def generate_palm2_long_prompt(title, data):
    return f"""
You are helping me generate 10 question and answer pairs based on a given movie and description. The questions and answers should be specific to the given movie and description but should be able to convey similarities between different movies that may have the same broad characteristics. Include the name of the movie in each question.
Movie: Lockout
Description: The International Space Station is now a prison - the ultimate black site. No one's getting out. And no one knows it's there. But when the imprisoned terrorists take over the Station and turn it into a missile aimed at Moscow, only a shuttle pilot and a rookie doctor can stop them. Their task is complicated by a rogue CIA agent (Scott Adkins) who has his own plans for the station and the terrorists within.
Q&A pairs: [
  {{
    "question": "What is the name of the movie?",
    "answer":"Lockout"
  }},
  {{
    "question": "Where does the movie take place?",
    "answer":"The International Space Station"
  }},
  {{
    "question": "What is the main conflict in the movie?",
    "answer":"Terrorists take over the International Space Station and turn it into a missile aimed at Moscow"
  }},
  {{
    "question": "Who are the main characters in the movie?",
    "answer":"A shuttle pilot and a rookie doctor"
  }},
  {{
    "question": "What is the tone of the movie?",
    "answer":"Thriller"
  }},
  {{
    "question": "What are the themes of the movie?",
    "answer":"Terrorism, redemption, survival"
  }},
  {{
    "question": "What are the genre of the movie?",
    "answer":"Action, sci-fi"
  }},
  {{
    "question": "What are similar movies to Lockout?",
    "answer":"Gravity, Elysium, The Martian"
  }},
  {{
    "question": "What is the significance of the title Lockout?",
    "answer":"The title refers to the fact that the International Space Station is now a prison"
  }},
  {{
    "question": "What is the ending of the movie?",
    "answer":"The shuttle pilot and the rookie doctor are able to stop the terrorists and save the International Space Station"
  }},
 ]
Movie: {title.strip()}
Description: {data.strip()}
Q&A pairs:
""".strip()


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    import os

    tqdm.pandas()

    input_path = "datasets/movie datasets/imdb"
    filename = "movie_sample_desc_2.csv"

    skip_chunks = 4
    df_complete = pd.read_csv(os.path.join(input_path, filename), chunksize=500)
    counter = 0
    for df in df_complete:
        if counter < skip_chunks:
            counter += 1
            continue
        print(f"Processing partition {counter}")
        # only for processing long desc
        df = df[~(df["long_desc"].isna())].reset_index(drop=True)

        print("Generating prompt")
        df["palm2_prompt"] = df.apply(
            lambda row: generate_palm2_long_prompt(row["originalTitle"], row["data"]),
            axis=1,
        )

        print("Generating questions")
        df["palm2_op"] = df["palm2_prompt"].progress_apply(generate_questions_palm2)

        df[["tconst", "originalTitle", "data", "palm2_prompt", "palm2_op"]].to_feather(
            os.path.join(input_path, f"movie_sample_long_desc_{counter}_2.feather")
        )
        counter += 1
