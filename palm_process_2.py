from palm_process import generate_questions_palm2
from bs4 import BeautifulSoup
import regex as re
from transformers import LlamaTokenizer
import regex as re


def clean_string(string):
    string_list = string.split("—")
    if len(string_list) > 1:
        string = " ".join(string_list[:-1])
    return re.sub("\s+", " ", string)


def generate_palm2_long_prompt2(title, data):
    return f"""
Instruction: Create a set of 2 question and answer pairs based on a provided movie description.
Description: How I failed, became the subject of my subject, made a self-reflexive film - or nearly all about religion, monsters and syncretism in Ghana? I started 2011 with filming about the West African concepts of monsters. But monsters don't seem to exist in Ghana and I was surprised that I went into the trap of universal ideas. But then I met David. And when he brought me to the shrine of Mamishie Rasta, he involved himself into a ritual. The conflict he faced after this was due to his Christian belief. This was the turning point of WE ARE THE OTHERS and I asked myself many questions as: What is African (Traditional) Religion at all? What happens when people combine different religions? And does monstrosity re-enters at some point?—Antje Akkermann
Q&A pairs:
[
 {{
   "question": "What is the movie We Are The Others about?",
   "answer":"The movie is a documentary about the filmmaker's experience in Ghana, where she discovered that there are no monsters as she had initially thought."
 }},
 {{
   "question": "Which location(s) are featured in the movie We Are The Others?",
   "answer":"Ghana"
 }},
  {{
   "question": "Which location(s) are featured in the movie We Are The Others?",
   "answer":"Ghana"
 }},
]
Instruction: Create a set of 2 question and answer pairs based on a provided movie description.
1. The movie name `{title.strip()}` should be mentioned in each question surrounded by double quotes
2. Output should be in a JSON list like [{{"question":"", "answer"}}, {{"question":"", "answer":""}},...].
Description: {data.strip()}
Q&A pairs:
""".strip()


def extract_additional_information(html):
    soup = BeautifulSoup(html, "html.parser")
    synopsis = soup.find("span", id="synopsis")
    if synopsis:
        synopsis_text = synopsis.find("div", {"class": "ipc-html-content-inner-div"})
        if synopsis_text:
            return synopsis_text.text
    summaries = [
        div.text
        for div in soup.find_all("div", {"class": "ipc-html-content-inner-div"})
        if div
    ]
    summaries = sorted(summaries, key=len, reverse=True)
    return summaries[0] if summaries else ""


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    import os
    from glob import glob

    tqdm.pandas()

    tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b")

    start_chunks = 3
    filelist = glob("datasets/movie_datasets/imdb/*_movies_html_subset.feather")[
        start_chunks:
    ]
    counter = start_chunks
    for file in filelist:
        try:
            df = pd.read_feather(file)

            print(f"Processing partition {counter}")

            df["synopsis"] = df["plot_html"].progress_apply(
                extract_additional_information
            )
            df["synopsis"] = df["synopsis"].progress_apply(clean_string)
            df["synopsis_len"] = (
                df["synopsis"]
                .progress_apply(lambda x: tokenizer(x)["input_ids"])
                .apply(len)
            )

            # only for processing long desc
            df = df[(df["synopsis_len"] > 50)].reset_index(drop=True)

            print("Generating prompt")
            df["palm2_prompt"] = df.apply(
                lambda row: generate_palm2_long_prompt2(
                    row["originalTitle"], row["synopsis"]
                ),
                axis=1,
            )

            print("Generating questions")
            df["palm2_op"] = df["palm2_prompt"].progress_apply(generate_questions_palm2)

            df[
                ["tconst", "originalTitle", "synopsis", "palm2_prompt", "palm2_op"]
            ].to_feather(
                os.path.join(
                    "datasets/movie_datasets/imdb",
                    f"movie_html_prompt_{counter}.feather",
                )
            )
        except Exception as e:
            pass
        finally:
            counter += 1
