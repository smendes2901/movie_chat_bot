from palm_process import generate_questions_palm2


def get_example_1():
    return """
Instruction: Create a maximum of 3 question and answer pairs that capture all the information in the given movie description, include the name of the movie in each question.
Movie: This Much
Description: "Russian mail order bride tried to balance her love life with her rich but morally bankrupt husband and her scruffy yet charming paramour"
Q&A pairs:
[
    {
        "question":"Can you describe the complex love triangle at the center of the movie “This Much”?",
        "answer":"The movie “This Much” revolves around a Russian mail order bride who finds herself in a complex love triangle. She is married to a rich man who, despite his wealth, is morally bankrupt. At the same time, she is drawn to a scruffy yet charming paramour. The main conflict of the movie lies in her struggle to balance her relationships with these two very different men."
    },
    {
        "question":"What is the main character’s challenge in “This Much” and how does it reflect on her personal journey?",
        "answer":"In “This Much”, the main character, a Russian mail order bride, faces the challenge of balancing her love life between her rich but morally bankrupt husband and her scruffy yet charming paramour. This challenge is a reflection of her personal journey as she navigates the complexities of love and relationships, and grapples with the moral implications of her choices."
    },
]
"""


def get_example_2():
    return """
Instruction: Create a maximum of 3 question and answer pairs that capture all the information in the given movie description, include the name of the movie in each question.
Movie: Photo de famille
Description: Gabrielle is a "statue" for tourists, much to the chagrin of her teenage son. Elsa is in angry at the world and desperate to become pregnant. Mao is a chronically depressed video game designer who drowns his melancholy in alcohol and psychoanalysis. They are brother and sisters but do not hang out. Ever. Their parents Pierre and Claudine, separated for a long time, have really done nothing to strengthen the bonds of the family - yet, at their grandfather's funeral, they are going to have to meet, and together answer the question: "What to do with grandma?"
Q&A pairs:
[
    {
        "question":"Can you describe the main characters and their personal struggles in the movie “Photo de famille”?",
        "answer":"In “Photo de famille”, the main characters are siblings Gabrielle, Elsa, and Mao. Gabrielle works as a “statue” for tourists, a job that her teenage son disapproves of. Elsa is angry at the world and is desperately trying to become pregnant. Mao is a chronically depressed video game designer who drowns his melancholy in alcohol and psychoanalysis."
    },
    {
        "question":"What is the family situation in “Photo de famille” and how does it affect the relationship between the siblings? ",
        "answer":"In “Photo de famille”, the siblings’ parents, Pierre and Claudine, have been separated for a long time and have done nothing to strengthen the family bonds. As a result, Gabrielle, Elsa, and Mao do not spend time together."
    },
    {
        "question":"What event brings the family together in “Photo de famille” and what challenge do they face?",
        "answer":"In “Photo de famille”, the death of their grandfather forces the siblings and their parents to meet. Together, they face the challenge of deciding what to do with their grandmother."
    },
]
    """


def get_example_3():
    return """
Instruction: Create a maximum of 3 question and answer pairs that capture all the information in the given movie description, include the name of the movie in each question.
Movie: Movies: Mr. Fukyô vs eiga-tachi
Description: A rental video shop, whose customers are few and afar, is visited by a mysterious man, Mr. Fukyo (meaning Mr. Recession). This man recommends closing the business. To protect the beloved shop, Tatsuya, the manager's son and successor, confronts the villain. He is joined by Mr. Violence, Mr. Human Drama, Mr. Horror and Miss Love Story; spirits of each movie genre. Mr. Fukyo, unable to compete against the spirits, brainwashes Mr. Violence into eradicating the movies. To rescue Mr. Violence from Mr. Fukyo's brainwashing, Tatsuya and the spirits will need to fight off Mr. Fukyo's forces. Can Tatsuya and the 'movie gurus' save their beloved shop? These spirits now appear 120 years after the creation of the first movie! Protect the video store!!! Strap on for 90 minutes of peculiar and spectacular scenes that will take your breath away! These fantastic movie spirits deliver an astounding last scene! Prepare yourselves!
Q&A pairs:
[
    {
        "question":"What is the main conflict in the movie “Mr. Fukyô vs eiga-tachi”?",
        "answer":"The main conflict in the movie “Mr. Fukyô vs eiga-tachi” is between Tatsuya, the manager’s son and successor of a rental video shop, and a mysterious man named Mr. Fukyo who recommends closing the business. Tatsuya is joined by spirits of each movie genre to protect the shop."
    },
    {
        "question":"Who are the characters that join Tatsuya in his fight against Mr. Fukyo in the movie “Mr. Fukyô vs eiga-tachi”?",
        "answer":"In the movie “Mr. Fukyô vs eiga-tachi”, Tatsuya is joined by Mr. Violence, Mr. Human Drama, Mr. Horror, and Miss Love Story, who are spirits of each movie genre."
    {
        "question":"What is the climax of the movie “Mr. Fukyô vs eiga-tachi”?",
        "answer":"The climax of the movie “Mr. Fukyô vs eiga-tachi” is when Tatsuya and the spirits fight off Mr. Fukyo’s forces to rescue Mr. Violence from Mr. Fukyo’s brainwashing and save their beloved shop."
    },
]
"""


def generate_palm2_long_prompt(title, data):
    return f"""
{get_example_1().strip()}
{get_example_2().strip()}
{get_example_3().strip()}
Instruction: Create a maximum of 3 question and answer pairs that capture all the information in the given movie description, include the name of the movie in each question.
Movie: {title.strip()}
Description: {data.strip()}
Q&A pairs:
"""


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    tqdm.pandas()

    filename = "./datasets/movie_datasets/imdb/movie_sample_desc_2.csv"

    df_complete = pd.read_csv(filename, chunksize=500)
    counter = 0
    for df in df_complete:
        print(f"Processing partition {counter}")
        # only for processing long desc
        df = df[~(df["data"].isna())].reset_index(drop=True)

        print("Generating prompt")
        df["palm2_prompt"] = df.apply(
            lambda row: generate_palm2_long_prompt(row["originalTitle"], row["data"]),
            axis=1,
        )

        print("Generating questions")
        df["palm2_op"] = df["palm2_prompt"].progress_apply(generate_questions_palm2)

        df[["tconst", "originalTitle", "data", "palm2_prompt", "palm2_op"]].to_feather(
            f"./datasets/movie_datasets/imdb/palm2_prompt3_reg_desc_output_{counter}.feather"
        )
        counter += 1
