import requests
from bs4 import BeautifulSoup
from requests.exceptions import ReadTimeout
import time
from random import randint

user_agent = {"User-agent": "Mozilla/5.0"}


def return_empty_on_error(func):
    def execute_code(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return ""

    return execute_code


@return_empty_on_error
def crawl_url(url):
    try:
        response = requests.get(url, headers=user_agent, timeout=5)
        return response.text
    except ReadTimeout:
        time.sleep(5)
        crawl_url(url)


@return_empty_on_error
def extract_short_desc(html: str):
    soup = BeautifulSoup(html, features="html.parser")
    return soup.findChild("span", class_="sc-466bb6c-2 chnFO").text


@return_empty_on_error
def extract_long_desc(html: str):
    soup = BeautifulSoup(html, features="html.parser")
    return soup.findChild("div", class_="ipc-html-content-inner-div").text


if __name__ == "__main__":
    from tqdm import tqdm
    import pandas as pd

    skip_chunks = list(range(15))

    movies_subset = pd.read_csv(
        "./datasets/movie datasets/imdb/movies_subset.csv", chunksize=500
    )
    counter = 0
    for subset in movies_subset:
        print(f"Processing chunk {counter}")
        if counter in skip_chunks:
            print(f"Skipping chunk {counter}")
            counter += 1
            continue
        subset["short_desc"] = ""
        subset["long_desc"] = ""
        for i, row in tqdm(subset.iterrows(), total=subset.shape[0]):
            try:
                tconst = row["tconst"]
                short_desc_url = f"https://www.imdb.com/title/{tconst}"
                html = crawl_url(short_desc_url)
                short_desc = extract_short_desc(html)
                subset.loc[i, "short_desc"] = str(short_desc)
                if short_desc.strip().endswith("Read all"):
                    long_desc_url = f"https://www.imdb.com/title/{tconst}/plotsummary/?ref_=tt_ov_pl"
                    html = crawl_url(long_desc_url)
                    long_desc = extract_long_desc(html)
                    subset.loc[i, "long_desc"] = str(long_desc)
            finally:
                time.sleep(randint(2, 5))
        subset.to_csv(
            f"./datasets/movie datasets/imdb/{counter}_movies_subset.csv", index=False
        )

        counter += 1
