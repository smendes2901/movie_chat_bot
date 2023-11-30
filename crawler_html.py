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


if __name__ == "__main__":
    from tqdm import tqdm
    import pandas as pd

    skip_chunks = list(range(62))

    movies_subset = pd.read_csv(
        "./datasets/movie_datasets/imdb/movies_subset.csv", chunksize=500
    )
    counter = 0
    for subset in movies_subset:
        print(f"Processing chunk {counter}")
        if counter in skip_chunks:
            print(f"Skipping chunk {counter}")
            counter += 1
            continue
        subset["main_html"] = ""
        subset["plot_html"] = ""
        for i, row in tqdm(subset.iterrows(), total=subset.shape[0]):
            try:
                tconst = row["tconst"]
                short_desc_url = f"https://www.imdb.com/title/{tconst}"
                subset.loc[i, "main_html"] = crawl_url(short_desc_url)
                long_desc_url = (
                    f"https://www.imdb.com/title/{tconst}/plotsummary/?ref_=tt_ov_pl"
                )
                subset.loc[i, "plot_html"] = crawl_url(long_desc_url)
            finally:
                time.sleep(randint(2, 5))
        subset.to_feather(
            f"./datasets/movie_datasets/imdb/{counter}_movies_html_subset.feather",
        )

        counter += 1
