import nlpaug.augmenter.char as nac
from random import choice
from bs4 import BeautifulSoup
import regex as re


def augment_string(string):
    augs = choice(
        [
            nac.KeyboardAug(),
            nac.RandomCharAug(action="insert"),
            nac.RandomCharAug(action="substitute"),
            nac.RandomCharAug(action="swap"),
            nac.RandomCharAug(action="delete"),
        ]
    )
    return augs.augment(string, n=1)[0]


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


def clean_synopsis(string):
    string_list = string.split("—")
    if len(string_list) > 1:
        string = " ".join(string_list[:-1])
    return re.sub("\s+", " ", string)
