from nltk import download as nltk_download
from nltk.data import find


def download_stopwords():
    try:
        find("corpora/stopwords.zip")
    except LookupError:
        nltk_download("stopwords")
