from ml.utils import download_stopwords
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer

MORPH = MorphAnalyzer()

download_stopwords()
STOPWORDS = set(stopwords.words("russian"))
