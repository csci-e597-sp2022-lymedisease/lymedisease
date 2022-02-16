import pandas as pd
from string import punctuation
import re
from collections import Counter
from itertools import chain
from nltk.tokenize import wordpunct_tokenize


def remove_special(string):
    for i in [r'@[^ ]+', r'http:[^ ]+', r'www.[^ ]+']:
        string = re.sub(i, '', string)
    for i in punctuation:
        string = string.replace(i, '')
    return string.lower()


def word_count_freq():
    all_text = []
    for year in [str(a) for a in range(2010, 2022)]:
        df = pd.read_csv(f'data/raw/{year}.csv', lineterminator='\n').fillna('')
        text = df['text']
        text = list(text.map(lambda x: remove_special(x)).map(lambda x: wordpunct_tokenize(x)))
        all_text = all_text + text
    return Counter(chain.from_iterable(all_text))


def save_word_to_txt(counts):
    with open('word_count.txt', 'w') as fp:
        fp.writelines((f'{k}:{v}\n' for k, v in counts.most_common()))


keywords = ['tick', 'ticks', 'bite', 'borreliosis', 'zoonotic', 'infection', 'forest', 'tickborne', 'erythema',
            'migrans', 'carditis', 'neuroborreliosis', 'borrelia', 'bacterium', 'ixodes', 'blackleg', 'blacklegged',
            'burgdorferi', 'borrelial', 'lymphocytoma', 'arthritis', 'deer', 'deertick', 'fever', 'headache',
            'headaches', 'paralysis', 'hearing', 'rash', 'fatigue', 'swollen', 'lymph', 'chill', 'chills', 'flu',
            'sweat', 'inflammatory', 'neck', 'knee', 'knees', 'stiffness', 'heart', 'palpitations', 'numbness',
            'tingling', 'nausea', 'vomiting', 'neurologic', 'vertigo', 'dizziness', 'sleepless', 'fogginess', 'nerve',
            'irritability', 'joint', 'depression', 'memory', 'malaise']


def save_keywords_count(counts):
    with open('keywords_count.txt', 'w') as fp:
        fp.writelines((f'{k}:{counts[k]}\n' for k in keywords))


if __name__ == "__main__":
    counts = word_count_freq()
    save_word_to_txt(counts)
    save_keywords_count(counts)