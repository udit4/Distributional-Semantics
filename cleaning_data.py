

from preparing_data import prepare
from collections import Counter
import pandas as pd
import os
import config
import re

path = os.getcwd()

vocab_words = pd.read_csv(r'{}/vocab_data.csv'.format(path))
vocab_words = dict(Counter(vocab_words['Words'].tolist()))


class cleaner:
    def __init__(self):
        self.contractions = config.contractions
        self.abbreviations = config.abbreviations
        return
    def clean_text(self, param_string):
        s = str(param_string)
        s = s.lower()
        s = self.expand_contractions(s)
        s = self.expand_abbreviations(s)
        s = self.clean_punctuations(s)
        return s
    # first, all contractions are expanded to their normal form
    def expand_contractions(self, s):
        #all_contractions_in_text = []
        for k,v in self.contractions.items():
            if(len(re.findall(k, s))!=0):
                s = re.sub(k, ' ' + v + ' ', s)
            else:
                continue
        s = re.sub(r'\s+', ' ', s)
        return s
    # second, expanding all the abbreviations in the text
    def expand_abbreviations(self, s):
        for k,v in self.abbreviations.items():
            if(len(re.findall(k, s))!=0):
                s = re.sub(k, ' ' + v + ' ', s)
            else:
                continue
        s = re.sub(r'\s+', ' ', s)
        return s
    # third, cleaning all the punctuations from the text
    def clean_punctuations(self, s):
        s = re.sub(r'\.|\,', '', s)
        s = re.sub(r'\W+', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        return s



data = prepare().data_from_brown_corpus()
obj = cleaner()
print(obj.clean_text(data))
