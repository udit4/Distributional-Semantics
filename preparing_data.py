
import os
from nltk.corpus import gutenberg
from pandas import read_csv
import re

brown_corpus_directory_path = r'{}/brown'.format(os.getcwd())

class prepare:
    def __init__(self):
        return
    #
    #removing pos-tags from the brown corpus
    #
    def clean_pos_tag_from_brown_corpus_data(self, data):
        data = re.sub(r'\/.+?\s', ' ', data)
        data = re.sub(r'\s+', ' ', data)
        return data
    #
    #making corpus from brown corpus data, a string object from all the files
    #
    def data_from_brown_corpus(self):
        complete_data = ''
        for file_name in os.listdir(brown_corpus_directory_path):
            f = open(r'{}/{}'.format(brown_corpus_directory_path, file_name), 'r')
            data = f.read()
            data = self.clean_pos_tag_from_brown_corpus_data(data)
            f.close()
            complete_data+=' {}'.format(data)
        complete_data = complete_data.strip()
        return complete_data
    #
    #cleaning data from abusive tweets
    #
    def clean_abusive_tweets_data(self, tweets):
        for i in range(0,len(tweets)):
            tweets[i] = re.sub(r'\n|\"|\!', ' ', tweets[i])
            tweets[i] = re.sub(r'RT \@.+?\:', ' ', tweets[i])
            tweets[i] = re.sub(r'\@.+\:', ' ', tweets[i])
            tweets[i] = re.sub(r'')
            tweets[i] = re.sub(r'\s+', ' ', tweets[i])
        return tweets
    #
    #making corpus from abusive tweets data
    #
    def data_from_abusive_tweets(self):
        df = read_csv(r'{}/abusive_data.csv'.format(os.getcwd()))
        tweets = self.clean_abusive_tweets_data(list(df['tweet']))
        print(len(tweets))
        return



obj = prepare()
print(obj.data_from_abusive_tweets())
