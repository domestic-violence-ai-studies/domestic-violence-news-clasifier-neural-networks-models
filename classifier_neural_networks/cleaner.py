#tar -cJf filename.tar.xz /path/to/folder_or_file ...
#https://blog.mimacom.com/text-classification/

from keras.datasets import imdb
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
import pandas as pd
import re
import numpy as np
import pickle



class Cleaner:
    def __init__(self):
        self.REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        self.BAD_SYMBOLS_RE = re.compile('[^\w\s]')
        self.STOPWORDS = set(stopwords.words('spanish'))
        self.tokenizer = None
    


    def clean_text(self, text):
        text = text.lower() # lowercase text
        text = self.REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
        text = self.BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    #    text = re.sub(r'\W+', '', text)
        text = ' '.join(word for word in text.split() if word not in self.STOPWORDS) # remove stopwors from text
        return text

    def clean_text_in_tags(self, tags):
        clean_tags = []
        for tag in tags:
            clean_tags = clean_tags + [self.clean_text(tag)]
        return clean_tags
        

    def clean_news(self, df):
        print("cleaning the text data")
        df = df.reset_index(drop=True)
        df.dropna(subset=['tags'], inplace=True)
        df['tags'] = df['tags'].apply(self.clean_text_in_tags)
        df['content'] = df['content'].apply(self.clean_text)
        df['content'] = df['content'].str.replace('\d+', '')
        return df
        
    def load_tokenizer(self, sentences):
        print("loading toikenizer")
        self.tokenizer = Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(sentences)

        # saving tokenizer
        with open('../data/neural_network_config/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def create_tokenizer_and_clean(self):
        filename1 = "../data/json_bundle_news_domestic_violence/no_rep_news-domestic-no-violence.json"
        filename2 = "../data/json_bundle_news_domestic_violence/no_rep_news-domestic-violence.json"
        
        
        df1 = pd.read_json(filename1)
        df2 = pd.read_json(filename2)

        

        self.df = df1.append(df2)
        print(self.df)
        print(self.df[["content"]])

        self.df = self.clean_news(self.df)
        self.df.to_json("../data/json_bundle_news_domestic_violence/clean_data.json",orient='records', force_ascii=False)
        
        sentences = self.df['content'].values
        self.load_tokenizer(sentences)


if __name__== "__main__":
    cleaner = Cleaner()
    cleaner.create_tokenizer_and_clean()