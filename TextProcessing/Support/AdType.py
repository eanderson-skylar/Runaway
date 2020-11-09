import keras
from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Embedding, Dropout, Concatenate, Bidirectional
from keras.layers import LSTM
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer
from keras import optimizers, layers, Input
from keras.utils import to_categorical
import numpy as np
from numpy import array
import pandas as pd
import sqlalchemy
import pickle
from datetime import datetime, date, timedelta
from sklearn.preprocessing import MinMaxScaler, Normalizer
from spellchecker import SpellChecker
from os import listdir
import os
import random
from termcolor import colored, cprint 
import re
import pyodbc
import urllib

pd.set_option("display.max_colwidth", 10000, "display.max_columns",10000)
spell = SpellChecker()

model_name = 'AdType v1'
class AdType:
    def __init__(self):
        self.data = None
    
    def predict(self, t, model, text):
        split_words = self.split_words
        n_size = 15
        
        clean_words = split_words(text).clean_words.tolist()[:n_size] #get words
        phrase = ' '.join(clean_words) #combine
            #encode phrase
        encoded = t.texts_to_sequences({phrase})
        x_phrase = sequence.pad_sequences(encoded, maxlen=(n_size))
        
        result = model.predict([x_phrase])
        
        if result[0,0] >= .5:
            return 'ENSLAVER'
        else:
            return 'JAILER'
    
    def split_words(self,tran):
        #formating
        tran = tran.replace(',',', ').replace(',  ',', ')
        tran = tran.replace('.','. ').replace('.  ','. ')
        tran = tran.replace('[illegible]','').replace('illegible','')
        tran = tran.replace('(illegible)','')
        tran = tran.replace('/r',' ').replace('/n',' ')
        tran = tran.replace('\r',' ').replace('\n',' ')
        tran = tran.replace('-',' ').replace('  ',' ')
        #break into words
        words = text_to_word_sequence(tran,filters='',lower=False)
        df = pd.DataFrame({'words':words})
        df['clean_words'] = None
        for index, row in df.iterrows():
            if len(text_to_word_sequence(row.words)) > 0:
                df.at[index,'clean_words'] = text_to_word_sequence(row.words)[0]
            else:
                df.at[index,'clean_words'] = ''
        
        #remove illegible
        df = df.loc[df.words != '[illegible]']
        
        #ad id
        df['id'] = df.index + 1
        
        #return words and clean_words
        return df
