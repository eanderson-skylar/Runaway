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

from DB.DB import DbOperation

def get_input_context(words_map, t, n_phrase, n_context, mapping):
    if 'clean_words' not in words_map:  # if no clean_words then just use words
        words_map['clean_words'] = words_map['words']

    x_phrase_ = np.empty([0, (n_phrase * 2) + 1])
    x_char_phrase_ = np.empty([0, 30])
    x_context_ = np.empty([0, (n_context * 2) + 1])
    y_group_ = np.empty([0, 1])
    y_gender_cat_ = np.empty([0, 3])
    y_ = np.empty([0, n_cat])
    for index, word in words_map.iterrows():
        # phrase
        ind_s = max(0, index - n_phrase)
        ind_e = min(words_map.shape[0], index + n_phrase)

        phrase = words_map.loc[ind_s:ind_e, ['clean_words']].values.tolist()
        phrase = [p[0] for p in phrase]  # unlist
        phrase = ' '.join(phrase)
        # encode phrase
        encoded = t.texts_to_sequences({phrase})
        x_phrase = sequence.pad_sequences(encoded, maxlen=(n_phrase * 2) + 1)
        x_phrase_ = np.append(x_phrase_, x_phrase, axis=0)

        # character phrase
        ind_s = max(0, index - n_phrase)
        ind_e = min(words_map.shape[0], index + n_phrase)

        phrase = words_map.loc[ind_s:ind_e, ['words']].values.tolist()
        phrase = [p[0] for p in phrase]  # unlist
        phrase = ' '.join(phrase)
        # encode character phrase
        encoded = [mapping[char] for char in phrase]
        x_char_phrase = sequence.pad_sequences([encoded], maxlen=30)
        x_char_phrase_ = np.append(x_char_phrase_, x_char_phrase, axis=0)

        # context
        ind_s = max(0, index - n_context)
        ind_e = min(words_map.shape[0], index + n_context)

        context = words_map.loc[ind_s:ind_e, ['words']].values.tolist()
        context = [p[0] for p in context]  # unlist
        context = ' '.join(context)
        # encode context
        encoded = t.texts_to_sequences({context})
        x_context = sequence.pad_sequences(encoded, maxlen=(n_context * 2) + 1)

        x_context_ = np.append(x_context_, x_context, axis=0)

        # y_group
        y_group = np.empty([1, 1])
        y_group[0, 0] = word['group']

        y_group_ = np.append(y_group_, y_group, axis=0)

        # y_gender_cat
        y_gender_cat = to_categorical(word['gender_cat'], num_classes=3)
        y_gender_cat_ = np.append(y_gender_cat_, [y_gender_cat], axis=0)

        # y
        y = to_categorical(word['target'], num_classes=n_cat)

        y_ = np.append(y_, [y], axis=0)

    return [x_phrase_, x_context_, x_char_phrase_, y_, y_group_, y_gender_cat_]


# break transcript into useable words
def trans_break(tran):
    # formating
    # tran = transcripts.loc[10,'transcript']
    tran = tran.replace(',', ', ').replace(',  ', ', ')
    tran = tran.replace('.', '. ').replace('.  ', '. ')
    # break into words
    words = text_to_word_sequence(tran, filters='', lower=False)
    df = pd.DataFrame({'words': words})
    df['clean_words'] = None
    for index, row in df.iterrows():
        if len(text_to_word_sequence(row.words)) > 0:
            df.at[index, 'clean_words'] = text_to_word_sequence(row.words)[0]
        else:
            df.at[index, 'clean_words'] = ''

    # remove illegible
    df = df.loc[df.words != '[illegible]']
    # bad_words = list(spell.unknown(words)) + ['illegible']
    # words = [word for word in words if word not in bad_words]

    return [df.clean_words.tolist(), df.words.tolist()]


def create_empty_words_map(words):
    word_map = pd.DataFrame({'words': words})
    word_map['target'] = 0

    return word_map

class FEModel:
    def __init__(self):
        self.tokenizer = None
        self.db = DbOperation()

    def create_tokenizer(self):
        # create tokenizer

        t = Tokenizer()
        transcripts = self.db.query_table(sql_query="select transcript from run_ad_complete")
        t.fit_on_texts(transcripts['transcript'])
        max_features = len(t.word_counts) + 1
        word_index = t.word_index

        # prepare pretrained embedding layer
        embeddings_index = {}
        f = open(os.path.join('./GloVe/', 'glove.6B.100d.txt'), encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))

        # calculate embedding matrix
        EMBEDDING_DIM = 100  # update embedding dim if Glove weights are changed
        MAX_SEQUENCE_LENGTH = 1000
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

                # Hey MTV Cribs, this is where the magic happens
        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)

if __name__ == "__main__":
    model = FEModel()
    model.create_tokenizer()