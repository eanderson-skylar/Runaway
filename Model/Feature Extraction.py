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

spell = SpellChecker()

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
    def __init__(self, model_name):
        self.model_name = model_name

        self.transcripts = None
        self.tokenizer = None
        self.mapping = None
        self.embedding_layer = None
        self.db = DbOperation()

        # get cat table
        self.cat = pd.read_csv("./Cat Table.csv")

        # set number of categories (always +1 for cat 0)
        self.n_cat = self.cat.loc[self.cat.cat >= 0].shape[0]


    def get_transcripts(self):
        self.transcripts = self.db.query_table(sql_query="select transcript from run_ad_complete")

    def create_tokenizer(self):
        # create tokenizer

        t = Tokenizer()
        if self.transcripts is None:
            self.get_transcripts()
        transcripts = self.transcripts
        t.fit_on_texts(transcripts['transcript'])
        max_features = len(t.word_counts) + 1
        word_index = t.word_index

        self.tokenizer = t

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
        self.embedding_layer = embedding_layer

    def create_character_mapping(self):
        t = self.tokenizer
        model_name = self.model_name

        if t is None:
            print('tokenizer missing, create')
            return

        chars = sorted(list(set(' '.join(self.transcripts['transcript']))))
        mapping = dict((c, i) for i, c in enumerate(chars))
        self.mapping = mapping

        #save
        with open('./Save Models/' + model_name + ' tokenizer.pickle',
                  'wb') as handle:  # save tokenizer
            pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./Save Models/' + model_name + ' mapping.pickle',
                  'wb') as handle:  # save mapping
            pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def import_training_data(self):
        # get training data
        sql_query = "select event_id, ar.[advertisement.id], batch_id, job_id, words, id,\
            target, [group], runaway_id, ar.eff_date \
            from run_annotate_result_max_eff ar\
            inner join run_event_processing ep on ep.[advertisement.id] = ar.[advertisement.id]"
        data = self.db.query_table(sql_query=sql_query)

        # create clean words column
        data['clean_words'] = [word.replace('[illegible]', '') for word in data.words]
        data['clean_words'] = [word.replace('.', ' ') for word in data.clean_words]  # periods become spaces
        data['clean_words'] = [word.replace('?', ' ') for word in data.clean_words]  # ? become spaces
        data['clean_words'] = [word.replace(',', ' ') for word in data.clean_words]  # commas become spaces
        data['clean_words'] = [word.replace(';', ' ') for word in data.clean_words]  # semicolons become spaces
        data['clean_words'] = [word.replace('-', '') for word in data.clean_words]  # hyphens are removed
        data['clean_words'] = [word.replace('\r', ' ') for word in data.clean_words]  # returns become spaces

        # handel multiple words in single value
        while any([word.find('  ') > -1 for word in data.clean_words]):
            data['clean_words'] = [word.replace('  ', '') for word in
                                   data.clean_words]  # replace double space with single

        data_add = pd.DataFrame(columns=['event_id', 'advertisement.id', 'batch_id', 'job_id', 'words', 'id',
                                         'target', 'group', 'runaway_id', 'eff_date', 'clean_words'])
        data['split'] = 0
        for index, row in data.iterrows():
            print(index/data.shape[0])
            if re.search('. .', row.clean_words) != None:
                split_words = row.clean_words.split(' ')
                split_words = [word for word in split_words if word != '']  # remove blank words

                # update data with first instance of split words
                data.clean_words.at[index] = split_words[0]
                data.split.at[index] = 1
                # if data is tagged then add group = 1
                if row.target > 0:
                    data.group.at[index] = 1
                # add extra words to data_add
                temp = data.loc[[index]].copy().reset_index(drop=True)
                for n, split in enumerate(split_words):
                    if n > 0:
                        temp['clean_words'] = split
                        temp['id'] += (1 / (n + 5))
                        if (len(split_words) > (n + 1)) & (temp.target.values[
                                                               0] > 0):  # if there are more words to add and target > 0 then group should be 1
                            temp['group'] = 1
                        else:
                            temp['group'] = 0
                        data_add = data_add.append(temp)

        data = data.append(data_add)
        data = data.sort_values(by=['batch_id', 'job_id', 'id']).reset_index(drop=True)

        # add gender_cat to data
        print('adding gender_cat to data...')
        data['gender_cat'] = 0
        ids = data['advertisement.id'].drop_duplicates().tolist()
        for id in ids:
            runs = data.loc[(data.target == -1) & (data['advertisement.id'] == id)]
            for run_index, run in runs.iterrows():
                if run.words == 'Male':
                    data.loc[(data['advertisement.id'] == id) & (data.target == 3) & (
                                data.runaway_id == run.runaway_id), 'gender_cat'] = 1
                else:
                    data.loc[(data['advertisement.id'] == id) & (data.target == 3) & (
                                data.runaway_id == run.runaway_id), 'gender_cat'] = 2
        self.data = data

    def create_tv_arrays(self, n_phrase=1, n_context=5):
        mapping = self.mapping
        data = self.data

        # seperate training/validation data by event id
        event_ids = data['event_id'].drop_duplicates().tolist()
        random.shuffle(event_ids)
        split_num = round(len(event_ids) * .2)

        ids_train = event_ids[split_num:]
        ids_valid = event_ids[:split_num]

        # generate arrays
        # train
        x_phrase_train = np.empty([0, (n_phrase * 2) + 1])
        x_context_train = np.empty([0, (n_context * 2) + 1])
        x_char_phrase_train = np.empty([0, 30])
        y_train = np.empty([0, n_cat])
        y_group_train = np.empty([0, 1])
        y_gender_cat_train = np.empty([0, 3])
        for event_id in ids_train:
            ad_ids = data['advertisement.id'].loc[data.event_id == event_id].drop_duplicates().tolist()
            for ad_id in ad_ids:
                words_map = data.loc[(data['advertisement.id'] == ad_id) & (data.id > -1)].copy().reset_index(drop=True)

                answer = get_input_context(words_map, t, n_phrase, n_context, mapping)
                x_phrase_train = np.append(x_phrase_train, answer[0], axis=0)
                x_context_train = np.append(x_context_train, answer[1], axis=0)
                x_char_phrase_train = np.append(x_char_phrase_train, answer[2], axis=0)
                y_train = np.append(y_train, answer[3], axis=0)
                y_group_train = np.append(y_group_train, answer[4], axis=0)
                y_gender_cat_train = np.append(y_gender_cat_train, answer[5], axis=0)

            # valid
        x_phrase_valid = np.empty([0, (n_phrase * 2) + 1])
        x_context_valid = np.empty([0, (n_context * 2) + 1])
        x_char_phrase_valid = np.empty([0, 30])
        y_valid = np.empty([0, n_cat])
        y_group_valid = np.empty([0, 1])
        y_gender_cat_valid = np.empty([0, 3])
        for event_id in ids_valid:
            ad_ids = data['advertisement.id'].loc[data.event_id == event_id].drop_duplicates().tolist()
            for ad_id in ad_ids:
                words_map = data.loc[(data['advertisement.id'] == ad_id) & (data.id > -1)].copy().reset_index(drop=True)

                answer = get_input_context(words_map, t, n_phrase, n_context, mapping)
                x_phrase_valid = np.append(x_phrase_valid, answer[0], axis=0)
                x_context_valid = np.append(x_context_valid, answer[1], axis=0)
                x_char_phrase_valid = np.append(x_char_phrase_valid, answer[2], axis=0)
                y_valid = np.append(y_valid, answer[3], axis=0)
                y_group_valid = np.append(y_group_valid, answer[4], axis=0)
                y_gender_cat_valid = np.append(y_gender_cat_valid, answer[5], axis=0)

        # additional prep/shaping for char_phrase
        vocab_size = len(mapping)
        x_char_phrase_train_shaped = [to_categorical(x, num_classes=vocab_size) for x in x_char_phrase_train]
        x_char_phrase_train_shaped = array(x_char_phrase_train_shaped)

        x_char_phrase_valid_shaped = [to_categorical(x, num_classes=vocab_size) for x in x_char_phrase_valid]
        x_char_phrase_valid_shaped = array(x_char_phrase_valid_shaped)

        #save arrays
        self.x_dict = {'x_phrase_train': x_phrase_train, 'x_context_train': x_context_train,
                           'x_char_phrase_train_shaped': x_char_phrase_train_shaped, 'x_phrase_valid': x_phrase_valid,
                           'x_context_valid': x_context_valid, 'x_char_phrase_valid_shaped': x_char_phrase_valid_shaped}
        self.y_dict = {'y_train': y_train, 'y_group_train': y_group_train, 'y_gender_cat_train': y_gender_cat_train,
                       'y_valid': y_valid, 'y_group_valid': y_group_valid, 'y_gender_cat_valid': y_gender_cat_valid}

if __name__ == "__main__":
    model = FEModel(model_name='Ad Runaway_all_feature v7')
    model.create_tokenizer()
    model.create_character_mapping()
    model.import_training_data()
    model.create_tv_arrays()