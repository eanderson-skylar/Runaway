# v4 ability to run local or cloud, ads weight, rewards, enslaver name, run date
# v4 also improved run_id assignment logic
# v5 simplfied run_num logic, improved create run ids logic to use alias instead of unknwon when available, added age and height validation, moved to_inch to module

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
from word2number import w2n
from text_to_num import text2num
import sys
import traceback
from scipy import stats
from functools import reduce

# custom modules
from Support.weightToNumber import weight2n
from Support.RunDate import RunDate
from Support.Reward import GetReward
from Support.EnslaverName import EnslaverName
from Support.AdType import AdType
from Support.to_inch import to_inch

dirname = os.path.dirname(__file__)

class Ad():
    def __init__(self):
        self.ad_id = None
        self.text = None
        self.words = pd.DataFrame(columns=['id', 'words'])
        self.clean_words = pd.DataFrame(columns=['id', 'clean_words'])
        self.cat = pd.read_csv(os.path.join(dirname, "../Model/Cat Table.csv"))

    # splits and cleans words
    def split_words(self, tran):
        self.text = tran

        # formating
        tran = tran.replace(',', ', ').replace(',  ', ', ')
        tran = tran.replace('.', '. ').replace('.  ', '. ')
        tran = tran.replace('[illegible]', '').replace('illegible', '')
        tran = tran.replace('(illegible)', '')
        tran = tran.replace('/r', ' ').replace('/n', ' ')
        tran = tran.replace('\r', ' ').replace('\n', ' ')
        tran = tran.replace('-', ' ').replace('  ', ' ')
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

        # ad id
        df['id'] = df.index + 1

        # return words and clean_words
        self.words = df[['id', 'words']]
        self.clean_words = df[['id', 'clean_words']]


class WordMap(Ad):
    def __init__(self, n_phrase, n_context):
        super().__init__()
        self.n_phrase = n_phrase
        self.n_context = n_context
        self.output = pd.DataFrame(columns=['id', 'group', 'target', 'gender_cat'])
        self.y_target = None
        self.y_group = np.empty([0, 1])
        self.y_gender = np.empty([0, 3])

    def merge_output(self):
        dfs = [self.words, self.clean_words, self.output]
        words_map = reduce(lambda left, right: pd.merge(left, right, on='id'), dfs)
        return words_map

    def group_cleanup(self):
        # reconcile target and group inconsistencies, pick most likley path
        # -Group = 1 but next word target does not match
        words_map = self.output
        y_group = self.y_group
        y_target = self.y_target

        for index, row in words_map.iterrows():
            if index < max(words_map.index):  # skip last row
                next_row = words_map.loc[index + 1]

                if (row['group'] == 1) & (
                        row.target != next_row.target):  # if group is one but current and next row do not match targets
                    # which is more likley? group is wrong or target is wrong?
                    row_group = y_group[index][0]
                    next_row_cat = y_target[index + 1][int(row.target)]  # next row cat
                    if row_group > (
                            1 - next_row_cat):  # if the prob of group is greater than the prob of next_row not being in same cat then update else remove group
                        words_map.at[index + 1, 'target'] = row.target
                    else:
                        words_map.at[index, 'group'] = 0
        self.output = words_map

    def predict(self, t, mapping, model):
        n_phrase = self.n_phrase
        n_context = self.n_context
        n_cat = self.cat
        n_cat = n_cat.loc[n_cat.cat >= 0].shape[0]

        # create words_map
        dfs = [self.words, self.clean_words]
        words_map = reduce(lambda left, right: pd.merge(left, right, on='id'), dfs)

        x_phrase_ = np.empty([0, (n_phrase * 2) + 1])
        x_char_phrase_ = np.empty([0, 30])
        x_context_ = np.empty([0, (n_context * 2) + 1])
        y_group_ = np.empty([0, 1])
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
            # encoded = [mapping[char] for char in phrase] #line below fixes issue of unmapped characters
            encoded = [mapping[char] for char in phrase if char in mapping]
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

        # return [x_phrase_,x_context_,x_char_phrase_,y_,y_group_]

        # character phrase shaping
        x_char_phrase_shaped = [to_categorical(x, num_classes=vocab_size) for x in x_char_phrase_]
        x_char_phrase_shaped = array(x_char_phrase_shaped)

        result = model.predict([x_phrase_, x_context_, x_char_phrase_shaped])
        result_target = [np.argmax(r) for r in result[0]]

        # raw outputs
        self.y_target = result[0]
        self.y_group = result[1]
        self.y_gender = result[2]

        # formated outputs
        self.output['target'] = result_target
        self.output['group'] = [round(r[0], 0) for r in result[1]]
        self.output['gender_cat'] = [np.argmax(r) for r in result[2]]
        self.output['id'] = self.output.index + 1


class Feature(WordMap):
    def __init__(self, n_phrase, n_context):
        super(Feature, self).__init__(n_phrase, n_context)
        self.run_num = 1  # cannot have run_num less than 1
        self.features = pd.DataFrame(columns=['cat', 'words', 'text', 'map_id'])
        self.run_date = None
        self.reward = None
        self.enslaver = pd.DataFrame({'first_name': [None], 'last_name': [None]})
        self.ad_type = None

    def validate_features(self):
        # only validates name, height and age (used in run_num logic)
        features = self.features

        # remove invalid features
        features['invalid'] = 0
        for index, row in features.iterrows():
            if row['cat'] in [3, 4]:  # name
                if row['words'] in ['', ' ', '  ']:
                    features.loc[index, 'invalid'] = 1
            if row['cat'] == 5:  # age
                try:
                    result = self.valid_age(row.words)
                except:
                    result = 'invalid'
                if result == 'invalid':
                    features.loc[index, 'invalid'] = 1
            if row['cat'] == 6:  # height
                try:
                    result = self.valid_height(row.words)
                except:
                    result = 'invalid'
                if result == 'invalid':
                    features.loc[index, 'invalid'] = 1
        features = features.loc[features.invalid == 0]
        features = features.reset_index(drop=True)
        features.drop(columns=['invalid'], inplace=True)

        self.features = features

    def valid_height(self, text):
        if to_inch(text)[0] is None:
            return 'invalid'
        else:
            return 'valid'

    def valid_age(self, text):
        def w2n_simple(text):
            try:
                text = w2n.word_to_num(text)
                return text
            except:
                return text

        temp_df = pd.DataFrame({'age_high': [None], 'age_low': [None]})
        temp = text
        # remove 'years'
        temp = temp.replace('years', '')
        temp = temp.replace("y'rs", "")

        # split words method
        split_words = [' and ', ' or ', ' to ']
        for split_word in split_words:
            if temp.find(split_word) != -1:
                temp_ = temp.split(split_word)
                answer_list = []
                for t_index, t_ in enumerate(temp_):
                    t_ = w2n_simple(t_)
                    t_ = w2n_simple(str(t_).replace(' ', ''))  # try removing spaces
                    t_ = w2n_simple(re.sub(r'[a-z]+', '', str(t_), re.I).replace(' ', ''))  # try removing alpha
                    answer_list = answer_list + [t_]

                answer_list = [x for x in answer_list if x != '']  # remove blank items from list
                temp_df.at[0, 'age_high'] = max(answer_list)
                temp_df.at[0, 'age_low'] = min(answer_list)
                if abs(min(answer_list) - max(answer_list)) > 10:
                    temp_df.at[0, 'age_low'] = max(
                        answer_list)  # if the low is more than 10 years away from the high then just use the high, to solve '41 or 2 years''

        # non split words method
        if (not str(temp_df.age_high.values[0]).isnumeric()):
            t_ = str(text)

            t_ = w2n_simple(t_)
            t_ = w2n_simple(str(t_).replace(' ', ''))  # try removing spaces
            t_ = w2n_simple(re.sub(r'[a-z]+', '', str(t_), re.I).replace(' ', ''))  # try removing alpha

            if str(t_).isnumeric():
                temp_df.at[0, 'age_high'] = int(t_)
                temp_df.at[0, 'age_low'] = int(t_)

        # final determination
        if (str(temp_df.age_high.values[0]).isnumeric()):
            if (temp_df.age_high.values[0] < 100) & (temp_df.age_low.values[0] > 0):
                return 'valid'
            else:
                return 'invalid'
        else:
            return 'invalid'

    def get_adType(self, t_at, model_at):
        text = self.text

        at = AdType()
        result = at.predict(t_at, model_at, text)

        self.ad_type = result

    def get_enslaver(self):
        features = self.features

        features = features.loc[features['cat'] == 16]

        # set first enslaver that works
        en = EnslaverName()

        def first_fit(features):
            for index, row in features.iterrows():
                enslaver = en.main(row.words)

                if (enslaver.first_name.values[0] is not None) | (enslaver.last_name.values[0] is not None):
                    return enslaver
            return pd.DataFrame({'first_name': [None], 'last_name': [None]})

        enslaver = first_fit(features)
        self.enslaver = enslaver

    def get_reward(self):
        features = self.features.copy()

        features = features.loc[features.cat.isin([1, 2])]
        response_df = pd.DataFrame(columns=['amount', 'unit', 'ps'])
        rew = GetReward()
        for index, row in features.iterrows():
            try:
                temp = rew.parseData(row.text)
                # assign primary v secondary
                if row['cat'] == 1:
                    temp['ps'] = 'p'
                else:
                    temp['ps'] = 's'
                response_df = response_df.append(temp)
            except:
                continue

        response_df = response_df.loc[response_df.amount != '']  # remove rewards without an amount
        if response_df.shape[0] > 0:
            response_df = response_df.reset_index(drop=True)
            response_df.amount = pd.to_numeric(response_df.amount)

            response = pd.DataFrame({'primary_reward': [response_df.loc[response_df.ps == 'p', 'amount'].max()],
                                     'secondary_reward': [response_df.loc[response_df.ps == 's', 'amount'].max()],
                                     'unit': [None]})

            # get unit
            units = response_df.groupby(by='unit', as_index=False).count()
            units = units.sort_values(by='amount', ascending=False).reset_index(drop=True)
            units = units.loc[units.unit != '']
            if units.shape[0] > 0:
                response['unit'] = units.unit.values[0]

        else:
            response = pd.DataFrame({'primary_reward': [None], 'secondary_reward': [None], 'unit': [None]})

        self.reward = response

    def set_run_date(self):
        self.run_date = self.__get_run_date()

    def __get_run_date(self):
        # return first good run_date
        features = self.features.copy()

        rd = RunDate()
        features = features.loc[features.cat == 17]
        for index, row in features.iterrows():
            try:
                # process text
                result_table = rd.parse(row.words)

                # checks
                if (result_table['month'][0] is not None) | (result_table['month_offset'][0] is not None) | (
                        result_table['week_offset'][0] is not None) | (result_table['day_offset'][0] is not None) | (
                        result_table['day_of_week_offset'][0] is not None):
                    return result_table

            except:
                continue

        result_table = rd.parse('')
        return result_table

    def get_features(self):
        word_map = self.merge_output()

        # remove possessive from names
        for index, row in word_map.iterrows():
            if row['target'] in [3, 4, 16]:
                word_map.loc[index, 'clean_words'] = word_map.loc[index, 'clean_words'].replace("'s", "")

        # list features
        words = ''
        text = ''
        features = pd.DataFrame(columns=['cat', 'words', 'text', 'map_id'])
        for word_index, word in word_map.iterrows():
            if word.target not in [0, -1]:
                if words == '': map_id = word.id
                words = ' '.join([words, str(word.clean_words)])
                text = ' '.join([text, str(word.words)])
                if word['group'] == 0:
                    features = features.append(
                        pd.DataFrame({'cat': [word.target], 'words': [words], 'text': [text], 'map_id': [map_id]}))
                    words = ''
                    text = ''

        features = features.reset_index(drop=True)
        self.features = features

        # mark duplicate names (names are assigned run_ids)
        self.features = self.__return_duplicate_names(self.features)

    def create_run_ids(self):
        self.validate_features()
        self.__get_run_num()
        self.__create_run_ids()

    def assign_run_ids(self):
        features = self.features

        assignable_cat_list = [4, 5, 6, 7, 8, 9, 10, 11, 12, 14]
        # get first run_id
        for index, row in features.iterrows():
            if (pd.notnull(row.run_id)) & (row['cat'] == 3):
                run_id_assign = row.run_id
                break
        # iterate through features, make assignments
        for index, row in features.iterrows():
            if (pd.notnull(row.run_id)) & (row['cat'] == 3):
                run_id_assign = row.run_id
            if row['cat'] in assignable_cat_list:  # feature cats assignable to runaways
                features.at[index, 'run_id'] = run_id_assign

        self.features = features

    def __get_run_num(self):
        features = self.features

        # checks the number of features occuring in the categories for name, age and height, uses max for run_num
        features_count = features.loc[features.remove == 0, ['cat', 'words']].groupby(['cat'], as_index=False).count()
        features_count = features_count.loc[features_count['cat'].isin([3, 6, 5]), ['cat', 'words']]

        if features_count.shape[0] > 0:
            self.run_num = features_count.words.max()
        else:
            # if no runaway can be detected then default to 1
            self.run_num = 1

    def __create_run_ids(self):
        features = self.features
        run_num = self.run_num
        y_cat = pd.DataFrame(self.y_target)
        y_cat['id'] = y_cat.index + 1

        # three possibilities (too many names, not enough names, just right)
        name_num = features.loc[(features.cat == 3) & (features.remove != 1)].shape[0]
        if run_num == name_num:
            features = features.reset_index(drop=True)
            self.features = features
        else:  # if number of runaways does not match number of names
            if name_num < run_num:  # not enough runaway names, USE ALIAS IF AVAILABLE or use 'unknown'
                for i in range(0, (run_num - name_num)):
                    if features.loc[features['cat'] == 4].shape[0] > 0:  # if any alias available use it
                        fv_index = features.loc[features['cat'] == 4].first_valid_index()
                        features.loc[fv_index, 'cat'] = 3
                        features = self.__return_duplicate_names(features)  # redo feature ids with new cat assignments
                        # if reassigned alias gets assigned to an existing run_id then use add unknown
                        if features.loc[features.run_id == features.loc[fv_index, 'run_id']].shape[0] > 1:
                            temp_unknown = pd.DataFrame({'cat': [3], 'words': ['unknown'], 'remove': [0],
                                                         'run_id': [features['run_id'].max() + 1]})
                            # if run_id is None then use 1
                            if pd.isnull(temp_unknown.run_id[0]):
                                temp_unknown.run_id = 1

                            features = features.append(temp_unknown)
                    else:  # add unknown name
                        temp_unknown = pd.DataFrame({'cat': [3], 'words': ['unknown'], 'remove': [0],
                                                     'run_id': [features['run_id'].max() + 1]})
                        # if run_id is None then use 1
                        if pd.isnull(temp_unknown.run_id[0]):
                            temp_unknown.run_id = 1

                        features = features.append(temp_unknown)

            else:  # too many runaway names, some must be nicknames, pick the most likely nicknames and reassign their cat
                # also need to get rid of all run_ids greater than run_num
                features = features.merge(y_cat[['id', 4]], how='left', left_on=['map_id'], right_on=['id'])
                for i in range(0, (name_num - run_num)):
                    # change the cat of all runaway names for run_id with the greatest prob of being a nickname
                    temp = features.loc[(features.cat == 3) & (features.remove == 0)]
                    temp_index = temp[4].idxmax()

                    max_run_id = features.loc[temp_index, 'run_id']

                    features.loc[features.run_id == max_run_id, 'cat'] = 4
                    features.loc[features.run_id == max_run_id, 'run_id'] = None

                features.drop(columns=['id', 4], inplace=True)
                features = self.__return_duplicate_names(features)  # redo feature ids with new cat assignments

        features = features.reset_index(drop=True)
        self.features = features

    def __return_duplicate_names(self, features):
        features = features.reset_index(drop=True)
        features['id'] = features.index + 1
        features['remove'] = 0
        features['run_id'] = None

        save_features = features.loc[features['cat'] != 3].reset_index(drop=True).copy()
        features = features.loc[features['cat'] == 3].reset_index(drop=True)
        features = features.loc[~features['words'].isin(['', ' '])]  # remove blank features
        features['first_name'] = None
        features['last_name'] = None

        if features.shape[0] > 1:
            # split into first and last
            for index, row in features.iterrows():
                name_list = str(row.words).split(' ')
                name_list = [n for n in name_list if n != '']  # remove blanks

                features.at[index, 'first_name'] = name_list[0]
                if len(name_list) > 1:
                    features.at[index, 'last_name'] = name_list[len(name_list) - 1]
            # mark duplicates based on rules
            name_id = 1
            for index, row in features.iterrows():
                if features.remove.loc[index] == 0:

                    # assign run_id
                    features.at[index, 'run_id'] = name_id

                    # simple match search
                    matches = features.loc[(features.words == row.words) & (features.id != row.id), 'id'].tolist()
                    if len(matches) == 1:
                        features.at[features['id'].isin(matches), 'remove'] = 1
                        features.at[features['id'].isin(matches), 'run_id'] = name_id
                    elif len(matches) > 1:
                        features.loc[features['id'].isin(matches), 'remove'] = 1
                        features.loc[features['id'].isin(matches), 'run_id'] = name_id

                    # first name search (first name can match only for first names without last name)
                    matches = features.loc[(features.first_name == row.first_name) & (pd.isnull(features.last_name)) & (
                                features.id != row.id), 'id'].tolist()
                    if len(matches) == 1:
                        features.at[features['id'].isin(matches), 'remove'] = 1
                        features.at[features['id'].isin(matches), 'run_id'] = name_id
                    elif len(matches) > 1:
                        features.loc[features['id'].isin(matches), 'remove'] = 1
                        features.loc[features['id'].isin(matches), 'run_id'] = name_id

                    name_id += 1

        else:
            # recombine
            features.drop(columns=['first_name', 'last_name'], inplace=True)
            features['run_id'] = 1

            features = features.append(save_features)
            features = features.sort_values(by=['id'])
            features.drop(columns=['id'], inplace=True)
            features = features.reset_index(drop=True)

            # check that run_ids are sequential and 0 based
            run_id_map = features.loc[pd.notnull(features.run_id), ['run_id']].drop_duplicates().sort_values(
                by='run_id')
            if run_id_map.shape[0] > 0:
                run_id_map = run_id_map.reset_index(drop=True)
                run_id_map['new'] = run_id_map.index + 1

                features.run_id.replace(pd.Series(run_id_map.new.values, index=run_id_map.run_id).to_dict(),
                                        inplace=True)

            return features

        features.drop(columns=['first_name', 'last_name'], inplace=True)

        # recombine
        features = features.append(save_features)
        features = features.sort_values(by=['id'])
        features.drop(columns=['id'], inplace=True)
        features = features.reset_index(drop=True)

        # check that run_ids are sequential and 0 based
        run_id_map = features.loc[pd.notnull(features.run_id), ['run_id']].drop_duplicates().sort_values(by='run_id')
        if run_id_map.shape[0] > 0:
            run_id_map = run_id_map.reset_index(drop=True)
            run_id_map['new'] = run_id_map.index + 1

            features.run_id.replace(pd.Series(run_id_map.new.values, index=run_id_map.run_id).to_dict(), inplace=True)

        return features


class Runaway():
    def __init__(self):
        self.ad_id = None
        self.run_id = None
        self.features = None
        self.output = None
        self.skin_tone_mapping = None
        self.occupation_mapping = None
        # clean features
        self.gender = None
        self.name = None
        self.age = None
        self.height = None
        self.skin_tone = None
        self.occupation = None
        self.weight = None
        self.reward = None
        self.ad_type = None

    def clean_features(self):
        # name (split into first and last)
        name_str = self.features.loc[(self.features.cat == 3) & (self.features.remove == 0), 'words'].values[0]
        name_list = name_str.split(' ')
        name_list = [name for name in name_list if name != '']  # remove blank items in list

        name_df = pd.DataFrame({'first_name': [None], 'last_name': [None]})
        if len(name_list) == 1:
            name_df['first_name'] = name_list[0]
        elif len(name_list) > 1:
            name_df['first_name'] = name_list[0]
            name_df['last_name'] = name_list[-1]

        self.name = name_df

        # gender
        if self.name.first_name.values[0] != 'unknown':
            gender_id = self.features.loc[(self.features.cat == 3) & (self.features.remove == 0), 'map_id'].values[0]
            gender_id = self.output.loc[self.output.id == gender_id, 'gender_cat'].values[0]

            if gender_id == 1:
                self.gender = 'male'
            elif gender_id == 2:
                self.gender = 'female'
            else:
                self.gender = 'unknown'
        else:
            self.gender = 'unknown'

        # age
        try:
            self.age = self.__clean_age()[['age_high', 'age_low']]
        except:
            self.age = pd.DataFrame({'age_high': [None], 'age_low': [None]})

        # height (use first height that is int)
        height_df = None
        temp = self.features.loc[self.features.cat == 6]
        for index, row in temp.iterrows():
            try:
                height_list = self.__to_inch(row.words)
                if height_list[0] is None:
                    height_df = None
                else:
                    height_df = pd.DataFrame({'min_height': [min(height_list)], 'max_height': [max(height_list)]})
                    break  # if height is not None then break
            except:
                continue

        if height_df is None:
            self.height = pd.DataFrame({'min_height': [None], 'max_height': [None]})
        else:
            # remove outlers
            if (height_df.min_height.values[0] < 36) | (height_df.max_height.values[0] > 96):
                height_df = pd.DataFrame({'min_height': [None], 'max_height': [None]})
            self.height = height_df

        # skin tone
        self.skin_tone = self.__get_skin_tone()

        # occupation
        self.occupation = self.__get_occupation()

        # weight
        self.weight = self.__get_weight()

    def __get_weight(self):
        # return first good weight, eliminate obviously incorrect weights
        features = self.features.copy()

        features = features.loc[features.cat == 8]
        for index, row in features.iterrows():
            try:
                # process text, convert to list of ints
                weights = weight2n().main(row.words).split(', ')
                weights = [int(weight) for weight in weights]

                # checks
                if (min(weights) < 50) | (max(weights) > 250):
                    continue

                return pd.DataFrame({'high_weight': [max(weights)], 'low_weight': [min(weights)]})
            except:
                continue

        return pd.DataFrame({'high_weight': [None], 'low_weight': [None]})

    def __get_occupation(self):  # returns first valid occupation for runaway using mapping
        features = self.features.copy()
        occupation_mapping = self.occupation_mapping

        occupation_mapping.text = occupation_mapping.text.replace({' ': ''},
                                                                  regex=True)  # remove spaces from mapping table
        features.words = features.words.copy().replace({' ': ''}, regex=True)  # remove spaces from features

        features = features.loc[features.cat == 14]
        if features.shape[0] > 0:
            features = features.merge(occupation_mapping, how='left', left_on='words', right_on='text')
            occupation = features.loc[pd.notnull(features.occupation), 'occupation']
            if len(occupation) > 0:
                return occupation.values[0]
            else:
                return None
        else:
            return None

    def __get_skin_tone(self):
        features = self.features
        skin_tone_mapping = self.skin_tone_mapping

        features = features.loc[features.cat == 7]
        if features.shape[0] > 0:
            features = features.merge(skin_tone_mapping, how='left', left_on='words', right_on='skin_tone')
            skin_tone = features.loc[pd.notnull(features.cat_y), 'cat_y']
            if len(skin_tone) > 0:
                return skin_tone.values[0]
            else:
                return None
        else:
            return None

    def __clean_age(self):
        age_df = pd.DataFrame(columns=['run_id', 'age_high', 'age_low'])
        features = self.features

        run_ids = features.loc[pd.notnull(features.run_id), 'run_id'].drop_duplicates().tolist()
        for run_id in run_ids:
            temp_feat = features.loc[features.run_id == run_id]
            for index, row in temp_feat.iterrows():
                if row['cat'] == 5:  # age
                    temp_df = pd.DataFrame({'run_id': [row.run_id], 'age_high': [None], 'age_low': [None]})
                    # remove 'years'
                    temp = row['words']
                    temp = temp.replace('years', '')
                    temp = temp.replace("y'rs", "")

                    # split words
                    split_words = [' and ', ' or ', ' to ']
                    for split_word in split_words:
                        if temp.find(split_word) != -1:
                            temp_ = temp.split(split_word)
                            answer_list = []
                            for t_index, t_ in enumerate(temp_):
                                t_ = self.__w2n_simple(t_)
                                t_ = self.__w2n_simple(str(t_).replace(' ', ''))  # try removing spaces
                                t_ = self.__w2n_simple(
                                    re.sub(r'[a-z]+', '', str(t_), re.I).replace(' ', ''))  # try removing alpha
                                answer_list = answer_list + [t_]

                            answer_list = [x for x in answer_list if x != '']  # remove blank items from list
                            temp_df.at[0, 'age_high'] = max(answer_list)
                            temp_df.at[0, 'age_low'] = min(answer_list)
                            if abs(min(answer_list) - max(answer_list)) > 10:
                                temp_df.at[0, 'age_low'] = max(
                                    answer_list)  # if the low is more than 10 years away from the high then just use the high, to solve '41 or 2 years''

                    # non split words
                    if (not str(temp_df.age_high.values[0]).isnumeric()):
                        t_ = str(row.words)

                        t_ = self.__w2n_simple(t_)
                        t_ = self.__w2n_simple(str(t_).replace(' ', ''))  # try removing spaces
                        t_ = self.__w2n_simple(
                            re.sub(r'[a-z]+', '', str(t_), re.I).replace(' ', ''))  # try removing alpha

                        if str(t_).isnumeric():
                            temp_df.at[0, 'age_high'] = int(t_)
                            temp_df.at[0, 'age_low'] = int(t_)
                        else:
                            print('could not find age')

                    # if data is good, add it then next run_id
                    if (str(temp_df.age_high.values[0]).isnumeric()):
                        age_df = age_df.append(temp_df)
                        break

        if age_df.shape[0] == 0:
            return pd.DataFrame({'age_high': [None], 'age_low': [None]})
        else:
            # remove impossible ages
            if (age_df.age_high.values[0] < 100) & (age_df.age_low.values[0] > 0):
                return age_df
            else:
                return pd.DataFrame({'age_high': [None], 'age_low': [None]})

    def __w2n_simple(self, text):
        try:
            text = w2n.word_to_num(text)
            return text
        except:
            return text

    def __to_inch(self, input):
        return to_inch(input)

if __name__ == "__main__":
    ad = Feature(n_phrase=1, n_context=5)
    pass