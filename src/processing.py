from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import io
import os
import sys
import numpy as np
import pandas as pd
import re
import time
import datetime

# Set the seed value all over the place to make this reproducible.
SEED_VAL = 42

# Decoder for tweets
decoder_dict = {"‚Äù": '"',
                "‚Äú": '"',
                "‚Ä¶": '...',
                "‚Äô": "'",
                "‚Äò": "'",
                "‚Äì": "-",
                "‚Äî": "-"}

# Columns of new training data
new_cols = ['text', 'factual_claim', 'sentiment', 'ideology', 'political',
            'immigration', 'macroeconomics', 'national_security',
            'crime_law_enforcement', 'civil_rights', 'environment',
            'education', 'healthcare', 'no_policy_content',
            'asks_for_donation', 'ask_to_watch_read_share_follow_s',
            'ask_misc', 'governance', 'id']

# Renaming new trainging data to old format
rename_to_old = {'ask_to_watch_read_share_follow_s': 'ask_to_etc',
                 'macroeconomics': 'macroeconomic',
                 'crime_law_enforcement': 'crime',
                 'healthcare': 'health_care',
                 'id': 'index'}

# List of different features
general_list = ['sentiment', 'ideology', 'political', 'factual_claim']
policy_list = ['immigration', 'macroeconomic', 'national_security', 'crime',
               'civil_rights', 'environment', 'education', 'health_care',
               'governance', 'no_policy_content']
ask_list = ['no_ask', 'asks_for_donation', 'ask_to_etc', 'ask_misc']
ordinal_list = ['policies', 'ask_requests']

y_cols = general_list + ordinal_list + policy_list + ask_list

class Logger(object):
    def __init__(self, output_dir):
        self.terminal = sys.stdout
        self.log = open(output_dir + "output_log.txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this handles the flush command by doing nothing.
        pass

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss

    params: elapsed - Elapsed time in seconds
    returns: Time formatted as hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def one_hot_decoder(row, one_hot_cols):
    """
    Decodes a One-Hot Encoded row : - )

    params: row - One-Hot Encoded row
            one_hot_cols - Decoded columns
    returns: The Chosen One
    """
    for i in range(len(one_hot_cols)):
        if row[one_hot_cols[i]] == 1:
            return i


def cleaner(txt, decoder_dict=decoder_dict):
    """
    Custom cleaner of text. Pretty ratchet tbh. XD bad coding and dirty
    as a dgjf,hslkjgblfhgkjv

    If you wanna see if you can improve it, pls do. Also check out:
    https://pypi.org/project/tweet-preprocessor
    When I programmed for this, it was decent but not everything I wanted.
    Hopefully it's better for you and you can just use that.

    Basically, I found a few weirdly coded stuff, like ‚Äù in tweets is
    really the character ". So that's why we got a decoder_dict. So we decode
    those with the decoder_dict, dirtily attach ends of contractions to the
    previous word (def not perfect, but seemed to be a little temporary
    band-aid), subbed out a few other stuff, and lowercased it all. I think for
    now, we gonna keep the punctuation cuz BERT seems to like it sometimes.

    params: txt - The unclean text
            decoder_dict - Dictionary that maps some wack symbols to correct
                punctuation
    returns: Cleaned text
    """
    txt = txt.split()
    for i in range(len(txt)):
        word = txt[i]
        # Decodes the wack characters
        while 'Ä' in word:
            char_i = word.index('Ä')
            to_decode = word[char_i - 1: char_i + 2]
            if to_decode in decoder_dict:
                word = word[:char_i - 1] + decoder_dict[to_decode]\
                        + word[char_i + 2:]
            else:
                word = word[:char_i - 1] + " " + word[char_i + 2:]
        # A little contraction floof
        if word in ['t', 's', 'd', 'll', 've', 're', 'm']:
            txt[i] = txt[i - 1] + "'" + word
            txt[i - 1] = ''
            continue
        link_pattern = re.compile('htt.*[^\s]*')
        word = link_pattern.sub(' ', word)
        amp_pattern = re.compile('&amp;')
        word = amp_pattern.sub('and', word)
        alpha_num_pattern = re.compile("[^a-zA-Z0-9_.,:']+")
        word = alpha_num_pattern.sub(' ', word)
        txt[i] = word.strip()
    txt = ' '.join(txt).strip()
    # Case of completely trash tweets
    if txt == '...' or txt == '':
        return np.NaN
    return txt



def pad_sequences_f(token_arr, maxlen, dtype="long",
                    value=0, truncating="post", padding="post"):
    return pad_sequences([token_arr], maxlen=maxlen, dtype=dtype,
                         value=value, truncating=truncating,
                         padding=padding)[0]


def get_data(data_dir, fp_list):
    """
    Collects and stores data into a list of dataframes
    :param data_dir: Data directories
    :param fp_list: List of filepaths of datasets
    :return: Resulting collected dataframes
    """
    res_dfs = []
    for curr_fp in fp_list:
        res_dfs.append(pd.read_csv(data_dir + curr_fp))
    return res_dfs


def split_data(df, data_dir):
    """
    Splits dataframe into X and y for training, validating, and testing
    :param df: Dataframe of tokenized tweets and labels
    :param data_dir: Directory for data
    :return: inputs, masks, and labels of train, validation, and test data
    """

    X_inputs = df['padded_tokenized_text']
    X_att_masks = df['att_masks']
    y = df[y_cols].astype(int)

    # Use 80% for training and 20% for test.
    X_train_inputs, X_test_inputs, y_train_labels, y_test_labels = \
        train_test_split(X_inputs, y, random_state=SEED_VAL, test_size=0.2)
    X_val_inputs, X_test_inputs, y_val_labels, y_test_labels = \
        train_test_split(X_test_inputs, y_test_labels, random_state=SEED_VAL,
                         test_size=0.5)
    # Do the same for the masks.
    X_train_masks, X_test_masks, _, _y_test_masks = \
        train_test_split(X_att_masks, y, random_state=SEED_VAL, test_size=0.2)
    X_val_masks, X_test_masks, _, _ = \
        train_test_split(X_test_masks, _y_test_masks, random_state=SEED_VAL,
                         test_size=0.5)

    # Stores them all into csv files
    X_train_inputs.to_csv(data_dir + 'X_train_inputs.csv', index=False)
    X_val_inputs.to_csv(data_dir + 'X_val_inputs.csv', index=False)
    X_test_inputs.to_csv(data_dir + 'X_test_inputs.csv', index=False)
    y_train_labels.to_csv(data_dir + 'y_train_labels.csv', index=False)
    y_val_labels.to_csv(data_dir + 'y_val_labels.csv', index=False)
    y_test_labels.to_csv(data_dir + 'y_test_labels.csv', index=False)
    X_train_masks.to_csv(data_dir + 'X_train_masks.csv', index=False)
    X_test_masks.to_csv(data_dir + 'X_test_masks.csv', index=False)
    X_val_masks.to_csv(data_dir + 'X_val_masks.csv', index=False)

    return X_train_inputs, X_test_inputs, X_val_inputs,\
        y_train_labels, y_test_labels, y_val_labels,\
        X_train_masks, X_test_masks, X_val_masks


def merge_dfs(prev_df, new_df):
    """
    Done in a very specific way in which the previous training data was a
    particular strange format, and ditto for the next training data

    Here, we do a quick and dirty way of just conforming the new dataset
    to the format of the old one. It's just easier to work with b/c of simpler
    labels, but definitely depends on whoever picks it up next : - )

    ***NOTE: Needs to be changed when handling differently formatted
    csv files. Hopefully they stay the same or are just standardized : - )***

    :param prev_df: Previous dataframe in old data format
    :param new_df: New dataframe in new data format
    :return: merged dataset just stacked on one another
    """
    prev_df = prev_df.drop(['state', 'Unnamed: 21'], axis=1)

    new_df = new_df.rename(columns=rename_to_old)[new_cols]
    new_df['opinion'] = new_df['factual_claim']\
        .apply(lambda x: 1 if not x else 0)

    complete_df = pd.concat([prev_df, new_df], ignore_index=True, sort=False)

    return complete_df


def clean_df(df):
    """
    Cleans dataframe. Does so by the following:
    1) converting negative labels to positive ones (necessary for BERT)
    *** NOTE: KEEP TRACK OF THIS AND CONVERT BACK IN THE PREDICTIONS ***
    :param df: Dataframe of tweets + labels
    :return: List of columns with negative labels and cleaned dataframe
    """
    df = df.drop_duplicates()
    df['index'] = df['index'].fillna(0)

    # Converts all negative labels to next greatest positive label
    # Stores all columns with negative labels in neg_labels
    neg_labels = {}
    for col in df.columns:
        if col in y_cols:
            curr_unique = df[col].unique()
            max_unique = max(curr_unique)
            for unique_label in sorted(curr_unique):
                if unique_label < 0:
                    if col not in neg_labels:
                        neg_labels[col] = int(max_unique)
                    max_unique += 1
                    df[col] = df[col].fillna(0)\
                        .apply(lambda x: max_unique if x == unique_label
                                else x)

    # Makes sure only one policy topic exists
    df = df.apply(
        lambda x: x if sum([x[policy] for policy in policy_list]) == 1
        else np.NaN, axis=1).dropna()
    # Makes sure a tweet is either labelled as opinion or factual claim
    df = df[df['opinion'] != df['factual_claim']]
    # Makes sure a tweet is only asking of one thing (never multiple things)
    df = df.drop(df[df['asks_for_donation'] == 1][df['ask_to_etc'] == 1].index)
    df = df.drop(df[df['ask_misc'] == 1][df['ask_to_etc'] == 1].index)
    df = df.drop(df[df['asks_for_donation'] == 1][df['ask_misc'] == 1].index)

    # Creates new column of no ask requests
    df['no_ask'] = df.apply(lambda x:
                            1 if not x['asks_for_donation']
                            and not x['ask_to_etc']
                            and not x['ask_misc']
                            else 0, axis=1)

    # Creates collective columns of policies and ask requests
    # i.e. ordinal encoding
    df['policies'] = df[policy_list].apply(one_hot_decoder,
                                           args=(policy_list,),
                                           axis=1)
    df['ask_requests'] = df[ask_list].apply(one_hot_decoder,
                                            args=(ask_list,),
                                            axis=1)

    return df, neg_labels


def get_classification_report(pred_df, curr_y_col, curr_y_labels):
    """
    Just prints and returns the Classification Report

    params: pred_df - Dataframe of predications
            curr_y_col - Name of feature
            curr_y_labels - True labels of desired feature
    returns: Clafficiation Report
    """
    pred_series = pred_df.idxmax(axis=1)
    print('========================================')
    print(' Classification Report for', curr_y_col)
    print('========================================')
    print(classification_report(curr_y_labels, pred_series))
    return classification_report(curr_y_labels, pred_series)
