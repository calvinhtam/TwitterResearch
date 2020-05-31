from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import io
import os
import sys
import numpy as np
import re
import time
import datetime

# Decoder for tweets
decoder_dict = {"‚Äù": '"',
                "‚Äú": '"',
                "‚Ä¶": '...',
                "‚Äô": "'",
                "‚Äò": "'",
                "‚Äì": "-",
                "‚Äî": "-"}

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

def bert_tokenize_f(sent, tokenizer):
    """
    Tokenize all of the sentences and map the tokens to thier word IDs.
        (1) Tokenize the sentence.
        (2) Prepend the `[CLS]` token to the start.
        (3) Append the `[SEP]` token to the end.
        (4) Map tokens to their IDs.
    """
    encoded_sent = tokenizer.encode(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                        # This function also supports truncation and conversion
                        # to pytorch tensors, but we need to do padding, so we
                        # can't use these features :( .
                        #max_length = 128,          # Truncate all sentences.
                        #return_tensors = 'pt',     # Return pytorch tensors.
                   )
    return encoded_sent

def pad_sequences_f(token_arr, maxlen, dtype="long",
                    value=0, truncating="post", padding="post"):
    return pad_sequences([token_arr], maxlen=maxlen, dtype=dtype,
                         value=value, truncating=truncating,
                         padding=padding)[0]

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
