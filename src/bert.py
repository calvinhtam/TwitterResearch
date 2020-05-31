import torch
from transformers import BertTokenizer
import os
import numpy as np
import random
from processing import decoder_dict, cleaner, pad_sequences_f

# Set the seed value all over the place to make this reproducible.
SEED_VAL = 42
MAX_LEN = 64


def prep():
    """
    Prepares the GPU settings, randomization, and tokenizer
    :return: GPU device, n_gpu (not really used), and the tokenizer
    """
    # specify GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print('n_gpu', n_gpu)
    torch.cuda.get_device_name(0)

    # Load the BERT tokenizer.
    random.seed(SEED_VAL)
    np.random.seed(SEED_VAL)
    torch.manual_seed(SEED_VAL)
    torch.cuda.manual_seed_all(SEED_VAL)

    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                              do_lower_case=True)

    return device, n_gpu, tokenizer


def bert_tokenize_f(sent, tokenizer):
    """
    Tokenize all of the sentences and map the tokens to their word IDs.
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
                        # max_length = 128,          # Truncate all sentences.
                        # return_tensors = 'pt',     # Return pytorch tensors.
                   )
    return encoded_sent


def convert_text(df, tokenizer, output_dir, max_len=64):
    """
    Converts text into tokenized values
    :param df: Dataframe of tweets + labels
    :return: Dataframe of convert tweets + labels
    """
    print('Pre-cleaning:')
    print(df['text'].tail(10))
    df['cleaned_text'] = df['text'].apply(cleaner, args=(decoder_dict,))
    print('Post-cleaning:')
    print(df['cleaned_text'].tail(10))
    df = df.dropna()

    print('Pre-tokenize:')
    print(df['cleaned_text'].tail(10))
    df['tokenized_text'] = df['cleaned_text'].apply(bert_tokenize_f,
                                                    args=(tokenizer,))
    print('Post-tokenize:')
    print(df['tokenized_text'].tail(10))

    if not os.path.exists(output_dir + 'tokenizer/'):
        os.makedirs(output_dir + 'tokenizer/')
    # Saves tokenizer_output
    tokenizer.save_pretrained(output_dir + 'tokenizer/')

    print('Max sentence length: ', max(df['tokenized_text'].apply(len)))

    # Set the maximum sequence length.
    # I've chosen 64 somewhat arbitrarily. It's slightly larger than the
    # maximum training sentence length of 48...

    print('Padding/truncating all sentences to %d values...' % max_len)

    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token,
                                                   tokenizer.pad_token_id))

    print('Pre-padding:')
    print(df['tokenized_text'].tail(10))

    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the
    # sequence, as opposed to the beginning.
    df['padded_tokenized_text'] = \
        df['tokenized_text'].apply(pad_sequences_f, args=(max_len,))

    print('Post-padding:')
    print(df['padded_tokenized_text'].tail(10))

    # Create attention masks
    df['att_masks'] = df['padded_tokenized_text'] \
        .apply(lambda x: [int(token_id > 0) for token_id in x])
    print('Attention masks')
    print(df['att_masks'].tail(10))

    return df
