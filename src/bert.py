import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler,\
    SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from pytorch_pretrained_bert import BertAdam
from transformers import BertTokenizer, get_linear_schedule_with_warmup,\
    BertForSequenceClassification
import pandas as pd
import io
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import re
import time
import datetime
import random
from helper_func import (format_time, cleaner, bert_tokenize_f,
                         pad_sequences_f, one_hot_decoder,
                         get_classification_report, Logger)

# Set the seed value all over the place to make this reproducible.
SEED_VAL = 42
MAX_LEN = 64

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
    df['no_ask'] = df.apply(lambda x: 1 if not x['asks_for_donation']
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


def convert_text(df, tokenizer, output_dir):
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

    print('Padding/truncating all sentences to %d values...' % MAX_LEN)

    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token,
                                                   tokenizer.pad_token_id))

    print('Pre-padding:')
    print(df['tokenized_text'].tail(10))

    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the
    # sequence, as opposed to the beginning.
    df['padded_tokenized_text'] = \
        df['tokenized_text'].apply(pad_sequences_f, args=(MAX_LEN,))

    print('Post-padding:')
    print(df['padded_tokenized_text'].tail(10))

    # Create attention masks
    df['att_masks'] = df['padded_tokenized_text'] \
        .apply(lambda x: [int(token_id > 0) for token_id in x])
    print('Attention masks')
    print(df['att_masks'].tail(10))

    return df


def split_data(df, data_dir):
    """
    Splits dataframe into X and y for training, validating, and testing
    :param df: Dataframe of tokenized tweets and labels
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


def setup_for_training(X_train_inputs, X_test_inputs, X_val_inputs,
                       y_train_labels, y_test_labels, y_val_labels,
                       X_train_masks, X_test_masks, X_val_masks,
                       epochs, batch_size, curr_y_col):
    # Intended for when your code fucks up and you have to rerun your
    # dumbass script over only a subset of columns
    # if curr_y_col in ['sentiment', 'political', 'ideology', 'policies',
    #                   'ask_requests', 'factual_claim']:
    #     print('Skipping', curr_y_col)
    #     continue
    curr_y_train_labels, curr_y_val_labels, curr_y_test_labels = \
        y_train_labels[curr_y_col], y_val_labels[curr_y_col],\
        y_test_labels[curr_y_col]

    # Convert all inputs and labels into torch tensors, the required
    # datatype for our model.
    X_train_inputs_tensor = torch.tensor(X_train_inputs.to_list())
    X_val_inputs_tensor = torch.tensor(X_val_inputs.to_list())

    curr_y_train_labels_tensor = torch.tensor(
        curr_y_train_labels.to_list())
    curr_y_val_labels_tensor = torch.tensor(curr_y_val_labels.to_list())

    X_train_masks_tensor = torch.tensor(X_train_masks.to_list())
    X_val_masks_tensor = torch.tensor(X_val_masks.to_list())

    num_labels = len(set(curr_y_train_labels.unique())\
        .union(set(curr_y_val_labels.unique()))\
        .union(set(curr_y_test_labels.unique())))

    # The DataLoader needs to know our batch size for training, so we
    # specify it here.

    # Create the DataLoader for our training set.
    train_data = TensorDataset(X_train_inputs_tensor, X_train_masks_tensor,
                               curr_y_train_labels_tensor)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=batch_size)

    # Create the DataLoader for our val set.
    val_data = TensorDataset(X_val_inputs_tensor, X_val_masks_tensor,
                             curr_y_val_labels_tensor)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler,
                                batch_size=batch_size)

    # Load BertForSequenceClassification, the pretrained BERT model with a
    # single linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        # Use the 12-layer BERT model, with an uncased vocab.
        pretrained_model_name_or_path="bert-base-cased",
        # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,
        # The number of output labels--3
        output_attentions=False,
        # Whether the model returns attentions weights.
        output_hidden_states=False,
        # Whether the model returns all hidden-states.
    )

    # Tell pytorch to run this model on the GPU.
    model.cuda()

    # Note: AdamW is a class from the huggingface library (as opposed to
    # pytorch) I believe the 'W' stands for 'Weight Decay fix"
    optimizer = BertAdam(model.parameters(),
                         lr=2e-5,
                         # args.learning_rate - default is 5e-5, our notebook
                         # had 2e-5
                         eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                         )

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                # Default value in run_glue.py
                                                num_training_steps=total_steps)

    return train_dataloader, val_dataloader, curr_y_val_labels, model,\
           optimizer, scheduler


def train_bert(model, train_dataloader, device, optimizer, scheduler,
               curr_y_col, epochs, epoch_i, loss_values):
    """
    Trains our model on training dataset
    :param model: Bert model
    :param train_dataloader: dataloader containing training X and y data
    :param device: GPU device
    :param curr_y_col: Current label
    :param epochs: Number of epochs
    :param epoch_i: Current epoch
    :param loss_values: loss values
    :return: updated model and loss values
    """
    print('========================================')
    print('          Training for', curr_y_col)
    print('========================================')

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1,
                                                     epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    # Put the model into training mode. Don't be mislead--the call to
    # 'train' just changes the *mode*, it doesn't *perform* the
    # training. 'dropout' and 'batchnorm' layers behave differently
    # during training vs. val
    # (source: https://stackoverflow.com/questions/51433378)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print(
                '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the
        # GPU using the 'to' method.
        #
        # 'batch' contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)


        # Always clear any previously calculated gradients before
        # performing a backward pass. PyTorch doesn't do this
        # automatically because accumulating the gradients is
        # "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training
        # batch).
        # This will return the loss (rather than the model output)
        # because we have provided the 'labels'.
        # The documentation for this 'model' function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/
        #       bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        # The call to `model` always returns a tuple, so we need to
        # pull the loss value out of the tuple.
        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that
        # we can calculate the average loss at the end. 'loss' is a
        # Tensor containing a single value; the `.item()` function
        # just returns the Python value from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed
        # gradient. The optimizer dictates the "update rule"--how the
        # parameters are modified based on their gradients,
        # the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    return model, loss_values


def val_bert(model, val_dataloader, device, curr_y_val_labels, curr_y_col,
             neg_labels):
    """
    Validates on our model on validation dataset
    :param model: Bert model
    :param val_dataloader: Dataloader containing validation X and y values
    :param device: GPU
    :param curr_y_col: Current label
    :return: Model and prediction dataframe
    """
    # Store predictions after epochs
    predictions = []

    print("")
    print("Running val eval...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave
    # differently during evaluation.
    model.eval()

    # Tracking variables
    eval_loss, eval_f1_score = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in val_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving
        # memory and speeding up val
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because
            # we have not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this 'model' function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/
            # bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the
        # output values prior to applying an activation function like
        # the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the classification report for this batch of val
        # sentences.
        tmp_f1_score = f1_score(np.argmax(logits, axis=1).flatten(),
                                label_ids.flatten(),
                                average='weighted')

        # Accumulate the f1_score
        eval_f1_score += tmp_f1_score

        # Track the number of batches
        nb_eval_steps += 1

        predictions.append(logits)

    # Report the final accuracy for this val run.
    print("  F1-Score: {0:.2f}".format(eval_f1_score / nb_eval_steps))
    pred_df = pd.DataFrame(np.concatenate(predictions))
    if curr_y_col in neg_labels:
        pred_df.rename({i: i - len(pred_df.columns)
                        for i in range(neg_labels[curr_y_col],
                                       int(max(pred_df.columns)))})
    get_classification_report(pred_df, curr_y_col, curr_y_val_labels)
    print("  Validation took: {:}".format(format_time(time.time() -
                                                      t0)))

    return model, pred_df


def trainer(X_train_inputs, X_test_inputs, X_val_inputs,
            y_train_labels, y_test_labels, y_val_labels,
            X_train_masks, X_test_masks, X_val_masks,
            epochs, batch_size, neg_labels, output_dir, device):
    """
    Trains BERT : - )
    It's saved to output_dir

    :param X_train_inputs: X data points for training
    :param X_test_inputs: X data points for testing
    :param X_val_inputs: X data points for validation
    :param y_train_labels: Y data labels for training
    :param y_test_labels: Y data labels for testing
    :param y_val_labels: Y data labels for validation
    :param X_train_masks: X masks for training
    :param X_test_masks: X masks for testing
    :param X_val_masks: X masks for validation
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param neg_labels: Dictionary that maps any feature with negative labels
                       to have the correct label on BERT predictions
    :param output_dir: Output directory
    :param device: Device to use
    :return: None
    """
    for curr_y_col in y_cols:

        train_dataloader, val_dataloader, curr_y_val_labels,\
        model, optimizer, scheduler =\
            setup_for_training(X_train_inputs, X_test_inputs, X_val_inputs,
                               y_train_labels, y_test_labels, y_val_labels,
                               X_train_masks, X_test_masks, X_val_masks,
                               epochs, batch_size, curr_y_col)

        # Store the average loss after each epoch so we can plot them.
        loss_values = []

        # Stores predictions after completed training
        pred_df = None

        curr_output_dir = '{}{}/'.format(output_dir, curr_y_col, "/")
        # Create output directory if needed
        if not os.path.exists(curr_output_dir):
            os.makedirs(curr_output_dir)

        # For each epoch...
        for epoch_i in range(epochs):

            model, loss_values = train_bert(model, train_dataloader, device,
                                            optimizer, scheduler, curr_y_col,
                                            epochs, epoch_i, loss_values)

            # ========================================
            #             Validation
            # ========================================
            # After the completion of each training epoch, measure our
            # performance on our val set.

            model, pred_df = val_bert(model, val_dataloader, device,
                                      curr_y_val_labels, curr_y_col,
                                      neg_labels)

        print("")
        print("Training complete for {}!".format(curr_y_col))

        # Save a trained model, configuration and tokenizer using
        # 'save_pretrained()'. They can then be reloaded using
        # 'from_pretrained()'

        # Takes care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(curr_output_dir)
        model.save_pretrained(curr_output_dir)
        pred_df.to_csv(curr_output_dir + 'predictions.csv')

        # Good practice: save your training arguments together with the trained
        # model
        # torch.save(args, os.path.join(curr_output_dir, 'training_args.bin'))

def training_driver(data_dir='data/', output_dir='model_save/',
           fp_list=('train_data.csv'), epochs=4, batch_size=32):
    """
    The main driver for everything skskksksk
    Sets everything up for training before the real trainer comes in : - )

    :param data_dir: Directory where the data be
    :param output_dir: Directory for the output : - )
    :param fp_list: The filepaths for the data
    :param epochs: Number of epochs
    :param batch_size: Size of batch
    :return: None
    """
    # Epochs 2-4
    # Batch_size 16 or 32

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Logs print output in file
    sys.stdout = Logger(output_dir)

    device, n_gpu, tokenizer = prep()

    res_dfs = get_data(data_dir, fp_list)

    if len(res_dfs) == 1:
        df = res_dfs[0]
    else:
        # Pretty useless, unnecessary and better practice would probably just
        # have everything in one training csv file in the first place : - )
        df = merge_dfs(res_dfs[0], res_dfs[1])

    df, neg_labels = clean_df(df)
    df = convert_text(df, tokenizer, output_dir)

    X_train_inputs, X_test_inputs, X_val_inputs, \
        y_train_labels, y_test_labels, y_val_labels, \
        X_train_masks, X_test_masks, X_val_masks = split_data(df, data_dir)

    trainer(X_train_inputs, X_test_inputs, X_val_inputs,
            y_train_labels, y_test_labels, y_val_labels,
            X_train_masks, X_test_masks, X_val_masks,
            epochs, batch_size, neg_labels, output_dir, device)


def predicting_driver(data_dir='data/', output_dir='model_save/',
           fp_list=('train_data.csv'), epochs=4, batch_size=32):
    """
    The main driver for everything skskksksk
    Sets everything up for training before the real trainer comes in : - )

    :param data_dir: Directory where the data be
    :param output_dir: Directory for the output : - )
    :param fp_list: The filepaths for the data
    :param epochs: Number of epochs
    :param batch_size: Size of batch
    :return: None
    """
    # Epochs 2-4
    # Batch_size 16 or 32

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Logs print output in file
    sys.stdout = Logger(output_dir)

    device, n_gpu, tokenizer = prep()

    res_dfs = get_data(data_dir, fp_list)

    if len(res_dfs) == 1:
        df = res_dfs[0]
    else:
        # Pretty useless, unnecessary and better practice would probably just
        # have everything in one training csv file in the first place : - )
        df = merge_dfs(res_dfs[0], res_dfs[1])

    df, neg_labels = clean_df(df)
    df = convert_text(df, tokenizer, output_dir)

    X_train_inputs, X_test_inputs, X_val_inputs, \
        y_train_labels, y_test_labels, y_val_labels, \
        X_train_masks, X_test_masks, X_val_masks = split_data(df, data_dir)

    trainer(X_train_inputs, X_test_inputs, X_val_inputs,
            y_train_labels, y_test_labels, y_val_labels,
            X_train_masks, X_test_masks, X_val_masks,
            epochs, batch_size, neg_labels, output_dir, device)
