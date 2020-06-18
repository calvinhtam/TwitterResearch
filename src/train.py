import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler,\
    SequentialSampler
from sklearn.metrics import f1_score
from pytorch_pretrained_bert import BertAdam
from transformers import get_linear_schedule_with_warmup,\
    BertForSequenceClassification
import pandas as pd
import os
import sys
import numpy as np
import time
from csv import writer
from processing import y_cols, format_time, get_classification_report,\
    get_data, split_data, merge_dfs, clean_df, logger
from bert import prep, bert_tokenize_f, convert_text

# Set the seed value all over the place to make this reproducible.
SEED_VAL = 42


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
    X_test_inputs_tensor = torch.tensor(X_test_inputs.to_list())
    X_val_inputs_tensor = torch.tensor(X_val_inputs.to_list())

    curr_y_train_labels_tensor = torch.tensor(curr_y_train_labels.to_list())
    curr_y_test_labels_tensor = torch.tensor(curr_y_test_labels.to_list())
    curr_y_val_labels_tensor = torch.tensor(curr_y_val_labels.to_list())

    X_train_masks_tensor = torch.tensor(X_train_masks.to_list())
    X_test_masks_tensor = torch.tensor(X_test_masks.to_list())
    X_val_masks_tensor = torch.tensor(X_val_masks.to_list())

    num_labels = len(set(curr_y_train_labels.unique())
                     .union(set(curr_y_val_labels.unique()))
                     .union(set(curr_y_test_labels.unique())))

    # The DataLoader needs to know our batch size for training, so we
    # specify it here.

    # Create the DataLoader for our training set.
    train_data = TensorDataset(X_train_inputs_tensor, X_train_masks_tensor,
                               curr_y_train_labels_tensor)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=batch_size)

    # Create the DataLoader for our test set.
    test_data = TensorDataset(X_test_inputs_tensor, X_test_masks_tensor,
                              curr_y_test_labels_tensor)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler,
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

    return train_dataloader, val_dataloader, test_dataloader,\
        curr_y_val_labels, curr_y_test_labels, model, optimizer, scheduler


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
            neg_labels, output_dir, device,
            epochs, batch_size):
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

        train_dataloader, val_dataloader, test_dataloader, curr_y_val_labels,\
            curr_y_test_labels, model, optimizer, scheduler = \
            setup_for_training(X_train_inputs, X_test_inputs, X_val_inputs,
                               y_train_labels, y_test_labels, y_val_labels,
                               X_train_masks, X_test_masks, X_val_masks,
                               epochs, batch_size, curr_y_col)

        # Store the average loss after each epoch so we can plot them.
        loss_values = []

        # Stores predictions after completed training
        val_pred_df = None

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

            model, val_pred_df = val_bert(model, val_dataloader, device,
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
        val_pred_df.to_csv(curr_output_dir + 'predictions.csv')

        # ========================================
        #             FINAL TEST
        # ========================================
        # After the completion of the last training epoch, measure our
        # performance on our test set.

        print('========================================')
        print(' FINAL TEST FOR', curr_y_col)
        val_bert(model, test_dataloader, device, curr_y_test_labels,
                 curr_y_col, neg_labels)

        # Good practice: save your training arguments together with the trained
        # model
        # torch.save(args, os.path.join(curr_output_dir, 'training_args.bin'))


def training_driver(data_dir='data/', output_dir='model_save/',
           fp_list=('train_data.csv'), epochs=4, batch_size=32, max_len=64):
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

    logger(output_dir='model_save/')

    device, n_gpu, tokenizer = prep()

    res_dfs = get_data(data_dir, fp_list)

    if len(res_dfs) == 1:
        df = res_dfs[0]
    else:
        # Pretty useless, unnecessary and better practice would probably just
        # have everything in one training csv file in the first place : - )
        df = merge_dfs(res_dfs[0], res_dfs[1])

    df, neg_labels = clean_df(df)
    df = convert_text(df, tokenizer, output_dir, max_len)

    X_train_inputs, X_test_inputs, X_val_inputs, \
        y_train_labels, y_test_labels, y_val_labels, \
        X_train_masks, X_test_masks, X_val_masks = split_data(df, data_dir)

    trainer(X_train_inputs, X_test_inputs, X_val_inputs,
            y_train_labels, y_test_labels, y_val_labels,
            X_train_masks, X_test_masks, X_val_masks,
            neg_labels, output_dir, device,
            epochs, batch_size)

    # Saves the configurations of the epoch, batch_size, and max_len as a csv
    configs_to_save_f = open(output_dir + 'config.csv', 'w', newline='')
    configs_to_save_csv_writer = writer(configs_to_save_f)
    configs_to_save_csv_writer.writerow(['Configurations', 'Value'])
    configs_to_save_csv_writer.writerow(['epochs', epochs])
    configs_to_save_csv_writer.writerow(['batch_size', batch_size])
    configs_to_save_csv_writer.writerow(['max_len', max_len])
