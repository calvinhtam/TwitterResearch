#!/usr/bin/env python

import sys
import json
import shutil

sys.path.insert(0, 'src') # add library code to path
# from src.etl import get_data
from src.bert_train import training_driver

DATA_PARAMS = 'config/data-params.json'
TEST_PARAMS = 'config/test-params.json'
TRAIN_PARAMS = 'config/train-params.json'

def load_params(fp):
    with open(fp) as fh:
        param = json.load(fh)
    return param

def main(targets):

    # make the clean target
    if 'clean' in targets:
        shutil.rmtree('data/temp', ignore_errors=True)
        shutil.rmtree('data/out', ignore_errors=True)
        shutil.rmtree('data/test', ignore_errors=True)

    # make the data target
    if 'data' in targets:
        cfg = load_params(DATA_PARAMS)
        get_data(**cfg)

    # cleans and prepares the data for training
    if 'process' in targets:
        cfg = load_params(PROCESS_PARAMS)
        clean(**cfg)

    # train
    if 'train' in targets:
        cfg = load_params(TRAIN_PARAMS)
        training_driver(**cfg)

    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
