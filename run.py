#!/usr/bin/env python

import sys
import json
import shutil

sys.path.insert(0, 'src') # add library code to path
from src.twitter_scraper import get_data
from src.train import training_driver


DATA_PARAMS = 'config/data-params.json'
TEST_PARAMS = 'config/test-params.json'
TRAIN_PARAMS = 'config/train-params.json'

def load_params(fp):
    with open(fp) as fh:
        param = json.load(fh)
    return param

def main(targets):

    # make the data target
    if 'data' in targets:
        cfg = load_params(DATA_PARAMS)
        get_data(**cfg)

    # train
    if 'train' in targets:
        cfg = load_params(TRAIN_PARAMS)
        training_driver(**cfg)

    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
