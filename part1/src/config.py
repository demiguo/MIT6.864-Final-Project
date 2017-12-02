import sys
import random
import torch 
import torch.autograd as autograd 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
import argparse
import torch.utils.data
import datetime
import logging

class Config:

    def init(self):
        args = {}

    def get_config_from_user(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-seed', '--seed', type=int, help="Torch Random Seed", required=False, default=1)

        parser.add_argument("-original_wordvec", "--original_wordvec", help="Raw Word Vectors File", required=False, default="../data/vector/vectors_pruned.200.txt")
        parser.add_argument("-pretrained_wordvec", "--pretrained_wordvec", help="Word Vectors Only File", required=False, default="../data/vector/vectors_only.txt")
        parser.add_argument("-question_file", "--question_file", help="Question File", required=False, default="../data/text_tokenized.txt")
        parser.add_argument("-train_file", "--train_file", help="Train File", required=False, default="../data/train_random.txt")
        parser.add_argument("-test_file", "--test_file", help="Test File", required=False, default="../data/test.txt")
        parser.add_argument("-dev_file", "--dev_file", help="Dev File", required=False, default="../data/dev.txt")
        parser.add_argument("-log_file", "--log_file", help="Log File", required=False, default="../log/tmp.txt")

        parser.add_argument("-max_body_len", "--max_body_len", help="Max Question Body Length", required=False, default=20)
        parser.add_argument("-max_title_len", "--max_title_len", help="Max Question Title Length", required=False, default=100)

        parser.add_argument("-embedding_dim", "--embedding_dim", help="Embedding Dimension", required=False, default=200)
        parser.add_argument("-hidden_dim", "--hidden_dim", help="Hidden Dimension", required=False, default=150)
        parser.add_argument("-final_dim", "--final_dim", help="Final Dimension", required=False, default=100)
       
        self.args = parser.parse_args()
        self.kwargs = {}
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        self.fh = logging.FileHandler(self.args.log_file)
        self.fh.setLevel(logging.DEBUG)
        self.ch = logging.StreamHandler(sys.stdout)
        self.ch.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S")
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)
        self.log.addHandler(self.fh)
        self.log.addHandler(self.ch)

        self.log.info("log to %s" % self.args.log_file)

