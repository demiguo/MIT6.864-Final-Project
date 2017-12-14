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

        parser.add_argument("-original_wordvec", "--original_wordvec", help="Raw Word Vectors File", required=False, default="../../data/vector/vectors_pruned.200.txt")
        parser.add_argument("-pretrained_wordvec", "--pretrained_wordvec", help="Word Vectors Only File", required=False, default="../../data/vector/vectors_only.txt")
        parser.add_argument("-pretrained_vocab", "--pretrained_vocab", help="Vocab Only File", required=False, default="../../data/vector/glove.vocab")
        parser.add_argument("-question_file", "--question_file", help="Question File", required=False, default="../../data/QR/text_tokenized.txt")
        parser.add_argument("-question_file_for_android", "--question_file_for_android", help="Question File", required=False, default="../../data/Android/corpus.tsv")

        parser.add_argument("-test_file", "--test_file", help="Test File", required=False, default="../../data/QR/test.txt")
        parser.add_argument("-dev_file", "--dev_file", help="Dev File", required=False, default="../../data/QR/dev.txt")

        parser.add_argument("-test_file_for_android", "--test_file_for_android", help="Test File _for_android", required=False, default="../../data/Android/test")
        parser.add_argument("-dev_file_for_android", "--dev_file_for_android", help="Dev File _for_android", required=False, default="../../data/Android/dev")

        parser.add_argument("-log_dir", "--log_dir", help="Log Directory", required=False, default="../../log")
        parser.add_argument("-model_dir", "--model_dir", help="Model Directory", required=False, default="../../models")
        parser.add_argument("-pretrained_encoder", "--pretrained_encoder", help="Pre-trianed Source Encoder", required=True)
        now = datetime.datetime.now()
        parser.add_argument('-model_suffix', '--model_suffix', help="Additional Model Information", required=False, default="%s-%s-%s-%s-%s" % (now.year, now.month, now.day, now.hour, now.minute))

        parser.add_argument("-max_body_len", "--max_body_len", type=int,help="Max Question Body Length", required=False, default=100)
        parser.add_argument("-max_title_len", "--max_title_len", type=int, help="Max Question Title Length", required=False, default=20)

        parser.add_argument("-batch_size", "--batch_size", type=int, help="Batch Size", required=False, default=128)
        parser.add_argument("-epochs", "--epochs", type=int, help="Epochs", required=False, default=2000)

        parser.add_argument("-embedding_dim", "--embedding_dim", type=int, help="Embedding Dimension", required=False, default=200)
        parser.add_argument("-delta_constant", "--delta_constant", type=float, help="Delta Constant", required=False, default=1.0)
        parser.add_argument("-model_type", "--model_type", help="Model Type", required=False, default="CNN")
       	
       	parser.add_argument("-mode", "--mode", help="Mode", required=False, default="train")
        parser.add_argument('-cuda', '--cuda', type=bool, help="Use CUDA or not", required=False, default=False)
        parser.add_argument("-use_glove", "--use_glove", type=bool, help="Use Glove", required=False, default=True)

        parser.add_argument("-log_step", "--log_step", type=int, help="#Steps Between Printing Logs", required=False, default=100)
        self.args = parser.parse_args()

        if torch.cuda.is_available() and self.args.cuda:
            self.use_cuda = True
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.set_device(1)
        else:
            self.use_cuda = False
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        self.kwargs = {}

        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.model_type == "CNN":
        	self.args.final_dim = 667
        else:
        	self.args.final_dim = 240
        if self.args.use_glove:
            self.args.embedding_dim = 300
            self.args.pretrained_wordvec = "../../data/vector/glove.vectors"
            self.args.pretrained_vocab = "../../data/vector/glove.vocab"
        self.args.discriminator_hidden_dim = 128
        self.args.model_file = "%s/1204-code-%s-%s" % (self.args.model_dir, self.args.model_type, self.args.model_suffix)
        self.args.log_file="%s/1204-code-%s-%s.log" % (self.args.log_dir, self.args.model_type, self.args.model_suffix)
        
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
        self.log.info("use_cuda=%d" % self.use_cuda)

