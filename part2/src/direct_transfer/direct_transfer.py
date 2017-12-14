import random
import torch 
import torch.autograd as autograd 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
import argparse
import torch.utils.data
import datetime
from tqdm import tqdm
import sys

from IPython import embed

from config import Config
import utils
import models

if __name__ == "__main__":
    config = Config()
    config.get_config_from_user()
    config.log.info("=> Finish Loading Configuration")

    # word processing (w2i, i2w, i2v)
    w2i, i2v, vocab_size = utils.word_processing(config)
    config.args.vocab_size = vocab_size
    config.log.info("=> Finish Word Processing")

    # get questions (question dictionary: id -> python array pair (title, body))
    i2q = utils.get_questions_for_android(config, w2i)
    config.log.info("=> Finish Retrieving Questions")

