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
from tqdm import tqdm

from config import config
from model import myCNN, myLSTM

""" Train: return model, optimizer """
def train_step(model, optimizer, data):
	print "not implemented"

""" Evaluate: return model """
def eval_step(model, data):
	print "not implemented"


if __name__ == "__main__":
	config = Config()
	config.get_config_from_user()

	# word processing (w2i, i2w, i2v)
	w2i, iw2, i2v, vocab_size = utils.word_processing(config)
	config.args.vocab_size = vocab_size 

	# get questions (question dictionary: id -> python array)
	i2q = utils.get_questions(config)

	# create dataset
	train_data = QRDataset(config.args.train_file, w2i, vocab_size)
	test_data = QRDataset(config.args.test_file, w2i, vocab_size)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.args.batch_size, shuffle=True, **config.kwargs)
	test_loader = torhc.utils.data.DataLoader(test_data, batch_size=config.args.batch_size, **config.kwargs)

	if config.model_type == "CNN":
		model = myCNN(config)
	else:
		model = myLSTM(config)

	optimizer = optim.Adam(model.get_train_parameters(), lr=0.1)
	for epoch in tqdm(range(config.args.epochs), desc="Running"):
			model, optimizer, avg_loss = train_step(model, optimizer, train_datA)
			