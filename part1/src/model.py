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


class myCNN(torch.module):
	def __init__(self, config):
		# TODO)(demi): super init
		self.config = config
		self.vocab_size = config.vocab_size
		self.embedding_dim = config.embedding_dim

		# CNN hyperparameters
		self.hidden_dim = config.hidden_dim
		self.final_dim = config.final_dim

		self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
		self.cnn1 = nn.Conv1d(self.embedding_dim, self.hidden_dim, 3, 1)
		self.pool1 = nn.MaxPool1d(2)
		self.cnn2 = nn.Conv1d(self.hidden_dim, self.final_dim, 3, 1)
		self.pool2 = nn.MaxPool1d(2)

		self.init_weight(config.pretrained_wordvec)

	def init_weight(self, pretrained_embedding_file):
		# update embeddings using pretrained embedding file
		word_vec = np.loadtxt(pretrained_embedding_file)
		self.word_embeds.weight.data.copy_(torch.FloatTensor(word_vec))
		self.word_embeds.weight.requires_grad = False

	def get_train_parameters(self):
		params = []
		for param in self.parameters():
			if param.requires_grad == True:
				params.append(param)
		return params

	def forward(self, text):
		self.batch_size, self.len = text.size()
		emb = self.word_embeds(text)
		assert emb.size() == (self.batch_size, self.len, self.embedding_dim)
		emb = torch.transpose(emb, (1, 2))
		assert emb.size() == (self.batch_size, self.embedding_dim, self.len)

		x1 = self.cnn1(emb)
		x1 = self.pool1(x1)
		x1 = torch.nn.ReLU()(x1)

		x2 = self.cnn2(x1)
		x2 = self.pool2(x2)
		x2 = torch.nn.ReLU()(x2)

		x2 = torch.max(x2, dim=2)[0]
		assert x2.size() == (self.batch_size, self.final_dim)

		return x2

class myLSTM(torch.module):
	def __init__(self, config):
		print "not implemented"

	def get_train_parameters(self):
		print "not implemented"

	def init_weight(self. pretrained_embedding_file):
		print "not implemented"

	def forward(self, text):
		print "not implemented"