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

	def init_weight(self, pretrained_embedding_file):
		# update embeddings using pretrained embedding file
		print "not implemented"

	def get_train_parameters(self):
		print "not implemented"

	def forward(self, text):
		self.batch_size, self.len = text.size()
		print "not implemented"


class myLSTM(torch.module):
	def __init__(self, config):
		print "not implemented"

	def get_train_parameters(self):
		print "not implemented"

	def init_weight(self. pretrained_embedding_file):
		print "not implemented"
		
	def forward(self, text):
		print "not implemented"