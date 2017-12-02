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

from sets import Set

NUM_SIMILAR_Q = 20
NUM_CANDIDATE_Q = 20

""" Return w2i, i2w, vocab_size """
def word_processing(config):
	w2i = {}
	i2w = {}

	f = open(config.args.original_wordvec)
	f_e = open(config.args.pretrained_wordvec, "w")

	w2i["<EMPTY>"] = 0
	i2w[0] = "<EMPTY>"
	vec = ["0"] * config.args.embedding_dim
	f_e.write(" ".join(vec))

	w2i["<UNK>"] = 1
	i2w[1] = "<UNK>"
	vec = ["0"] * config.args.embedding_dim
	f_e.write(" ".join(vec))

	vocab_size = 2

	lines = f.readlines()
	for line in tqdm(lines):
		l = line.split(" ")
		word = l[0]
		vec = l[1:]
		if word in w2i:
			print "[warning]word %s already in w2i" % word
			continue
		w2i[word] = vocab_size
		i2w[vocab_size] = word
		vocab_size += 1
		f_e.write(" ".join(vec))

	f_e.close()
	f.close()
	return w2i, i2w, vocab_size


""" Return i2q: index to a python array pair (title, body) """
def get_questions(config, w2i):
	i2q = {}
	f = open(config.args.question_file)

	def getw2i(word):
		if word not in w2i:
			return w2i["<UNK>"]
		else:
			return w2i["<UNK>"]

	# question 0 (special case)
	title_ids = [0] * config.args.max_title_len
	body_ids = [0] * config.args.max_body_len
	i2q[0] = (title_ids, body_ids)

	lines = f.readlines()
	for line in tqdm(lines):
		qid, title, body = line.split("\t")

		qid = int(qid) + 1 # make sure it's 0 indexed

		title = title.split(" ")
		body = body.split(" ")

		title_ids = []
		for i in range(config.args.max_title_len):
			if i < len(title):
				title_ids.append(getw2i(title[i]))
			else:
				title_ids.append(0)

		body_ids = []
		for i in range(config.args.max_body_len):
			for i in range(len(body)):
				body_ids.append(getw2i(body[i]))
			else:
				body_ids.append(0)
		if qid in i2q:
			print "qid %d already in i2q" % qid
			continue
		i2q[qid] = (title_ids, body_ids)
	f.close()
	return i2q


""" qid, similar_q, candidate_q, label """
class QRDataset(torch.utils.data.Dataset):
	def __init__(self, config, data_file, w2i, vocab_size):
		f = open(data_file)
		lines = f.readlines()
		f.close()

		self.data_size = len(lines)
		qid = np.zeros((self.data_size))
		# TODO(demi): let's first pad it to 20 similar & candidate questions
		similar_q = np.zeros((self.data_size, NUM_SIMILAR_Q), dtype=int)
		candidate_q = np.zeros((self.data_size, NUM_CANDIDATE_Q), dtype=int) 
		label = np.zeros((self.data_size, NUM_CANDIDATE_Q), dtype=int)

		for i in range(self.data_size):
			line = lines[i]
			q, s, c = line.split("\t")
			s = s.split(" ")
			c = c.split(" ")
			if len(s) > 20 or len(c) != 20:
				config.log.warning("# of similar/candidate questions not matches expectation\n    s:%s\n     c:%s" %(" ".join(s)," ".join(c)))
			qid[i] = int(q)
			s_set = Set([])
			for j in range(len(s)):
				similar_q[i][j] = int(s[j])
				s_set.add(int(s[j]))

			for j in range(len(c)):
				candidate_q[i][j] = int(c[j])
				label[i][j] = 1 if int(c[j]) in s_set else 0

	def __getitem__(self, index):
		return (qid[index], similar_q[index], candidate_q[index], label[index])

	def __len__(self):
		return self.data_size