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

from IPython import embed
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
	f_e.write(" ".join(vec) + "\n")

	w2i["<UNK>"] = 1
	i2w[1] = "<UNK>"
	vec = ["0"] * config.args.embedding_dim
	f_e.write(" ".join(vec) + "\n")

	vocab_size = 2

	lines = f.readlines()
	for line in tqdm(lines, desc="Word Processing"):
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


""" Return i2q: index to a python array pair (padded title, padded body, len title, body title) """
def get_questions(config, w2i):
	i2q = {}
	f = open(config.args.question_file)

	def getw2i(word):
		if word not in w2i:
			return w2i["<UNK>"]
		else:
			return w2i[word]

	# question 0 (special case)
	title_ids = [0] * config.args.max_title_len
	body_ids = [0] * config.args.max_body_len
	i2q[0] = (title_ids, body_ids, 1, 1)

	lines = f.readlines()
	if config.args.mode == "debug":
		lines = lines[:15]

	config.log.info("# of questions: %d" % len(lines))
	for i, line in tqdm(enumerate(lines), desc="Retrieving Questions"):
		qid, title, body = line.split("\t")

		qid = int(qid) 
		if config.args.mode == "debug":
			qid = i + 1

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
			if i < len(body):
				body_ids.append(getw2i(body[i]))
			else:
				body_ids.append(0)
		if qid in i2q:
			print "qid %d already in i2q" % qid
			continue

		assert len(title_ids) == config.args.max_title_len
		assert len(body_ids) == config.args.max_body_len

		i2q[qid] = (title_ids, body_ids, len(title), len(body))
	f.close()
	return i2q


""" qid, similar_q, candidate_q, label """
class QRDataset(torch.utils.data.Dataset):
	def __init__(self, config, data_file, w2i, vocab_size, i2q, is_train=True, prune_positive_sample=10, K_neg=20):
		f = open(data_file)
		lines = f.readlines()
		f.close()

		self.data_size = len(lines)
		self.qid = np.zeros((self.data_size), dtype=int)
		self.similar_q = np.zeros((self.data_size, NUM_SIMILAR_Q), dtype=int)
		self.candidate_q = np.zeros((self.data_size, NUM_CANDIDATE_Q), dtype=int) 
		self.label = np.zeros((self.data_size, NUM_CANDIDATE_Q), dtype=int)
		self.similar_num = np.zeros((self.data_size), dtype=int)
		self.candidate_num = np.zeros((self.data_size), dtype=int)

		cur_index = 0
		for i in tqdm(range(self.data_size), desc="Dataset"):
			line = lines[i]
			if len(line.split("\t")) == 3:
				q, s, c = line.split("\t")
			elif len(line.split("\t")) == 4:
				q, s, c, _ = line.split("\t")
			else:
				config.log.info("line split not right: %s" % line)
				continue
			s = s.split(" ")
			c = c.split(" ")
			if config.args.mode == "debug":
				if i > 300:
					break

			if is_train:
				if len(s) > prune_positive_sample:
					continue
			else:
				if len(s) > 20:
					print "test/dev: positive examples #= %d" % len(s)
				if len(s) == 1 and s[0] == "":
					continue

			q = int(q)
			s = [int(o) for o in s]
			c = [int(o) for o in c]

			if config.args.mode == "debug":
				s = [o % 10 for o in s]
				c = [o % 10 for o in c]
				q = int(q) % 10

			if not q in i2q:
				config.log.warning("case line %d not found q %d" % (i,q))
			new_s = []
			for o in s:
				if o in i2q:
					new_s.append(o)
				else:
					config.log.warning("case line %d | s | %d not found" % (i, o))
			new_c = []
			for o in c:
				if o in i2q:
					new_c.append(o)
				else:
					config.log.warning("case line %d | c | %d not found" % (i, o))
			if len(new_s) == 0 or len(new_c) == 0:
				config.log.warning("case line %d now has empty s or c\nq=%d\ns=%s\nc=%s\n" % (i,q,str(s),str(c)))
				continue
			s = new_s
			c = new_c

			# sort s in terms of question length
			self.qid[cur_index] = int(q)
			s_set = Set([])
			for j in range(len(s)):
				s_set.add(int(s[j]))
			
			for j in range(len(s)):
				self.similar_q[cur_index][j] = int(s[j])


			if is_train:
				candidate_ids = np.random.choice(len(c), min(len(c), K_neg), replace=False)
			else:
				candidate_ids = range(len(c))
				assert len(c) == NUM_CANDIDATE_Q, "len(c) [%d] != NUM_CANDIDATE_Q" % (len(c))

			# TODO(demi): sort candidate_ids in terms of question length
			for j in range(len(candidate_ids)):
				idx = candidate_ids[j]
				self.candidate_q[cur_index][j] = int(c[idx])
				self.label[cur_index][j] = 1 if int(c[idx]) in s_set else 0

			self.similar_num[cur_index] = len(s)
			self.candidate_num[cur_index] = len(candidate_ids)

			cur_index += 1

		self.data_size = cur_index

		# TODO(demi): add similar_num and candidate_num
	def __getitem__(self, index):
		return (self.qid[index], self.similar_q[index], self.candidate_q[index], self.label[index], self.similar_num[index], self.candidate_num[index])

	def __len__(self):
		return self.data_size