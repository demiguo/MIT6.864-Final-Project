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

import utils
from config import Config
from model import myCNN, myLSTM

""" Train: return model, optimizer """
def train(config, model, optimizer, data_loader, i2q):
	# TODO(demi): currently, this only works for CNN model. In the future, make it compatible for LSTM model.
	model.train()
	
	for batch_idx, (qid, similar_q, candidate_q, label, similar_num, candidate_num) in tqdm(enumerate(data_loader), desc="Training"):
		# qid: batch_size (tensor)
		# similar_q: batch_size * num_similar_q (tensor)
		# candidate_q: batch_size * 20 (tensor)
		# label: batch_size * 20 (tensor)
		num_similar_q = 1
		num_candidate_q = 20                                              

		batch_size = qid.size(0)
		assert qid.size() == (batch_size,)
		assert similar_q.size() == (batch_size, num_similar_q)
		assert candidate_q.size() == (batch_size, num_candidate_q)
		assert label.size() == (batch_size, num_candidate_q)
		

		""" Retrieve Question Text """

		# get question title and body
		q_title = torch.zeros((batch_size, config.args.title_max_len)).long()
		q_body = torch.zeros((batch_size, config.args.body_max_len)).long()
		q_title_len = torch.zeros((batch_size)).long()
		q_body_len = torch.zeros((batch_size)).long()
		for i in range(batch_size):
			t, b, t_len, b_len = i2q[qid[i]]
			q_title[i] = t
			q_body[i] = b
			q_title_len[i] = t_len
			q_body_len[i] = b_len
		q_title = autograd.Variable(q_title)
		q_body = autograd.Variable(q_body)
		q_title_len = autograd.Variable(q_title_len)
		q_body_len = autograd.Variable(q_body_len)

		# get similar question title and body
		similar_title = torch.zeros((batch_size, num_similar_q, config.args.title_max_len)).long()
		similar_body = torch.zeros((batch_size, num_similar_q, config.args.body_max_len)).long()
		similar_title_len = torch.zeros((batch_size, num_similar_q)).long()
		similar_body_len = torch.zeros((batch_size, num_similar_q)).long()
		for i in range(batch_size):
			l = similar_num[i]
			similar_ids = np.random.choice(l, num_similar_q, replace=False)
			for j in range(num_similar_q):
				idx = similar_ids[j]
				t, b, t_len, b_len = i2q[similar_q[i][idx]]
				similar_title[i][j] = t
				similar_body[i][j] = b
				similar_title_len[i][j] = t_len
				similar_body_len[i][j] = b_len
		similar_title = autograd.Variable(similar_title)
		similar_body = autograd.Variable(similar_body)
		similar_title_len = autograd.Variable(similar_title_len)
		similar_body_len = autograd.Variable(similar_body_len)


		# get candidate question title and body
		candidate_title = torch.zeros((batch_size, num_candidate_q, config.args.title_max_len)).long()
		candidate_body = torch.zeros((batch_size, num_candidate_q, config.args.body_max_len)).long()
		candidate_title_len = torch.zeros((batch_size, num_candidate_q)).long()
		candidate_title_len = torch.zeros((batch_size, num_candidate_q)).long()
		for i in range(batch_size):
			l = candidate_num[i]
			candidate_ids = np.random.choice(l, num_candidate_q, replace=False)
			for j in range(num_candidate_q):
				idx = candidate_ids[j]
				t, b, t_len, b_len = i2q[candidate_q[i][idx]]
				candidate_title[i][j] = t
				candidate_body[i][j] = b
				candidate_title_len[i][j] = t_len
				candidate_body_len[i][j] = b_len
		candidate_title = autograd.Variable(candidate_title)
		candidate_body = autograd.Variable(candidate_body)
		candidate_title_len = autograd.Variable(candidate_title_len)
		candidate_body_len = autograd.Variable(candidate_body_len)

		""" Retrieve Question Embeddings """

		q_title_emb = model(q_title)
		q_body_emb = model(q_body)
		q_emb = 0.5 * (model(q_title, q_title_len)+ model(q_body, q_body_len))
		assert q_emb.size() == (batch_size, config.args.final_dim)

		similar_title = similar_title.contiguous().view(batch_size * num_similar_q, config.args.title_max_len)
		similar_body = similar_body.contiguous().view(batch_size * num_similar_q, config.args.body_max_len)
		similar_title_len = similar_title_len.contiguous().view(batch_size * num_similar_q)
		similar_body_len = similar_body_len.contiguous().view(batch_size * num_similar_q)
		similar_emb = 0.5 * (model(similar_title, similar_title_len) + model(similar_body, similar_body_len))
		similar_emb = similar_emb.contiguous().view(batch_size, num_similar_q, config.args.final_dim)

		candidate_title = candidate_title.contiguous().view(batch_size * num_candidate_q, config.args.title_max_len)
		candidate_body = candidate_body.contiguous().view(batch_size * num_candidate_q, config.args.body_max_len)
		candidate_title_len = candidate_title_len.contiguous().view(batch_size * num_candidate_q)
		candidate_body_len = candidate_body_len.contiguous().view(batch_size * num_candidate_q)
		candidate_emb = 0.5 * (model(candidate_title, candidate_title_len) + model(candidate_body, candidate_body_len))
		candidate_emb = candidate_emb.contiguous().view(batch_size, num_candidate_q, config.args.final_dim)


		""" Calculate Loss """
		optimizer.zero()
		# TODO(demi): make it batch operations
		max_margin_loss = autograd.Variable(torch.zeros((batch_size, num_candidate_q)))
		for i in tqdm(range(batch_size), desc="BatchMaxMarginCalculation"):
			qi = q_emb[i].view(cnofig.args.final_dim)
			pp_ind = random.randint(0, num_similar_q)
			pp = similar_emb[i][pp_ind].view(config.args.final_dim)
			s2 = torch.nn.Cosine_Similarity(dim=0)(qi, pp).view(1)
			for j in range(num_candidate_q):
				p = candidate_emb[i][j].view(config.args.final_dim)
				s1 = torch.nn.Consine_Similarity(dim=0)(qi, p).view(1)
				delta = config.args.delta_constant if label[i][j] == 0 else 0
				margin = s1 - s2 + delta
				max_margin_loss[i][j] += margin
		max_margin_loss = torch.max(max_margin_loss, dim=1)[0].view(batch_size)
		loss = torch.mean(max_margin_loss)

		loss.backward()
		optimizer.step()



""" Evaluate: return model """
def evaluate(model, data_loader):
	model.eval()
	print "not implemented"
	return 0, 0, 0, 0

if __name__ == "__main__":
	config = Config()
	config.get_config_from_user()
	config.log.info("=> Finish Loading Configuration")


	# word processing (w2i, i2w, i2v)
	w2i, i2v, vocab_size = utils.word_processing(config)
	config.args.vocab_size = vocab_size 
	config.log.info("=> Finish Word Processing")

	# get questions (question dictionary: id -> python array pair (title, body))
	i2q = utils.get_questions(config, w2i)
	config.log.info("=> Finish Retrieving Questions")

	# create dataset
	train_data = utils.QRDataset(config, config.args.train_file, w2i, vocab_size, is_train=True)
	config.log.info("=> Building Dataset: Finish Train")
	test_data = utils.QRDataset(config, config.args.test_file, w2i, vocab_size, is_train=False)
	config.log.info("=> Building Dataset: Finish Test")
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.args.batch_size, shuffle=True, **config.kwargs)
	test_loader = torhc.utils.data.DataLoader(test_data, batch_size=config.args.batch_size, **config.kwargs)
	config.log.info("=> Building Dataset: Finish All")

	if config.args.model_type == "CNN":
		model = myCNN(config)
	else:
		model = myLSTM(config)

	optimizer = optim.Adam(model.get_train_parameters(), lr=0.1)
	for epoch in tqdm(range(config.args.epochs), desc="Running"):
			model, optimizer, avg_loss = train(model, optimizer, train_loader)
			MAP, MRR, P1, P5 = evaluate(model, optimizer, test_loader)
			print "EPOCH[%d] Train Loss", avg_loss.data[0]
			print "EPOCH[%d] TEST: MAP %.3lf MRR %.3lf P@1 %.3lf P@5 %.3lf" % (MAP, MRR, P1, P5)

			# TODO(demi): save checkpoint