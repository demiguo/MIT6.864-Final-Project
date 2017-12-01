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
def train(config, model, optimizer, data_loader, i2q):
	model.train()
	
	for batch_idx, (qid, similar_q, candidate_q, label) in tqdm(enumerate(data_loader), desc="Training"):
		# qid: batch_size (tensor)
		# similar_q: batch_size * num_similar_q (tensor)
		# candidate_q: batch_size * 20 (tensor)
		# label: batch_size * 20 (tensor)
		num_similar_q = similar_q.size(1)
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
		for i in range(batch_size):
			t, b = i2q[qid[i]]
			q_title[i] = autograd.Variable(torch.LongTensor(t))
			q_body[i] = autograd.Variable(torch.LongTensor(b))

		# get similar question title and body
		similar_title = torch.zeros((batch_size, num_similar_q, config.args.title_max_len)).long()
		similar_body = torch.zeros((batch_size, num_similar_q, config.args.body_max_len)).long()
		for i in range(batch_size):
			for j in range(num_similar_q):
				t, b = i2q[similar_q[i][j]]
				similar_title[i][j] = autograd.Variable(torch.LongTensor(t))
				simlar_body[i][j] = autograd.Variable(torch.LongTensor(b))

		# get candidate question title and body
		candidate_title = torch.zeros((batch_size, num_candidate_q, config.args.title_max_len)).long()
		candidate_body = torch.zeros((batch_size, num_candidate_q, config.args.body_max_len)).long()
		for i in range(batch_size):
			for j in range(num_candidate_q):
				t, b = i2q[candidate_q[i][j]]
				candidate_title[i][j] = autograd.Variable(torch.LongTensor(t))
				candidate_body[i][j] = autograd.Variable(torch.LongTensor(b))

		""" Retrieve Question Embeddings """

		q_title_emb = model(q_title)
		q_body_emb = model(q_body)
		q_emb = 0.5 * (model(q_title)+ model(q_body))
		assert q_emb.size() == (batch_size, config.args.final_dim)

		similar_title = similar_title.contiguous().view(batch_size * num_similar_q, config.args.title_max_len)
		similar_body = similar_body.contiguous().view(batch_size * num_similar_q, config.args.body_max_len)
		similar_emb = 0.5 * (model(similar_title) + model(similar_body))
		similar_emb = similar_emb.contiguous().view(batch_size, num_similar_q, config.args.final_dim)

		candidate_title = candidate_title.contiguous().view(batch_size * num_candidate_q, config.args.title_max_len)
		candidate_body = candidate_body.contiguous().view(batch_size * num_candidate_q, config.args.body_max_len)
		candidate_emb = 0.5 * (model(candidate_title) + model(candidate_body))
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

	# word processing (w2i, i2w, i2v)
	w2i, iw2, i2v, vocab_size = utils.word_processing(config)
	config.args.vocab_size = vocab_size 

	# get questions (question dictionary: id -> python array pair (title, body))
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
			model, optimizer, avg_loss = train(model, optimizer, train_loader)
			MAP, MRR, P1, P5 = evaluate(model, optimizer, test_loader)
			print "EPOCH[%d] Train Loss", avg_loss.data[0]
			print "EPOCH[%d] TEST: MAP %.3lf MRR %.3lf P@1 %.3lf P@5 %.3lf" % (MAP, MRR, P1, P5)

			# TODO(demi): save checkpoint