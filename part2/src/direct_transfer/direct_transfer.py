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
from model import myCNN, myLSTM

from meter import AUCMeter 

""" Evaluate: AUC(0.05) on Android """
def evaluate_for_android(model, data_loader, i2q):
    model.eval()

    meter = AUCMeter()
    for batch_idx, (q1_ids, q2_ids, labels) in tqdm(enumerate(data_loader), desc="Evaluate"):
        batch_size = q1_ids.size(0)

        # Q1
        q1_title = torch.zeros((batch_size, config.args.max_title_len)).long()
        q1_body = torch.zeros((batch_size, config.args.max_body_len)).long()
        q1_title_len = torch.zeros((batch_size)).long()
        q1_body_len = torch.zeros((batch_size)).long()
        for i in range(batch_size):
            t, b, t_len, b_len = i2q[q1_ids[i]]
            q1_title[i] = torch.LongTensor(t)
            q1_body[i] = torch.LongTensor(b)
            q1_title_len[i] = t_len
            q1_body_len[i] = b_len
        q1_title = autograd.Variable(q1_title)
        q1_body = autograd.Variable(q1_body)
        q1_title_len = autograd.Variable(q1_title_len)
        q1_body_len = autograd.Variable(q1_body_len)
        if config.use_cuda:
            q1_title, q1_title_len, q1_body, q1_body_len = q1_title.cuda(), q1_title_len.cuda(), q1_body.cuda(), q1_body_len.cuda()
        q1_emb = 0.5 * (model(q1_title, q1_title_len)+ model(q1_body, q1_body_len))

        # Q2
        q2_title = torch.zeros((batch_size, config.args.max_title_len)).long()
        q2_body = torch.zeros((batch_size, config.args.max_body_len)).long()
        q2_title_len = torch.zeros((batch_size)).long()
        q2_body_len = torch.zeros((batch_size)).long()
        for i in range(batch_size):
            t, b, t_len, b_len = i2q[q2_ids[i]]
            q2_title[i] = torch.LongTensor(t)
            q2_body[i] = torch.LongTensor(b)
            q2_title_len[i] = t_len
            q2_body_len[i] = b_len
        q2_title = autograd.Variable(q2_title)
        q2_body = autograd.Variable(q2_body)
        q2_title_len = autograd.Variable(q2_title_len)
        q2_body_len = autograd.Variable(q2_body_len)
        if config.use_cuda:
            q2_title, q2_title_len, q2_body, q2_body_len = q2_title.cuda(), q2_title_len.cuda(), q2_body.cuda(), q2_body_len.cuda()
        q2_emb = 0.5 * (model(q2_title, q2_title_len)+ model(q2_body, q2_body_len))

        # q1_emb, q2_emb: (batch_size, final_dim)
        scores = nn.CosineSimilarity()(q1_emb, q2_emb).view(-1)
        print "labels.size()=", labels.size(), " scores.data.size=", scores.data.size()
        meter.add(scores.data, labels)

    auc = meter.value(0.05)
    return auc 

if __name__ == "__main__":
    config = Config()
    config.get_config_from_user()
    config.log.info("=> Finish Loading Configuration")

    # word processing (w2i, i2w, i2v)
    w2i, i2v, vocab_size = utils.word_processing(config)
    config.args.vocab_size = vocab_size
    config.log.info("=> Finish Word Processing")

    # get questions (question dictionary: id -> python array pair (title, body))
    i2q = utils.get_questions_for_android(config, w2i, lower=True)
    config.log.info("=> Finish Retrieving Questions")

    # build android dataset
    dev_data = utils.AndroidDataset(config, config.args.dev_file_for_android, w2i, vocab_size)
    test_data = utils.AndroidDataset(config, config.args.test_file_for_android, w2i, vocab_size)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=1024, **config.kwargs)  # TODO(demi): make test/dev batch size super big
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024, **config.kwargs)  # TODO(demi): make test/dev batch size super big

    # load model
    if config.args.model_type == "CNN":
        config.log.info("=> Running CNN Model")
        model = myCNN(config)
    else:
        config.log.info("=> Running LSTM Model")
        model = myLSTM(config)

    if config.use_cuda:
        model = model.cuda()

    checkpoint = torch.load(config.args.load_model)
    model.load_state_dict(checkpoint["model"])


    # evaluate function (similar to main.py)
    dev_auc = evaluate_for_android(model, dev_loader, i2q)
    test_auc = evaluate_for_android(model, test_loader, i2q)

    print "Dev AUC: %.3lf || Test AUC: %.3lf" % (dev_auc, test_auc)

