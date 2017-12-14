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

import utils
from config import Config
from model import myCNN, myLSTM, myMLP, myFNN
from meter import AUCMeter 

""" Train: LOSS1 (max margin) - \lambda LOSS2 (discriminator loss) """
def train(config, encoder, discriminator, optimizer1, optimizer2, src_data_loader, tgt_data_loader, src_i2q, tgt_i2q):
    encoder.train()
    discriminator.train()
    avg_loss = 0
    total = 0

    combined_data_loader = zip(src_data_loader, tgt_data_loader)
    max_iteration_per_epoch = min(len(src_data_loader), len(tgt_data_loader))

    for batch_idx, (src_batch, tgt_batch) in tqdm(enumerate(combined_data_loader), desc="Training"):
        # first, calculate loss 1: (mean per batch)
        qid, similar_q, candidate_q, label, similar_num, candidate_num = src_batch

        num_similar_q = 1
        num_candidate_q = config.args.data_neg_num

        batch_size = qid.size(0)
        total += batch_size
        assert qid.size() == (batch_size,)


        """ Retrieve Question Text """

        # get question title and body
        q_title = torch.zeros((batch_size, config.args.max_title_len)).long()
        q_body = torch.zeros((batch_size, config.args.max_body_len)).long()
        q_title_len = torch.zeros((batch_size)).long()
        q_body_len = torch.zeros((batch_size)).long()
        for i in range(batch_size):
            #print "qid=", qid[i]
            #print "i2q[qid[i]]=", i2q[qid[i]]
            #embed()
            t, b, t_len, b_len = i2q[qid[i]]
            q_title[i] = torch.LongTensor(t)
            q_body[i] = torch.LongTensor(b)
            q_title_len[i] = t_len
            q_body_len[i] = b_len
            #embed()
        q_title = autograd.Variable(q_title)
        q_body = autograd.Variable(q_body)
        q_title_len = autograd.Variable(q_title_len)
        q_body_len = autograd.Variable(q_body_len)
        if config.use_cuda:
            q_title, q_title_len, q_body, q_body_len = q_title.cuda(), q_title_len.cuda(), q_body.cuda(), q_body_len.cuda()

        # get similar question title and body
        similar_title = torch.zeros((batch_size, num_similar_q, config.args.max_title_len)).long()
        similar_body = torch.zeros((batch_size, num_similar_q, config.args.max_body_len)).long()
        similar_title_len = torch.zeros((batch_size, num_similar_q)).long()
        similar_body_len = torch.zeros((batch_size, num_similar_q)).long()
        for i in range(batch_size):
            l = similar_num[i]
            similar_ids = np.random.choice(l, num_similar_q, replace=False)
            for j in range(num_similar_q):
                idx = similar_ids[j]
                t, b, t_len, b_len = i2q[similar_q[i][idx]]
                similar_title[i][j] = torch.LongTensor(t)
                similar_body[i][j] = torch.LongTensor(b)
                similar_title_len[i][j] = t_len
                similar_body_len[i][j] = b_len
        similar_title = autograd.Variable(similar_title)
        similar_body = autograd.Variable(similar_body)
        similar_title_len = autograd.Variable(similar_title_len)
        similar_body_len = autograd.Variable(similar_body_len)
        if config.use_cuda:
            similar_title, similar_title_len, similar_body, similar_body_len =\
                similar_title.cuda(), similar_title_len.cuda(), similar_body.cuda(), similar_body_len.cuda()

        # get candidate question title and body
        candidate_title = torch.zeros((batch_size, num_candidate_q, config.args.max_title_len)).long()
        candidate_body = torch.zeros((batch_size, num_candidate_q, config.args.max_body_len)).long()
        candidate_title_len = torch.zeros((batch_size, num_candidate_q)).long()
        candidate_body_len = torch.zeros((batch_size, num_candidate_q)).long()
        for i in range(batch_size):
            l = candidate_num[i]
            candidate_ids = np.random.choice(l, num_candidate_q, replace=False)
            for j in range(num_candidate_q):
                idx = candidate_ids[j]
                t, b, t_len, b_len = i2q[candidate_q[i][idx]]
                candidate_title[i][j] = torch.LongTensor(t)
                candidate_body[i][j] = torch.LongTensor(b)
                candidate_title_len[i][j] = t_len
                candidate_body_len[i][j] = b_len
        candidate_title = autograd.Variable(candidate_title)
        candidate_body = autograd.Variable(candidate_body)
        candidate_title_len = autograd.Variable(candidate_title_len)
        candidate_body_len = autograd.Variable(candidate_body_len)
        if config.use_cuda:
            candidate_title, candidate_title_len, candidate_body, candidate_body_len =\
                candidate_title.cuda(), candidate_title_len.cuda(), candidate_body.cuda(), candidate_body_len.cuda()
        """ Retrieve Question Embeddings """

        q_emb = 0.5 * (model(q_title, q_title_len)+ model(q_body, q_body_len))
        assert q_emb.size() == (batch_size, config.args.final_dim)

        similar_title = similar_title.contiguous().view(batch_size * num_similar_q, config.args.max_title_len)
        similar_body = similar_body.contiguous().view(batch_size * num_similar_q, config.args.max_body_len)
        similar_title_len = similar_title_len.contiguous().view(batch_size * num_similar_q)
        similar_body_len = similar_body_len.contiguous().view(batch_size * num_similar_q)
        similar_emb = 0.5 * (model(similar_title, similar_title_len) + model(similar_body, similar_body_len))
        similar_emb = similar_emb.contiguous().view(batch_size, num_similar_q, config.args.final_dim)

        candidate_title = candidate_title.contiguous().view(batch_size * num_candidate_q, config.args.max_title_len)
        candidate_body = candidate_body.contiguous().view(batch_size * num_candidate_q, config.args.max_body_len)
        candidate_title_len = candidate_title_len.contiguous().view(batch_size * num_candidate_q)
        candidate_body_len = candidate_body_len.contiguous().view(batch_size * num_candidate_q)
        candidate_emb = 0.5 * (model(candidate_title, candidate_title_len) + model(candidate_body, candidate_body_len))
        candidate_emb = candidate_emb.contiguous().view(batch_size, num_candidate_q, config.args.final_dim)
       
        d = config.args.final_dim
        q_emb = q_emb.view(batch_size, 1, d)
        # NB(demi): assume num_similar_q = 1
        similar_emb = similar_emb.view(batch_size, 1, d)
        pos_score = nn.CosineSimilarity(dim=2,eps=1e-6)(q_emb, similar_emb).view(batch_size, 1)
        #embed()n
        assert pos_score.size() == (batch_size, 1), "pos_score.size()=%s" % str(pos_score.size())

        pos_score = pos_score.expand(batch_size, num_candidate_q)

        q_expand_emb = q_emb.expand(batch_size, num_candidate_q, d)
        assert q_expand_emb.size() == candidate_emb.size()
        neg_scores = nn.CosineSimilarity(dim=2, eps=1e-6)(q_expand_emb, candidate_emb)
        #assert neg_scores.size() == (batch_size, num_candidate_q)

        loss1 = neg_scores - pos_score + config.args.delta_constant
        #assert loss1.size() == (batch_size, num_candidate_q)
        loss1 = torch.max(loss1, dim=1)[0].view(batch_size,1)
        loss1 = (loss1 > 0).float() * loss1
        loss1 = torch.mean(loss1) # mean batch loss1

        ########### second, calculate loss 2: (mean per batch) ###########

        # add loss2: domain 0
        emb1 = q_emb
        emb2 = similar_emb.view(batch_size * num_similar_q, configs.args.final_dim)
        emb3 = candidate_emb.view(batch_size * num_candidate_q, config.args.final_dim)

        src_emb = torch.cat((emb1, emb2, emb3), 0)
        src_num = batch_size + batch_size * num_similar_q + batch_size * num_candidate_q
        src_target = torch.autograd.Variable(torch.zeros((src_num)).long())  # domain 0
        if config.use_cuda:
            src_target = src_target.cuda()
        assert src_emb.size() == (src_num, config.args.final_dim)
        loss2 = discriminator.loss(src_emb, src_target)

        # add loss2: domain 1
        q1_ids = tgt_batch
        batch_size2 = q1_ids.size(0)

        q1_title = torch.zeros((batch_size2, config.args.max_title_len)).long()
        q1_body = torch.zeros((batch_size2, config.args.max_body_len)).long()
        q1_title_len = torch.zeros((batch_size2)).long()
        q1_body_len = torch.zeros((batch_size2)).long()
        for i in range(batch_size2):
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
        assert q1_emb.size() == (batch_size2, config.args.final_dim)
        tgt_emb = q1_emb
        tgt_target = torch.autograd.Variable(torch.ones((batch_size2)).long())
        if config.use_cuda:
            tgt_target = tgt_target.cuda()
        loss2 += discriminator.loss(tgt_emb, tgt_target)


        # gradient
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss = loss1 - config.loss_delta * loss2
        acg_loss += loss.data[0]
        total += 1
        loss.backward()
        optimizer1.step()
        optimizer2.step()

    avg_loss /= total   # TODO(demi): verify such average is a ok average
    return encoder, discriminator, optimizer1, optimizer1, avg_loss

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
        meter.add(scores.data, labels)

    auc = meter.value(0.05)
    return auc 
if __name__ == "__main__":
    config = Config()
    config.get_config_from_user()
    config.log.info("=> Finish Loading Configuration")


    # word processing (w2i, i2w, i2v)
    if config.args.use_glove:
        w2i, i2v, vocab_size = utils.word_processing_glove(config)
    else:
        w2i, i2v, vocab_size = utils.word_processing(config)
    config.args.vocab_size = vocab_size
    config.log.info("=> Finish Word Processing")

    # get questions (question dictionary: id -> python array pair (title, body))
    src_i2q = utils.get_questions(config, w2i)
    tgt_i2q = utils.get_questions_for_android(config, w2i)
    config.log.info("=> Finish Retrieving Questions")


    src_train_data = utils.QRDataset(config, config.args.train_file, w2i, vocab_size, src_i2q, is_train=True)
    src_train_loader = torch.utils.data.DataLoader(src_train_data, batch_size=config.args.batch_size, shuffle=True, **config.kwargs)
    tgt_train_data = utils.QuestionList(config.args.question_file)
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=config.args.batch_size, **config.kwargs)
    config.log.info("=> Building Dataset: Finish Train")

    
    dev_data = utils.AndroidDataset(config, config.args.dev_file_for_android, w2i, vocab_size)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=1024, **config.kwargs)  # TODO(demi): make test/dev batch size super big
    test_data = utils.AndroidDataset(config, config.args.test_file_for_android, w2i, vocab_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024, **config.kwargs)  # TODO(demi): make test/dev batch size super big
    config.log.info("=> Building Dataset: Finish All")


    if config.args.model_type == "CNN":
        config.log.info("=> Running CNN Model")
        encoder = myCNN(config)
    else:
        config.log.info("=> Running LSTM Model")
        encoder = myLSTM(config)

    discriminator = myFNN(config)

    if config.use_cuda:
        encoder = encoder.cuda()
        discriminator = discriminator.cuda()

    optimizer1 = optim.Adam(encoder.get_train_parameters(), lr=config.args.init_lr, weight_decay=1e-8)
    optimizer2 = optim.Adam(discriminator.get_train_parameters(), lr=-config.args.init_lr, weight_decay=1e-8)

    best_epoch = -1
    best_dev_auc = -1
    best_test_auc = -1

    for epoch in tqdm(range(config.args.epochs), desc="Running"):
        encoder, discriminator, optimizer1, optimizer2, avg_loss = train(config, encoder, discriminator, optimizer1, optimizer2, src_train_loader, tgt_train_loader, src_i2q, tgt_i2q)

        dev_auc = evaluate_for_android(encoder, dev_loader, tgt_i2q)
        test_auc = evaluate_for_android(encoder, test_loader, tgt_i2q)

        config.log.info("EPOCH[%d] Train Loss %.3lf" % (epoch, avg_loss))
        config.log.info("EPOCH[%d] ANDROID DEV: AUC %.3lf" % (epoch, dev_auc))
        config.log.info("EPOCH[%d] ANDROID TEST: AUC %.3lf" % (epoch, test_auc))
        if dev_auc > best_dev_auc:
            best_epoch = epoch
            best_dev_auc = dev_auc
            best_test_auc = test_auc
            config.log.info("=> Update Best Epoch to %d, based on Android Dev Score" % best_epoch)
            config.log.info("=> Update Model: Dev AUC %.3lf || Test AUC %.3lf || Saved at %s " % (best_dev_auc, best_test_auc, "%s-epoch%d" % (config.args.model_file, epoch)))

        def save_checkpoint():
            checkpoint = {"encoder":encoder.state_dict(), 
                          "discriminator":discriminator.state_dict(),
                          "optimizer1":optimizer1.state_dict(),
                          "optimizer2":optimizer2.state_dict(),
                          "auc": "Dev AUC %.3lf || Test AUC %.3lf" % (dev_auc, test_auc),
                          "args":config.args}
            checkpoint_file = "%s-epoch%d" % (config.args.model_file, epoch)
            config.log.info("=> saving checkpoint @ epoch %d to %s" % (epoch, checkpoint_file))
            torch.save(checkpoint, checkpoint_file)
        save_checkpoint()

    config.log.info("=> Best Model: Dev AUC %.3lf || Test AUC %.3lf || Saved at %s " % (best_dev_auc, best_test_auc, "%s-epoch%d" % (config.args.model_file, epoch)))
