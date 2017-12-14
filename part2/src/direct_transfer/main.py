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
from model import myCNN, myLSTM
from direct_transfer import evaluate_for_android

""" Train: return model, optimizer """
def train(config, model, optimizer, data_loader, i2q):
    # TODO(demi): currently, this only works for CNN model. In the future, make it compatible for LSTM model.
    model.train()

    avg_loss = 0
    total = 0
    for batch_idx, (qid, similar_q, candidate_q, label, similar_num, candidate_num) in tqdm(enumerate(data_loader), desc="Training"):
        # qid: batch_size (tensor)
        # similar_q: batch_size * num_similar_q (tensor)
        # candidate_q: batch_size * 20 (tensor)
        # label: batch_size * 20 (tensor)
        num_similar_q = 1
        num_candidate_q = 20

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


        """ Calculate Loss """
        optimizer.zero_grad()
        # TODO(demi): make it batch operations
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
        assert neg_scores.size() == (batch_size, num_candidate_q)

        loss = neg_scores - pos_score + config.args.delta_constant
        assert loss.size() == (batch_size, num_candidate_q)
        loss = torch.max(loss, dim=1)[0].view(batch_size,1)
        loss = (loss > 0).float() * loss
        loss = torch.mean(loss) # mean batch loss

        """
        DEPRECATE: non-batch way to calculate max-margin loss
        max_margin_loss = autograd.Variable(torch.zeros((batch_size, num_candidate_q+1)))
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
            max_margin_loss[i][num_candidate_q] += 0 # positive example loss
        max_margin_loss = torch.max(max_margin_loss, dim=1)[0].view(batch_size)
        loss = torch.mean(max_margin_loss)
        """
        avg_loss += loss.data[0] * batch_size
        loss.backward()
        optimizer.step()
    avg_loss /= total
    return model, optimizer, avg_loss


""" Evaluate: return model """
def evaluate(model, optimizer, data_loader, i2q):
    model.eval()
    # TODO(demi): currently, this only works for CNN model. In the future, make it compatible for LSTM model.

    total = 0

    MAP = 0
    MRR = 0
    P1 = 0
    P5 = 0
    for batch_idx, (qid, similar_q, candidate_q, label, similar_num, candidate_num) in tqdm(enumerate(data_loader), desc="Evaluating"):
        # qid: batch_size (tensor)
        # similar_q: batch_size * num_similar_q (tensor)
        # candidate_q: batch_size * 20 (tensor)
        # label: batch_size * 20 (tensor)
        num_candidate_q = 20

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


        # get candidate question title and body
        candidate_title = torch.zeros((batch_size, num_candidate_q, config.args.max_title_len)).long()
        candidate_body = torch.zeros((batch_size, num_candidate_q, config.args.max_body_len)).long()
        candidate_title_len = torch.zeros((batch_size, num_candidate_q)).long()
        candidate_body_len = torch.zeros((batch_size, num_candidate_q)).long()
        for i in range(batch_size):
            l = candidate_num[i]
            assert l == num_candidate_q
            candidate_ids = list(range(l))
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

        candidate_title = candidate_title.contiguous().view(batch_size * num_candidate_q, config.args.max_title_len)
        candidate_body = candidate_body.contiguous().view(batch_size * num_candidate_q, config.args.max_body_len)
        candidate_title_len = candidate_title_len.contiguous().view(batch_size * num_candidate_q)
        candidate_body_len = candidate_body_len.contiguous().view(batch_size * num_candidate_q)
        candidate_emb = 0.5 * (model(candidate_title, candidate_title_len) + model(candidate_body, candidate_body_len))
        candidate_emb = candidate_emb.contiguous().view(batch_size, num_candidate_q, config.args.final_dim)


        """ Compute Metrics """
        d = config.args.final_dim
        q_emb = q_emb.view(batch_size, 1, d)
        q_expand_emb = q_emb.expand(batch_size, num_candidate_q, d)
        assert q_expand_emb.size() == candidate_emb.size()

        scores = nn.CosineSimilarity(dim=2, eps=1e-6)(q_expand_emb, candidate_emb)
        assert scores.size() == (batch_size, num_candidate_q)
        scores = scores.cpu().data.numpy()
        assert scores.shape == (batch_size, num_candidate_q)

        # TODO(demi): move these metrics calculation to other files
        for batch_id in range(batch_size):
            batch_scores = scores[batch_id]
            batch_ranks = np.argsort(-batch_scores)

            # MAP
            batch_MAP = 0
            correct = 0
            tmp = []
            for i in range(num_candidate_q):
                idx = batch_ranks[i]
                assert label[batch_id][idx] == 0 or label[batch_id][idx] == 1
                if label[batch_id][idx] == 1:
                    correct += 1
                    tmp.append(1.0 * correct / (i + 1))
            tmp = np.array(tmp)
            batch_MAP = 0 if np.size(tmp) == 0 else tmp.mean()
            MAP += batch_MAP

            # MRR
            first_correct = num_candidate_q + 1
            for i in range(num_candidate_q):
                idx = batch_ranks[i]
                if label[batch_id][idx] == 1:
                    first_correct = i + 1
                    break
            MRR += 1.0 / first_correct

            # P@1
            batch_P1 = 0
            for i in range(1):
                idx = batch_ranks[i]
                if label[batch_id][idx] == 1:
                    batch_P1 += 1
            batch_P1 /= 1.0
            P1 += batch_P1

            # P@5
            batch_P5 = 0
            for i in range(5):
                idx = batch_ranks[i]
                if label[batch_id][idx] == 1:
                    batch_P5 += 1
            batch_P5 /= 5.0
            P5 += batch_P5


    return MAP / total, MRR / total, P1 / total, P5 / total


if __name__ == "__main__":
    config = Config()
    config.get_config_from_user()
    config.log.info("=> Finish Loading Configuration")


    # word processing (w2i, i2w, i2v)
    w2i, i2v, vocab_size = utils.word_processing_glove(config)
    config.args.vocab_size = vocab_size
    config.log.info("=> Finish Word Processing")

    # get questions (question dictionary: id -> python array pair (title, body))
    i2q = utils.get_questions(config, w2i)
    i2q_for_android = utils.get_questions_for_android(config, w2i)
    config.log.info("=> Finish Retrieving Questions")


    # create dataset
    train_data = utils.QRDataset(config, config.args.train_file, w2i, vocab_size, i2q, is_train=True)
    config.log.info("=> Building Dataset: Finish Train")
    dev_data = utils.QRDataset(config, config.args.dev_file, w2i, vocab_size, i2q, is_train=False)
    config.log.info("=> Building Dataset: Finish Dev")
    test_data = utils.QRDataset(config, config.args.test_file, w2i, vocab_size, i2q, is_train=False)
    config.log.info("=> Building Dataset: Finish Test")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.args.batch_size, shuffle=True, **config.kwargs)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=1024, **config.kwargs)  # TODO(demi): make test/dev batch size super big
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024, **config.kwargs)  # TODO(demi): make test/dev batch size super big
    
    dev_data_for_android = utils.AndroidDataset(config, config.args.dev_file_for_android, w2i, vocab_size)
    dev_loader_for_android = torch.utils.data.DataLoader(dev_data, batch_size=1024, **config.kwargs)  # TODO(demi): make test/dev batch size super big
    test_data_for_android = utils.AndroidDataset(config, config.args.test_file_for_android, w2i, vocab_size)
    test_loader_for_android = torch.utils.data.DataLoader(test_data, batch_size=1024, **config.kwargs)  # TODO(demi): make test/dev batch size super big
    config.log.info("=> Building Dataset: Finish All")

    if config.args.model_type == "CNN":
        config.log.info("=> Running CNN Model")
        model = myCNN(config)
    else:
        config.log.info("=> Running LSTM Model")
        model = myLSTM(config)

    if config.use_cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.get_train_parameters(), lr=0.001, weight_decay=1e-8)
    best_epoch = -1
    best_dev_auc = -1
    best_test_auc = -1
    for epoch in tqdm(range(config.args.epochs), desc="Running"):
            model, optimizer, avg_loss = train(config, model, optimizer, train_loader, i2q)
            dev_MAP, dev_MRR, dev_P1, dev_P5 = evaluate(model, optimizer, dev_loader, i2q)
            test_MAP, test_MRR, test_P1, test_P5 = evaluate(model, optimizer, test_loader, i2q)

            dev_auc = evaluate_for_android(model, dev_loader_for_android, i2q_for_android)
            test_auc = evaluate_for_android(model, test_loader_for_android, i2q_for_android)

            config.log.info("EPOCH[%d] Train Loss %.3lf" % (epoch, avg_loss))
            config.log.info("EPOCH[%d] DEV: MAP %.3lf MRR %.3lf P@1 %.3lf P@5 %.3lf" % (epoch, dev_MAP, dev_MRR, dev_P1, dev_P5))
            config.log.info("EPOCH[%d] TEST: MAP %.3lf MRR %.3lf P@1 %.3lf P@5 %.3lf" % (epoch, test_MAP, test_MRR, test_P1, test_P5))
            config.log.info("EPOCH[%d] ANDROID DEV: AUC %.3lf" % (epoch, dev_auc))
            config.log.info("EPOCH[%d] ANDROID TEST: AUC %.3lf" % (epoch, test_auc))
            if dev_auc > best_dev_auc:
                best_epoch = epoch
                best_dev_auc = dev_auc
                best_test_auc = test_auc
                config.log.info("=> Update Best Epoch to %d, based on Android Dev Score" % best_epoch)
                config.log.info("=> Update Model: Dev AUC %.3lf || Test AUC %.3lf || Saved at %s " % (best_dev_auc, best_test_auc, "%s-epoch%d" % (config.args.model_file, epoch)))


            def save_checkpoint():
                checkpoint = {"model":model.state_dict(), 
                              "optimizer":optimizer.state_dict(),
                              "dev_eval":"MAP %.3lf MRR %.3lf P@1 %.3lf P@5 %.3lf" % (dev_MAP, dev_MRR, dev_P1, dev_P5),
                              "test_eval":"MAP %.3lf MRR %.3lf P@1 %.3lf P@5 %.3lf" % (test_MAP, test_MRR, test_P1, test_P5),
                              "args":config.args}
                checkpoint_file = "%s-epcoh%d" % (config.args.model_file, epoch)
                config.log.info("=> saving checkpoint @ epoch %d to %s" % (epoch, checkpoint_file))
                torch.save(checkpoint, checkpoint_file)
            
            save_checkpoint()
    config.log.info("=> Best Model: Dev AUC %.3lf || Test AUC %.3lf || Saved at %s " % (best_dev_auc, best_test_auc, "%s-epoch%d" % (config.args.model_file, epoch)))
