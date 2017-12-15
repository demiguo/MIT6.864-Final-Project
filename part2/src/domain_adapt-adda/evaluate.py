import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data
from tqdm import tqdm

from meter import AUCMeter
from sklearn.metrics import roc_auc_score

""" Evaluate: AUC(0.05) on Android """
def evaluate_for_android(model, data_loader, i2q, config):
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

def evaluate_for_qr(model, data_loader, i2q, config):
    model.eval()
    # TODO(demi): currently, this only works for CNN model. In the future, make it compatible for LSTM model.

    total = 0

    MAP = 0
    MRR = 0
    P1 = 0
    P5 = 0
    AUC_label = []
    AUC_score = []
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

        """ Retrieve Question Embeddings """

        q_emb = 0.5 * (model(q_title, q_title_len)+ model(q_body, q_body_len))
        assert q_emb.size() == (batch_size, config.args.final_dim)

        candidate_title = candidate_title.contiguous().view(batch_size * num_candidate_q, config.args.max_title_len)
        candidate_body = candidate_body.contiguous().view(batch_size * num_candidate_q, config.args.max_body_len)
        candidate_title_len = candidate_title_len.contiguous().view(batch_size * num_candidate_q)
        candidate_body_len = candidate_body_len.contiguous().view(batch_size * num_candidate_q)
        if config.use_cuda:
            candidate_title, candidate_title_len, candidate_body, candidate_body_len =\
                candidate_title.cuda(), candidate_title_len.cuda(), candidate_body.cuda(), candidate_body_len.cuda()
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

            # AUC (global)
            for i in range(num_candidate_q):
                idx = batch_ranks[i]
                assert label[batch_id][idx] == 0 or label[batch_id][idx] == 1
                AUC_label.append(label[batch_id][idx])
                AUC_score.append(batch_scores[idx])

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

    AUC = roc_auc_score(np.array(AUC_label), np.array(AUC_score)) 
    return AUC, MAP / total, MRR / total, P1 / total, P5 / total

