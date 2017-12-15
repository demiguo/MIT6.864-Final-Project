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

    acc_total = 0
    avg_acc = 0

    combined_data_loader = zip(src_data_loader, tgt_data_loader)
    max_iteration_per_epoch = min(len(src_data_loader), len(tgt_data_loader))

    for batch_idx, (src_batch, tgt_batch) in tqdm(enumerate(combined_data_loader), desc="Training"):

        """ Retrieve Question Text """

        src_batch_size = src_batch.size(0)

        src_title = torch.zeros((src_batch_size, config.args.max_title_len)).long()
        src_body = torch.zeros((src_batch_size, config.args.max_body_len)).long()
        src_title_len = torch.zeros((src_batch_size)).long()
        src_body_len = torch.zeros((src_batch_size)).long()
        for i in range(src_batch_size):
            t, b, t_len, b_len = src_i2q[src_batch[i]]
            src_title[i] = torch.LongTensor(t)
            src_body[i] = torch.LongTensor(b)
            src_title_len[i] = t_len
            src_body_len[i] = b_len
        src_title = autograd.Variable(src_title)
        src_body = autograd.Variable(src_body)
        src_title_len = autograd.Variable(src_title_len)
        src_body_len = autograd.Variable(src_body_len)
        if config.use_cuda:
            src_title, src_title_len, src_body, src_body_len = src_title.cuda(), src_title_len.cuda(), src_body.cuda(), src_body_len.cuda()
       
        tgt_batch_size = tgt_batch.size(0)

        tgt_title = torch.zeros((tgt_batch_size, config.args.max_title_len)).long()
        tgt_body = torch.zeros((tgt_batch_size, config.args.max_body_len)).long()
        tgt_title_len = torch.zeros((tgt_batch_size)).long()
        tgt_body_len = torch.zeros((tgt_batch_size)).long()
        for i in range(tgt_batch_size):
            t, b, t_len, b_len = src_i2q[src_batch[i]]
            tgt_title[i] = torch.LongTensor(t)
            tgt_body[i] = torch.LongTensor(b)
            tgt_title_len[i] = t_len
            tgt_body_len[i] = b_len
        tgt_title = autograd.Variable(tgt_title)
        tgt_body = autograd.Variable(tgt_body)
        tgt_title_len = autograd.Variable(tgt_title_len)
        tgt_body_len = autograd.Variable(tgt_body_len)
        if config.use_cuda:
            tgt_title, tgt_title_len, tgt_body, tgt_body_len = tgt_title.cuda(), tgt_title_len.cuda(), tgt_body.cuda(), tgt_body_len.cuda()

        src_emb = 0.5 * (encoder(src_title, src_title_len) + encoder(src_body, src_body_len))
        tgt_emb = 0.5 * (encoder(tgt_title, tgt_title_len) + encoder(tgt_body, tgt_body_len))

        src_target = torch.autograd.Variable(torch.zeros((src_batch_size)).long())
        tgt_target = torch.autograd.Variable(torch.ones((tgt_batch_size)).long())
        if config.use_cuda:
            src_target = src_target.cuda()
            tgt_target = tgt_target.cuda()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        src_loss, src_acc = discriminator.loss(src_emb, src_target, acc=True)
        tgt_loss, tgt_acc = discriminator.loss(tgt_emb, tgt_target, acc=True)
        loss = src_loss + tgt_loss
        avg_loss += loss.data[0]
        total += 1
        loss.backward()

        torch.nn.utils.clip_grad_norm(encoder.get_train_parameters(), config.args.max_norm)
        
        #if batch_idx % 10 == 0:
        
            #embed()
        optimizer1.step()
        optimizer2.step()

        avg_acc += src_acc * src_batch_size + tgt_acc * tgt_batch_size
        acc_total += src_batch_size + tgt_batch_size

    avg_loss /= total  
    avg_acc /= acc_total
    return encoder, discriminator, optimizer1, optimizer2, avg_loss, avg_acc


def evaluate(config, encoder, discriminator, src_data_loader, tgt_data_loader, src_i2q, tgt_i2q):
    encoder.eval()
    discriminator.eval()

    acc_total = 0
    avg_acc = 0

    combined_data_loader = zip(src_data_loader, tgt_data_loader)
    max_iteration_per_epoch = min(len(src_data_loader), len(tgt_data_loader))

    for batch_idx, (src_batch, tgt_batch) in tqdm(enumerate(combined_data_loader), desc="Evaluating"):

        """ Retrieve Question Text """

        src_batch_size = src_batch.size(0)

        src_title = torch.zeros((src_batch_size, config.args.max_title_len)).long()
        src_body = torch.zeros((src_batch_size, config.args.max_body_len)).long()
        src_title_len = torch.zeros((src_batch_size)).long()
        src_body_len = torch.zeros((src_batch_size)).long()
        for i in range(src_batch_size):
            t, b, t_len, b_len = src_i2q[src_batch[i]]
            src_title[i] = torch.LongTensor(t)
            src_body[i] = torch.LongTensor(b)
            src_title_len[i] = t_len
            src_body_len[i] = b_len
        src_title = autograd.Variable(src_title)
        src_body = autograd.Variable(src_body)
        src_title_len = autograd.Variable(src_title_len)
        src_body_len = autograd.Variable(src_body_len)
        if config.use_cuda:
            src_title, src_title_len, src_body, src_body_len = src_title.cuda(), src_title_len.cuda(), src_body.cuda(), src_body_len.cuda()
       
        tgt_batch_size = tgt_batch.size(0)

        tgt_title = torch.zeros((tgt_batch_size, config.args.max_title_len)).long()
        tgt_body = torch.zeros((tgt_batch_size, config.args.max_body_len)).long()
        tgt_title_len = torch.zeros((tgt_batch_size)).long()
        tgt_body_len = torch.zeros((tgt_batch_size)).long()
        for i in range(tgt_batch_size):
            t, b, t_len, b_len = src_i2q[src_batch[i]]
            tgt_title[i] = torch.LongTensor(t)
            tgt_body[i] = torch.LongTensor(b)
            tgt_title_len[i] = t_len
            tgt_body_len[i] = b_len
        tgt_title = autograd.Variable(tgt_title)
        tgt_body = autograd.Variable(tgt_body)
        tgt_title_len = autograd.Variable(tgt_title_len)
        tgt_body_len = autograd.Variable(tgt_body_len)
        if config.use_cuda:
            tgt_title, tgt_title_len, tgt_body, tgt_body_len = tgt_title.cuda(), tgt_title_len.cuda(), tgt_body.cuda(), tgt_body_len.cuda()

        src_emb = 0.5 * (encoder(src_title, src_title_len) + encoder(src_body, src_body_len))
        tgt_emb = 0.5 * (encoder(tgt_title, tgt_title_len) + encoder(tgt_body, tgt_body_len))

        src_target = torch.autograd.Variable(torch.zeros((src_batch_size)).long())
        tgt_target = torch.autograd.Variable(torch.zeros((tgt_batch_size)).long())
        if config.use_cuda:
            src_target = src_target.cuda()
            tgt_target = tgt_target.cuda()

        src_loss, src_acc = discriminator.loss(src_emb, src_target, acc=True)
        tgt_loss, tgt_acc = discriminator.loss(tgt_emb, tgt_target, acc=True)
        
        avg_acc += src_acc * src_batch_size + tgt_acc * tgt_batch_size
        acc_total += src_batch_size + tgt_batch_size

    avg_acc /= acc_total
    return avg_acc


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

    src_train_data = utils.QuestionList("../../data/domain_classifier/src.train")
    src_train_loader = torch.utils.data.DataLoader(src_train_data, batch_size=config.args.src_batch_size, shuffle=True, **config.kwargs)
    tgt_train_data = utils.QuestionList("../../data/domain_classifier/tgt.train")
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=config.args.tgt_batch_size, shuffle=True, **config.kwargs)
    
    src_test_data = utils.QuestionList("../../data/domain_classifier/src.test")
    src_test_loader = torch.utils.data.DataLoader(src_test_data, batch_size=1024, **config.kwargs)
    tgt_test_data = utils.QuestionList("../../data/domain_classifier/tgt.test")
    tgt_test_loader = torch.utils.data.DataLoader(tgt_test_data, batch_size=1024, **config.kwargs)

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

    optimizer1 = optim.Adam(encoder.get_train_parameters(), lr=config.args.init_src_lr, weight_decay=1e-8)
    optimizer2 = optim.Adam(discriminator.get_train_parameters(), lr=-config.args.init_tgt_lr, weight_decay=1e-8)


    for epoch in tqdm(range(config.args.epochs), desc="Running"):
        encoder, discriminator, optimizer1, optimizer2, avg_loss, avg_acc = \
                train(config, encoder, discriminator, optimizer1, optimizer2, src_train_loader, tgt_train_loader, src_i2q, tgt_i2q)
        
        test_acc = evaluate(config, encoder, discriminator, src_test_loader, tgt_test_loader, src_i2q, tgt_i2q)
        config.log.info("EPOCH[%d] Train Loss %.3lf || Discriminator Avg ACC %.3lf" % (epoch, avg_loss, avg_acc))
        config.log.info("EPOCH[%d] TEST: ACC %.3lf" % (epoch,test_acc))
        
        def save_checkpoint():
            checkpoint = {"encoder":encoder.state_dict(), 
                          "discriminator":discriminator.state_dict(),
                          "optimizer1":optimizer1.state_dict(),
                          "optimizer2":optimizer2.state_dict(),
                          "auc": "test acc %.3lf" % (test_acc),
                          "args":config.args}
            checkpoint_file = "%s-domain-classifier-epoch%d" % (config.args.model_file, epoch)
            config.log.info("=> saving checkpoint @ epoch %d to %s" % (epoch, checkpoint_file))
            torch.save(checkpoint, checkpoint_file)
        save_checkpoint()

    config.log.info("=> Best Model: Dev AUC %.3lf || Test AUC %.3lf || Saved at %s " % (best_dev_auc, best_test_auc, "%s-domain-classifier-epoch%d" % (config.args.model_file, epoch)))
