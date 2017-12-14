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
from model import myCNN, myLSTM, myMLP
from evaluate import evaluate_for_android, evaluate_for_qr

""" Train: return model, optimizer """
def train(config, src_encoder, tgt_encoder, discriminator, encoder_optimizer, discriminator_optimizer, \
        src_data_loader, tgt_data_loader, src_i2q, tgt_i2q):

    src_encoder.eval()
    tgt_encoder.train()
    discriminator.train()

    avg_loss = 0
    total = 0

    zipped_data_loader = zip(src_data_loader, tgt_data_loader)
    max_iteration_per_epoch = min(len(src_data_loader), len(tgt_data_loader))
    for batch_idx, ((src_qid), (tgt_qid)) in tqdm(enumerate(zipped_data_loader), desc="Training"):
        # qid: batch_size (tensor)

        batch_size = min(src_qid.size(0), tgt_qid.size(0))
        total += batch_size
        if src_qid.size() != (batch_size,) or tgt_qid.size() != (batch_size,): 
            src_qid = src_qid[:batch_size]
            tgt_qid = tgt_qid[:batch_size]
            #print(batch_size)

        """ Retrieve Question Text """

        # get question title and body from SOURCE domain
        q_title = torch.zeros((batch_size, config.args.max_title_len)).long()
        q_body = torch.zeros((batch_size, config.args.max_body_len)).long()
        q_title_len = torch.zeros((batch_size)).long()
        q_body_len = torch.zeros((batch_size)).long()
        
        # get question title and body from TARGET domain
        tgt_q_title = torch.zeros((batch_size, config.args.max_title_len)).long()
        tgt_q_body = torch.zeros((batch_size, config.args.max_body_len)).long()
        tgt_q_title_len = torch.zeros((batch_size)).long()
        tgt_q_body_len = torch.zeros((batch_size)).long()

        for i in range(batch_size):
            #print "qid=", qid[i]
            #print "i2q[qid[i]]=", i2q[qid[i]]
            #embed()
            t, b, t_len, b_len = src_i2q[src_qid[i]]
            q_title[i] = torch.LongTensor(t)
            q_body[i] = torch.LongTensor(b)
            q_title_len[i] = t_len
            q_body_len[i] = b_len
        
            t, b, t_len, b_len = tgt_i2q[tgt_qid[i]]
            tgt_q_title[i] = torch.LongTensor(t)
            tgt_q_body[i] = torch.LongTensor(b)
            tgt_q_title_len[i] = t_len
            tgt_q_body_len[i] = b_len

        q_title = autograd.Variable(q_title)
        q_body = autograd.Variable(q_body)
        q_title_len = autograd.Variable(q_title_len)
        q_body_len = autograd.Variable(q_body_len)
        
        tgt_q_title = autograd.Variable(tgt_q_title)
        tgt_q_body = autograd.Variable(tgt_q_body)
        tgt_q_title_len = autograd.Variable(tgt_q_title_len)
        tgt_q_body_len = autograd.Variable(tgt_q_body_len)
       
        if config.use_cuda:
            q_title, q_title_len, q_body, q_body_len = q_title.cuda(), q_title_len.cuda(), q_body.cuda(), q_body_len.cuda()
            tgt_q_title, tgt_q_title_len, tgt_q_body, tgt_q_body_len = tgt_q_title.cuda(), tgt_q_title_len.cuda(), tgt_q_body.cuda(), tgt_q_body_len.cuda()

        """ Retrieve Question Embeddings """

        src_q_emb = 0.5 * (src_encoder(q_title, q_title_len) + src_encoder(q_body, q_body_len))
        assert src_q_emb.size() == (batch_size, config.args.final_dim)

        tgt_q_emb = 0.5 * (tgt_encoder(tgt_q_title, tgt_q_title_len) + tgt_encoder(tgt_q_body, tgt_q_body_len))
        assert tgt_q_emb.size() == (batch_size, config.args.final_dim)

        """ Train Discriminator """
        discriminator_optimizer.zero_grad()

        # concat features of src_q and tgt_q
        concat_q_emb = torch.cat((src_q_emb, tgt_q_emb), dim=0)
        
        # get predicted domain labels
        domain_label_pred = discriminator(concat_q_emb.detach())
        
        # create ground-truth domain labels
        domain_label_src = autograd.Variable(torch.ones(src_q_emb.size(0)).long())
        domain_label_tgt = autograd.Variable(torch.zeros(tgt_q_emb.size(0)).long())
        if config.use_cuda:
            domain_label_src, domain_label_tgt = domain_label_src.cuda(), domain_label_tgt.cuda()
        concat_domain_label = torch.cat((domain_label_src, domain_label_tgt), dim=0)
        
        # calculate loss for discriminator
        loss_discriminator = torch.nn.CrossEntropyLoss()(domain_label_pred, concat_domain_label)
        loss_discriminator.backward()
        
        # optimize discriminator
        discriminator_optimizer.step()

        # calculate accuracy of discriminator
        # NOTE(jason): it is reasonable to conjure that the acc should be 50%, since we 
        # want to deceive the discriminator
        domain_label_pred_cls = torch.squeeze(domain_label_pred.max(1)[1])
        accuracy = (domain_label_pred_cls == concat_domain_label).float().mean()

        """ Train Target Encoder """
        discriminator_optimizer.zero_grad()
        encoder_optimizer.zero_grad()

        # NOTE(jason): I re-compute features of target domain because I'm not sure 
        # if is correct to use pre-computed features
        tgt_q_emb = 0.5 * (tgt_encoder(tgt_q_title, tgt_q_title_len) + tgt_encoder(tgt_q_body, tgt_q_body_len))
        assert tgt_q_emb.size() == (batch_size, config.args.final_dim)

        target_domain_label_pred = discriminator(tgt_q_emb)
        # create fake target label (0 -> 1) to optimize tgt_encoder
        fake_label_tgt = autograd.Variable(torch.ones(tgt_q_emb.size(0)).long())
        if config.use_cuda:
            fake_label_tgt = fake_label_tgt.cuda()

        # calculate loss for tgt_encoder
        loss_encoder = torch.nn.CrossEntropyLoss()(target_domain_label_pred, fake_label_tgt)
        loss_encoder.backward()

        # optimize tgt_encoder
        encoder_optimizer.step()
        
        if ((batch_idx + 1) % config.args.log_step == 0):
            print('Iter: {}/{}, dis_loss: {:.3f}, enc_loss: {:.3f}, dis_acc: {:.3f}'
                    .format(batch_idx + 1, max_iteration_per_epoch, 
                        loss_discriminator.data[0],
                        loss_encoder.data[0],
                        accuracy.data[0]))

        # NOTE(jason): I'm not sure which loss should be returned
        avg_loss += 0 * batch_size
    avg_loss /= total
    return tgt_encoder, discriminator, encoder_optimizer, discriminator_optimizer, avg_loss


if __name__ == "__main__":
    config = Config()
    config.get_config_from_user()
    config.log.info("=> Finish Loading Configuration")


    # word processing (w2i, i2w, i2v)
    w2i, i2v, vocab_size = utils.word_processing(config)
    config.args.vocab_size = vocab_size
    config.log.info("=> Finish Word Processing")

    # get questions (question dictionary: id -> python array pair (title, body))
    src_i2q = utils.get_questions(config, w2i)
    tgt_i2q = utils.get_questions_for_android(config, w2i, lower=True)
    config.log.info("=> Finish Retrieving Questions")

    # create dataset
    src_train_data = utils.QuestionList(config.args.question_file)
    tgt_train_data = utils.QuestionList(config.args.question_file_for_android)
    config.log.info("=> Building Dataset: Finish Train")
    
    src_dev_data = utils.QRDataset(config, config.args.dev_file, w2i, vocab_size, src_i2q, is_train=False)
    dev_data = utils.AndroidDataset(config, config.args.dev_file_for_android, w2i, vocab_size)
    config.log.info("=> Building Dataset: Finish Dev")
    
    src_test_data = utils.QRDataset(config, config.args.test_file, w2i, vocab_size, src_i2q, is_train=False)
    test_data = utils.AndroidDataset(config, config.args.test_file_for_android, w2i, vocab_size)
    config.log.info("=> Building Dataset: Finish Test")
    
    src_train_loader = torch.utils.data.DataLoader(src_train_data, batch_size=config.args.batch_size, shuffle=True, **config.kwargs)
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=config.args.batch_size, shuffle=True, **config.kwargs)
    src_dev_loader = torch.utils.data.DataLoader(src_dev_data, batch_size=256, **config.kwargs)  # TODO(demi): make test/dev batch size super big
    src_test_loader = torch.utils.data.DataLoader(src_test_data, batch_size=256, **config.kwargs)  # TODO(demi): make test/dev batch size super big

    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=256, **config.kwargs)  # TODO(demi): make test/dev batch size super big
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, **config.kwargs)  # TODO(demi): make test/dev batch size super big
    config.log.info("=> Building Dataset: Finish All")

    if config.args.model_type == "CNN":
        config.log.info("=> Running CNN Model")
        src_encoder = myCNN(config)
        tgt_encoder = myCNN(config)
    else:
        config.log.info("=> Running LSTM Model")
        src_encoder = myLSTM(config)
        tgt_encoder = myLSTM(config)
    
    config.log.info("=> Loading Pre-trained Weights for Source and Target Encoder")
    src_encoder = utils.init_model(src_encoder, config.args.pretrained_encoder)
    tgt_encoder = utils.init_model(tgt_encoder, config.args.pretrained_encoder)
    
    config.log.info("=> Running MLP Model for discriminator")
    discriminator = myMLP(config)
    
    if config.use_cuda:
        src_encoder = src_encoder.cuda()
        tgt_encoder = tgt_encoder.cuda()
        discriminator = discriminator.cuda()

    # Evaluate pre-trained source encoder on source domain
    dev_AUC, dev_MAP, dev_MRR, dev_P1, dev_P5 = evaluate_for_qr(src_encoder, src_dev_loader, src_i2q, config)
    test_AUC, test_MAP, test_MRR, test_P1, test_P5 = evaluate_for_qr(src_encoder, src_test_loader, src_i2q, config)
    config.log.info("SOURCE DEV: AUC %.3lf MAP %.3lf MRR %.3lf P@1 %.3lf P@5 %.3lf" % (dev_AUC, dev_MAP, dev_MRR, dev_P1, dev_P5))
    config.log.info("SOUTCE TEST: AUC %.3lf MAP %.3lf MRR %.3lf P@1 %.3lf P@5 %.3lf" % (test_AUC, test_MAP, test_MRR, test_P1, test_P5))
    
    dev_AUC = evaluate_for_android(tgt_encoder, dev_loader, tgt_i2q, config)
    test_AUC = evaluate_for_android(tgt_encoder, test_loader, tgt_i2q, config)
    config.log.info("TARGET Dev AUC(.05): %.3lf || Test AUC(.05): %.3lf" % (dev_AUC, test_AUC))
    
    # NOTE(jason): Here suppose we have had source_encoder, we need to train a target_encoder with a domain discriminator
    # The target encoder will be trained using encoder_optimizer, while the domain discriminator will be trained using 
    # discriminator_optimizer.
    
    encoder_optimizer = optim.Adam(tgt_encoder.get_train_parameters(), lr=1e-4, weight_decay=1e-9)
    discriminator_optimizer = optim.Adam(discriminator.get_train_parameters(), lr=1e-4, weight_decay=1e-9)

    for epoch in tqdm(range(config.args.epochs), desc="Running"):
            tgt_encoder, discriminator, encoder_optimizer, discriminator_optimizer, avg_loss = train(config, \
                    src_encoder, tgt_encoder, discriminator, encoder_optimizer, discriminator_optimizer, \
                    src_train_loader, tgt_train_loader, src_i2q, tgt_i2q)
            
            if (epoch + 1) % 10 == 0:
                dev_AUC = evaluate_for_android(tgt_encoder, dev_loader, tgt_i2q, config)
                test_AUC = evaluate_for_android(tgt_encoder, test_loader, tgt_i2q, config)
                
                # TODO(demi): change this, so only evaluate test on best dev model
                config.log.info("EPOCH[%d] Train Loss %.3lf" % (epoch, avg_loss))
                config.log.info("EPOCH[%d] Dev AUC: %.3lf || Test AUC: %.3lf" % (epoch, dev_AUC, test_AUC))

            def save_checkpoint():
                pass
                '''
                checkpoint = {"model":model.state_dict(), 
                              "optimizer":optimizer.state_dict(),
                              "dev_eval":"Dev AUC %.3lf" % (dev_AUC),
                              "test_eval":"Test AUC %.3lf" % (test_AUC),
                              "args":config.args}
                checkpoint_file = config.args.model_file
                config.log.info("=> saving checkpoint @ epoch %d to %s" % (epoch, checkpoint_file))
                torch.save(checkpoint, checkpoint_file)
                '''
            
            save_checkpoint()
            # TODO(demi): save checkpoint
