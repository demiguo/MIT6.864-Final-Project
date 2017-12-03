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


class myCNN(torch.nn.Module):
    def __init__(self, config):
        super(myCNN, self).__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim

        # CNN hyperparameters
        self.hidden_dim = config.hidden_dim
        self.final_dim = config.final_dim

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.cnn1 = nn.Conv1d(self.embedding_dim, self.final_dim, 3, 1)

        self.init_weight(config.pretrained_wordvec)

    def init_weight(self, pretrained_embedding_file):
        # update embeddings using pretrained embedding file
        word_vec = np.loadtxt(pretrained_embedding_file)
        self.word_embeds.weight.data.copy_(torch.FloatTensor(word_vec))
        self.word_embeds.weight.requires_grad = False

    def get_train_parameters(self):
        params = []
        for param in self.parameters():
            if param.requires_grad == True:
                params.append(param)
        return params

    def forward(self, text, text_len):
        self.batch_size, self.len = text.size()

        emb = self.word_embeds(text)
        assert emb.size() == (self.batch_size, self.len, self.embedding_dim), "1:emb.size()=%s" % str(emb.size())
        emb = torch.transpose(emb, (1, 2))
        assert emb.size() == (self.batch_size, self.embedding_dim, self.len), "2:emb.size()=%s" % str(emb.size())

        x = self.cnn1(emb)
        x = torch.nn.Tanh()(x)

        assert x.size(0) == self.batch_size and x.size(1) == self.final_dim, "1:x.size()=%s" % str(x.size())
        x = torch.sum(x, dim=2)
        assert x.size() == (self.batch_size, self.final_dim), "2:x.size()=%s" % str(x.size())

        text_len = text_len.view(self.batch_size, 1)
        text_len = text_len.repeat(1, self.final_dim)
        x = x / text_len.float()
        assert x.size() == (self.batch_size, self.final_dim), "3:x.size()=%s" % str(x.size())
        
        return x

class myLSTM(torch.nn.Module):
    def __init__(self, config):
        super(myLSTM, self).__init__()
        print "not implemented"

    def get_train_parameters(self):
        print "not implemented"

    def init_weight(self, pretrained_embedding_file):
        print "not implemented"

    def forward(self, text):
        print "not implemented"