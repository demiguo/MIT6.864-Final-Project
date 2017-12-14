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



class myFNN(torch.nn.Module):
    def __init__(self, config):
        super(myFNN, self).__init__()
        self.config = config
        self.input_dim = config.args.final_dim
        self.hidden_dim = config.args.discriminator_hidden_dim
        self.restored = False

        self.MLP = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
            nn.LogSoftmax(dim=1)
        ) 

    def get_train_parameters(self):
        params = []
        for param in self.parameters():
            if param.requires_grad == True:
                params.append(param)
        return params
    
    def forward(self, input):
        return self.MLP(input)

    def loss(self, input, target, acc=False):
        output = self(input)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        if not acc:
            return loss

        mean_acc = (output == target).float().mean().data[0]
        return loss, mean_acc
        
class myMLP(torch.nn.Module):
    def __init__(self, config):
        super(myMLP, self).__init__()
        self.config = config
        self.input_dim = config.args.final_dim
        self.hidden_dim = config.args.discriminator_hidden_dim
        self.restored = False

        self.MLP = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
            nn.LogSoftmax(dim=1)
        ) 

    def get_train_parameters(self):
        params = []
        for param in self.parameters():
            if param.requires_grad == True:
                params.append(param)
        return params
    
    def forward(self, input):
        return self.MLP(input)


class myCNN(torch.nn.Module):
    def __init__(self, config):
        super(myCNN, self).__init__()
        self.config = config
        self.vocab_size = config.args.vocab_size
        self.embedding_dim = config.args.embedding_dim

        # CNN hyperparameters
        self.final_dim = config.args.final_dim

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.cnn1 = nn.Conv1d(self.embedding_dim, self.final_dim, 3, 1)

        self.init_weight(config.args.pretrained_wordvec)

    def init_weight(self, pretrained_embedding_file):
        # update embeddings using pretrained embedding file
        word_vec = torch.FloatTensor(np.loadtxt(pretrained_embedding_file))
        if self.config.use_cuda:
            word_vec = word_vec.cuda()
        self.word_embeds.weight.data.copy_(word_vec)
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
        emb = torch.transpose(emb, 1, 2)
        assert emb.size() == (self.batch_size, self.embedding_dim, self.len), "2:emb.size()=%s" % str(emb.size())

        x = self.cnn1(emb)
        x = torch.nn.Tanh()(x)

        x = torch.sum(x, dim=2)

        text_len = text_len.view(self.batch_size, 1)
        text_len = text_len.expand(self.batch_size, self.final_dim)
        x = x / text_len.float()

        return x

class myLSTM(torch.nn.Module):
    def __init__(self, config):
        super(myLSTM, self).__init__()
        self.config = config
        self.vocab_size = config.args.vocab_size
        self.embedding_dim = config.args.embedding_dim
        self.final_dim = config.args.final_dim

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.final_dim // 2, num_layers=1, batch_first=True, bidirectional=True, dropout=0.1)

        self.init_weight(config.args.pretrained_wordvec)

    def init_weight(self, pretrained_embedding_file):
        # update embeddings using pretrained embedding file
        word_vec = torch.FloatTensor(np.loadtxt(pretrained_embedding_file))
        if self.config.use_cuda:
            word_vec = word_vec.cuda()
        self.word_embeds.weight.data.copy_(word_vec)
        self.word_embeds.weight.requires_grad = False

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.randn(2, batch_size, self.final_dim // 2))
        c = autograd.Variable(torch.randn(2, batch_size, self.final_dim // 2))
        if self.config.use_cuda:
            h, c = h.cuda(), c.cuda()
        return (h, c)

    def get_train_parameters(self):
        params = []
        for param in self.parameters():
            if param.requires_grad == True:
                params.append(param)
        return params

    """ Sort text data in nonincreasing order by length, and return new text, text_len """
    def preprocess(self, text, text_len):
        text = text.cpu()
        text_len = text_len.cpu()
        indices = np.argsort(-text_len.data.numpy())

        #print "text_len = ", text_len
        #print "indices=", indices
        #print "text_len.data.numpy=",text_len.data.numpy()

        new_text = np.zeros((self.batch_size, self.max_len), dtype=int)
        new_text_len = np.zeros((self.batch_size), dtype=int)

        for i in range(self.batch_size):
            new_text[i] = text.data[indices[i]].numpy()
            new_text_len[i] = text_len.data[indices[i]]

        new_text = torch.autograd.Variable(torch.from_numpy(new_text))
        new_text_len = torch.autograd.Variable(torch.from_numpy(new_text_len))

        if self.config.use_cuda:
            new_text, new_text_len = new_text.cuda(), new_text_len.cuda()
        return new_text, new_text_len, indices


    def forward(self, text, text_len):
        self.batch_size, self.max_len = text.size()
        text, text_len, indices = self.preprocess(text, text_len)
        text_len_list = text_len.cpu().data.numpy().tolist()

        # Model
        emb = self.word_embeds(text)
        assert emb.size() == (self.batch_size, self.max_len, self.embedding_dim)

        #print("emb=", emb)
        #print("text_len_list=", text_len_list)
        pack_emb_input = torch.nn.utils.rnn.pack_padded_sequence(emb, text_len_list, batch_first=True)
        #self.hidden = self.init_hidden(self.batch_size)
        lstm_out, _  = self.lstm(pack_emb_input)

        #print "lstm_out=", lstm_out
        pad_lstm_out, lstm_lens = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # NB(demi): in theory lstm_lens should be the same as text_len
        
        #print "self.batch_size=", self.batch_size
        #print "self.max_len=", self.max_len
        #print "self.final_dim=", self.final_dim
        #print "pad_lstm_out.size=", pad_lstm_out.size()
        # TODO(demi): do we want to add nonlinearity here?

        # Average Pooling
        outputs = torch.sum(pad_lstm_out, dim=1)
        assert outputs.size() == (self.batch_size, self.final_dim)

        text_len = text_len.view(self.batch_size, 1)
        text_len = text_len.expand(self.batch_size, self.final_dim)
        outputs = outputs / text_len.float()
        
        inverse_indices = [0] * self.batch_size
        for i in range(self.batch_size):
            inverse_indices[indices[i]] = i
        inverse_indices = torch.LongTensor(inverse_indices)
        if self.config.use_cuda:
            inverse_indices = inverse_indices.cuda()

        inverse_outputs = outputs[inverse_indices]
        return inverse_outputs

