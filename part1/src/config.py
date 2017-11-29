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


class Config:
    def init():
        args = {}
    def get_config_from_user(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-seed', '--seed', type=int, help="Torch Random Seed", required=False, default=1)
        # include vector file name
        # include questions file name
        self.args = vars(parser.parse_args())
