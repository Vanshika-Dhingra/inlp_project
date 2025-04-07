import math
import pickle
import sys
import time
import copy
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import wandb
import torch.nn.utils.prune as prune

from utils import read_corpus, batch_iter, LabelSmoothingLoss
from vocab import Vocab, VocabEntry

def get_layers(model):
    arr =[]
    for i,j in model.named_parameters():
        a = i.split('.')
        arr.append(tuple(a))
        
    layers = []
    for name, weight in arr:
        for i,j in model.named_children():
            if i == name:
                layers.append([j,weight])
    return layers