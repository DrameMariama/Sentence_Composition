##credit goes to Guillaume
import argparse
import sys
import os
import fastBPE
import torch
from types import SimpleNamespace
import numpy as np
import subprocess
import tempfile
import torch.nn as nn

SOURCE_PATH = os.environ['HOME']

LASER_PATH = SOURCE_PATH + '/devfair_sentence_composition/wiki-split/Data/Laser_vectors/LASER' 
FAIRSEQ_PATH = SOURCE_PATH + '/devfair_sentence_composition/wiki-split/Data/Laser_vectors/fairseq-sentemb/'

LASER_V1 = '/checkpoint/mariama20/laser_models' # whereever laser_models is

sys.path.extend([SOURCE_PATH,
                 FAIRSEQ_PATH,
                 LASER_PATH + '/tools-external',
                 LASER_PATH + '/source',
                 LASER_PATH + '/source/tools'])

os.environ['LASER'] = LASER_PATH

from embed import Encoder, SentenceEncoder, EncodeLoad, EncodeFile
#from text_processing import Token, TokenLine, BPEfastApply

from fairseq import options, tasks, utils
import fairseq.mysg as sg
#import fairseq.sequence_generator as sg

model_path=LASER_V1 + '/bilstm.93langs.2018-12-26.seq2seq'
codes_path=LASER_V1 + '/laser.v1.fcodes'
vocab_path=LASER_V1 + '/laser.v1.fvocab'
cfg_path=LASER_V1 + '/bilstm.93langs.2018-12-26.conf'

args = SimpleNamespace(
                       task='multitask',
                       left_pad_target='False',
                       data=cfg_path,
                       input=model_path,
                       left_pad_source=True)

task = tasks.setup_task(args)

full_model = utils.load_ensemble_for_inference([args.input], tasks.setup_task(args))[0][0].cuda()

dct = task.dictionary
sgen = sg.SequenceGenerator([full_model], dct)

def beam_decode(emb, maxlen=70, beam_size=5):
    # returns only best
    # emb must be of shape bsize x 1024, on cuda
    out = []
    for e in emb:
        a = e.unsqueeze(0)
        print(a.repeat(beam_size, 1))
        toks = sgen.generate(e.unsqueeze(0), maxlen=maxlen, beam_size=beam_size)[0][0]['tokens']
        out.append(dct.string(toks, bpe_symbol='@@ '))
    return out

if __name__ == "__main__":
    
    dim = 1024
    #embed_test_data = np.fromfile('/private/home/mariama20/devfair_sentence_composition/wiki-split/Data/Laser_vectors/valid_complex_sent.raw', dtype=np.float32, count=-1)
    embed_test_data = np.loadtxt('/private/home/mariama20/devfair_sentence_composition/wiki-split/Data/Laser_vectors/mlp_laser_vect3')
    print(embed_test_data.shape)
    #embed_test_data.resize(embed_test_data.shape[0] // dim, dim)
    # embed_test_data = embed_test_data[:10]
    # embed_test_data = torch.from_numpy(embed_test_data).cuda().float()
    
    #print(embed_test_data[0].unsqueeze(0).size())
    # print(beam_decode(embed_test_data))
