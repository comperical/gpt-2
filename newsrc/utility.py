import os 
import sys
import pickle

import numpy as np
import pandas as pd
from functools import lru_cache

sys.path.append("src")
import encoder, model

SAMPLE_LENGTH = 10

MODEL_NAME = '117M'

MODES = ['sample', 'modsample', 'modhcode']

def assert_zero(diff, epsilon=1e-6):
    flatdiff = diff.flatten()
    maxabs = max(abs(flatdiff))
    assert maxabs == 0, "Got max absolute value of {}".format(maxabs)

def assert_small(diff, epsilon=1e-6):
    flatdiff = diff.flatten()
    maxabs = max(abs(flatdiff))
    assert maxabs < epsilon, "Got max abs of {}".format(maxabs)

def softmax(logits):
    mxlgt = max(logits)
    nonorm = np.exp(logits - mxlgt)
    nrmfct = sum(nonorm)
    return nonorm / nrmfct

@lru_cache(maxsize=1)
def get_encoder():
    return encoder.get_encoder(MODEL_NAME)

def get_start_token():
    return get_encoder().encoder['<|endoftext|>'],

@lru_cache(maxsize=1)
def get_hparams():
    import json
    hparams = model.default_hparams()
    with open(os.path.join('models', MODEL_NAME, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    return hparams

#def get_sentences():
#    return ["the man bought a new car", "the car bought a new man", "the elephant bought a new man"]

"""
def get_sentences():

    def allwords():
        yield 'argument'
        yield 'conviction'
        yield 'belief'
        yield 'concern'
        yield 'statement'

        yield 'machine'
        yield 'business'
        yield 'structure'
        yield 'cognition'
        yield 'apparatus'

    def sentences():
        for w in allwords():
            yield "The lawyer explained his {} that prisoners should have the right to vote.".format(w)

    return list(sentences())

def get_sentences():

    def allwords():
        yield 'money'
        yield 'cat'
        yield 'dog'
        yield 'shoes'
        yield 'documents'

        yield 'tree'
        yield 'house'
        yield 'liberty'
        yield 'corporation'
        yield 'finances'
        yield 'elephant'

    def sentences():
        for w in allwords():
            yield "We put the {} in the car.".format(w)

    return list(sentences())
"""


def get_sentences():

    def allwords():
        yield 'pizza'
        yield 'pasta'
        #yield 'hamburgers'
        yield 'cereal'
        yield 'steak'
        #yield 'steaks'

        yield 'trees'
        yield 'houses'
        yield 'liberty'
        yield 'corporation'
        yield 'finances'

    def sentences():
        for w in allwords():
            yield "The family ate {} for dinner.".format(w)

    return list(sentences())

def get_encoded_sents(enc=None):
    if enc == None:
        enc = encoder.get_encoder('117M')

    return np.array([enc.encode(sent) for sent in get_sentences()])


def get_mode_info(argv):
    modstr = argv[0]
    assert modstr in MODES, "Invalid mode string {}, expected one of {}".format(modstr, MODES)

    theseed = None

    return modstr, theseed


def top_k_logits(logits, k):
    import sample
    return sample.top_k_logits(logits, k)

def check_start_context(start_token, context, batch_size):
    import tensorflow as tf

    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        return context

    assert context is None, 'Specify exactly one of start_token and context!'
    return tf.fill([batch_size, 1], start_token)

class PickleData:

    def __init__(self, module, mdcode, seedid=10000, result=None):
        
        if mdcode == 'modhcode':
            assert seedid == 0

        self.mdcode = mdcode
        self.seedid = seedid
        self.result = result
        self.module = module


    def savepath(self):
        mdstr = "{}{}".format(self.mdcode, self.seedid)
        filepath = "{}__{}.pkl".format(self.module.__name__, mdstr)
        return os.path.sep.join(['pkldata', filepath])

    def load(self):
        assert self.result == None

        with open(self.savepath(), 'rb') as fh:
            self.result = pickle.load(fh)

        return self.result

    def save(self):

        with open(self.savepath(), 'wb') as fh:
            pickle.dump(self.result, fh)

        print("Pickled result to path {}".format(self.savepath()))

