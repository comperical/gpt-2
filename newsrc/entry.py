import os 
import sys
import time
import pickle
import importlib

import numpy as np
import pandas as pd
import tensorflow as tf

from collections import Counter

import utility

sys.path.append('/userdata/crm/src/python/shared3')
from arg_map import ArgMap


class ShowSentenceTest:

    def run_op(self, argmap):

        sents = utility.get_sentences()
        print(sents)

        encsents = utility.get_encoded_sents()
        print(encsents)

class ShowSentenceProbTest:

    def run_op(self, argmap):
        import bimodel
        _, adata, aprobs, _ = utility.PickleData(bimodel, 'modhcode', 0).load()

        utility.assert_small(aprobs[:, 0] + 1.513)
        aprobs = aprobs[:, 1:]

        for idx in range(adata.shape[0]):
            thesent = utility.get_encoder().decode(adata[idx, 1:])
            logprob = -np.sum(np.log(aprobs[idx, :]))
            print("Sentence/LogProb: \n\t{}\n\t{:.05f}".format(thesent, logprob))

class HcodeModelTest:
    """
    For implementations that have modeling ability, run the model against the hardcoded sentences.
    """

    def run_op(self, argmap):

        modname = argmap.get_str(('modname', 'original'))
        relmod = importlib.import_module(modname)        
        print("Going model hard-coded data with module {}".format(modname))

        # Load the hard-coded sample data
        origdata = utility.get_encoded_sents()

        #enc = utility.get_encoder()
        hparams = utility.get_hparams()

        with tf.Session(graph=tf.Graph()) as sess:

            # Notice!!! You don't need these set-seed operations here!!!
            # np.random.seed(seed)
            # tf.set_random_seed(seed)

            tfop = relmod.model_or_sample(origdata.shape[0], origdata)

            ckpt = tf.train.latest_checkpoint(os.path.join('models', utility.MODEL_NAME))
            tf.train.Saver().restore(sess, ckpt)

            alpha = time.time()
            result = sess.run(tfop)
            print("Basic Model successful, took {:.03f} seconds".format(time.time()-alpha))

            utility.PickleData(relmod, 'modhcode', 0, result=result).save()

class BasicModelTest:

    def run_op(self, argmap):

        modname = argmap.get_str(('modname', 'original'))
        seed = argmap.get_int(('seed', 10000))
        relmod = importlib.import_module(modname)        
        print("Going to do Basic Model with module {}".format(modname))

        # Load the sample data, and peel off the initial layer of <start_token> data.
        origdata = self.load_data(seed)
        origdata = origdata[:, 1:]
        assert origdata.shape[1] == utility.SAMPLE_LENGTH

        enc = utility.get_encoder()
        hparams = utility.get_hparams()

        with tf.Session(graph=tf.Graph()) as sess:
            np.random.seed(seed)
            tf.set_random_seed(seed)

            tfop = relmod.model_or_sample(origdata.shape[0], origdata)

            ckpt = tf.train.latest_checkpoint(os.path.join('models', utility.MODEL_NAME))
            tf.train.Saver().restore(sess, ckpt)

            alpha = time.time()
            result = sess.run(tfop)
            print("Basic Model successful, took {:.03f} seconds".format(time.time()-alpha))

            utility.PickleData(relmod, 'modsample', seed, result=result).save()



    def load_data(self, seedid):
        # Load the data from the original model
        import original
        return utility.PickleData(original, 'sample', seedid).load()



class BasicSampleTest:

    def run_op(self, argmap):

        modname = argmap.get_str(('modname', 'original'))
        seed = argmap.get_int(('seed', 10000))
        batch_size = argmap.get_int(('batchsize', 100))

        enc = utility.get_encoder()
        hparams = utility.get_hparams()
        relmod = importlib.import_module(modname)

        with tf.Session(graph=tf.Graph()) as sess:
            np.random.seed(seed)
            tf.set_random_seed(seed)

            tfop = relmod.model_or_sample(batch_size)

            ckpt = tf.train.latest_checkpoint(os.path.join('models', utility.MODEL_NAME))
            tf.train.Saver().restore(sess, ckpt)

            alpha = time.time()
            result = sess.run(tfop)
            print("Sample successful, took {:.03f} seconds".format(time.time()-alpha))

            utility.PickleData(relmod, 'sample', seed, result=result).save()



def get_command_list():
    return [sym[:-len("Test")] for sym in dir(sys.modules[__name__]) if sym.endswith("Test")]

if __name__ == '__main__':

    if len(sys.argv) >= 2:
        testname = sys.argv[1]
        testname = testname[:-4] if testname.endswith("Test") else testname
        assert testname in get_command_list(), "Unknown test name {}".format(testname)
        testlist = [testname]
    else:
        testlist = get_command_list()


    for onetest in testlist:
        try:
            print("Running Test {}".format(onetest))
            evalstr = "{}Test()".format(onetest)
            tool2run = eval(evalstr)
        except:
            print("Problem creating tool {}Test, did you follow naming convention?".format(onetest))
            quit()

        argmap = ArgMap.build_from_args(sys.argv[2:])
        tool2run.run_op(argmap)

