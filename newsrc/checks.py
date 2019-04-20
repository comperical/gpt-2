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

class CheckFinalBiModelTest:

    def run_op(self, argmap):
        for opcode in utility.MODES:
            self.check_against_augm(opcode)

    def check_against_augm(self, opcode):
        import bmdl_augm
        import bimodel

        seed = 0 if opcode == 'modhcode' else 10000

        # Okay, load the SAMPLES data from the augmented operation
        _, adata, aprobs, astep = utility.PickleData(bimodel, opcode, seed).load()

        # Load the MODSAMPLE data from the expl_model operation
        _, bdata, bprobs, _, bstep = utility.PickleData(bmdl_augm, opcode, seed).load()

        utility.assert_zero(adata - bdata)
        utility.assert_zero(astep - bstep)
        utility.assert_small(aprobs - bprobs)

        print("Success, checked against augmented model for opcode {}".format(opcode.upper()))       

class CheckBiModelAugmTest:

    def run_op(self, argmap):


        self.check_sample('sample')
        self.check_sample('modsample')

        self.check_hcode_model()


    def check_sample(self, opcode):
        import bmdl_augm
        import augmented

        # Okay, load the SAMPLES data from the augmented operation
        _, adata, aprobs, aloggs = utility.PickleData(augmented, 'sample').load()

        # Load the MODSAMPLE data from the expl_model operation
        _, bdata, bprobs, bloggs, bstep = utility.PickleData(bmdl_augm, opcode).load()

        utility.assert_zero(adata - bdata)
        utility.assert_small(aprobs - bprobs)
        utility.assert_small(aloggs - bloggs)

        assert bstep.shape == (1, ) and bstep[0] == utility.SAMPLE_LENGTH
        print("Success, explicit model operation produces exactly the same probabilities as sample operation")        

    def check_hcode_model(self):
        import expl_model
        import bmdl_augm

        # Okay, load the SAMPLES data from the augmented operation
        _, adata, aprobs, aloggs = utility.PickleData(expl_model, 'modhcode', 0).load()

        # Load the MODSAMPLE data from the expl_model operation
        _, bdata, bprobs, bloggs, bstep = utility.PickleData(bmdl_augm, 'modhcode', 0).load()

        utility.assert_zero(adata - bdata)
        utility.assert_small(aprobs - bprobs)
        utility.assert_small(aloggs - bloggs)

        print("Success, explicit model and augmented bimodel produce identical results on hard-coded inputs")        


class CheckExplModelTest:

    def run_op(self, argmap):
        import expl_model
        import augmented

        # Okay, load the SAMPLES data from the augmented operation
        _, adata, aprobs, aloggs = utility.PickleData(augmented, 'sample').load()

        # Load the MODSAMPLE data from the expl_model operation
        _, bdata, bprobs, bloggs = utility.PickleData(expl_model, 'modsample').load()

        utility.assert_zero(adata - bdata)
        utility.assert_small(aprobs - bprobs)
        utility.assert_small(aloggs - bloggs)

        print("Success, explicit model operation produces exactly the same probabilities as sample operation")


class CheckAugmentedTest:

    def run_op(self, argmap):
        import original
        import augmented

        # Original data just returns the actual samples
        origdata = utility.PickleData(original, 'sample').load()

        # Augmented saves a bunch more stuff.
        _, augmdata, tknprbs, loggies = utility.PickleData(augmented, 'sample').load()

        utility.assert_zero(origdata - augmdata)
        print("Success, checked data of size {} against original sample".format(origdata.shape))

        for bi in range(augmdata.shape[0]):
            for pi in range(1, augmdata.shape[1]):
                logits = loggies[bi, :, pi]
                token = augmdata[bi, pi]
                probs = utility.softmax(logits)

                aprb = probs[token]
                bprb = tknprbs[bi, pi]

                utility.assert_small(aprb - bprb, epsilon=1e-4)

        print("Checked correspondence between logit values and token probabilities")




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

