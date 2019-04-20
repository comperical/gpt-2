
import time
import fire
import json
import os
import numpy as np
import tensorflow as tf
import pickle

import model, sample, encoder

def explicit_steps(sess, hparams, start_token, input_tokens):

    batch_size = 1
    temperature = 1

    tknprbs = tf.fill([batch_size, 1], -1.513)
    output = tf.fill([batch_size, 1], start_token)

    loggies = tf.fill([1, hparams.n_vocab], -1.5655)    

    with tf.name_scope('explicit_sequence'):

        pastshape = model.past_shape(hparams=hparams, batch_size=batch_size)

        past, _ = model.upmodel(hparams=hparams, X=output[:, :-1], reuse=tf.AUTO_REUSE)
        past.set_shape(pastshape)

        for step in range(len(input_tokens)):

            xtok = output[:, -1]

            presents, logits = model.upmodel(hparams=hparams, X=xtok[:, tf.newaxis], past=past, reuse=tf.AUTO_REUSE)
            presents.set_shape(pastshape)

            logits = logits[:, -1, :hparams.n_vocab] / tf.to_float(temperature)

            past = tf.concat([past, presents], axis=-2)
            loggies = tf.concat([loggies, logits], axis=0)

            ps_samples = tf.constant(input_tokens[step], shape=(1, 1))
            justidx = tf.constant(input_tokens[step], shape=(1, ))
            #print("PS/S is shape {}".format(ps_samples.shape))

            # This is the full prob distribution
            probs = tf.reshape(logits, (batch_size, hparams.n_vocab))
            probs = tf.nn.softmax(probs, axis=1)
            tknprb = tf.gather(probs, justidx, axis=1)
            #print("Full  Prob shape is {}".format(probs.shape))
            #print("Token Prob is {}".format(tknprb.shape))
            assert tknprb.shape == (batch_size, 1)

            tknprbs = tf.concat([tknprbs, tknprb], axis=1)
            #allprbs = tf.concat([allprbs, probs], axis=0)
            output = tf.concat([output, ps_samples], axis=1)

    return sess.run([loggies, output, tknprbs])


def model_loop(
    model_name='117M',
    seed=None,
    nsamples=0,
    length=None,
    temperature=1,
    top_k=0,
):
    alpha = time.time()
    print("Loading encoder...", end='')
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    print("... done, took {:.03f} seconds".format(time.time()-alpha))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    batch_size = 1
    full_len = 10

    with tf.Session(graph=tf.Graph()) as sess:

        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams, length=full_len,
            #context=context,
            start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        tf.train.Saver().restore(sess, ckpt)

        past, prev, mainout, loggies = sess.run(output)

        print(mainout)
        text = enc.decode(mainout[0])
        print(text)

        print("Past is {}".format(past.shape))
        print("Prev is {}, shape is {}".format(prev, prev.shape))
        print("Loggies shape is {}".format(loggies.shape))

        fullcalc = explicit_steps(sess, hparams, enc.encoder['<|endoftext|>'], mainout[0][1:])
        #explicit, explout, allprbs = explicit_steps(sess, hparams, enc.encoder['<|endoftext|>'], mainout[0][1:])
        #print(explicit.shape)

        """
        print(explout)
        text = enc.decode(explout[0])
        print(text)

        print("AL loggies: ")
        for d in range(len(loggies)):
            print(loggies[d, 1:10])

        print("Expl loggies: ")
        for d in range(len(explicit)):
            print(explicit[d, 1:10])
        """

        with open('fullcalc.pkl', 'wb') as fh:
            pickle.dump(fullcalc, fh)

        with open('mainout.pkl', 'wb') as fh:
            pickle.dump(mainout, fh) 

if __name__ == '__main__':
    fire.Fire(model_loop)

