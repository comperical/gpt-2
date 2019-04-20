
import time
import fire
import json
import os
import numpy as np
import tensorflow as tf
import pickle

import model, sample, encoder, utility

def model_sequence(sess, hparams, start_token, input_tokens):

    temperature = 1
    batch_size, numiters = input_tokens.shape
    context = tf.fill([batch_size, 1], start_token)
    tftokens = tf.convert_to_tensor(input_tokens, dtype=tf.int32)

    with tf.name_scope('model_sequence'):

        pastshape = model.past_shape(hparams=hparams, batch_size=batch_size)

        def body(past, output, tknprbs, idxlogs, loggies, curstep):

            # I still don't understand why this line is necessary
            xtok = output[:, -1]

            presents, logits = model.upmodel(hparams=hparams, X=xtok[:, tf.newaxis], past=past, reuse=tf.AUTO_REUSE)
            presents.set_shape(pastshape)

            logits = logits[:, -1, :hparams.n_vocab]  / tf.to_float(temperature)
            probs = tf.reshape(logits, (batch_size, hparams.n_vocab))
            probs = tf.nn.softmax(probs, axis=1)

            print("Logits shape is {}".format(logits.shape))

            #ps_samples = tf.constant(input_tokens[:, step], shape=(batch_size, 1), dtype=tf.int32)

            # this is the current step of the data
            justidx = tf.gather(tftokens, curstep, axis=1)
            print("Output has shape {}".format(output.shape))

            newprb = tf.gather(probs, justidx, axis=1)
            newprb = tf.reshape(tf.linalg.diag_part(newprb), (batch_size, 1))
            print("Full  Prob shape is {}, idx shape is {}".format(probs.shape, justidx))
            print("Token Prob is {}".format(newprb.shape))

            #samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32, seed=1000)            
            assert newprb.shape == (batch_size, 1), "Token shape is {}".format(newprb.shape)

            logits = tf.reshape(logits, shape=(batch_size, hparams.n_vocab, 1))

            return [
                tf.concat([past, presents], axis=-2),
                tf.concat([output, justidx], axis=1),
                tf.concat([tknprbs, newprb], axis=1),
                tf.concat([idxlogs, justidx], axis=1),
                tf.concat([loggies, logits], axis=2),
                tf.add(curstep, tf.constant(1))
            ]

        def cond(*args):
            return True

        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.

        history, _ = model.upmodel(hparams=hparams, X=context[:, :-1], past=None, reuse=tf.AUTO_REUSE)
        history.set_shape(pastshape)
        tknprbs = tf.fill([batch_size, 1], -1.56)
        idxlogs = tf.fill([batch_size, 1], -5)
        curstep = tf.constant(0, shape=(1, ))
        loggies = tf.fill([batch_size, hparams.n_vocab, 1], -1.5655)    

        result = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=numiters,
            loop_vars=[
                history,
                context,
                tknprbs,
                idxlogs,
                loggies, 
                curstep
            ],
            shape_invariants=[
                tf.TensorShape(pastshape),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, hparams.n_vocab, None]),
                tf.TensorShape((1, ))
            ],
            back_prop=False,
        )

        return sess.run(result)

def multi_probe(
    model_name='117M',
    seed=None,
    nsamples=0,
    length=None,
    temperature=1,
    top_k=0,
):


    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

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
            start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        tf.train.Saver().restore(sess, ckpt)

        encsents = utility.get_encoded_sents(enc)

        alpha = time.time()
        results = model_sequence(sess, hparams, enc.encoder['<|endoftext|>'], encsents)
        print("calc took {:.03f} for {} sentences".format(time.time()-alpha, len(get_sentences())))

        with open('likely_calc.pkl', 'wb') as fh:
            pickle.dump(results, fh)

if __name__ == '__main__':
    fire.Fire(multi_probe)

