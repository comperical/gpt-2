#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def model_sequence(*, hparams, context):

    assert context is not None, "Context is not optional"

    batch_size = 1
    temperature = 1
    length = 1



    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])

        print("Context is {}, shape is {}".format(context, context.shape))

        def step(hparams, tokens, past=None):
            lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

            logits = lm_output['logits'][:, :, :hparams.n_vocab]
            presents = lm_output['present']
            presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
            return {
                'logits': logits,
                'presents': presents,
            }


        def body(past, prev, output):
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            logits = top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
            ]

        def cond(*args):
            return True

        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tokens


def probe_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    length=None,
    temperature=1
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])


        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = probe_model(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = "My father was sad that the Patriots lost the game."

            context_tokens = enc.encode(raw_text)
            generated = 0

            for idx, tkn in enumerate(context_tokens):

                history = context_tokens[:idx-1]

                out = sess.run(output, feed_dict={context: history })

                print(out)
                quit()

if __name__ == '__main__':
    fire.Fire(interact_model)

