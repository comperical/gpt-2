
import time
import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def explicit_steps(sess, hparams, context_tokens, start_token):

    batch_size = 1
    temperature = 1

    print("Context tokens are {}".format(context_tokens))

    context = tf.fill([batch_size, 1], start_token)

    loggies = tf.fill([1, hparams.n_vocab], -1.5655)    

    with tf.name_scope('explicit_sequence'):

        initial_out = model.model(hparams=hparams, X=context[:, :-1], reuse=tf.AUTO_REUSE)
        past = initial_out['present']
        past.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))

        for step in range(len(context_tokens)):

            tokens = context[:, -1]
            tokens = tokens[:, tf.newaxis]
            print("X/tokens shape is {}".format(tokens.shape))
            lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)
            print("Logits shape is {}, X/tokens is {}".format(lm_output['logits'].shape, tokens.shape))

            logits = lm_output['logits'][:, :, :hparams.n_vocab]
            print("Logits shape is NOW {}".format(logits.shape))
            logits = logits[:, -1, :]  / tf.to_float(temperature)

            presents = lm_output['present']
            presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))

            past = tf.concat([past, presents], axis=-2)
            loggies = tf.concat([loggies, logits], axis=0)

            ps_samples = tf.constant(context_tokens[step], shape=(1, 1))
            print("PS/S is shape {}".format(ps_samples.shape))

            context = tf.concat([context, ps_samples], axis=1)
            print("Context shape is now {}".format(context.shape))

    return sess.run([loggies, context])


def replicate_loop(
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
    full_len = 4
    base_text = "Three men walked into a bar. The first one ordered a drink, while "

    with tf.Session(graph=tf.Graph()) as sess:

        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams, length=full_len,
            #context=context,
            start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        alpha = time.time()
        print("Loading model....", end='')
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)
        print("...done, took {:03f} seconds".format(time.time()-alpha))

        context_tokens = enc.encode(base_text)
        #past, prev, mainout, loggies = sess.run(output, feed_dict={ context: [context_tokens for _ in range(batch_size)] })
        past, prev, mainout, loggies = sess.run(output)

        print(mainout)
        text = enc.decode(mainout[0])
        print(text)

        print("Past is {}".format(past.shape))
        print("Prev is {}, shape is {}".format(prev, prev.shape))
        print("Loggies shape is {}".format(loggies.shape))

        """
        onestep = sample.sample_sequence(
            hparams=hparams, length=1,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        s1tokens = mainout[0, :13]
        _, _, oneout, s1loggies = sess.run(onestep, feed_dict={ context: [s1tokens for _ in range(batch_size)] })


        """

        explicit, explout = explicit_steps(sess, hparams, mainout[0][1:], enc.encoder['<|endoftext|>'])
        print(explicit.shape)

        print(explout)
        text = enc.decode(explout[0])
        print(text)

        print("AL loggies: ")
        for d in range(len(loggies)):
            print(loggies[d, 1:10])

        print("Expl loggies: ")
        for d in range(len(explicit)):
            print(explicit[d, 1:10])



if __name__ == '__main__':
    fire.Fire(replicate_loop)

