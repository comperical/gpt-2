
import sys
import tensorflow as tf

import utility

sys.path.append("src")
import model

def model_or_sample(batch_size):

    return explicit_loop_sample(hparams=utility.get_hparams(),
                            length=utility.SAMPLE_LENGTH,
                            start_token=utility.get_start_token()[0],
                            batch_size=batch_size)

def explicit_loop_sample(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0):

    context = utility.check_start_context(start_token, context, batch_size)
    pastshape = model.past_shape(hparams=hparams, batch_size=batch_size)

    with tf.name_scope('explicit_sequence'):

        past, _ = model.upmodel(hparams=hparams, X=context[:, :-1], past=None, reuse=tf.AUTO_REUSE)
        past.set_shape(pastshape)

        loggies = tf.fill([batch_size, hparams.n_vocab, 1], -1.5655)
        tknprbs = tf.fill([batch_size, 1], -1.513)

        for step in range(length):

            xtok = context[:, -1]

            presents, logits = model.upmodel(hparams=hparams, X=xtok[:, tf.newaxis], past=past, reuse=tf.AUTO_REUSE)
            presents.set_shape(pastshape)

            logits = logits[:, -1, :hparams.n_vocab]  / tf.to_float(temperature)
            logits = utility.top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32, seed=1000)

            # This is the full prob distribution
            probs = tf.reshape(logits, (batch_size, hparams.n_vocab))
            probs = tf.nn.softmax(probs, axis=1)
            justidx = tf.reshape(samples, shape=(batch_size, ))            
            tknprb = tf.gather(probs, justidx, axis=1)
            tknprb = tf.reshape(tf.linalg.diag_part(tknprb), (batch_size, 1))

            past = tf.concat([past, presents], axis=-2)
            context = tf.concat([context, samples], axis=1)
            tknprbs = tf.concat([tknprbs, tknprb], axis=1)
            loggies = tf.concat([loggies, tf.reshape(logits, shape=(batch_size, hparams.n_vocab, 1))], axis=2)  

    return past, context, tknprbs, loggies


