import tensorflow as tf

import model




def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    with tf.name_scope('sample_sequence'):

        pastshape = model.past_shape(hparams=hparams, batch_size=batch_size)

        def body(past, output):

            # I still don't understand why this line is necessary
            xtok = output[:, -1]

            presents, logits = model.upmodel(hparams=hparams, X=xtok[:, tf.newaxis], past=past, reuse=tf.AUTO_REUSE)
            presents.set_shape(pastshape)

            logits = logits[:, -1, :hparams.n_vocab]  / tf.to_float(temperature)
            logits = top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32, seed=1000)

            return [
                tf.concat([past, presents], axis=-2),
                tf.concat([output, samples], axis=1)
            ]

        def cond(*args):
            return True

        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.

        history, _ = model.upmodel(hparams=hparams, X=context[:, :-1], past=None, reuse=tf.AUTO_REUSE)
        history.set_shape(pastshape)

        _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                history,
                context
            ],
            shape_invariants=[
                tf.TensorShape(pastshape),
                tf.TensorShape([batch_size, None])
            ],
            back_prop=False,
        )

        return tokens