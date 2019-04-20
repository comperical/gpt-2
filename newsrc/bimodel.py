
import sys

import tensorflow as tf

import utility

sys.path.append("src")
import model

def model_or_sample(batch_size, modeldata=None):

    print(modeldata)
    length = utility.SAMPLE_LENGTH if modeldata is None else modeldata.shape[1]

    return tf_loop_bimodel(hparams=utility.get_hparams(),
                            length=length,
                            modeldata=modeldata,                            
                            start_token=utility.get_start_token()[0],
                            batch_size=batch_size)

def tf_loop_bimodel(*, hparams, length, modeldata, start_token=None, batch_size=None, context=None, temperature=1, top_k=0):

    context = utility.check_start_context(start_token, context, batch_size)
    pastshape = model.past_shape(hparams=hparams, batch_size=batch_size)
    assert modeldata is None or modeldata.shape == (batch_size, length)

    with tf.name_scope('tf_loop_bimodel'):

        mdltensor = None if modeldata is None else tf.constant(modeldata, dtype=tf.int32)
        #print("MDL tensor shape is {}".format(mdltensor.shape))
        pastshape = model.past_shape(hparams=hparams, batch_size=batch_size)

        def body(past, output, tknprbs, curstep):

            # I still don't understand why this line is necessary
            xtok = output[:, -1]

            presents, logits = model.upmodel(hparams=hparams, X=xtok[:, tf.newaxis], past=past, reuse=tf.AUTO_REUSE)
            presents.set_shape(pastshape)

            logits = logits[:, -1, :hparams.n_vocab]  / tf.to_float(temperature)
            logits = utility.top_k_logits(logits, k=top_k)

            if modeldata is None:
                items = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32, seed=1000)
            else:
                items = tf.gather(mdltensor, curstep, axis=1)
                items = tf.reshape(items, shape=(batch_size, 1))

            assert items.shape == (batch_size, 1)

            # This is the full prob distribution
            probs = tf.reshape(logits, (batch_size, hparams.n_vocab))
            probs = tf.nn.softmax(probs, axis=1)
            justidx = tf.reshape(items, shape=(batch_size, ))            
            tknprb = tf.gather(probs, justidx, axis=1)
            tknprb = tf.reshape(tf.linalg.diag_part(tknprb), (batch_size, 1))

            return [
                tf.concat([past, presents], axis=-2),
                tf.concat([output, items], axis=1),
                tf.concat([tknprbs, tknprb], axis=1),
                tf.add(curstep, tf.constant(1))             
            ]

        def cond(*args):
            return True

        history, _ = model.upmodel(hparams=hparams, X=context[:, :-1], past=None, reuse=tf.AUTO_REUSE)
        history.set_shape(pastshape)

        loggies = tf.fill([batch_size, hparams.n_vocab, 1], -1.5655)
        tknprbs = tf.fill([batch_size, 1], -1.513)
        curstep = tf.constant(0, shape=(1,))

        result = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                history,
                context,
                tknprbs,
                curstep
            ],
            shape_invariants=[
                tf.TensorShape(pastshape),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),              
                tf.TensorShape([1])
            ],
            back_prop=False
        )

        return result
