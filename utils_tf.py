# Based on code from https://github.com/tensorflow/cleverhans

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
import math
import numpy as np
import os
import six
import tensorflow as tf
import time
import warnings

from utils import batch_indices, _ArgsWrapper

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class _FlagsWrapper(_ArgsWrapper):
    """
    Wrapper that tries to find missing parameters in TensorFlow FLAGS
    for backwards compatibility.

    Plain _ArgsWrapper should be used instead if the support for FLAGS
    is removed.
    """
    def __getattr__(self, name):
        val = self.args.get(name)
        if val is None:
            warnings.warn('Setting parameters ({}) from TensorFlow FLAGS is '
                          'deprecated.'.format(name))
            val = FLAGS.__getattr__(name)
        return val


def model_loss(y, model, mean=True):
    """
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :param mean: boolean indicating whether should return mean of loss
                 or vector of losses for each input of the batch
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """

    op = model.op
    #print(op)
    if "softmax" in str(op).lower():
        logits, = op.inputs
    else:
        logits = model

    out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    if mean:
        out = tf.reduce_mean(out)
    return out


class RobustTraining()

    def __init__(self, sess, model, X_train, Y_train):
        self.sess = sess
        self.predictions = model
        self.X_nat = X_train
        self.Y_nat = Y_train
        self.X_adv = X_train


    def train(x, y, save=False,
                    predictions_adv=None, evaluate=None, verbose=True, args=None):
        args = _FlagsWrapper(args or {})

        # Check that necessary arguments were given (see doc above)
        assert args.nb_epochs, "Number of epochs was not given in args dict"
        assert args.learning_rate, "Learning rate was not given in args dict"
        assert args.batch_size, "Batch size was not given in args dict"

        if save:
            assert args.train_dir, "Directory for save was not given in args dict"
            assert args.filename, "Filename for save was not given in args dict"

        # Define loss
        loss = model_loss(y, self.predictions)
        if predictions_adv is not None:
            p = 1.0
            loss = ((1-p)*loss + p*model_loss(y, predictions_adv))

        train_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

        with self.sess.as_default():
            if hasattr(tf, "global_variables_initializer"):
                tf.global_variables_initializer().run()
            else:
                self.sess.run(tf.initialize_all_variables())

            for epoch in six.moves.xrange(args.nb_epochs):
                if verbose:
                    print("Epoch " + str(epoch))

                # Compute number of batches
                nb_batches = int(math.ceil(float(len(X_train)) / args.batch_size))
                assert nb_batches * args.batch_size >= len(X_train)

                prev = time.time()
                for batch in range(nb_batches):

                    # Compute batch start and end indices
                    start, end = batch_indices(batch, len(X_train), args.batch_size)

                    # update adversarial examples
                    self.update_adv(start, end)
                    # Perform one training step (on adversarial examples)
                    train_step.run(feed_dict={x: X_adv[start:end],
                                              y: Y_train[start:end]})
                assert end >= len(X_train)  # Check that all examples were used
                cur = time.time()
                if verbose:
                    print("\tEpoch took " + str(cur - prev) + " seconds")
                prev = cur
                if evaluate is not None:
                    evaluate()

            if save:
                save_path = os.path.join(args.train_dir, args.filename)
                saver = tf.train.Saver()
                saver.save(sess, save_path)
                print("Completed model training and saved at:" + str(save_path))
            else:
                print("Completed model training.")

        return True
        
        
    def update_adv(self, start, end, preds, y=None, eps=0.3, ord=2, model=None, steps=15):
        y = y / tf.reduce_sum(y, 1, keep_dims=True)

        x_adv = tf.stop_gradient(self.X_adv[start:end])
        x_nat = tf.stop_gradient(self.X_nat[start:end])

        for t in xrange(steps):
            loss = model_loss(y, model(x_adv), mean=False)
            grad, = tf.gradients(eps*loss, x_adv)
            grad2, = tf.gradients(tf.nn.l2_loss(x_adv-x_nat), x_adv)
            grad = grad - grad2
            x_adv = tf.stop_gradient(x_adv+1./np.sqrt(t+1)*grad)

        # return x_adv
        self.X_adv[start:end] = x_adv




def model_eval(sess, x, y, model, X_test, Y_test, args=None):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param model: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    :return: a float with the accuracy value
    """
    args = _FlagsWrapper(args or {})

    assert args.batch_size, "Batch size was not given in args dict"

    # Define symbol for accuracy
    # Keras 2.0 categorical_accuracy no longer calculates the mean internally
    # tf.reduce_mean is called in here and is backward compatible with previous
    # versions of Keras
    acc_value = tf.reduce_mean(keras.metrics.categorical_accuracy(y, model))

    # Init result var
    accuracy = 0.0

    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
        assert nb_batches * args.batch_size >= len(X_test)

        for batch in range(nb_batches):
            #if batch % 100 == 0 and batch > 0:
                #print("Batch " + str(batch))

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * args.batch_size
            end = min(len(X_test), start + args.batch_size)
            cur_batch_size = end - start

            # The last batch may be smaller than all others, so we need to
            # account for variable batch size here
            cur_acc = acc_value.eval(
                feed_dict={x: X_test[start:end],
                           y: Y_test[start:end]})

            accuracy += (cur_batch_size * cur_acc)

        assert end >= len(X_test)

        # Divide by number of examples to get final value
        accuracy /= len(X_test)

    return accuracy
