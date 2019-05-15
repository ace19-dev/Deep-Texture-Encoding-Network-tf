"""Learnable Residual Encoding Layer

With the texture encoding layer, the visual descriptors X are pooled into a
set of N residual encoding vectors E = {e1 , ...en }

The encoding layer can capture more texture details by increasing the number of
learnable codewords.

Given a set of N visual descriptors X = {x 1 , ..x N }
and a learned codebook C = {c 1 , ...c K } containing K codewords that
are D-dimensional, each descriptor x i can be assigned with a
weight a ik to each codeword c k and the corresponding
residual vector is denoted by r ik = x i − c k , where i =
1, ...N and k = 1, ...K. Given the assignments and the
residual vector, the residual encoding model applies an aggregation operation
for every single codeword c k :

ek = ∑Ni=1 eik = ∑Ni=1 aik * rik

The resulting encoder outputs a fixed length representation E = {e 1 , ...e K }
(independent of the number of input descriptors N ).

Encoding Layer
---------------
assigning a descriptor to each codeword
aik = exp(−sk‖rik‖2) / ∑Kj=1 exp(−sj‖rij‖2)

"""

import numpy as np

import tensorflow as tf

slim = tf.contrib.slim


OP_NAME = 'encoding_op'


class EncodingLayer(tf.keras.layers.Layer):
    def __init__(self, D, K):
        super(EncodingLayer, self).__init__()

        # init codewords and smoothing factor (learnable parameters)
        std1 = 1. / ((K * D) ** (1 / 2))
        self.codewords = slim.model_variable(name='codewords',
                                        initializer=tf.random_uniform(shape=(K, D), minval=-std1, maxval=std1),
                                        regularizer=slim.l2_regularizer(0.05))
        self.scale = slim.model_variable(name='scale',
                                    initializer=tf.random_uniform(shape=(K,), minval=-1, maxval=0),
                                    regularizer=slim.l2_regularizer(0.05))


    def build(self, input_shape):
        self.batch_size = input_shape[0]


    # def call(self, input):
    #     X = tf.reshape(input, [-1, input.shape[1] * input.shape[2], input.shape[3]], name='input')
    #
    #     s = tf.reshape(self.scale, (1, 1, self.codewords.shape[0]))
    #     x = tf.broadcast_to(tf.expand_dims(X, axis=2),
    #                         [self.batch_size, X.shape[1], self.codewords.shape[0], self.codewords.shape[1]])
    #     c = tf.expand_dims(tf.expand_dims(self.codewords, axis=0), axis=0)
    #
    #     SL = tf.multiply(s, tf.reduce_sum(tf.pow(tf.subtract(x, c), 2), axis=3))
    #     A = tf.nn.softmax(SL, axis=2)
    #
    #     a = tf.expand_dims(A, axis=3)
    #     E = tf.reduce_sum(tf.multiply(a, tf.subtract(x, c)), axis=1, name='encoding_vectors')
    #
    #     return E

    def call(self, input):
        X = tf.reshape(input, [-1, input.shape[1] * input.shape[2], input.shape[3]], name='input')

        E = encoding_op(python_func,
                        [X, self.codewords, self.scale, self.batch_size],
                        grad_func,
                        name=OP_NAME)
        return E


def python_func(inps, name=None):
    with tf.name_scope(name):
        X = inps[0]
        C = inps[1]
        S = inps[2]
        batch_size = inps[3]

        forward_func = encoding(X, C, S, batch_size)
        # victim_op
        backward_func, _X, _C, _S, _b = tf.identity_n([forward_func, X, C, S, batch_size])
        return backward_func + tf.stop_gradient(forward_func - backward_func)


def encoding_op(func, inp, grad, name=None, victim_op='IdentityN'):
    # Need to generate a unique name to avoid duplicates.
    rnd_name = 'my_gradient' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({victim_op: rnd_name}):
        return func(inp, name=name)


def encoding(X, C, S, batch_size):
    '''
    :param X:
    :param C:
    :param S:
    :return E (N residual encoding vectors, B X K X D)
    '''

    global A

    s = tf.reshape(S, (1, 1, C.shape[0]))
    x = tf.broadcast_to(tf.expand_dims(X, axis=2),
                        [batch_size, X.shape[1], C.shape[0], C.shape[1]])
    c = tf.expand_dims(tf.expand_dims(C, axis=0), axis=0)

    SL = tf.multiply(s, tf.reduce_sum(tf.pow(tf.subtract(x, c), 2), axis=3))
    A = tf.nn.softmax(SL, axis=2)

    a = tf.expand_dims(A, axis=3)
    E = tf.reduce_sum(tf.multiply(a, tf.subtract(x, c)), axis=1, name='encoding_vectors')

    return E


# Return custom gradient wrt each input of the op.
def grad_func(op, gradE, tmp1, tmp2, tmp3, tmp4):
    X = op.inputs[1]
    C = op.inputs[2]
    S = op.inputs[3]
    batch_size = op.inputs[4]

    e = tf.expand_dims(gradE, axis=1)
    x = tf.broadcast_to(tf.expand_dims(X, axis=2),
                        [batch_size, X.shape[1], C.shape[0], C.shape[1]])
    c = tf.expand_dims(tf.expand_dims(C, axis=0), axis=0)
    gradSL = tf.reduce_sum(tf.multiply(e, tf.subtract(x, c)), axis=3)

    gradX = tf.matmul(A, gradE) # Batch matrix multiplication
    a = tf.expand_dims(tf.reduce_sum(A, axis=1), axis=2)
    gradC = tf.reduce_sum(tf.multiply(-gradE, a), axis=0)

    s = tf.reshape(S, [1, 1, C.shape[0]])
    t = tf.expand_dims((2 * gradSL * s), axis=3)
    tmp = tf.multiply(t, tf.subtract(x, c))

    GX = tf.multiply(tf.reduce_sum(tmp, axis=2), gradX)
    GC = tf.multiply(tf.reduce_sum(tf.reduce_sum(tmp, axis=0), axis=0), gradC)
    GS = tf.reduce_sum(tf.reduce_sum(tf.multiply(gradSL, tf.divide(A, s)), axis=0), axis=0)

    return None, GX, GC, GS, None
