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


def encoding_layer(inputs, D, K):
    """
    Encoding Layer accepts 3D or 4D inputs.

    It considers an input feature maps with the shape of C×H×W
    as a set of C-dimensional input features inputs={x1,...xN}, where N is total number
    of features given by H×W, which learns an inherent codebook D={d1,...dK} and
    a set of smoothing factor of visual centers S={s1,...sK}.

    Encoding Layer outputs the residuals with soft-assignment weights ek=∑Ni=1 eik,
    where eik = (exp(−sk‖rik‖2) / ∑Kj=1 exp(−sj‖rij‖2)) * rik and
    the residuals are given by rik=xi−dk. The output encoders are E={e1,...eK}.

    Parameters:
    :param inputs: BxHxWxC
    :param D – dimension of the features or feature channels
    :param K – number of codewords
    :return: E (residual encoding vectors - B X K X D)

    Shape:
    Input: B×N×D or B×H×WxD (where B is batch, N is total number of features Or H×W.)
    Output: B×K×D

    Variables:
    self.codewords (Tensor) – the learnable codewords of shape (K×D)
    self.scale (Tensor) – the learnable scale factor of visual centers

    Examples:
        B,C,H,W,K = 2,3,4,5,6
        inputs = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), requires_grad=True)
        layer = encoding.Encoding(C,K).double().cuda()
        E = layer(inputs)
    """
    global batch_size
    batch_size = inputs.get_shape().as_list()[0]

    # init codewords and smoothing factor (learnable parameters)
    std1 = 1. / ((K * D) ** (1 / 2))
    codewords = slim.model_variable(name='codewords',
                                    initializer=tf.random_uniform(shape=(K, D), minval=-std1, maxval=std1),
                                    regularizer=slim.l2_regularizer(0.05))
    scale = slim.model_variable(name='scale',
                                initializer=tf.random_uniform(shape=(K,), minval=-1, maxval=0),
                                regularizer=slim.l2_regularizer(0.05))

    if inputs.get_shape().ndims == 4:
        # BxHxWxD => Bx(HW)xD (BxNxD)
        X = tf.reshape(inputs, [-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3]], name='input')
    else:
        raise RuntimeError('Encoding Layer unknown input dims!')


    return encoding(X, codewords, scale)


@tf.custom_gradient
def encoding(X, C, S):
    '''
    :param X:
    :param C:
    :param S:
    :return E (N residual encoding vectors, B X K X D)
    '''

    s = tf.reshape(S, (1, 1, C.shape[0]))
    x = tf.broadcast_to(tf.expand_dims(X, axis=2),
                        [batch_size, X.shape[1], C.shape[0], C.shape[1]])
    c = tf.expand_dims(tf.expand_dims(C, axis=0), axis=0)

    SL = tf.multiply(s, tf.reduce_sum(tf.pow(tf.subtract(x, c), 2), axis=3))
    A = tf.nn.softmax(SL, axis=2)

    a = tf.expand_dims(A, axis=3)
    E = tf.reduce_sum(tf.multiply(a, tf.subtract(x, c)), axis=1, name='encoding_vectors')

    def grad(gradE):
        e = tf.expand_dims(gradE, axis=1)
        x = tf.broadcast_to(tf.expand_dims(X, axis=2),
                            [batch_size, X.shape[1], C.shape[0], C.shape[1]])
        c = tf.expand_dims(tf.expand_dims(C, axis=0), axis=0)
        gradSL = tf.reduce_sum(tf.multiply(e, tf.subtract(x, c)), axis=3)

        gradX = tf.matmul(A, gradE)  # Batch matrix multiplication
        a = tf.expand_dims(tf.reduce_sum(A, axis=1), axis=2)
        gradC = tf.reduce_sum(tf.multiply(-gradE, a), axis=0)

        s = tf.reshape(S, [1, 1, C.shape[0]])
        t = tf.expand_dims((2 * gradSL * s), axis=3)
        tmp = tf.multiply(t, tf.subtract(x, c))

        GX = tf.multiply(tf.reduce_sum(tmp, axis=2), gradX)
        GC = tf.multiply(tf.reduce_sum(tf.reduce_sum(tmp, axis=0), axis=0), gradC)
        GS = tf.reduce_sum(tf.reduce_sum(tf.multiply(gradSL, tf.divide(A, s)), axis=0), axis=0)

        return GX, GC, GS   # <- Is it correct to return this value?

    return E, grad
