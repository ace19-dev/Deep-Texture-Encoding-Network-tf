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


# OP_NAME_SCALED_L2 = 'scaled_l2'
# OP_NAME_AGGREGATE = 'aggregate'
OP_NAME = 'encoding_op'


def encoding_layer(X, D, K):
    """
    Encoding Layer accepts 3D or 4D inputs.

    It considers an input feature maps with the shape of C×H×W
    as a set of C-dimensional input features X={x1,...xN}, where N is total number
    of features given by H×W, which learns an inherent codebook D={d1,...dK} and
    a set of smoothing factor of visual centers S={s1,...sK}.

    Encoding Layer outputs the residuals with soft-assignment weights ek=∑Ni=1 eik,
    where eik = (exp(−sk‖rik‖2) / ∑Kj=1 exp(−sj‖rij‖2)) * rik and
    the residuals are given by rik=xi−dk. The output encoders are E={e1,...eK}.

    Parameters:
    :param X: BxHxWxC
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
        X = Variable(torch.cuda.DoubleTensor(B,C,H,W).uniform_(-0.5,0.5), requires_grad=True)
        layer = encoding.Encoding(C,K).double().cuda()
        E = layer(X)
    """
    global batch_size
    batch_size = X.get_shape().as_list()[0]

    # init codewords and smoothing factor (learnable parameters)
    std1 = 1. / ((K * D) ** (1 / 2))
    codewords = slim.model_variable(name='codewords',
                                    initializer=tf.random_uniform(shape=(K, D), minval=-std1, maxval=std1),
                                    regularizer=slim.l2_regularizer(0.05))
    scale = slim.model_variable(name='scale',
                                initializer=tf.random_uniform(shape=(K,), minval=-1, maxval=0),
                                regularizer=slim.l2_regularizer(0.05))

    if X.get_shape().ndims == 4:
        # BxHxWxD => Bx(HW)xD (BxNxD)
        x = tf.reshape(X, [-1, X.shape[1] * X.shape[2], X.shape[3]], name='input')
    else:
        raise RuntimeError('Encoding Layer unknown input dims!')

    E = encoding_op(python_func, [x, codewords, scale], grad_func, name=OP_NAME)
    # E = encoding(x, codewords, scale)
    return E


def python_func(inps, name=None):
    with tf.name_scope(name):
        X = inps[0]
        C = inps[1]
        S = inps[2]

        forward_func = encoding(X, C, S)
        # victim_op
        backward_func, _X, _C, _S = tf.identity_n([forward_func, X, C, S])
        return backward_func + tf.stop_gradient(forward_func - backward_func)


def encoding_op(func, inp, grad, name=None, victim_op='IdentityN'):
    # Need to generate a unique name to avoid duplicates.
    rnd_name = 'my_gradient' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({victim_op: rnd_name}):
        return func(inp, name=name)


def encoding(X, C, S):
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
    # x = tf.broadcast_to(tf.expand_dims(X, axis=2),
    #                     [batch_size, X.shape[1], C.shape[0], C.shape[1]])
    # c = tf.expand_dims(tf.expand_dims(C, axis=0), axis=0)

    E = tf.reduce_sum(tf.multiply(a, tf.subtract(x, c)), axis=1, name='encoding_vectors')

    return E


# Return custom gradient wrt each input of the op.
def grad_func(op, gradE, tmp1, tmp2, tmp3):
    X = op.inputs[1]
    C = op.inputs[2]
    S = op.inputs[3]

    # gradA = (gradE.unsqueeze(1) *
    #           (X.unsqueeze(2).expand({X.size(0), X.size(1), C.size(0), C.size(1)}) -
    #            C.unsqueeze(0).unsqueeze(0))).sum(3);
    e = tf.expand_dims(gradE, axis=1)
    x = tf.broadcast_to(tf.expand_dims(X, axis=2),
                        [batch_size, X.shape[1], C.shape[0], C.shape[1]])
    c = tf.expand_dims(tf.expand_dims(C, axis=0), axis=0)
    gradSL = tf.reduce_sum(tf.multiply(e, tf.subtract(x, c)), axis=3)

    # gradX = at::bmm(A, gradE); -> Batch matrix multiplication
    gradX = tf.matmul(A, gradE)
    # gradC = (-gradE * A.sum(1).unsqueeze(2)).sum(0);
    a = tf.expand_dims(tf.reduce_sum(A, axis=1), axis=2)
    gradC = tf.reduce_sum(tf.multiply(-gradE, a), axis=0)

    # tmp = (2 * gradSL * S.view({1, 1, C.size(0)})).unsqueeze(3) * \
    #       (X.unsqueeze(2).expand({X.size(0), X.size(1), C.size(0), C.size(1)}) -
    #        C.unsqueeze(0).unsqueeze(0));
    s = tf.reshape(S, [1, 1, C.shape[0]])
    t = tf.expand_dims((2 * gradSL * s), axis=3)
    # x = tf.broadcast_to(tf.expand_dims(X, axis=2),
    #                     [batch_size, X.shape[1], C.shape[0], C.shape[1]])
    # c = tf.expand_dims(tf.expand_dims(C, axis=0), axis=0)
    tmp = tf.multiply(t, tf.subtract(x, c))

    # GX = tmp.sum(2);
    GX = tf.multiply(tf.reduce_sum(tmp, axis=2), gradX)
    # GC = tmp.sum(0).sum(0);
    GC = tf.multiply(tf.reduce_sum(tf.reduce_sum(tmp, axis=0), axis=0), gradC)
    # GS = (gradSL * (SL / S.view({1, 1, C.size(0)}))).sum(0).sum(0);
    GS = tf.reduce_sum(tf.reduce_sum(tf.multiply(gradSL, tf.divide(A, s)), axis=0), axis=0)

    return None, GX, GC, GS
