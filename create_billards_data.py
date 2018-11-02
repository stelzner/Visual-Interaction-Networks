"""
This script comes from the RTRBM code by Ilya Sutskever from
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
"""

from numpy import *
from scipy import *
import scipy.io

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

shape_std = shape


def shape(A):
    if isinstance(A, ndarray):
        return shape_std(A)
    else:
        return A.shape()


size_std = size


def size(A):
    if isinstance(A, ndarray):
        return size_std(A)
    else:
        return A.size()


det = linalg.det


def new_speeds(m1, m2, v1, v2):
    new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2


def norm(x): return sqrt((x ** 2).sum())


def sigmoid(x): return 1. / (1. + exp(-x))


SIZE = 10


# size of bounding box: SIZE X SIZE.

def bounce_n(T=128, n=2, r=None, m=None):
    if r is None:
        r = array([1.2] * n)
    if m is None:
        m = array([1] * n)
    # r is to be rather small.
    X = zeros((T, n, 2), dtype='float')
    y = zeros((T, n, 2), dtype='float')
    v = randn(n, 2)
    v = v / norm(v) * .5
    good_config = False
    while not good_config:
        x = 2 + rand(n, 2) * 8
        good_config = True
        for i in range(n):
            for z in range(2):
                if x[i][z] - r[i] < 0:
                    good_config = False
                if x[i][z] + r[i] > SIZE:
                    good_config = False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(x[i] - x[j]) < r[i] + r[j]:
                    good_config = False

    eps = .5
    for t in range(T):
        # for how long do we show small simulation

        v_prev = copy(v)

        for i in range(n):
            X[t, i] = x[i]
            y[t, i] = v[i]

        for mu in range(int(1 / eps)):

            for i in range(n):
                x[i] += eps * v[i]

            for i in range(n):
                for z in range(2):
                    if x[i][z] - r[i] < 0:
                        v[i][z] = abs(v[i][z])  # want positive
                    if x[i][z] + r[i] > SIZE:
                        v[i][z] = -abs(v[i][z])  # want negative

            for i in range(n):
                for j in range(i):
                    if norm(x[i] - x[j]) < r[i] + r[j]:
                        # the bouncing off part:
                        w = x[i] - x[j]
                        w = w / norm(w)

                        v_i = dot(w.transpose(), v[i])
                        v_j = dot(w.transpose(), v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)

                        v[i] += w * (new_v_i - v_i)
                        v[j] += w * (new_v_j - v_j)

    return X, y


def ar(x, y, z):
    return z / 2 + arange(x, y, z, dtype='float')


def draw_image(X, res, r=None):
    T, n = shape(X)[0:2]
    if r is None:
        r = array([1.2] * n)

    A = zeros((T, res, res, 3), dtype='float')

    [I, J] = meshgrid(ar(0, 1, 1. / res) * SIZE, ar(0, 1, 1. / res) * SIZE)

    for t in range(T):
        for i in range(n):
            A[t, :, :, i] += exp(-(((I - X[t, i, 0]) ** 2 +
                                    (J - X[t, i, 1]) ** 2) /
                                   (r[i] ** 2)) ** 4)

        A[t][A[t] > 1] = 1
    return A


def bounce_mat(res, n=2, T=128, r=None):
    if r is None:
        r = array([1.2] * n)
    x, y = bounce_n(T, n, r)
    A = draw_image(x, res, r)
    return A, y


def bounce_vec(res, n=2, T=128, r=None, m=None):
    if r is None:
        r = array([1.2] * n)
    x, y = bounce_n(T, n, r, m)
    V = draw_image(x, res, r)
    y = concatenate((x, y), axis=2)
    return V.reshape(T, res, res, 3), y


# make sure you have this folder
logdir = './img'


def show_sample(V):
    T = V.shape[0]
    for t in range(T):
        plt.imshow(V[t])
        # Save it
        fname = logdir + '/' + str(t) + '.png'
        plt.savefig(fname)


if __name__ == "__main__":
    res = 32
    T = 100
    N = 1000
    dat = empty((N, T, res, res, 3), dtype=float)
    dat_y = empty((N, T, 3, 4), dtype=float)
    for i in range(N):
        dat[i], dat_y[i] = bounce_vec(res=res, n=3, T=T)
        print('training example {} / {}'.format(i, N))
    data = dict()
    data['X'] = dat
    data['y'] = dat_y
    scipy.io.savemat('billards_balls_training_data.mat', data)

    N = 200
    dat = empty((N, T, res, res, 3), dtype=float)
    dat_y = empty((N, T, 3, 4), dtype=float)
    for i in range(N):
        dat[i], dat_y[i] = bounce_vec(res=res, n=3, T=T)
        print('test example {} / {}'.format(i, N))
    data = dict()
    data['X'] = dat
    data['y'] = dat_y
    scipy.io.savemat('billards_balls_testing_data.mat', data)

    # show one video
    show_sample(dat[0])
    print(dat_y[0, :])
