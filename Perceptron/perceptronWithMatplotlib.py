#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - mTengine <mtj334510983@163.com>

import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Perceptron(object):

    def __init__(self):
        self.trainingSets = {"1": [np.array([1.5, 5.2]),
                                   np.array([3.1, 3.2]),
                                   np.array([5.1, 2.2]),
                                   np.array([4.1, 4.2]),
                                   np.array([1.1, 5.2])],
                             "-1": [np.array([1.7, 1.6]),
                                    np.array([2.1, 2.2]),
                                    np.array([0.5, 2.7]),
                                    np.array([2.4, 2.2])]}
        self.w = np.array([1., -1.])
        self.b = 0
        for y, xList in self.trainingSets.items():
            for x in xList:
                ax.plot(x[0], x[1], 'ro' if y == "1" else 'bo')

    def checkX(self, y, x):
        return True if y * (self.w.dot(x) + self.b) > 0 else False

    def checkClassify(self):
        for y, xList in self.trainingSets.items():
            for x in xList:
                if not self.checkX(int(y), x):
                    self.refreshModel(int(y), x)
                    return False
        return True

    def refreshModel(self, y, x):
        self.w += y * x
        self.b += y
        print("{0}x1 + {1}x2 + {2}".format(self.w[0], self.w[1], self.b))

fig, ax = plt.subplots()
xdata = np.arange(-2, 7)
line, = ax.plot(xdata, xdata, lw=2)
p = Perceptron()


def data_gen():
    while not p.checkClassify():
        yield p


def init():
    ax.set_ylim(-2, 6)
    ax.set_xlim(-2, 6)
    return line,


def run(p):
    line.set_ydata((-p.b - p.w[0] * xdata) / p.w[1])

    return line,

ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=250, init_func=init)
plt.show()
