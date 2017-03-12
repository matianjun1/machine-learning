#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - mTengine <mtj334510983@163.com>

import numpy as np


class Perceptron(object):

    def __init__(self):
        self.trainingSets = {"1": [np.array([3, 3]), np.array([4, 3])], "-1": [np.array([1, 1])]}
        self.w = np.array([0, 0])
        self.b = 0

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

    def run(self):
        while not self.checkClassify():
            pass
        print("{0}x1 + {1}x2 + {2}".format(self.w[0], self.w[1], self.b))

p = Perceptron()
p.run()
