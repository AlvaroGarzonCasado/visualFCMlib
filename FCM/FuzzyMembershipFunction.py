# -*- coding: utf-8 -*-

"""
This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation, either version 3 of the License, or (at your option) any later 
version.

This program is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program. If not, see http://www.gnu.org/licenses/

Author: Pablo Cano Marchal

"""

from __future__ import division
import numpy as np
#import matplotlib.pyplot as plt
from collections import OrderedDict as odict
from itertools import izip
import scipy.optimize as op


def ajusta_tolerancia(e):

    if np.abs(e) < 1e-5:
        return 0

    else:
        return e


def find_all_zeros(f, a, b):

    zeros = []

    if np.abs(a - b) > 1e-3:
        x = np.linspace(a, b, 100)
        y = [ajusta_tolerancia(f(xx)) for xx in x]

        zero_crossings2 = list(np.where(np.diff(np.sign(y)))[0])

        if not zero_crossings2:
            return [] 
        zero_crossings2.append(len(x) - 1)

        for i in range(len(zero_crossings2) - 1):
            aa = x[zero_crossings2[i]]
            bb = x[zero_crossings2[i + 1]]

            zeros.append(op.brentq(f, aa, bb))
    else:

        if np.abs(f(a)) < 1e-5:
            zeros.append(a)
    return zeros


def get_mu_triang(x, c, s1, s2):
    return np.max([0, np.min([(x - s1) / (c - s1), (x - s2) / (c - s2)])])


class TriangularMembershipFunction(object):

    def __init__(self,
                 min_soporte,
                 max_soporte,
                 pico_soporte,
                 name='triangular'):

        self.min_s = min_soporte
        self.max_s = max_soporte
        self.peak = pico_soporte
        self.s_d = self.peak - self.min_s
        self.s_i = self.max_s - self.peak

        self.name = name

    def __call__(self, value):

        if hasattr(value, '__iter__'):
            if not hasattr(value, '__call__'):
                return [self._evaluate_membership_crisp_value(v)
                        for v in value]
            else:
                return [self._evaluate_ns_fuzzification(v) for v in value]
        else:
            if not hasattr(value, '__call__'):
                return self._evaluate_membership_crisp_value(value)
            else:
                return self._evaluate_ns_fuzzification(value)

    def _evaluate_membership_crisp_value(self, crisp_value):

        if crisp_value < self.min_s or crisp_value > self.max_s:
            return 0
        else:
            x = self.peak - crisp_value

            if crisp_value == self.peak:
                return 1
            elif crisp_value == self.max_s or crisp_value == self.min_s:
                return 0
            elif x >= 0:  # hay que incluir el igual
                return 1 - x / self.s_d
            else:
                return 1 + x / self.s_i

    def _evaluate_ns_fuzzification(self, value_):

        def dif(x):
            cmf = self._evaluate_membership_crisp_value(x)
            if np.abs(cmf) < 1e-6:
                return 0
            else:
                return cmf - value_(x)

        intersecciones = []


        intersecciones.extend(find_all_zeros(dif, self.min_s, self.peak))
        intersecciones.extend(find_all_zeros(dif, self.peak, self.max_s))

        return np.max([self._evaluate_membership_crisp_value(e)
                       for e in intersecciones])

    def __str__(self):
        s = self.name
        s += '  --  min: ' + str(self.min_s)
        s += ' peak: ' + str(self.peak)
        s += ' max: ' + str(self.max_s)
        return s


class FuzzyVariable(object):


    def __init__(self,
                 universe_discourse,
                 labels,
                 membership_function_type,
                 name='Fuzzy Variable',
                 anchors=[1, 2, 3, 4, 5],
                 ns_fuzzification_func=None):

        self.universe_discourse = universe_discourse
        self.labels = labels
        self.membership_function_type = membership_function_type
        self.name = name
        self.anchors = anchors
        self.ns_fuzzification_func = ns_fuzzification_func

        self._genera_membership_functions_value_set()

    def _genera_membership_functions_value_set(self):

        num_elementos = len(self.labels)
        umin, umax = self.universe_discourse
        rango = umax - umin
        ancho_intervalo = rango / (num_elementos - 1)
        self.membership_functions = odict()
        for i, label in zip(self.anchors, self.labels):

            i = i - 1 
            min_s = umin + (i - 1) * ancho_intervalo
            peak = umin + i * ancho_intervalo
            max_s = umin + (i + 1) * ancho_intervalo


            self.membership_functions[label] =\
                    TriangularMembershipFunction(min_s, max_s, peak, label)

    def fuzzify(self, input_):

        if not self.ns_fuzzification_func:
            return self.s_fuzzify(input_)
        else:
            return self.ns_fuzzify(input_)

    def ns_fuzzify(self, input_):

        membership_values_array = [
            memb_func(lambda x: self.ns_fuzzification_func(x, input_))
            for memb_func in self.membership_functions.itervalues()
        ]
        return membership_values_array

    def s_fuzzify(self, crisp_value):

        membership_values_array = [
            memb_func(crisp_value)
            for memb_func in self.membership_functions.itervalues()
        ]
        return membership_values_array

    def defuzzify(self, membership_values_array):

        f = 0
        for membership, membership_func\
            in izip(membership_values_array,
                    self.membership_functions.itervalues()):

            f += membership * membership_func.peak
        den = sum([np.abs(m) for m in membership_values_array])
        if np.abs(den) < 1e-4:
            return f
        else:
            return f / den


def test_TriangularMembershipFunction(a=5, b=8, c=6):
    t = TriangularMembershipFunction(a, b, c)
    x = np.linspace(a, b, num=20)
    res = [t(xx) for xx in x]
    print res

    return t


def test_FuzzyVariable():
    from DefinicionVariables import UNIVERSE_DISCOURSE_DEFAULT
    from DefinicionVariables import LISTA_LABELS_DEFAULT
    s = FuzzyVariable(UNIVERSE_DISCOURSE_DEFAULT, LISTA_LABELS_DEFAULT,
                      TriangularMembershipFunction)

    for x in np.linspace(*s.universe_discourse, num=20):
        fuzzified = s.fuzzify(x)
        print 'valor', x, 'fuzzy: ', fuzzified, 'defuzzified: ',\
          s.defuzzify(fuzzified)
    return s


if __name__ == '__main__':

    s = test_FuzzyVariable()