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
import unidecode
import numpy as np
from collections import OrderedDict as odict
from tabulate import tabulate
from pdb import set_trace as bp

import FuzzyMembershipFunction as FMF

lang = 'ingles'

LISTA_LABELS_DEFAULT = ['VL', 'L', 'M', 'H', 'VH']
UNIVERSE_DISCOURSE_DEFAULT = (1, 5)
parts = {'espanol': 'Nodo ', 'ingles': 'Node '}
part  = parts[lang]

def estandariza_nombre(nombre):
    '''
    Función auxiliar para estandarizar el nombre de las variables
    pasando a mayúsculas y eliminando las tildes
    '''
    nnn = nombre
    nnn = unidecode.unidecode(unicode(nombre))
    nnn = nnn.replace(' ', '_')
    return nnn.upper()


class Variable(object):
    '''
    Clase para almacenar la informacion relativa a las variables
    '''
    def __init__(self, espanol, ingles='', latex='',
                 unidades='uds',
                 universe_discourse=UNIVERSE_DISCOURSE_DEFAULT,
                 labels=LISTA_LABELS_DEFAULT,
                 membership_func=FMF.TriangularMembershipFunction,
                 anchors=[1, 2, 3, 4, 5],
                 ns_fuzzification_func=None,
                 normalizar_sc=False):

        self.nombres               = {'espanol': unidecode.unidecode(unicode(espanol)), 'ingles': unidecode.unidecode(unicode(ingles))} 
        self.nombre                = self.nombres[lang]
        self.unidades              = unidades.replace('%', '\\%')
        self.universe_discourse    = universe_discourse
        self.labels                = labels
        self.anchors               = anchors
        self.membership_func       = membership_func
        self.ns_fuzzification_func = ns_fuzzification_func
        self.fuzzy_variable        = FMF.FuzzyVariable(
            self.universe_discourse, self.labels,
            self.membership_func,
            ns_fuzzification_func=self.ns_fuzzification_func,
            anchors=self.anchors)

        if anchors is None:
            self.anchors = [1, 2, 3, 4, 5]

        else:
            self.anchors = anchors
        
        self.normalizar_sc = normalizar_sc

    def __str__(self): 

        return self.nombre.encode('ascii', 'replace')

    def __repr__(self): 

        return self.nombre.encode('ascii', 'replace')

    def __call__(self, argument):

        if hasattr(argument, '__iter__'):
            return self.defuzzify(argument)

        else:
            return self.fuzzify(argument)

    def fuzzify(self, argument):

        if not self.normalizar_sc:
            return self.fuzzy_variable.fuzzify(argument)

        else:
            a = np.array(self.fuzzy_variable.fuzzify(argument))

            if sum(a) > 0:
                return a/sum(a)

            else:
                return a

    def defuzzify(self, argument):
 
        return self.fuzzy_variable.defuzzify(argument)

    def iter_universe(self, npoints=21):

        for i in np.linspace(*self.universe_discourse,
                             num=npoints):
            yield i

    def show_fuzzy_properties(self):

        print self.fuzzy_variable

    def __hash__(self):

        return hash((self.nombre))

    def __eq__(self, other):

        try:
            return (self.nombre) == (other.nombre)

        except AttributeError:
            return (self.nombre) == (other)

    def _get_nombre_acortado(self):

        ns = self.nombre.split(' ')
        return '_'.join((s[0:4] for s in ns))