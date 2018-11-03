#!/usr/bin/env python
# -*- coding: utf-8 -*-


from collections import defaultdict as ddict
import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd
import networkx as nx
import copy
import itertools
import os
import logtools
from logtools import timethis, log, logger
from RelationMatrixPortfolio import DEFAULT_DIC_MAPPING_VALORES

from pdb import set_trace as bp
from PIL import Image
from math import fsum as fs
import sys
import Variable as v 

__version__ = '3.0'
part = v.part

def iterator_diccionario_escenario(diccionario_base,
                                   variables_barridas,
                                   numero_puntos_en_universo=3):

    lista_combinacion_valores = iterator_cartesian_product(
        variables_barridas,
        numero_puntos_en_universo)

    for escenario in lista_combinacion_valores:
        dic_escenario = diccionario_base
        for variable, valor in escenario:
            dic_escenario[variable] = valor
        yield dic_escenario


def iterator_cartesian_product(lista_variables, numero_puntos=5):

    lista_auxiliar = []
    for variable in lista_variables:
        lista_auxiliar.append(
            [(variable, i) for i in
                variable.iter_universe(numero_puntos)])
    return itertools.product(*lista_auxiliar)


class Relation(object):


    def __init__(self,
                 predecessor,
                 successor,
                 Rp,
                 omega,
                 dic_mapping_valores=DEFAULT_DIC_MAPPING_VALORES):
        self.predecessor = predecessor
        self.successor = successor
        self.omega = omega
        self.dic_mapping_valores = dic_mapping_valores
        
        if type(Rp) is tuple:
            R = Rp[0] 
            self.rel_type = Rp[1]
        else:
            R = Rp
            self.rel_type = 'c'

        self.R = self._parse_relational_matrix_string_to_matrix(R)
        self.Romega = self.R * self.omega
        self.Romega_sparse = coo_matrix(self.Romega)

    def _parse_relational_matrix_string_to_matrix(self, rel_string):

        if type(rel_string) is str:

            filas = [fila.strip()
                     for fila in rel_string.split('\n') if fila.strip()]
            rel_matrix = [[self._helper_convert_value(elemento)
                           for elemento in fila.split(' ')]
                          for fila in filas]
            return np.matrix(rel_matrix)
        else:  
            return rel_string

    def _helper_convert_value(self, valor):

        try:
            return self.dic_mapping_valores[valor]
        except KeyError:
            return float(valor)

    def get_sparse_representation(self):

        data = self.Romega_sparse.data
        rows_i, cols_i = self.Romega.nonzero()
        return [(r, c, d) for r, c, d in zip(rows_i, cols_i, data)]

    def __str__(self):

        a = self.Romega.tolist()
        s = '\n'.join(' '.join('{:.2f} '.format(cell) for cell in row) for row in a)
        return s


class Nodo(object):


    def __init__(self, variable):

        self.variable = copy.deepcopy(variable)

        self.iter_universe = self.variable.iter_universe
        self.name = self.variable.nombre

        self.nombre = self.variable.nombre 

        self.impact = []
        self.impact_normalized = []
        self.impact_normalization_factor = -1
        self.fuzzy_membership_vector = []
        self.crisp_value = -1
        self.crisp_input = -1
        self.lista_R = []
        self.Rtotal = None
        self.vec_pesos = []
        self.ones = None
        self.kernels = None
        self.layer = -1
        self.predecessors = []


    def populate_ones_and_kernels(self):

        if self.Rtotal is not None:
            self.ones = np.ones((self.Rtotal.shape[0], 1))
            self.kernels = np.array(self.variable.anchors).reshape((
                self.Rtotal.shape[0], 1))
        else:
            raise ValueError(
                'Rtotal must be assigned before calling populate_ones_and_kernels')

    @log
    def compute_value(self, xtotal):

        logger.debug('Computing value of node %s', self.name)
        
        self.impact = np.dot(self.Rtotal, xtotal)

        self.impact_normalization_factor = np.dot(self.ones.transpose(),
                                                  self.impact)
        self.impact_normalized = self.impact / self.impact_normalization_factor
        
        self.crisp_value = np.asscalar(np.dot(self.kernels.transpose(),
                                              self.impact_normalized))

        self.fuzzy_membership_vector = np.array(self.variable(
            self.crisp_value)).reshape((-1, 1))
        
        logger.debug('Final value of node %s: %s %s', self.name,
                      self.crisp_value, self.fuzzy_membership_vector)


        xp = np.reshape(xtotal,(-1,5))

        xparcial       = [xp[i] for i in range(len(xp))]
        Romega_parcial = [self.lista_R[i] for i in range(len(self.lista_R))]


        impacto_parcial = []
        for i in range(len(self.lista_R)): 
            impacto_parcial.append(np.dot(Romega_parcial[i], xparcial[i]) / self.impact_normalization_factor)

        for i in range(len(impacto_parcial)):
            nuevo_vec = np.squeeze(np.asarray(impacto_parcial[i]))
            tam_vec = len(nuevo_vec)

            for j in range(tam_vec):
                if nuevo_vec[j] != 0:
                    nuevo_vec[j] = round(nuevo_vec[j], 4)
                 
            self.vec_pesos.append(nuevo_vec)

        return self.vec_pesos

    def load_value(self, value):

        if not hasattr(value, '__iter__'):
            self.crisp_value = value
            self.fuzzy_membership_vector = np.array(self.variable(
                self.crisp_value)).reshape((-1, 1))
        else:
            self.fuzzy_membership_vector = np.array(value).reshape((
                self.Rtotal.shape[0], 1))
            self.crisp_value = self.variable(self.fuzzy_membership_vector)

    def _gen_str(self):

        s = part + str(self.variable.nombre)
        return s

    def __str__(self):

        return self._gen_str()

    def __repr__(self):

        return self._gen_str()

    def __hash__(self):

        return hash((self.variable.nombre))

    def __eq__(self, other):

        try:
            return (self.name) == (other.nombre)
        except AttributeError:
            return (self.name) == (other)

    
class FuzzyCognitiveMap(object):


    def __init__(self,
                 relations_list,
                 name='sistema',
                 output_path='./',
                 verbose=False,
                 extra_string='',
                 kernels = range(1,6)):
        self.name = name
        self.output_path = output_path
        self.verbose = verbose
        self.relations_list, lista_nodos = (
            self._adapt_relation_lists(relations_list))
        self.datos_nodos = {nodo: nodo for nodo in lista_nodos}
        self.G = nx.DiGraph()
        self.vis = 'ko'
        self.part = part
        
        for nodo in lista_nodos:
            self.G.add_node(nodo, d=nodo)

        for relation in self.relations_list:
            self.G.add_edge(relation.predecessor,
                            relation.successor,
                            rel=relation)

        self.nodos_salida = [n
                             for n in self.G.nodes()
                             if self.G.out_degree(n) == 0]
        self.nodos_sucesores = [n
                                for n in self.G.nodes()
                                if self.G.in_degree(n) > 0]
        self.nodos_predecesores = [n
                                   for n in self.G.nodes()
                                   if n not in self.nodos_salida]
        self.nodos_entrada = list(set(self.G.nodes()) - set(
            self.nodos_sucesores))
        self.nodos_intermedios = list(set(self.nodos_sucesores) - set(
            self.nodos_salida))


        for nodo in self.nodos_sucesores:
            self.datos_nodos[nodo].Rtotal, self.datos_nodos[nodo].lista_R = self._build_Rtotal(nodo)
            self.datos_nodos[nodo].populate_ones_and_kernels()
        
        self._assign_node_layer()

        self.extra_string = extra_string

        self.dic_nod_pred = {}
        self.aux = []
        self.kernels = kernels
        
        self.grafo = []
        
        logger.info('Model %s successfully built', self.name)

    def _assign_node_layer(self):

        self.dict_layer = ddict(list)
        self.dict_layer[0] = list(self.nodos_entrada)
        for nodo in self.nodos_entrada:
            nodo.layer = 0
        for nodo in self.nodos_salida:
            layer = self._assign_node_layer_low_level_recursive(nodo)
            self.dict_layer[layer].append(nodo)
        for nodo in self.nodos_intermedios:
            self.dict_layer[nodo.layer].append(nodo)
        self.layer_max = np.max(self.dict_layer.keys())

    def _assign_node_layer_low_level_recursive(self, nodo):

        niveles_nodos_anteriores = [
            x.layer
            if x.layer > -1 else self._assign_node_layer_low_level_recursive(x)
            for x, _ in self.G.in_edges(nodo)
        ]
        self.datos_nodos[nodo].layer = np.max(niveles_nodos_anteriores) + 1
        return self.datos_nodos[nodo].layer

    def _load_values(self, dict_values):

        for nodo, valor in dict_values.iteritems():
            self.datos_nodos[nodo].load_value(valor)

    def _adapt_relation_lists(self, relations_list):

        lista_variables = [[rel.predecessor, rel.successor]
                           for rel in relations_list]
        variables = set(itertools.chain(*lista_variables))
        dic_variable_nodo = {k: Nodo(k) for k in variables}
        new_rels = [Relation(dic_variable_nodo[rel.predecessor],
                             dic_variable_nodo[rel.successor], (rel.R, rel.rel_type),
                             rel.omega) for rel in relations_list]
        
        return new_rels, dic_variable_nodo.values()

    def _build_Rtotal(self, nodo):

        lista_R = []
        for antecesor, sucesor in self.G.in_edges(nodo):
            lista_R.append(self.G[antecesor][sucesor]['rel'].Romega)
        Rtotal = np.concatenate(lista_R, axis=1)
        return Rtotal, lista_R

    def _get_predecessor_values(self, nodo):

        lista_x = []
        for antecesor, sucesor in self.G.in_edges(nodo):
            lista_x.append(self.datos_nodos[antecesor].fuzzy_membership_vector)
        xtotal = np.concatenate(lista_x, axis=0)
        logger.debug('_get_predecessor_values: %s', nodo)
        logger.debug('lista_x: %s', lista_x)
        logger.debug('xtotal: %s', xtotal)
        return xtotal


    def compute(self, dic_valores, do_checks=True):


        self.dic_valores = dic_valores

        try:
            if do_checks:
                set_nodos_valores = set(dic_valores.keys())
                set_nodos_entrada = set(self.nodos_entrada)

                if not set_nodos_valores.issubset(set_nodos_entrada):
                    logger.warning(
                        'Existen nodos con valor asignado que no son nodos de entrada\n %s',
                        set_nodos_valores.difference(set_nodos_entrada))


                if not set_nodos_entrada.issubset(set_nodos_valores):
                    logger.warning(
                        'Faltan valores para calcular el sistema\n %s',
                        set_nodos_entrada.difference(set_nodos_valores))

            self._load_values(dic_valores)
            
            for layer in range(1, self.layer_max + 1):
                logger.debug('Computing layer  %s .......', layer)
                for nodo in self.dict_layer[layer]:
                    logger.debug('Computing node %s ', nodo)
                    xtotal = self._get_predecessor_values(nodo)
                    logger.debug('xtotal en compute: %s', xtotal)
                    vec_pesos = nodo.compute_value(xtotal)
                    
            for nodo in self.nodos_sucesores:

                self.dic_nod_pred[nodo] = {'predecesores': [pred for pred in self.G.predecessors(nodo)], 'vec_pesos': nodo.vec_pesos}

            self.aux = self.dic_nod_pred
            self.dic_nod_pred = {}

            for key in self.aux.iterkeys():
                itera = range(len(self.aux[key]['predecesores']))
                self.dic_nod_pred[key] = {self.aux[key]['predecesores'][i]: self.aux[key]['vec_pesos'][i] for i in itera}

            self.dic_pesos_back = {}
            for nodo in self.nodos_sucesores:
                partial = dict()
                for pred in self.dic_nod_pred[nodo].iterkeys():
                    partial[pred] = fs(self.dic_nod_pred[nodo][pred]) * 100
                self.dic_pesos_back[nodo] = partial                  

            self.dic_pesos_for = dict()
            for nodo in self.nodos_entrada + self.nodos_intermedios:
                partial = dict()
                for key in self.dic_pesos_back.iterkeys():
                    for value in self.dic_pesos_back[key]:     
                        if value == nodo:
                            partial[key] = self.dic_pesos_back[key][value]
                self.dic_pesos_for[nodo] = partial

            return copy.deepcopy(self.values())
            

        except KeyError as e:
            logger.error(
                '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            %s no pertenece al modelo. No se puede calcular el valor del sistema.'
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!''',
                e.message)
            return -1

    def gen_graph(self,
                  path=None,
                  show_relation_matrices=False,
                  show_node_values=False,
                  show_contribution_values=False,
                  recompute_node_locations=True,
                  orden_grafo='LR',
                  fontsize=9,
                  func_texto_nodo=lambda x: unicode(x.ingles)):
        

        G = nx.nx_agraph.to_agraph(self.G)
        G.graph_attr['overlap'] = 'voronoi'
        G.graph_attr['splines'] = 'true'
        G.graph_attr['rankdir'] = orden_grafo
        G.layout(prog='dot')
        if path is None:
            path = os.path.join(self.output_path, self.name)
        G.draw(path + '.pdf')
        G.write(path + '.dot')

    def values(self, crisp=True):

        return {nodo: nodo.crisp_value
                if crisp else nodo.fuzzy_membership_vector
                for nodo in self.G.nodes()}         

    def study_model_output(self,
                          study_variables,
                          fixed_value_nodes={},
                          default_value=3,
                          num_partition_points=5):

        base_scenario_dict = {nodo: fixed_value_nodes.get(nodo, default_value)
                              for nodo in self.nodos_entrada}
        logger.debug('base_scenario_dict %s', base_scenario_dict)
        l = []
        for escenario in iterator_diccionario_escenario(
                base_scenario_dict, study_variables, num_partition_points):
            l.append(self.compute(escenario))

        df = pd.DataFrame(l)
        return df