# -*- coding: utf-8 -*-
import sys
sys.path.append('./FCM')

import FuzzyCognitiveMap as fcm
import RelationMatrixPortfolio as mr
import random
import unidecode
import numpy as np
from collections import OrderedDict as odict
from tabulate import tabulate
from Variable import Variable
import DefinicionVariables as v
from pdb import set_trace as bp
import VisualFCMlib as vf

RELACIONES_RITMO_PROD = [
    fcm.Relation(v.RITMO_PRODUCCION, v.TIEMPO_RESIDENCIA,
     mr.relacion_bivalente_inversa(),
     mr.NORMAL),

    fcm.Relation(v.RITMO_PRODUCCION, v.CAUDAL_ENTRADA_AGUA,
     mr.relacion_bivalente(),
     mr.NORMAL),

    fcm.Relation(v.RITMO_PRODUCCION, v.CAUDAL_ENTRADA_SOLIDOS,
     mr.relacion_bivalente(),
     mr.NORMAL)]

RELACIONES_OIL_CONTENT = [
    ]

RELACIONES_HUMEDAD_CONTENT = [

    ]

RELACIONES_AGUA_DECANTER = [
    fcm.Relation(v.ADICION_AGUA_DECANTER, v.CAUDAL_ENTRADA_AGUA,
     mr.relacion_univalente_incremento(),
     mr.NORMAL)]

RELACIONES_NIVEL_SOLIDOS = [
    fcm.Relation(v.CAUDAL_ENTRADA_SOLIDOS, v.LINEA_SEPARACION,
     mr.relacion_bivalente(),
     mr.NORMAL),

    fcm.Relation(v.VELOCIDAD_DIFERENCIAL, v.LINEA_SEPARACION,
     mr.relacion_bivalente_inversa(),
     mr.NORMAL)]

RELACIONES_NIVEL_AGUA = [
    fcm.Relation(v.CAUDAL_ENTRADA_AGUA, v.LINEA_SEPARACION,
     mr.relacion_bivalente(),
     mr.NORMAL)]

RELACIONES_VELOCIDAD = [
    fcm.Relation(v.VELOCIDAD_PRINCIPAL, v.ANCHURA_INTERFASE_SEPARACION,
     mr.relacion_bivalente_inversa(),
     mr.FUERTE),

    fcm.Relation(v.ANCHURA_INTERFASE_SEPARACION, v.AGOTAMIENTO,
     mr.relacion_bivalente_inversa(),
     mr.NORMAL)]

RELACIONES_VISCOSIDAD = [
    fcm.Relation(v.VISCOSIDAD_PASTA, v.FACILIDAD_MOVIEMIENTO_FLUIDOS,
     mr.relacion_bivalente_inversa(),
     mr.NORMAL)]

RELACIONES_MOVIMIENTO = [
    fcm.Relation(v.ESTADO_BATIDO, v.FACILIDAD_MOVIEMIENTO_FLUIDOS,
     mr.relacion_bivalente(['2', '2', '+', '+', '+']),
     mr.NORMAL),

    fcm.Relation(v.FACILIDAD_MOVIEMIENTO_FLUIDOS, v.ANCHURA_INTERFASE_SEPARACION,
     mr.relacion_bivalente_inversa(['+', '+', '+', '2', '2']),
     mr.FUERTE),

    fcm.Relation(v.FACILIDAD_MOVIEMIENTO_FLUIDOS, v.LIMPIEZA_ACEITE,
     mr.relacion_bivalente(['2', '2', '+', '+', '+']),
     mr.NORMAL)]

RELACIONES_NIVEL_ACEITE = [

    ]

RELACIONES_OFFSET_PRESILLA = [
    fcm.Relation(v.LINEA_SEPARACION, v.OFFSET_PRESILLA_LINEA_SEPARACION,
     mr.relacion_bivalente(),
     mr.FUERTE),

    fcm.Relation(v.POSICION_PRESILLAS, v.OFFSET_PRESILLA_LINEA_SEPARACION,
     mr.relacion_bivalente(),
     mr.FUERTE),

    fcm.Relation(v.OFFSET_PRESILLA_LINEA_SEPARACION, v.AGOTAMIENTO,
     '''
     + 0 0 0 0
     0 + 0 0 0 
     0 0 0 0 0 
     0 0 0 0 0 
     0 0 + + +
     ''',
     mr.FUERTE),

    fcm.Relation(v.OFFSET_PRESILLA_LINEA_SEPARACION, v.LIMPIEZA_ACEITE,
     '''
     0 0 0 0 +
     0 0 0 + 0 
     0 0 0 0 0 
     0 0 + 0 0 
     + + 0 0 0
     ''',
     mr.FUERTE)]

RELACIONES_TIEMPO_RESIDENCIA = [
    fcm.Relation(v.TIEMPO_RESIDENCIA, v.AGOTAMIENTO,
     mr.relacion_bivalente(),
     mr.NORMAL),

    fcm.Relation(v.TIEMPO_RESIDENCIA, v.LIMPIEZA_ACEITE,
     mr.relacion_bivalente(),
     mr.NORMAL)]

RELACIONES_COMPLETO = RELACIONES_RITMO_PROD\
                        + RELACIONES_OIL_CONTENT\
                        + RELACIONES_HUMEDAD_CONTENT\
                        + RELACIONES_AGUA_DECANTER\
                        + RELACIONES_NIVEL_SOLIDOS\
                        + RELACIONES_NIVEL_AGUA\
                        + RELACIONES_VELOCIDAD\
                        + RELACIONES_VISCOSIDAD\
                        + RELACIONES_MOVIMIENTO\
                        + RELACIONES_NIVEL_ACEITE\
                        + RELACIONES_OFFSET_PRESILLA\
                        + RELACIONES_TIEMPO_RESIDENCIA\

modelo = fcm.FuzzyCognitiveMap(RELACIONES_COMPLETO,
                               name='sis_base_sin')

# Genera la figura 7
modelo.gen_graph(orden_grafo = 'TB')  

dic_valores_iniciales = {
    'Production Rate': 2,
    'Water Addition to Decanter': 1.8,
    'Main velocity': 2,
    'Overflow Weirs Position': 1,
    'Differential Speed': 5,
	'Kneading State': 4,
	'Paste Viscosity': 3
    }

dic_res = modelo.compute(dic_valores_iniciales)

# Genera la figura 8
vf.visualize_results_graph(modelo, '_sis_simp_base', show_bars = False, palette = 'paper', show_relation_matrices = False, orden_grafo = 'TB', show_node_values = False) 
# Genera la figura 9
vf.visualize_results_graph(modelo, '_sis_simp', show_bars = True, palette = 'paper', show_relation_matrices = 'Impact', orden_grafo = 'LR') 

# Genera la figura 10
vf.extract_node(modelo, ['Yield'], 'extback1', mode = 'back', grade = 1) 
# Genera la figura 11
vf.extract_node(modelo, ['Overflow Weirs Position', 'Interphase width'], 'extfor2', mode = 'for', grade = 2) 
# Genera la figura 12
vf.extract_node(modelo, ['Water income flow'], 'extcomp2', mode = 'comp') 

# Genera la figura 13
vf.get_total_weight(modelo, 'Production Rate', 'Yield','weight_w')
# Genera los datos de la tabla IV
dic = vf.obtain_weights(modelo, dic_valores_iniciales.keys(), ['Oil Cleannes','Yield'])

#vf.parametric(modelo, ['Production Rate', 'Water Addition to Decanter','Main velocity'], ['Interphase width','Yield'], 0.5)  

# Genera la figura 14
vf.simplify(modelo, grade = 1, mode = 'backward', name = 'simp_back_1_w')
# Genera la figura 15
vf.simplify(modelo, grade = 2, mode = 'backward', name = 'simp_back_2_w')

#vf.simplify(modelo, grade = 1, mode = 'forward', name = 'simp_for_1_w')