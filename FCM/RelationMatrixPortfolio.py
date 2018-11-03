# -*- coding: utf-8 -*-

DEBIL = 0.25
NORMAL = 0.5
FUERTE = 0.75
MUY_FUERTE = 1

DEFAULT_DIC_MAPPING_VALORES = {
    '++': 1,
    '+': 1,
    '0': 0,
    'a': 0.25,
    'b': 0.5, 'c': 0.75, 'd': 1, 'C': 1, 'D': 1.75, 'E': 3, 'S': 20,
    '02': 0.2, '04': 0.4, '06': 0.6, '08': 0.8, '05': 0.5, '075': 0.75,
    '1': 1, '1.5': 1.5, '2': 2, '3': 3}


def relacion_bivalente(lista_simbolos=['++', '++', '++', '++', '++']):
    base = '''
    {0} 0 0 0 0
    0 {1} 0 0 0
    0 0 {2} 0 0
    0 0 0 {3} 0
    0 0 0 0 {4}
    '''

    return base.format(*lista_simbolos), 'b'


def relacion_bivalente_inversa(lista_simbolos=['++', '++', '++', '++', '++']):
    base = '''
    0 0 0 0 {0}
    0 0 0 {1} 0
    0 0 {2} 0 0
    0 {3} 0 0 0
    {4} 0 0 0 0
    '''

    return base.format(*lista_simbolos), 'bi'


def relacion_univalente_decremento(lista_simbolos=['0', 'a', 'b', 'c', 'd']):
    base = '''
    {0} {1} {2} {3} {4}
    0 0 0 0 0
    0 0 0 0 0
    0 0 0 0 0
    0 0 0 0 0
    '''
    return base.format(*lista_simbolos), 'ud'


def relacion_univalente_incremento(lista_simbolos=['0', 'a', 'b', 'c', 'd']):
    base = '''
    0 0 0 0 0
    0 0 0 0 0
    0 0 0 0 0
    0 0 0 0 0
    {0} {1} {2} {3} {4}
    '''

    return base.format(*lista_simbolos), 'ui'


def relacion_sweetpoint(lista_simbolos=['++', '++', '++', '++', '++']):
    base = '''
    {0} 0 0 0 {4}
    0 0 0 0 0
    0 {1} 0 {3} 0
    0 0 0 0 0
    0 0 {2} 0 0
    '''

    return base.format(*lista_simbolos), 'sp'
