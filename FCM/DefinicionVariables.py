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


from Variable import Variable


LISTA_LABELS_DEFAULT = ['VL', 'L', 'M', 'H', 'VH']
UNIVERSE_DISCOURSE_DEFAULT = (1, 5)


COND_ECONOMICAS = Variable(u'Condiciones Económicas',
                           'Economic Aspects',
                           'C_e')
LIMPIEZA = Variable(u'Suciedad',
                    'Dirtiness',
                    'D_t')
ESTADO_FRUTO = Variable(u'Estado Fruto',
                        'Fruit State',
                        'E_f')

ESTADO_FRUTO_ENTRADA = Variable(u'Estado Fruto Entrada',
                        'Incoming Fruit State',
                        'E_f^I')

MADUREZ = Variable(u'Madurez',
                   'Ripeness',
                   'R_f')
OBJETIVO_ELABORACION = Variable(u'Objetivo Elaboración',
                                'Elaboration Objective',
                                'O_E')
HUMEDAD_ACEITUNA = Variable(u'Humedad Aceituna',
                            'Olive Moisture',
                            'H_o',
                            '%')
HUMEDAD_ACEITUNA_ENTRADA = Variable(u'Humedad Aceituna Entrada',
                            'Incoming Olive Moisture',
                            'H_o^I',
                            '%')
CRIBA = Variable(u'Tamaño Criba',
                 'Sieve Size',
                 'C_s',
                 'mm')

CRIBA_EQUIVALENTE = Variable(u'Tamano Criba Equivalente',
                             'Eq Sieve Size',
                             'C_{se}',
                             'mm')
ATRANQUE_MOLINO = Variable(u'Atranque Molino',
                           'Mill Blocking',
                           'B_m')
GRADO_MOLIENDA = Variable(u'Grado Molienda',
                          'Crushing Degree',
                          'G_m')
TEMPERATURA_BATIDO = Variable(u'Temperatura Batido',
                              'Kneading Temperature',
                              'T_b',
                              'ºC')
TIEMPO_BATIDO = Variable(u'Tiempo Batido',
                         'Kneading Time',
                         't_b',
                         'min')
ADICION_COADYUVANTE = Variable(u'Adición Coadyuvantes',
                               'Coadjuvant Addition',
                               'A_c',
                               '%')
ESTADO_BATIDO = Variable(u'Estado Batido',
                         'Kneading State',
                         'K_s')
FLUIDEZ_PASTA = Variable(u'Fluidez Pasta',
                         'Paste Fluidity',
                         'F_p')
ADICION_AGUA_DECANTER = Variable(u'Adición Agua Decanter',
                                 'Water Addition to Decanter',
                                 'F_W',
                                 '%')
POSICION_PRESILLAS = Variable(u'Posicion Presillas',
                              'Overflow Weirs Position',
                              'r_1',
                              'mm')
RITMO_PRODUCCION = Variable(u'Ritmo Producción',
                            'Production Rate',
                            'F',
                            '%')
VELOCIDAD_DIFERENCIAL = Variable(u'Velocidad Diferencial',
                                 'Differential Speed',
                                r'\Delta\omega',
                                'rpm')
AGOTAMIENTO = Variable(u'Agotamiento',
                       'Yield',
                       'X',
                       '% gsms')
FRUTADO = Variable(u'Nivel Frutado',
                   'Fruity',
                   'F')
AMARGO = Variable(u'Nivel Amargo',
                  'Bitter',
                  'B')
PICANTE = Variable(u'Nivel Picante',
                   'Pungent',
                   'P')
DEFECTO = Variable(u'Nivel Defecto',
                   'Defect',
                   'D')
ACIDEZ = Variable(u'Acidez',
                  'Acidity',
                  'A',
                  'º')
HUMEDAD_PASTA = Variable(u'Humedad Pasta',
                         'Paste Moisture Content',
                         'P_H',
                         '%')
SOLIDOS_PASTA = Variable(u'Solidos Pasta',
                         'Paste Solid Content',
                         'P_S',
                         '%')
ADICION_AGUA_BATIDORA = Variable(u'Adicion Agua Batidora',
                                 'Thermomixer Water Addition',
                                 'A_B',
                                 '%')
ADICION_AGUA_MOLINO = Variable(u'Adicion Agua Molino',
                               'Mill Water Addition',
                               'M_W',
                               '%')
EMULSION_PASTA = Variable(u'Emulsión Pasta',
                          'Paste Emulsion',
                          'P_E')
EMULSION_PASTA_CORREGIDA = Variable(u'Emulsión Pasta Corregida',
                                    'Corrected Paste Emulsion',
                                    'P_{EC}')
ACEITE_PASTA = Variable(u'Contenido Aceite Pasta',
                        'Paste Oil Content',
                        'X_o',
                        '%')
ACEITE_ACEITUNA = Variable(u'Contenido Aceite Aceituna',
                        'Oil Content of Olives',
                        'X_o',
                        '%')
CAUDAL_ENTRADA_ACEITE = Variable(u'Caudal entrada aceite',
                                 'Oil Income Flow',
                                 'F_o',
                                 '%')
CAUDAL_ENTRADA_AGUA = Variable(u'Caudal entrada agua',
                               'Water income flow',
                               'F_w',
                               '%')
CAUDAL_ENTRADA_SOLIDOS = Variable(u'Caudal entrada sólidos',
                                  'Solid income flow',
                                  'F_s',
                                  '%')
NIVEL_ACEITE = Variable(u'Nivel Aceite',
                        'Oil Pool Width',
                        'h_o')
NIVEL_AGUA = Variable(u'Nivel Agua',
                      'Water Pool Width',
                      'h_w')
NIVEL_SOLIDOS = Variable(u'Nivel Sólidos',
                         'Solid Width',
                         'h_s')
TIEMPO_RESIDENCIA = Variable(u'Tiempo de residencia',
                             'Residence Time',
                             't_r')
FACILIDAD_MOVIEMIENTO_FLUIDOS = Variable(u'Facilidad movimiento fluidos',
                                         'Fluid movement ease',
                                         'E')
LIMPIEZA_ACEITE = Variable(u'Limpieza Aceite',
                           'Oil Cleannes',
                           'O_c')
ATRANQUE_DECANTER = Variable(u'Atranque decánter',
                             'Decanter Blocking',
                             'B_d')
VISCOSIDAD_PASTA = Variable(u'Viscosidad pasta',
                            'Paste Viscosity',
                            r'mu_p')

RUPTURA_CELDAS = Variable(u'Ruptura celdas',
                          'Cells Breakage',
                          'R_c')

ALMACENAMIENTO_TOLVA = Variable(u'Tiempo Almacenamiento Tolva',
                          'Storage Time in Hopper',
                          'T_s')

FIRMEZA_PULPA = Variable(u'Firmeza Pulpa',
                          'Pulp Firmness',
                          'P_F')

PAR_DECANTER = Variable(u'Par Decanter',
                        'Decanter Torque',
                        r'\Tau_d')

VELOCIDAD_PRINCIPAL = Variable(u'Velocidad Principal',
                               'Main velocity',
                               r'\Omega')

ANCHURA_INTERFASE_SEPARACION = Variable(u'Anchura Interfase Separacion',
                                        'Interphase width',
                                        'W_{wo}')

FENOLES_TOTALES = Variable(u'Fenoles Totales',
                           'Total phenols',
                           'C_F')

VOLATILES = Variable(u'Contenido en Volatiles',
                     'Volatile Content',
                     'V_c')

VARIEDAD = Variable(u'Variedad',
                    'Variety',
                    'V')

DESGASTE_CRIBA = Variable(u'Desgaste Criba',
                          'Sieve Worn',
                          'D_c')

DESGASTE_MARTILLOS = Variable(u'Desgaste Martillos',
                              'Hammer Worn',
                              'D_h')

VELOCIDAD_MOLINO = Variable(u'Velocidad Molino',
                            'Mill Speed',
                            'V_m')

RELACION_PULPA_HUESO = Variable(u'Relacion Pulpa-Hueso',
                                'Pit-Flesh Ratio',
                                'R_p')

TIPO_CRIBA = Variable(u'Tipo de Criba',
                      'Sieve Type',
                      'S_t')

EMULSION_PASTA_SIN_CORREGIR = Variable(u'Emulsion Pasta Sin Corregir',
                                       'Uncorrected Paste Emulsion',
                                       'P_{EU}')

INCREMENTO_TEMPERATURA_MOLINO = Variable(u'Incremento Temperatura Molino',
                                         'Milling Temperature Increase',
                                         r'\Delta_{Tm}')

FLUJO_ENTRADA_MOLINO = Variable(u'Flujo Entrada Molino',
                                'Milling Production Rate',
                                r'M_R')

OFFSET_PRESILLA_LINEA_SEPARACION = Variable(
                             u'Offset Presilla - Linea Separacion',
                             'Weirs-Separation Interphase Offset',
                             '\Delta r')

TAMANO_GOTAS = Variable(u'Tamano Gotas',
                        'Drop Size',
                        'D_s')

LINEA_SEPARACION = Variable(u'Linea Separacion',
                            'Separation Interphase',
                            'r_s')

ENFERMEDAD_ACEITUNA = Variable(u'Enfermedad Aceituna',
                               'Olive Illnes',
                               'O_I')

ENFERMEDAD_ACEITUNA = Variable(u'Enfermedad Aceituna',
                               'Olive Illnes',
                               'O_I')
