#!/usr/bin/env python
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

Main author: Álvaro Garzón Casado
Secondary author: Pablo Cano Marchal

"""

import os
from collections import defaultdict as ddict
import networkx as nx
from unidecode import unidecode
from math import fsum as fs
import matplotlib.pylab as plt
import numpy as np
from logtools import timethis, log, logger
from tqdm import tqdm
import webcolors as wc
from paretochart import pareto
from collections import OrderedDict
from operator import itemgetter
from pdb import set_trace as bp
import pandas as pd

__version__ = '1.5'

FUENTE = 'Helvetica'
GROSOR_LINEA_NODO = 2
nivel_gris = '#404040'
alpha = 0.8
desat = 0.8

""" ___________________________
	FUNCIONES DE REPRESENTACIÓN
	___________________________ """

def visualize_results_graph(model,
							nombre_graf,
							list_nodos_fijos = [],
							dic_costos = {},
							path=None,
							show_relation_matrices=False,
							show_node_values=True,
							show_cost=False,
							show_contribution_values=False,
							show_bars = False,
							recompute_node_locations=True,
							palette = 'paper',
							orden_grafo='LR',
							fontsize=9,
							func_texto_nodo=lambda x: unicode(x.nombre,),
							rewrite=False,
							dic_clusters = {}):
	'''
	Defines a graph based on the system and plots it,
	saving it in a file with path defined by model.output_path
	'''
	global part
	part                   = model.part
	dic_impactos           = dict()
	dic_values             = model.values()
	model.dic_color_nodes  = dict()
	model.list_nodos_fijos = list_nodos_fijos

	if palette == 'basics' and show_bars == True:
		show_bars = False

	mc = _asign_colors(model, palette, show_bars)

	if type(mc) is tuple:
		print 'La paleta de colores empleada no es lo suficientemente grande. Se necesitan ' + str(mc[0] - mc[1]) + ' colores más.\n'
		return
	
	model.nombre_graf = nombre_graf
	trans             = hex(int(alpha * 255.))[-2:].upper()

	if path is None:
		path = os.path.join(model.output_path, nombre_graf)

	folder_sis = _create_image_folder(path, '_folder')

	if show_bars:
		image_folder = _create_image_folder(os.path.join(folder_sis, 'Bar folder'),'')     
		print ('Generando gráficos de barras...\n')
		logger.info('Generando gráficos de barras...')
		prop        = 0.6
		filecounter = len(model.dic_nod_pred.keys())

		for nodo in tqdm(model.dic_nod_pred.iterkeys(), total = filecounter, unit=" graphs"):
			list_colores = list()

			for key in model.dic_nod_pred[nodo].iterkeys():
				list_colores.append(model.dic_color_nodes[key])

			list_colores.append(model.dic_color_nodes[nodo])
			_gen_bars(nodo.name, model.dic_nod_pred[nodo], range(1,6), nodo.crisp_value, list_colores, image_folder, prop, mc, alpha)
			plt.close()
            
		print ('Generación finalizada.\n')
		logger.info('Generación finalizada.')

	dic_nodos  = {unicode(part) + unicode(n.nombre): n
				for n in dic_values.iterkeys()}
	dic_values = {unicode(part) + unicode(n.nombre): {'valor': round(v,3)}
				for n, v in dic_values.iteritems()}

	dic_rels = ddict(dict)

	for antecesor, sucesor in model.G.edges():
		dic_rels[unicode(part)
				+ unicode(antecesor.nombre)][unicode(part)
				+ unicode(sucesor.nombre)] = (
					str(model.G[antecesor][sucesor]['rel']))

	G                         = nx.nx_agraph.to_agraph(model.G)
	G.graph_attr['overlap']   = 'voronoi'
	G.graph_attr['splines']   = 'true'
	G.graph_attr['rankdir']   = orden_grafo
	G.graph_attr['fontname']  = FUENTE
	G.graph_attr['bgcolor']   = '#E6E6E6'
	G.graph_attr['fontcolor'] = '#111111'

	html_tabla_template_entrada = r'''<
	<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
	<TR><TD BGCOLOR="{color}">{nombre_nodo}</TD></TR>'''
	html_tabla_template_sucesores = r'''<
	<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
	<TR><TD BGCOLOR="{color}">{nombre_nodo}</TD></TR>'''
	model.snv = 0

	if show_node_values:

		if show_bars == True:
			html_tabla_template_entrada += '<TR><TD BGCOLOR="{color}">{valor}</TD></TR>'
			model.snv = 1

		else:
			html_tabla_template_entrada += '<TR><TD BGCOLOR="{color}">{valor}</TD></TR>'
			html_tabla_template_sucesores += '<TR><TD BGCOLOR="{color}">{valor}</TD></TR>'

	if show_cost:
		html_tabla_template_entrada += '<TR><TD BGCOLOR="{color}">c: {costo}</TD></TR>'
		html_tabla_template_sucesores += '<TR><TD BGCOLOR="{color}">c: {costo}</TD></TR>'
            

	if show_contribution_values:
		dic_impactos = {}
		html_tabla_template_entrada += '<TR><TD BGCOLOR="{color}">{impacto}</TD></TR>'
		html_tabla_template_sucesores += '<TR><TD BGCOLOR="{color}">{impacto}</TD></TR>'

		for nodo in G.nodes():
			comp = unidecode(nodo)[5:]

			if comp in model.nodos_sucesores:
				vec_impactos = np.zeros(len(model.kernels))

				for pos in range(5):
					valor = 0

					for nodo_pred in model.dic_nod_pred[comp].keys():
						valor += model.dic_nod_pred[comp][nodo_pred][pos]

					vec_impactos[pos] = valor
				dic_impactos[nodo] = vec_impactos

			else:
				dic_impactos[nodo] = list()

			if type(dic_impactos[nodo]) != list:
				dic_impactos[nodo].tolist()

			else:
				dic_impactos[nodo] = ''

	html_tabla_template_entrada += '</TABLE>>'
	html_tabla_template_sucesores += '</TABLE>>'

	for edge in G.edges():
		antecesor, sucesor = edge
		nod_pred           = unidecode(edge[0])[5:]
		nod_suc            = unidecode(edge[1])[5:]
		nn                 = model.dic_nod_pred[nod_suc][nod_pred]
		grosor             = fs(nn) * 10
		
		if grosor == 0:
			edge.attr['pendwidth'] = 5
			edge.attr['style']     = 'dashed'

		else:
			edge.attr['penwidth'] = grosor
			edge.attr['style']    = 'bold'

		if show_relation_matrices == True:
			edge.attr['label'] = dic_rels[antecesor][sucesor]

		elif show_relation_matrices == 'Impact':
			edge.attr['label'] = str(grosor * 10) + '%'

		edge.attr['labeldistance'] = 0
		edge.attr['fontsize']      = 20
		edge.attr['fontname']      = FUENTE	
		edge.attr['fontcolor']     = nivel_gris
		
		if palette == 'basics':
					
			for rel in model.relations_list:

				if rel.predecessor == nod_pred and rel.successor == nod_suc:
					tipo = rel.rel_type

					if tipo == 'b':
						edge.attr['color'] = '#83E87E'

					elif tipo == 'bi':
						edge.attr['color'] = '#CF1C1C'

					elif tipo == 'ud':
						edge.attr['color'] = '#0C1ACC'

					elif tipo == 'ui':
						edge.attr['color'] = '#EC0AC6'

					elif tipo == 'sp':
						edge.attr['color'] = '#EC960A'

					else:
						edge.attr['color'] = '#DBEC0A'

		else:
			edge.attr['color'] = model.dic_color_nodes[nod_pred]
            
	for nodo in G.nodes():
		nodo.attr['fontsize'] = 20
		nodo.attr['fontname'] = FUENTE
		nodo.attr['penwidth'] = GROSOR_LINEA_NODO
		nodo.attr['style']    = 'filled'
		comp                  = unidecode(nodo)[5:]
            
		dic_format = {
			'nombre_nodo': func_texto_nodo(dic_nodos[nodo]),
			'valor': dic_values[nodo]['valor'],
			'costo': dic_costos.get(nodo, 0),
			'impacto': dic_impactos.get(nodo, ''),
			'color': 'transparent'
		}

		if comp in model.nodos_sucesores:

			if show_bars == True:
				nodo.attr['shape']    = 'rectangle'
				nodo.attr['image']    = os.path.join(image_folder, comp + '.png') 
				nodo.attr['imagepos'] = 'bc'
				nodo.attr['labelloc'] = 't'               
				nodo.attr['label']    = html_tabla_template_sucesores.format(
				**dic_format
				)

			else:
				nodo.attr['shape'] = 'ellipse'           
				nodo.attr['label'] = html_tabla_template_sucesores.format(
				**dic_format
				)
                
		else:

			if comp in list_nodos_fijos:
				nodo.attr['shape'] = 'hexagon'

			else:
				nodo.attr['shape'] = 'ellipse'

			nodo.attr['label'] = html_tabla_template_entrada.format(
			**dic_format
			)
		
		nodo.attr['fontcolor'] = _node_font_color(model.dic_color_nodes[comp] + trans) 
		nodo.attr['color']     = model.dic_color_nodes[comp]
		nodo.attr['fillcolor'] = model.dic_color_nodes[comp] + trans


	for cluster in dic_clusters.iterkeys():
		cluster_fin = [unicode(part + nodo) for nodo in dic_clusters[cluster]]
		G.add_subgraph(cluster_fin, name = 'cluster ' + cluster, color = '#9A2828', style = 'dashed', label = 'Set ' + cluster, labelloc = 'b') 	

	G.layout(prog='dot')
	G.draw(os.path.join(folder_sis, nombre_graf) + '.pdf')
	model.grafo      = G
	model.folder_sis = folder_sis
	model.vis        = 'ok'

def _asign_colors(model, palette, sb):

	"""
	Function that formerly asigns colors to every node in the system.


	Inputs:

	- palette: colors palette we want to apply to our grahic representation.

	Outputs:

	- mc: color of the crisp value marker.

	""" 

	len_suc = list()	

	for node in model.nodos_sucesores:
		len_suc.append(len(model.dic_nod_pred[node]))

	max_len = max(len_suc)

	if palette == 'rand':
		import random
		r = lambda: random.randint(0,255)

		for key in model.datos_nodos.iterkeys():
			model.dic_color_nodes[key] = '#%02X%02X%02X' % (r(), r(), r())
			colhex1                    = [nivel_gris]

		modo = 1

	elif palette == 'warm':
		from colors_palette import warm_pal as colhex1
		modo = 2

	elif palette == 'cold':
		from colors_palette import cold_pal as colhex1
		modo = 2

	elif palette == 'brilliant':
		from colors_palette import brilliant_pal as colhex1
		modo = 2

	elif palette == 'autumn':
		from colors_palette import autumn_pal as colhex1
		modo = 2

	elif palette == 'paper':
		from colors_palette import paper_pal as colhex1
		modo = 2   

	elif palette == 'automatic':
		import seaborn as sns
		modo    = 3
		colrgb  = sns.color_palette(palette = 'Set2', n_colors = max_len, desat = desat).as_hex()
		colhex1 = [unidecode(color.upper()) for color in colrgb]
		colhex1.append(nivel_gris)

	elif palette == 'basics':
		model.dic_color_nodes = {nodo: '#FFFFFF' for nodo in model.datos_nodos}
		mc = nivel_gris
		return mc
		
	colhex = colhex1[:]
	mc     = colhex.pop() 

	if modo == 2 and len(colhex) < max_len:
		mc = (max_len,len(colhex))
		return mc

	if modo == 2 or modo == 3:        
		cont_sal = -1 

		for nodo in model.G.nodes(): 

			if nodo in model.nodos_sucesores: 
				cont_col       = -1
				colhex_aux     = colhex[:] 
				colhex_removed = {key: 0 for key in colhex}

				for key in model.G.predecessors(nodo): 

					if key in model.dic_color_nodes.keys(): 

						try:
							colhex_aux.remove(model.dic_color_nodes[key]) 
							colhex_removed[model.dic_color_nodes[key]] += 1 

						except ValueError:

							if colhex_removed[model.dic_color_nodes[key]] == 1:
								colores_empleados = set([model.dic_color_nodes[nodo_pred] for nodo_suc in model.G.successors(key) for nodo_pred in model.G.predecessors(nodo_suc) if nodo_pred in model.dic_color_nodes.keys()])
								cont              = -1

								while True:
									cont += 1

									if colhex_aux[cont] not in colores_empleados:
										model.dic_color_nodes[key] = colhex_aux[cont]
										colhex_aux.remove(model.dic_color_nodes[key])
										break                                           

				for key in model.G.predecessors(nodo): 

					if key not in model.dic_color_nodes.keys(): 
						cont_col                  += 1
						model.dic_color_nodes[key] = colhex_aux[cont_col]

				if nodo in model.nodos_salida: 
					cont_sal                   += 1
					model.dic_color_nodes[nodo] = colhex[cont_sal]

	return mc

def _gen_bars(name, dic_w, kernels, crisp_value, list_colores, folder, prop, markercolor, alpha):

	eje_x                  = kernels
	dic_bar                = {}
	crisp_value            = round(crisp_value, 3)
	dic_bar['node_labels'] = dic_w.keys()

	for key in eje_x:   
		dic_bar[key] = [dic_w[col][key - 1] for col in dic_w.iterkeys() ]

	tam1       = 8 * prop
	tam2       = 6 * prop
	f          = plt.figure(figsize = (tam1, tam2))
	ax1        = f.add_subplot(111)
	rect       = f.patch
	font_color = _node_font_color(list_colores[-1])
	rect.set_facecolor(list_colores.pop())
	bar_width  = 0.8

	bot   = dict()
	eje_y = list()

	for kernel in range(len(dic_bar) - 1):
		bot[kernel + 1] = 0

	for it in range(1, len(dic_bar['node_labels']) + 1):

		for kernel in range(1, len(dic_bar)):

			if dic_bar[kernel][it - 1] != 0:
				color_rgb_aux = wc.hex_to_rgb(list_colores[it - 1]);
				color_rgb     = tuple()

				for i in range(len(color_rgb_aux)):
					color_rgb = color_rgb + (1 - alpha*(1 - color_rgb_aux[i]/255.),)

				ax1.bar(kernel, dic_bar[kernel][it - 1], bottom = bot[kernel], align = 'center', edgecolor = list_colores[it - 1], color = color_rgb)
				eje_y.append(dic_bar[kernel][it - 1] + bot[kernel])
				bot[kernel] += dic_bar[kernel][it - 1]

	ax1.set_axis_bgcolor = '#BDBDBD'
	plt.xlim([min(kernels) - bar_width, max(kernels) + bar_width])
	plt.ylim([0, min([1, 1.1 * max(bot.values())])])
	
   	cmy = 0
	nz  = 0

	for key in bot.iterkeys():
		cmy += bot[key]

		if bot[key] != 0:
			nz += 1

	cmy = cmy / (2 * nz)
	ax1.plot(crisp_value, cmy, marker = '^', markerfacecolor = markercolor, markersize = 15.)	
	eje_y = sorted(list(set(eje_y)))

	plt.xticks(eje_x, ('VS','S','M','H','VH'))
	puntero = plt.annotate(s = crisp_value, xy = (crisp_value, cmy), xytext = (crisp_value + 0.15, cmy * 0.92), fontsize = 15., color = markercolor)

	ax1.tick_params(axis = 'x', colors = font_color)
	ax1.tick_params(axis = 'y', colors = font_color)
	
	plt.savefig(os.path.join(folder, name + '.png'), facecolor = f.get_facecolor())

def _node_font_color(color_str):

	red   = int(color_str[1:3], 16)
	green = int(color_str[3:5], 16)
	blue  = int(color_str[5:7], 16)

	gray_scale_value = nivel_gris if (red * 0.2126 + green * 0.7152 + blue * 0.0722) > 0.5 * 255  else  '#FFFFFF'

	return gray_scale_value

""" ___________________________
	FUNCIONES DE SIMPLIFICACIÓN
	___________________________ """

def extract_all(model, grade = float('inf'), mode = 'backward'):

	""" 
	Extraction of complete subgraphs with maximum degree of all the nodes of the system. 
	A directory is generated and all subgraphs are stored inside.
            

	Inputs: 

	- grade: degree of influence to represent in the extraction. 
	Infinity is used as default value, so that if novalue is given, 
	all related nodes will be extracted.

	- mode: mode of analysis. Backward is used as default value. Forward and complete are the other possible options.
		- Backward: backward propagation of the node. It will be run for intermediate and output nodes.
		- Forward: forward propagation of the node. It will be run for input and intermediate nodes.
		- Complete: backward and forward propagations of the node. It will be run for every node in the system.

	"""

	if model.vis is not 'ok':
		print 'Before execute an extract_all, a visualize_results_graph should be execute.\n'
		return		

	string = 'Extract all.\nMode: ' + mode + '.\nGrade: ' + str(grade) 
	folder = _create_image_folder(os.path.join(model.folder_sis, string), '')

	if mode in ['backward', 'back', 'bw']:
		sel_set = model.nodos_sucesores[:]

	elif mode in ['forward', 'for', 'fw']:
		sel_set = model.nodos_predecesores[:]

	elif mode == 'complete':
		sel_set = model.G.nodes()

	else:
		print 'Modo de extracción no reconocido.\n'

	tam = len(sel_set)

	for nodo in tqdm(sel_set, total = tam, unit = ' extracts'):
		extract_node(model,[nodo.name], nodo.name, mode = mode, path = folder, grade = grade)

	print 'Generada extracción completa (grado ' + str(grade) + ') de nodos. Modo ' + mode + '.\n'

def extract_node(model, lista_nodos_extraidos, nombre_graf, grade = float('inf'), mode = 'backward', path = None):
        
	""" 
	Extraction of subgraphs from a list of nodes given as input, being possible to choose a degree of influence and mode.


	Inputs:

	- lista_nodos_extraidos: list that contains the node or nodes that we want to extract.

	- nombre_graf: name we will give to pdf file with final graph.

	- grade: degree of influence to represent in the extraction. 
	Infinity is used as default value, so that if novalue is given, 
	all related nodes will be extracted.

	- mode: mode of analysis. Backward is used as default value.

	- path: path of the directory where we save the pdf file. 
	Predetermined, if none is chosen, the method itmodel assigns as 
	the directory the same chosen in the creation of the class.


	Outputs:

	- rest: nodes set that compound the extraction subgraph.

	"""
	if model.vis is not 'ok':
		print 'Before execute an extract_node, a visualize_results_graph should be execute.\n'
		return

	try:
		rest = []

		if mode in ['backward', 'back', 'bw'] or mode in ['forward', 'for', 'fw']:
			
			for nodo in lista_nodos_extraidos:
				nodo_extraido = part + nodo
				rest.extend(_obtain_subgraph(model, nodo_extraido, [], grade, mode))
            
		elif mode in ['complete', 'comp']:
			
			for nodo in lista_nodos_extraidos:
				nodo_extraido = part + nodo
				rest.extend(_obtain_subgraph(model, nodo_extraido, [], grade, 'backward'))
				rest.extend(_obtain_subgraph(model, nodo_extraido, [], grade, 'forward'))

		else:
			print 'Modo de extracción no reconocido.\n'
			return

	except KeyError:
		print 'Fallo en ' + nombre_graf + '. El parámetro lista_nodos_extraidos debe ser una lista.\n'
       
	rest = set(rest)
	subG = model.grafo.subgraph(rest).copy()

	if path is None:
		path = model.folder_sis

	n_cluster = 0

	for nodo in subG.nodes():
		
		if unidecode(nodo[5:]) in lista_nodos_extraidos:
			n_cluster += 1
			subG.add_subgraph(nodo, name = 'cluster ' + str(n_cluster), style = 'dashed', color = '#9A2828')

	subG.layout(prog = 'dot')
	subG.draw(os.path.join(path, nombre_graf) + '.pdf')
	
	porc = round(100 - float(len(rest)) / model.G.size() * 100, 2)
	print 'Archivo: ' + nombre_graf + '\n- Porcentaje de simplificación: ' + str(porc) + ' %.\n'
        
	return rest, porc


def get_total_weight(model, init, end, graph = False):
     
	"""
	Allows calculus of total weight of a single node over another one, not necessary a successor one.

 
	Inputs:

	- init: initial node that acts as path source. 

	- end: final node that acts as path target.

	- graph: binary input that determines if we want to draw a graph (True) or not (False)

	Outputs:

	- total_weight: total weight between nodes.

	"""

	if model.vis is not 'ok':
		print 'Before execute a get_total_weight, a visualize_results_graph should be execute.\n'
		return
        
	paths = list(nx.all_simple_paths(model.G, init, end))
	nodos = list()
	
	for path in paths:
		
		for nodo in path:
			nodos.append(nodo)
        
	nodos_aux = list()
	
	for nodo in nodos:
		
		if nodo == init:
			
			for gnode in model.G.nodes():
				
				if gnode.name == init:
					nodos_aux.append(gnode)
		
		elif nodo == end:
			
			for gnode in model.G.nodes():
				
				if gnode.name == end:
					nodos_aux.append(gnode)
		
		else:
			nodos_aux.append(nodo)
        
	if nodos_aux:
		nodos_aux = model.grafo.subgraph(set(nodos_aux)).copy()
		nodos_aux.add_subgraph(unicode(part + init), name = 'cluster 1', style = 'dashed', color = '#9A2828')
		nodos_aux.add_subgraph(unicode(part + end), name = 'cluster 2', style = 'dashed', color = '#9A2828')
		
		pos_init  = nodos_aux.nodes().index(unicode(part + init))
		pos_fin   = nodos_aux.nodes().index(unicode(part + end))
		nodo_init = nodos_aux.nodes()[pos_init]
		nodo_fin  = nodos_aux.nodes()[pos_fin]

		nodos_aux.add_node(unicode(init), fontname = nodo_init.attr['fontname'], color = nodo_init.attr['color'], style = nodo_init.attr['style'], fontsize = str(int(nodo_init.attr['fontsize'])*2), fillcolor = nodo_init.attr['fillcolor'], fontcolor = nodo_init.attr['fontcolor'], label = unicode(init))#label = nodo_init.attr['label'])
		nodos_aux.add_node(unicode(end), fontname = nodo_fin.attr['fontname'], color = nodo_fin.attr['color'], style = nodo_fin.attr['style'], fontsize = str(int(nodo_fin.attr['fontsize'])*2), fillcolor = nodo_fin.attr['fillcolor'], fontcolor = nodo_fin.attr['fontcolor'], label = unicode(end))#label = nodo_fin.attr['label'])

		lista_w_paths = list()
		
		for path in paths:
			lista_partial = list()
			lista_partial = [model.dic_pesos_for[path[pos]][path[pos + 1]] / 100 for pos in range(len(path) - 1)]
			lista_w_paths.append(_prod(lista_partial))
            
		total_weight = round(fs(lista_w_paths * 100), 2)
		str_peso     = str(total_weight) + ' %'

		if graph:
			nodos_aux.add_edge(unicode(init), unicode(end))
			nodos_aux.edges()[-1].attr['penwidth'] = total_weight / 10.
			nodos_aux.edges()[-1].attr['style'] = 'bold'
			nodos_aux.edges()[-1].attr['label'] = str_peso
			nodos_aux.edges()[-1].attr['labeldistance'] = 0
			nodos_aux.edges()[-1].attr['fontsize'] = 40
			nodos_aux.edges()[-1].attr['fontname'] = FUENTE	
			nodos_aux.edges()[-1].attr['fontcolor'] = nivel_gris
			nodos_aux.edges()[-1].attr['color'] = nodo_init.attr['color']

			cadena = 'Weight from ' + str(init) + ' to ' + str(end) + ': ' + str_peso + '.pdf'
			nodos_aux.layout(prog = 'dot')
			nodos_aux.draw(path = os.path.join(model.folder_sis, cadena), format = 'pdf')

		return total_weight

	else:
		return None

def obtain_weights(model, lista_entrada, lista_salida, graph = False):

	if model.vis is not 'ok':
		print 'Before execute an obtain_weights, a visualize_results_graph should be execute.\n'
		return

	for nodo in lista_salida:
		
		if nodo in model.nodos_entrada:
			print nodo + ' no puede emplearse como nodo de salida.\n'
			lista_salida.remove(nodo)

	dic_pesos = ddict(dict)

	for nodo_salida in lista_salida:

		f    = plt.figure()
		ax1  = f.add_subplot(111)

		for nodo_entrada in lista_entrada:

			peso = get_total_weight(model, nodo_entrada, nodo_salida, graph = False)

			if peso > 0: # Haciendo que el peso se mayor que 0 evitamos también a los nodos que no tienen relación, ya que su peso sería de typo Nonetype y daría error.

				dic_pesos[nodo_salida][nodo_entrada] = peso

		
		if graph == True:
			data        = OrderedDict(sorted(dic_pesos[nodo_salida].items(), key=itemgetter(1)))
			values      = [x for x in data.values() if x is not None]
			labels      = [y for y in data.keys() if data[y] is not None]
			con         = -1
			labels_prep = list()

			for nodo in labels:
				con += 1
				ax1.bar(con + 1, values[con], align = 'center', edgecolor = model.dic_color_nodes[nodo], color = model.dic_color_nodes[nodo])
				lbl      = nodo.split()
				lbl_prep = str()

				for i in lbl:
					lbl_prep = lbl_prep + i + '\n'

				labels_prep.append(lbl_prep)

			bar_width = 8 / (len(values) * 1.2)

			plt.xticks(range(1,len(values)+1), labels_prep)
			plt.xlim([1 - bar_width, len(values) + bar_width])
			ax1.tick_params(axis = 'x', colors = nivel_gris)
			ax1.tick_params(axis = 'y', colors = nivel_gris)
			plt.xticks(rotation = 15, size = 8)
			plt.savefig(os.path.join(model.folder_sis, nodo_salida + '_weights'))
			plt.close()

	df  = pd.DataFrame(dic_pesos)
	dfc = df.copy()

	if len(df.columns) > 1:

		dfc['Total'] = pd.Series(df.fillna(0).mean(axis = 1).values, index = df.index)

	print dfc
	print '\n'
	
	return dic_pesos
			

def parametric(model, lista_entrada, lista_salida, paso):
	
	""" 
	Function that allows a graphical representation of the 
	relation among a list of input nodes and a list of output nodes.


	Inputs:

	- lista_entrada: list containing nodes that we can use as inputs (independent variables).
	If any of them isn't an initial node of the system, it's removed.

	- lista_salida: list containing nodes that we can use as outputs (dependent variables).
	If any of them is an initial node of the system, it's removed.

	- datos_iniciales: dictionary including default values applied to initial nodes.

	- dic_res: dictionary including model initial results.

	- paso: step between two consecutives values of the output variables. 
	Calculus range is set by minimum and maximum values of the universe of discourse.


	"""

	dic_res         = model.values().copy()
	datos_iniciales = model.dic_valores.copy()

	if model.vis is not 'ok':
		print 'Before execute a parametric, a visualize_results_graph should be execute.\n'
		return

	div = float((max(model.kernels) - min(model.kernels)) / paso)

	if abs(div) - abs(int(div)) != 0:
		print str(paso) + ' no es un valor adecuado para el paso. Emplee un valor divisor de ' + str((max(model.kernels) - min(model.kernels))) + '.\n'   
	
	else:

		for nodo in lista_entrada:
			
			if nodo not in model.nodos_entrada:
				print nodo + ' no puede emplearse como nodo de entrada.\n'
				lista_entrada.remove(nodo)
                 
		for nodo in lista_salida:
			
			if nodo in model.nodos_entrada:
				print nodo + ' no puede emplearse como nodo de salida.\n'
				lista_salida.remove(nodo)

		if len(lista_entrada) > 0 and len(lista_salida) > 0:

			tam  = len(lista_salida) * len(lista_entrada)
			init = 0

			for nodo_salida in lista_salida:
				dic_res_loop = dic_res.copy()
				dat_ini_loop = datos_iniciales.copy()
				plt.plot([min(model.kernels), max(model.kernels)],[dic_res_loop[nodo_salida], dic_res_loop[nodo_salida]], 'k--', linewidth = 3)
								
				
				for nodo_entrada in tqdm(lista_entrada, total = tam, unit = ' calcs', initial = init):
					init += 1
					vec_x = list()
					vec_y = list()
					dic_res_loop = dic_res.copy()
					dat_ini_loop = datos_iniciales.copy()
					
					for j in np.arange(min(model.kernels), max(model.kernels) + paso, paso):
						dat_ini_loop[nodo_entrada] = j
						dic_res_new = model.compute(dat_ini_loop)
						vec_x.append(j)
						vec_y.append(dic_res_new[nodo_salida])

					plt.plot(vec_x, vec_y, label = nodo_entrada)

				plt.xlim([min(model.kernels), max(model.kernels)])
				plt.xlabel('Input data')
				plt.ylabel('Output data')
				plt.title(nodo_salida)
				plt.legend(loc = 'best', fontsize = '12')
				plt.savefig(os.path.join(model.folder_sis, nodo_salida))
				plt.close()
				print 'Generado gráfico paramétrico para ' + nodo_salida + '.\n'
              
def simplify(model, grade, mode = 'backward', name = None):

	"""
	Extraction depended on mode of all the initials or finals nodes of the system according with a grade.

	Inputs:

	- grade: degree of influence to represent in the extraction. 

	- mode: mode of analysis. Backward is used as default value.

	- name: name we will give to pdf file with final graph.
    
	"""
	if model.vis is not 'ok':
		print 'Before execute a simplify, a visualize_results_graph should be execute.\n'
		return

	lista = list()

	if mode in ['backward', 'back', 'bw']:
		mode  = 'backward'
		lista = [nodo.name for nodo in model.nodos_salida]
	
	elif mode in ['forward', 'for', 'fw']:
		mode  = 'forward'
		lista = [nodo.name for nodo in model.nodos_entrada]
	
	else:
		print 'Modo de extracción no reconocido.\n'
		return   

	if name is None:
		name = 'Simplificación ' + model.nombre_graf + ',\nMode: ' + mode + ',\nGrade: ' + str(grade)

	extract_node(model, lista, nombre_graf = name, grade = grade, mode = mode)     


""" _______________
	OTRAS FUNCIONES
	_______________ """

def _create_image_folder(path, base = '_images_folder'):

	""" 
	Creates a folder properly named.


	Inputs:

	- path: first part of the folder name.

	- base: second part of the folder name. Predetermined, it sets _images_folder.

	"""

	listdir      = os.listdir(os.getcwd())
	image_folder = path + base

	while True:

		if image_folder[2:] in listdir:

			if image_folder[-1] != ')':
				image_folder = image_folder + '(1)'

			else:
				prev = image_folder.find('(') + 1
				post = image_folder[prev:].find(')') + prev
				last = int(image_folder[prev:post])
				image_folder = image_folder[:prev] + str(last + 1) + ')'
		
		else:
			os.mkdir(image_folder)  
			break
        
	return image_folder


def _obtain_subgraph(model, nodo_extraido, rest, grade, mode = 'backward'):

	""" 
	Recursive function that allows to obtain a subset of the model since
	a determined node, depend on the given degree of influence.
        

	Inputs:
 
	- nodo_extraido: node that we want to extract within each loop 	iteration in the extract_node algorithm.
 
	- rest: list containing actual extracted nodes.

	- grade: degree of influence to represent in the extraction. 

	- mode: mode of analysis. Backward is used as default value.
	"""

	if mode in ['backward', 'back', 'bw']:
		
		if len(model.grafo.predecessors(nodo_extraido)) == 0 and nodo_extraido not in rest:
			rest.append(nodo_extraido)

		comp = nodo_extraido[5:]

		if comp not in model.nodos_entrada:
			aux = model.dic_pesos_back[comp].copy()
			num_pred = len(model.grafo.predecessors(nodo_extraido))

			if num_pred <= grade:
				predecesores = model.grafo.predecessors(nodo_extraido)
		
			else:
				predecesores = list()
		
				for it in range(grade):
					max_value = max(aux.values())
		
					for key, value in aux.iteritems():
		
						if value == max_value:
							predecesores.append(key)
							aux.pop(key)
							break
		
		else:
			predecesores = list()
   
		for nodo_pred in predecesores:
		
			if type(nodo_pred.name) is unicode:
				nodo_pred = unidecode(nodo_pred)
		
			else:
				nodo_pred = part + nodo_pred.name
		
			rest = _obtain_subgraph(model,nodo_pred, rest, grade, mode)

		if nodo_extraido not in rest:
			rest.append(nodo_extraido)

		return rest

	elif mode in ['forward', 'for', 'fw']:
            
		if len(model.grafo.successors(nodo_extraido)) == 0 and nodo_extraido not in rest:
			rest.append(nodo_extraido)

		comp = nodo_extraido[5:]

		if comp not in model.nodos_salida:
			aux = model.dic_pesos_for[comp].copy()
			num_suc = len(model.grafo.successors(nodo_extraido))

			if num_suc <= grade:
				sucesores = model.grafo.successors(nodo_extraido)
		
			else:
				sucesores = list()
		
				for it in range(grade):
					max_value = max(aux.values())
		
					for key, value in aux.iteritems():
		
						if value == max_value:
							sucesores.append(key)
							aux.pop(key)
							break
		
		else:
			sucesores = list()
   
		for nodo_suc in sucesores:
		
			if type(nodo_suc.name) is unicode:
				nodo_suc = unidecode(nodo_suc)
		
			else:
				nodo_suc = part + nodo_suc.name
		
			rest = _obtain_subgraph(model, nodo_suc, rest, grade, mode)

		if nodo_extraido not in rest:
			rest.append(nodo_extraido)

		return rest


def _prod(lista):

	""" 
	Returns an accurate floating point prod of values in the iterable.


	Inputs: 

	- lista: list with iterable values.

	"""

	acum = 1

	for pos in range(len(lista)):
		acum *= lista[pos]
	
	return acum