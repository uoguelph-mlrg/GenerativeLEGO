import numpy as np
import helpers
from pyFiles import LegoGraph, utils
import helpers
from helpers import load_gin
from DGL_DGMG.dgmg_helpers import DGMGEvaluationWithGIN 
from helpers import GINDataset
from DGL_GIN.dataloader import GraphDataLoader, collate
import torch
import numpy as np
import pickle
import os
import networkx as nx
import ast
import argparse
import copy


class GraphPermuter():
	def __init__(self, args):
		if args.force_valid:
			self.add_random_node = self.__add_node_valid
			self.get_edges_func = self.__get_random_valid_source_dest_shifts
		else:
			self.add_random_node = self.__add_random_node
			self.get_edges_func = self.__get_random_source_dest_shifts

		if args.remove_disjoint == True:
			self.delete_random_node = self.__delete_node_no_disjoint
		else:
			self.delete_random_node = self.__delete_node

		self.implied_edges = utils.ImpliedEdgesUtil()

	def permute_graph(self, g):
		np.random.seed()
		action = np.random.choice(2)
		if action == 0:
			# delete random node
			if g.number_of_nodes() > 2:
				self.delete_random_node(g)
			else:
				# Don't want graphs to get too small but still want to perform a permutation
				self.add_random_node(g)

		elif action == 1:
			# Add random node
			self.add_random_node(g)

		elif action == 2:
			if g.number_of_nodes() > 2:
				# Move node to another location in the graph
				deleted_node_label = self.delete_random_node(g)
				self.add_random_node(g, new_node_label = deleted_node_label)
			else:
				self.add_random_node(g)
			

		helpers.add_node_attributes(g)
		helpers.add_edge_attributes(g)


	def __add_node_label(self, node_type, g):
		g.node_labels[g.number_of_nodes() - 1] = 'Brick(4, 2)' if node_type == 1 else 'Brick(2, 4)'

	def __get_brick_size(self, node_type):
		return (4, 2) if node_type == 1 else (2, 4)


	def __add_edge_with_dirn(self, g, dirn, new_node_size, node_type):
		res = self.get_edges_func(g, dirn, new_node_size)
		if res is None:
			return False

		src, dest, shifts = res
		g.add_generated_node(node_type)
		self.__add_node_label(node_type, g)
		return self.__add_edge(g, src, dest, shifts)


	def __add_edge(self, g, src, dest, shifts):
		if not g.has_edge_between(src, dest):
			g.add_generated_edge(src, dest, shifts[0], shifts[1])
			return True
		return False


	def __get_random_valid_source_dest_shifts(self, g, dirn, new_node_size):
		if dirn == 0:
			possible_nodes = g.get_nodes_that_can_connect_underneath_of(new_node_size)
			if len(possible_nodes) == 0:
				return
			src = possible_nodes[np.random.choice(len(possible_nodes))]
			dest = g.number_of_nodes()
			possible_shifts = g.get_valid_connections_old_underneath(src, new_node_size)

		else:
			possible_nodes = g.get_nodes_that_can_connect_on_top_of(new_node_size)
			if len(possible_nodes) == 0:
				return
			dest = possible_nodes[np.random.choice(len(possible_nodes))]
			src = g.number_of_nodes()
			possible_shifts = g.get_valid_connections_old_on_top(dest, new_node_size)

		shifts = possible_shifts[np.random.choice(len(possible_shifts))]
		return src, dest, shifts


	def __get_random_source_dest_shifts(self, g, dirn, *args):
		if dirn == 0:
			src = np.random.choice(g.number_of_nodes() - 1)
			dest = g.number_of_nodes() - 1
		else:
			src = g.number_of_nodes() - 1
			dest = np.random.choice(g.number_of_nodes() - 1)

		shifts = np.random.choice(7, 2) - 3
		return src, dest, shifts


	def __repeat_if_invalid(self, g, new_node_label = None, counter = 0):
		if g.valid_graph == False:
			print('repeating invalid, ', counter)
			g.valid_graph = True
			g.overconstrained_brick = False
			g.merged_brick = False
			g.invalid_shift = False
			self.__delete_node(g, node = g.number_of_nodes() - 1)
			if counter < 10:
				self.__add_node_valid(g, new_node_label = new_node_label, counter = counter)


	def __add_node_valid(self, g, new_node_label = None, counter = 0):
		self.__add_random_node(g, new_node_label = new_node_label, counter = counter)
		self.__repeat_if_invalid(g, new_node_label = new_node_label, counter = counter + 1)


	def __add_random_node(self, g, new_node_label = None, counter = 0):
		if new_node_label is None:
			node_type = np.random.choice(2) + 1
		elif new_node_label == 'Brick(4, 2)':
			node_type = 1
		elif new_node_label == 'Brick(2, 4)':
			node_type = 2
		else:
			raise Exception('Unsupported node label {}'.format(new_node_label))

		size = self.__get_brick_size(node_type)

		dirn = np.random.choice(2)

		if not self.__add_edge_with_dirn(g, dirn, size, node_type) and counter < 10:
			# So that we don't throw an error in delete node
			# g.lego_assembly[g.number_of_nodes() - 1] = 0
			# self.__delete_node(g, node = g.number_of_nodes() - 1)
			self.__add_random_node(g, new_node_label = new_node_label, counter = counter + 1)
		elif counter > 10:
			# So that we don't throw an error in delete node
			# g.lego_assembly[g.number_of_nodes() - 1] = 0
			# self.__delete_node(g, node = g.number_of_nodes() - 1)
			pass
		else:
			# g.update_nx_graphs()
			g.convert_to_LDraw_and_verify(None)
			self.implied_edges.add_implied_edges(g)

			
	def __delete_node_no_disjoint(self, g):
		node = np.random.choice(g.number_of_nodes())
		deleted_edges = list(g.g_undirected.edges(node))
		g.g_undirected.remove_node(node)
		while(nx.is_connected(g.g_undirected) == False):
			g.g_undirected.add_node(node)
			g.g_undirected.add_edges_from(deleted_edges)
			node = np.random.choice(g.number_of_nodes())
			deleted_edges = list(g.g_undirected.edges(node))
			g.g_undirected.remove_node(node)

		return self.__delete_node(g, node = node)


	def __delete_node(self, g, node = None):
		if node is None:
			node = np.random.choice(g.number_of_nodes())

		deleted_node_label = g.node_labels[node]
		g.remove_nodes(node)
		self.__update_node_labels(g, node)
		self.__update_edge_labels(g, node)

		g.update_nx_graphs()
		g.convert_to_LDraw_and_verify(None)

		return deleted_node_label
		
		
	def __update_node_labels(self, g, node):
		del g.lego_assembly.bricks[node]
		del g.node_labels[node]
		node_labels = {}
		for key, val in g.node_labels.items():
			node_labels[key - (key > node)] = val
		g.node_labels = node_labels

		bricks = {}
		for key, val in g.lego_assembly.items():
			bricks[key - (key > node)] = val
		g.lego_assembly.bricks = bricks
		

	def __update_edge_labels(self, g, node):
		g.edge_labels = {k: v for k, v in g.edge_labels.items() if node not in k}
		edge_labels = {}
		for key, val in g.edge_labels.items():
			edge_labels[key[0] - (key[0] > node), key[1] - (key[1] > node)] = val
		g.edge_labels = edge_labels


class Runner():
	def __init__(self, ds, evaluation, dir, start_iter, args):
		self.evaluation = evaluation
		self.dir = dir
		self.ds = ds
		self.permuter = GraphPermuter(args)
		self.start_iter = start_iter

		self.__initialize_values()
		if start_iter == 0:
			self.__initialize_results()
			self.__evaluate_graphs()
			self.__print_results(0)
		self.include_random = args.include_random


	def perform_run(self, n_iter = 100):
		for i in range(self.start_iter + 1, n_iter):
			self.num_invalid = 0
			self.num_disjoint = 0
			self.num_nodes = 0
			save_builds = (i % self.n_iters_to_save == 0) or (i % (n_iter - 1) == 0)
			if save_builds:
				ldr_path = self.__create_ldr_dir(i)
			graph_path = self.__create_graph_dir(i)

			for sample in self.ds:
				g = sample['graph']
				filename = sample['filename']
				g.valid_graph = True
				g.overconstrained_brick = False
				g.merged_brick = False
				g.invalid_shift = False
				
				self.permuter.permute_graph(g)
				self.__increment_counters(g)

				if save_builds and g.valid_graph:
					filename = os.path.join(ldr_path, filename)
					g.write_to_file(filename)

			if save_builds:
				os.system('zip -r -j {}/ldrs.zip . {}/*.ldr'.format(ldr_path, ldr_path))
				os.system('rm {}/*.ldr'.format(ldr_path))

			with open(os.path.join(graph_path, 'graphs.h5'), 'wb') as f:
				pickle.dump(self.ds, f)

			self.__evaluate_graphs()
			self.__print_results(i)

		return self.results	


	def __initialize_values(self):
		metrics = ['fid', 'GIN_accuracy', 'kid', 'density', 'coverage', 'precision',
			'recall', 'invalid_ratio', 'num_disjoint']
		self.results = {}
		for metric in metrics:
			self.results[metric] = []
		self.num_samples = len(ds_two)
		self.num_invalid = 0
		self.num_disjoint = 0
		self.num_nodes = 0
		self.n_iters_to_save = 25


	def __initialize_results(self):
		ldr_dir = self.__create_ldr_dir(0)
		graph_dir = self.__create_graph_dir(0)

		for graph in self.ds:
			g = graph['graph']
			g.convert_to_LDraw_and_verify(os.path.join(ldr_dir, graph['filename']))
			self.num_nodes += g.number_of_nodes()

		with open(os.path.join(graph_dir, 'graphs.h5'), 'wb') as f:
			pickle.dump(self.ds, f)


	def __create_ldr_dir(self, iter):
		iter_path = os.path.join(self.dir, 'iter_{:03d}'.format(iter))
		ldr_path = os.path.join(iter_path, 'ldr_files')
		try:
			os.mkdir(iter_path)
			os.mkdir(ldr_path)
		except FileExistsError:
			pass

		return ldr_path

	def __create_graph_dir(self, iter):
		iter_path = os.path.join(self.dir, 'iter_{:03d}'.format(iter))
		graph_path = os.path.join(iter_path, 'graphs')
		try:
			os.mkdir(iter_path)
		except FileExistsError:
			pass
		try:
			os.mkdir(graph_path)
		except FileExistsError:
			pass

		return graph_path


	def __evaluate_graphs(self):
		ds = GINDataset(list_of_graphs = self.ds)
		metrics = self.evaluation.evaluate_all(ds, calculate_accuracy = True)
		metrics['invalid_ratio'] = (self.num_invalid / self.num_samples) * 100
		metrics['num_disjoint'] = (self.num_disjoint / self.num_samples) * 100

		for key, val in metrics.items():
			self.results[key].append(val)

	
	def __increment_counters(self, g):
		if not g.valid_graph:
			self.num_invalid += 1
		if g.is_disjoint():
			self.num_disjoint += 1
		self.num_nodes += g.number_of_nodes()




	def __print_results(self, iter):
		print('fid: {}, invalid: {}, disjoint: {}, acc: {}, avg nodes: {}, iter: {}'.format(self.results['fid'][-1], 
			self.results['invalid_ratio'][-1], self.results['num_disjoint'][-1], self.results['GIN_accuracy'][-1], self.num_nodes / len(self.ds), iter))


def save_results_to_file(res, path):
	for key, val in res.items():
		with open(os.path.join(path, '{}.dat'.format(key)), 'wb') as f:
			pickle.dump(val, f)

def load_ds_two(args, dataset):
	if args.resume_from == 'None':
		if args.dataset_split == 'split':
			ds_two = np.array(dataset)[ix[split:]]
		elif args.dataset_split == 'copy':
			ds_two = []
			for i in dataset[:200]:
				g = i['graph']; filename = i['filename']; target = i['target']
				g = LegoGraph.GeneratedLegoGraph(g)
				g.update_nx_graphs()
				helpers.add_node_attributes(g)
				helpers.add_edge_attributes(g)
				ds_two.append({'graph': g, 'filename': filename, 'target': target})
				# ds_two.append(copy.deepcopy(dataset[i]))
		elif args.dataset_split == 'strip':
			split = 200
			ds_two = np.array(dataset)[ix[:split]]
		elif args.dataset_split == 'copy-full':
			ds_two = []
			for i in dataset:
				g = i['graph']; filename = i['filename']; target = i['target']
				g = LegoGraph.GeneratedLegoGraph(g)
				g.update_nx_graphs()
				helpers.add_node_attributes(g)
				helpers.add_edge_attributes(g)
				ds_two.append({'graph': g, 'filename': filename, 'target': target})
				# ds_two.append(copy.deepcopy(i))
		iter = 0

	else:
		dir = os.path.join('permutation_results', args.resume_from, 'run_00')
		files = sorted(os.listdir(dir))
		iter = int(files[-1].split('_')[-1])
		with open(os.path.join(os.getcwd(), 'permutation_results', args.resume_from, 'run_00', files[-1], 'graphs', 'graphs.h5'), 'rb') as f:
			ds_two = pickle.load(f)

	return ds_two, iter

def update_args(args):
	split = args.resume_from.split('_')
	args.force_valid = ast.literal_eval(split[1])
	args.remove_disjoint = ast.literal_eval(split[2])
	args.dataset_split = split[3]

def parse_args():
	parser = argparse.ArgumentParser(description='Permutations')

	parser.add_argument('--dataset_split', default = 'copy-full', choices = ['split', 'copy-full', 'copy', 'strip'])
	parser.add_argument('--num_permutations', default = 500, type = int)
	parser.add_argument('--include_random', action = 'store_true')
	parser.add_argument('--resume_from', default = 'None')

	args = parser.parse_args()
	args.force_valid = True; args.remove_disjoint = True
	return parser.parse_args()


def split_dataset(args):
	if args.resume_from != 'None':
		update_args(args)

	config = {'include_augmented_ninety': False, 'include_augmented': True, 'include_random': args.include_random, 'use_bfs': False, 'use_reset': False}
	dataset = GINDataset(config = config, with_edge_types = True)
	num_samples = len(dataset.dataset)
	ix = list(range(num_samples))
	np.random.seed(42)
	np.random.shuffle(ix)

	if args.dataset_split == 'split':
		split = num_samples // 2
		ds_one = np.array(dataset.dataset)[ix[:split]]

	elif args.dataset_split == 'copy':
		ds_one = np.array(dataset.dataset)[ix]

	elif args.dataset_split == 'strip':
		split = 200
		ds_one = np.array(dataset.dataset)[ix[split:]]

	elif args.dataset_split == 'copy-full':
		ds_one = np.array(dataset.dataset)[ix]

	ds_two, start_iter = load_ds_two(args, dataset.dataset)

	ds_one = GINDataset(list_of_graphs = list(ds_one))

	return ds_one, ds_two, start_iter


def get_run_dir(run_num, args):
	if args.resume_from == 'None':
		res = 'results_{}_{}_{}'.format(args.force_valid, args.remove_disjoint, args.dataset_split)
		res_dir = os.path.join(os.getcwd(), 'permutation_results', res)
		run_dir = os.path.join(res_dir, 'run_{:02d}'.format(run_num))
		
		i = 0
		while(os.path.isdir(res_dir)):
			res = 'results_{}_{}_{}_{}'.format(args.force_valid, args.remove_disjoint, args.dataset_split, i)
			res_dir = os.path.join(os.getcwd(), 'permutation_results', res)
			i += 1

		run_dir = os.path.join(res_dir, 'run_{:02d}'.format(run_num))

		os.mkdir(res_dir)
		os.mkdir(run_dir)
	else:
		run_dir = os.path.join(os.getcwd(), 'permutation_results', args.resume_from, 'run_00')
	return run_dir


if __name__ == '__main__':
	args = parse_args()

	ds_one, ds_two, start_iter = split_dataset(args)

	#setup gin evaluation metrics
	gin = load_gin()
	evaluation = DGMGEvaluationWithGIN(ds_one, gin, embed_func = 'get_graph_embed_concat')
	print('Permuted dataset size: ', len(ds_two))

	n_runs = 1
	n_steps = args.num_permutations
	for i in range(n_runs):
		path = get_run_dir(i, args)
		np.random.seed()
		runner = Runner(ds_two, evaluation, path, start_iter, args)
		results = runner.perform_run(n_iter = n_steps)
		
		save_results_to_file(results, path)
