import json
import networkx as nx
import numpy as np
from pyFiles import LDraw, LegoGraph
import importlib
import ast
import rebrick
import dgl

global DEBUG
DEBUG = False


class RebrickableDatabase():
	#Open the database
	with open('database/brick_type_db.json', 'r') as fp:
		brick_types = json.load(fp)

	#Open the database
	with open('database/brick_size_db.json', 'r') as fp:
		brick_sizes = json.load(fp)



	def update_brick_size(self, new_key, new_val):
		self.brick_size[new_key] = new_val
		with open('database/brick_size_db.json', 'w') as fp:
			json.dump(new_db, fp)



	def update_brick_size(self, new_key, new_val):
		self.brick_types[new_key] = new_val
		with open('database/brick_type_db.json', 'w') as fp:
			json.dump(new_db, fp)



	def get_transformation_and_brick_name(self, brick_size):
		transformation_matrix_and_brick_name = self.brick_sizes[brick_size]
		transformation_matrix = ' '.join(transformation_matrix_and_brick_name.split()[:-1])
		brick_name = transformation_matrix_and_brick_name.split()[-1]

		return transformation_matrix, brick_name



def first_is_on_top_of_second(g, first, second):
	return first in g.g_directed.neighbors(second) #True if there's a directed edge from cur to prev



def get_brick_size_tuple(g, brick):
	#Retrieve a tuple representing the size of the brick
	size = g.node_labels[brick] 
	brick_type = size.split('(')[0]
	size = '(' + size.split('(')[1]
	size = ast.literal_eval(size)
	if brick_type == 'Brick' or brick_type == 'Slope':
		height = 3
	else:
		height = 1
	return (int(size[0]), int(size[1]), height) #Tuple of (width x length)



def get_brick_type(g, ix):
	return ''.join([c for c in g.node_labels[ix] if c.isalpha()])



def strip_node_labels(g):
	stripped_labels = {}
	for i, label in g.node_labels.items():
		stripped_labels[i] = ''.join(c for c in label if not c.isalpha())
	return stripped_labels


def get_brick_size(g, ix):
	return ''.join(c for c in g.node_labels[ix] if not c.isalpha())

def write_to_file(g, fileName):
	file_lines = g.get_LDraw_representation()
	with open(fileName, 'w') as file:
		for line in file_lines:
			file.write('{}\n'.format(line))


class LegoGraphToActionSequence():

	"""Summary: Converts graphs to an action sequence to be used in training the generative
	model.
	
	Attributes:
	    node_action_to_type (dict): Maps actions (ints) to node labels (strs)
	    node_type_to_action (dict): Maps node labels (strs) to actions (ints)
	"""
	
	def __init__(self):
		self.node_type_to_action = {}
		self.node_type_to_action['Brick(2, 4)'] = 1
		self.node_type_to_action['Brick(4, 2)'] = 2

		self.node_action_to_type = {v: k for k, v in self.node_type_to_action.items()}



	def to_action_sequence(self, g, class_num):
		"""Summary: Converts the given graph with the given class number to an action sequence.
		
		Args:
		    g (LegoGraph): The graph to convert
		    class_num (int): The class number of the graph
		
		Returns:
		    list: The action sequence to build the given graph
		"""
		action_sequence = []
		action_sequence.append(class_num) # Used for class conditioning
		action_sequence.append(self.node_type_to_action[g.node_labels[0]]) # Action to create first node
		action_sequence.append(0) # Action to not add edges to this node
		# diffs = []

		for node in range(1, len(g.g_undirected.nodes)): # Iterate over all nodes
			action_sequence.append(self.node_type_to_action[g.node_labels[node]]) #Add node action
			#Get list of neighbors to be added
			neighbors = np.array(list(g.g_undirected.neighbors(node)))
			neighbors_to_add = neighbors[neighbors < node]
			#Add them to action sequence
			for neighbor in neighbors_to_add:
				# diffs.append(node - neighbor)
				if first_is_on_top_of_second(g, first = neighbor, second = node):
					action_sequence.append(1) # Action to indicate the edge is from node to neighbor
					edge_embedding = ast.literal_eval(g.edge_labels[node, neighbor])

				else:
					action_sequence.append(2) # Action to indicate the edge is from neighbor to node
					edge_embedding = ast.literal_eval(g.edge_labels[neighbor, node])

				action_sequence.append(neighbor) # Action to connect edge to neighbor
				action_sequence.append(edge_embedding) # Action to determine positional shifts
			
			action_sequence.append(0) #Action to stop adding edges
		
		action_sequence.append(0) #Action to stop adding nodes
		
		return action_sequence#, diffs



	def to_action_sequence_subgraphs(self, g, class_num):
		ret = []
		subgraph = dgl.DGLGraph()
		action_sequence = []
		action_sequence.append(class_num)
		action_sequence.append(self.node_type_to_action[g.node_labels[0]]) #Action to create first node
		action_sequence.append(0) #Action to not add edges to this node
		ret.append({'graph': subgraph, 'actions': action_sequence})

		for node in range(1, g.number_of_nodes()):
			# print(node)
			action_sequence = []
			# Class conditioning
			action_sequence.append(class_num)
			# Add node action
			action_sequence.append(self.node_type_to_action[g.node_labels[node]])
			subgraph = LegoGraph.LegoGraph(g.subgraph(range(node)))
			for i in range(node):
				subgraph.node_labels[i] = g.node_labels[i]
			edges = subgraph.edges(); num_edges = len(edges[0])
			for i in range(num_edges):
				src = edges[0][i].item(); dest = edges[1][i].item()
				subgraph.edge_labels[src, dest] = g.edge_labels[src, dest]
			next_subgraph = g.subgraph(range(node + 1))
			srcs = next_subgraph.edges()[0]; dests = next_subgraph.edges()[1]
			# print(srcs, dests)
			for ix, src in enumerate(srcs):
				src = src.item()
				if src == node:
					# print('src ', src)
					action_sequence.append(1)
					action_sequence.append(dests[ix].item())
					action_sequence.append(ast.literal_eval(g.edge_labels[src, dests[ix].item()]))
			for ix, dest in enumerate(dests):
				dest = dest.item()
				if dest == node:
					# print('dest ', dest)
					action_sequence.append(2)
					action_sequence.append(srcs[ix].item())
					action_sequence.append(ast.literal_eval(g.edge_labels[srcs[ix].item(), dest]))
			action_sequence.append(0) #Action to stop adding edges
			ret.append({'graph': subgraph, 'actions': action_sequence})
		ret.append({'graph': g, 'actions': [0]}) #EOS
		return ret




#-------------------------------------------------------------------------------------------



class ImpliedEdgesUtil():

	"""Summary: Class for working with implied edges in a graph. Provides methods for adding all implied
	edges, implied edges missing from a node, retrieving implied edges without connecting them, and so on.
	"""
	
	def add_implied_edges(self, g):
		"""Summary: Adds all implied edges to the given graph.
		
		Args:
		    g (LegoGraph): The LegoGraph we want to add implied edges to.
		"""
		self.__assert_already_converted_to_LDraw(g)

		for node in list(g.g_directed.nodes):
			self.add_edges_implied_by_node(g, node)



	def add_edges_implied_by_node(self, g, node_ix):
		"""Summary: Add all missing implied edges to the given node.
		
		Args:
		    g (LegoGraph): The LegoGraph we want to add implied edges to.
		    node_ix (int): The index of the node we want to add missing implied edges to
		"""
		implied_edges = self.get_edges_implied_by_node(g, node_ix)
		implied_edges.add_all_edges()



	def get_implied_edges(self, g):
		"""Summary: Function to return all implied edges without adding them to the graph.
		
		Args:
		    g (LegoGraph): The graph whose implied edges we want to retrieve.
		
		Returns:
		    list, list: A list containing information about the implied edges (src, dest, shifts), and
		    a corresponding list of [brick1, brick2] pairs which are missing an implied edge.
		"""
		self.__assert_already_converted_to_LDraw(g)

		implied_edges = ImpliedEdges()
		for node in list(g.g_directed.nodes):
			implied_edges += self.get_edges_implied_by_node(g, node, check_top_and_bottom=False)

		return implied_edges



	def get_edges_implied_by_node(self, g, node_ix, check_top_and_bottom=True):
		"""Summary: Get all implied edges missing from the node at node_ix.
		
		Args:
		    g (LegoGraph): The LegoGraph we want to get missing implied edges for.
		    node_ix (int): The index of the node in g that we want to get missing implied edges for.
		
		Returns:
		    list, list: A list containing information about the implied edges (src, dest, shifts), and
		    a corresponding list of [brick1, brick2] pairs which are missing an implied edge.
		"""
		if (g.number_of_nodes() == 0 or (g.number_of_edges() == 0 and not isinstance(g, LegoGraph.GeneratedLegoGraphSubgraph))):
			return []

		if nx.is_connected(g.g_undirected) == False:
			return self.__get_edges_implied_by_node_disjoint_graph(g, node_ix, check_top_and_bottom=check_top_and_bottom)
		else:
			return self.__get_edges_implied_by_node_connected_graph(g, node_ix, check_top_and_bottom=check_top_and_bottom)



	def __get_edges_implied_by_node_disjoint_graph(self, g, node_ix, check_top_and_bottom=True):
		for subgraph in g.lego_subgraphs:
			if node_ix in subgraph.lego_assembly.bricks:
				return self.get_edges_implied_by_node(subgraph, node_ix, check_top_and_bottom=check_top_and_bottom)
 


	def __get_edges_implied_by_node_connected_graph(self, g, node_ix, check_top_and_bottom=True):
		implied_edges = ImpliedEdges()
		bricks = g.lego_assembly.bricks
		brick = bricks[node_ix]
		connection_heights = brick.get_connection_heights()

		top_brick = {'data': brick, 'ix': node_ix}
		for bottom_brick in g.lego_assembly.bricks_indexed_by_height[connection_heights[0]]:
			implied_edges += self.__get_edge_info_if_bricks_can_connect(g, top_brick, bottom_brick)

		if check_top_and_bottom:
			bottom_brick = {'data': brick, 'ix': node_ix}
			for top_brick in g.lego_assembly.bricks_indexed_by_height[connection_heights[1]]:
				implied_edges += self.__get_edge_info_if_bricks_can_connect(g, top_brick, bottom_brick)

		return implied_edges



	def __get_edge_info_if_bricks_can_connect(self, g, top_brick, bottom_brick):
		bottom_brick_ix = bottom_brick['ix']
		bottom_brick = bottom_brick['data']
		top_brick_ix = top_brick['ix']
		top_brick = top_brick['data']
		if top_brick == bottom_brick:
			return []

		try:
			if top_brick.can_connect_on_top_of(bottom_brick) and (bottom_brick_ix, top_brick_ix) not in g.edge_labels:
				return [self.__get_edge_info(g, top_brick, bottom_brick, top_brick_ix, bottom_brick_ix)]

		except MergedBrick:
			pass

		return []



	def __get_edge_info(self, g, top_brick, bottom_brick, top_brick_ix, bottom_brick_ix):
		x_shift, z_shift = top_brick.get_connection_points(bottom_brick)
		return ImpliedEdge(g, bottom_brick_ix, top_brick_ix, x_shift, z_shift)



	def __assert_already_converted_to_LDraw(self, g):
		if len(g.lego_assembly) != g.number_of_nodes() \
			or len(g.lego_assembly.bricks_indexed_by_height_list) != g.number_of_nodes():
			g.convert_to_LDraw()

		assert len(g.lego_assembly) == g.number_of_nodes(), 'Issue converting to LDraw: got lens {} and {}'.format(len(g.lego_assembly), g.number_of_nodes())
		assert len(g.lego_assembly.bricks_indexed_by_height_list) == g.number_of_nodes(), 'Bricks indexed by height != num nodes: {} and {}'.format(len(g.lego_assembly.bricks_indexed_by_height_list), g.number_of_nodes())



#-------------------------------------------------------------------------------------------



class ImpliedEdges(list):
	def add_all_edges(self):
		for edge in self:
			edge.insert_implied_edge()



#-------------------------------------------------------------------------------------------



class ImpliedEdge():
	def __init__(self, g, src, dest, x_shift, z_shift):
		self.src = src; self.dest = dest; self.x_shift = x_shift; self.z_shift = z_shift
		if isinstance(g, LegoGraph.GeneratedLegoGraphSubgraph):
			self.g = g.parent_lego_graph
		else:
			self.g = g


	def insert_implied_edge(self):
		if not self.g.has_edge_between(self.src, self.dest):
			self.g.add_edges([self.src], [self.dest])
			self.g.edge_labels[self.src, self.dest] = '({}, {})'.format(self.x_shift, self.z_shift)



#-------------------------------------------------------------------------------------------




class LegoGraphValidation():

	"""Summary: Used for validating LegoGraphs
	
	Attributes:
	    implied_edges (ImpliedEdges): Used for determining if graphs are missing implied edges.
	"""
	
	implied_edges = ImpliedEdgesUtil()

	def check_if_brick_overconstrained(self, g):
		# Return if the given graph has an overconstrained brick
		g.assert_already_converted_to_LDraw()

		if self.__graph_is_too_small_for_error(g) or g.overconstrained_brick:
			return g.overconstrained_brick

		if g.is_disjoint():
			self.__check_if_subgraphs_overconstrained(g)

		return g.overconstrained_brick



	def __graph_is_too_small_for_error(self, g):
		return g.number_of_nodes() <= 2 or (g.number_of_edges() == 0 and not isinstance(g, LegoGraph.GeneratedLegoGraphSubgraph))



	def __check_if_subgraphs_overconstrained(self, g):
		assert len(g.lego_subgraphs) > 0, 'len(lego_subgraphs) == 0, must convert to ldraw before this function'
		for subgraph in g.lego_subgraphs:
			if subgraph.overconstrained_brick == True:
				g.overconstrained_brick = True
				g.valid_graph = False
				break



	def check_if_bricks_merged(self, g):
		# Returns if the given graph has bricks occupying the same space
		g.assert_already_converted_to_LDraw()

		if self.__graph_is_too_small_for_error(g) or g.merged_brick:
			return g.merged_brick

		if g.is_disjoint():
			self.__check_if_subgraphs_merged(g)

		else:
			self.__check_if_connected_graph_merged(g)
		
		return g.merged_brick



	def __check_if_subgraphs_merged(self, g):
		for subgraph in g.lego_subgraphs:
			if self.check_if_bricks_merged(subgraph) == True:
				g.valid_graph = False
				g.merged_brick = True
				break



	def __check_if_connected_graph_merged(self, g):
		bricks = g.lego_assembly.bricks
		g.merged_brick = not LDraw.verify_LDraw_from_graph(g)
		g.valid_graph = False if g.merged_brick else g.valid_graph



	def check_if_missing_implied_edges(self, g):
		# Checks if the given graph is missing implied edges
		g.assert_already_converted_to_LDraw()

		if self.__graph_is_too_small_for_error(g) or g.missing_implied_edges:
			return g.missing_implied_edges

		elif g.is_disjoint():
			self.__check_if_subgraphs_missing_implied_edges(g)

		else:
			self.__check_if_missing_implied_edges_connected(g)

		return g.missing_implied_edges



	def __check_if_subgraphs_missing_implied_edges(self, g):
		for subgraph in g.lego_subgraphs:
			if self.check_if_missing_implied_edges(subgraph) == True:
				g.missing_implied_edges = True
				break



	def __check_if_missing_implied_edges_connected(self, g):
		implied_edges = self.implied_edges.get_implied_edges(g)
		if len(implied_edges) > 0:
			g.missing_implied_edges = True



	def check_if_graph_has_invalid_shift(self, g):
		# Returns if the graph contains invalid shifts
		res = True
		for edge in g.g_directed.edges:
			res = res and self.check_if_shift_invalid(g, edge[0], edge[1])

		g.invalid_shift = not res
		g.valid_graph = False if g.invalid_shift else g.valid_graph
		return g.invalid_shift



	def check_if_shift_invalid(self, g, src, dest):
		# Returns if the edge between the given nodes is invalid
		assert g.has_edge_between(src, dest), 'Checking unconnected nodes for invalid shift, nodes {} and {}'.format(src, dest)
		
		shifts = ast.literal_eval(g.edge_labels[src, dest])
		x_shift = shifts[0]; z_shift = shifts[1]

		src_brick_size = get_brick_size_tuple(g, src)
		dest_brick_size = get_brick_size_tuple(g, dest)
		ret = True
		
		if src_brick_size == (4, 2, 3) and dest_brick_size == (4, 2, 3):
			if abs(x_shift) > 3 or abs(z_shift) > 1:
				g.invalid_shift = True
				g.valid_graph = False
				ret = False

		elif src_brick_size == (2, 4, 3) and dest_brick_size == (2, 4, 3):
			if abs(x_shift) > 1 or abs(z_shift) > 3:
				g.invalid_shift = True
				g.valid_graph = False
				ret = False

		elif src_brick_size == (4, 2, 3) and dest_brick_size == (2, 4, 3):
			if abs(x_shift) > 2 or abs(z_shift) > 2:
				g.invalid_shift = True
				g.valid_graph = False
				ret = False

		elif src_brick_size == (2, 4, 3) and dest_brick_size == (4, 2, 3):
			if abs(x_shift) > 2 or abs(z_shift) > 2:
				g.invalid_shift = True
				g.valid_graph = False
				ret = False
		else:
			raise Exception('Shouldnt be able to get here')

		return ret



	def invalid_shift(self, g):
		return g.invalid_shift



	def is_disjoint(self, g):
		if len(list(g.g_undirected.nodes)) <= 1:
			return False
		return not nx.is_connected(g.g_undirected)



	def is_valid(self, g):
		if g.valid_graph == False or g.number_of_edges() == 0:
			return False
		return True


#-------------------------------------------------------------------------------------------



class TestingHelpers():
	def is_same_lego_build(self, g, g_test):
		if self.__check_if_obvious_difference(g, g_test):
			return False
		
		g.convert_to_LDraw()
		g_test.convert_to_LDraw()
		res = True
		if g.is_disjoint():
			res = self.__check_if_subgraphs_are_same_lego_build(g, g_test)

		else:
			res = self.__check_if_connected_graph_is_same_lego_build(g, g_test)

		return res



	def __check_if_obvious_difference(self, g, g_test):
		if g.number_of_nodes() != g_test.number_of_nodes() or g.number_of_edges() != g_test.number_of_edges():
			return True

		cur_is_disjoint = g.is_disjoint()
		test_is_disjoint = g_test.is_disjoint()
		if cur_is_disjoint != test_is_disjoint:
			return True



	def __check_if_subgraphs_are_same_lego_build(self, g, g_test):
		res = True
		if len(g.lego_subgraphs) != len(g_test.lego_subgraphs):
			res = False

		same_subgraph_counter = 0
		for cur_subgraph in g.lego_subgraphs:
			for test_subgraph in g_test.lego_subgraphs:
				same_subgraph_counter += cur_subgraph.is_same_lego_build(g_test)
		res = same_subgraph_counter == len(g.lego_subgraphs)

		return res



	def __check_if_connected_graph_is_same_lego_build(self, g, g_test):
		cur_file_lines = g.get_LDraw_representation()
		test_file_lines = g_test.get_LDraw_representation()
		# assert len(cur_file_lines) != 0, 'generated file lines is zero'
		# assert len(test_file_lines) != 0, 'test file lines is zero'
		res = True
		file_lines_counter = {}
		for line in cur_file_lines:
			if line[0] != '1':
				continue
			line = line[4:]
			try:
				file_lines_counter[line] += 1
			except:
				file_lines_counter[line] = 1
		for line in test_file_lines:
			if line[0] != '1':
				continue
			line = line[4:]
			try:
				file_lines_counter[line] -= 1
			except:
				res = False
		for counter in file_lines_counter.values():
			if counter != 0:
				res = False
				break

		return res



	def is_same_graph(self, g, g_test):
		if self.__check_if_obvious_difference(g, g_test):
			return False

		g.convert_to_LDraw()
		g_test.convert_to_LDraw()
		res = True
		if g.is_disjoint():
			res = self.__check_if_subgraphs_are_same_graph(g, g_test)

		else:
			res = self.__check_if_connected_graph_is_same_graph(g, g_test)

		return res



	def __check_if_subgraphs_are_same_graph(self, g, g_test):
		res = True
		for i, subgraph in enumerate(g.lego_subgraphs):
			res = res and subgraph.is_same_graph(g_test.lego_subgraphs[i])
		return res



	def __check_if_connected_graph_is_same_graph(self, g, g_test):
		if list(g.node_labels.values()) != list(g_test.node_labels.values()):
			return False

		if (g.all_edges()[0] != g_test.all_edges()[0]).any().item() or (g.all_edges()[1] != g_test.all_edges()[1]).any().item():
			return False

		if list(g.edge_labels.values()) != list(g_test.edge_labels.values()):
			return False
		return True



#-------------------------------------------------------------------------------------------



class MergedBrick(Exception):
	pass