import networkx as nx
import dgl
import numpy as np
import json
import ast
import os
import importlib
import copy

from pyFiles import LDraw
from pyFiles import utils
from pyFiles import Brick

importlib.reload(LDraw)
importlib.reload(utils)
importlib.reload(Brick)


#Comments at top of add_bricks_to_ldr function and figure below this cell explain more about what these 3 numbers mean
LDR_UNITS_PER_STUD = 20 #width/length of stud in LDR units
LDR_UNITS_PER_PLATE = 8 #height of a lego plate

global DEBUG
DEBUG = utils.DEBUG


class LegoGraph(dgl.DGLGraph):        
	"""
	Class: LegoGraph
	Summary: Inherits from dgl.DGLGraph (ie it can do everything a dgl graph does, plus the functionality I've added below).
	Main purpose right now is for converting to LDraw, verifying the graph. 
	"""

	graph_to_action_sequence = utils.LegoGraphToActionSequence()
	graph_validation = utils.LegoGraphValidation()
	def __init__(self, *args):
		super().__init__(*args)
		if len(args) > 0:
			if isinstance(args[0], LegoGraph):
				self.edge_labels = copy.deepcopy(args[0].edge_labels)
				self.node_labels = copy.deepcopy(args[0].node_labels)
		else:
			#Edge labels, where edge_labels[src, dest] gives label in the form '(x_shift, z_shift)'
			#Both edge and node labels are stored as a string to make life easy with visualizing the graph
			self.edge_labels = {}
			#Node labels, where node_labels[node] gives label in the form 'Brick(4, 2)'
			self.node_labels = {}

		#Networkx does a few things better than DGL, so it's helpful to carry around a networkx equivalent to our graph
		self.update_nx_graphs()

		# Stores LEGO assembly/LDraw representation
		self.lego_assembly = Brick.LEGOAssembly()

		self.LDraw_conversion = LegoGraphToLDraw()

		#For keeping track of validity
		self.invalid_shift = False # Keeps track of when we add a shift that is out of range.
		self.overconstrained_brick = False # Keeps track of when a brick is required to be in multiple locations at once.
		self.merged_brick = False # Keeps track of when two bricks occupy the same space.
		self.missing_implied_edges = False # Keeps track of when the graph is missing implied edges. This doesn't invalidate a graph.
		self.valid_graph = True



	def get_nodes_that_can_connect_underneath_of(self, size):
		"""Summary: Get a list of nodes that can connect underneath of a brick with the given size
		(i.e an edge from a pre-existing brick to a new brick with given size).
		All nodes in the returned list can form a valid connection underneath of a brick with the
		given size.
		
		Args:
			size (tuple of ints): Tuple of ints representing the brick we want to find valid connection
			nodes for.
		
		Returns:
			list of ints: A list where each element is the node number of a node that can connect underneath
			of a brick with the given size.
		"""

		# Need to convert to LDraw first (to obtain all the LEGO brick positions)
		self.assert_already_converted_to_LDraw()

		if self.is_disjoint() == False:
			return self.lego_assembly.get_bricks_with_open_top_connections_for(size)
		else:
			bricks = []
			for subgraph in self.lego_subgraphs:
				bricks += subgraph.lego_assembly.get_bricks_with_open_top_connections_for(size)
			return bricks



	def get_nodes_that_can_connect_on_top_of(self, size):
		"""Summary: Get a list of nodes that can connect on top of a brick with the given size
		(i.e an edge from a new brick with given size to a pre-existing brick).
		All nodes in the returned list can form a valid connection on top of a brick with the
		given size.
		
		Args:
			size (tuple of ints): Tuple of ints representing the brick we want to find valid connection
			nodes for.
		
		Returns:
			list of ints: A list where each element is the node number of a node that can connect on top
			of a brick with the given size.
		"""
		self.assert_already_converted_to_LDraw()

		if self.is_disjoint() == False:
			return self.lego_assembly.get_bricks_with_open_bottom_connections_for(size)
		else:
			bricks = []
			for subgraph in self.lego_subgraphs:
				bricks += subgraph.lego_assembly.get_bricks_with_open_bottom_connections_for(size)
			return bricks



	def get_valid_connections_old_underneath(self, old_node, new_size):
		"""Summary: Returns a list of (x_shfit, z_shift) pairs that can be used to form a valid
		connection from old_node to a brick of size new_size.
		
		Args:
		    old_node (int): Integer representing the node index for the node we want to find valid
		    connections for.
		    new_size (tuple): Size of the new brick we want to connect (i.e (4, 2))
		
		Returns:
		    list: List of (x_shift, z_shift) pairs that can be used to form a valid connection
		    from old_node to a brick of size new_size.
		"""
		self.assert_already_converted_to_LDraw()

		if self.is_disjoint() == False:
			return self.lego_assembly.get_open_top_connections_for(old_node, new_size)

		else:
			for subgraph in self.lego_subgraphs:
				if old_node in list(subgraph.g_undirected.nodes):
					return subgraph.lego_assembly.get_open_top_connections_for(old_node, new_size)
	


	def get_valid_connections_old_on_top(self, old_node, new_size):
		"""Summary: Returns a list of (x_shfit, z_shift) pairs that can be used to form a valid
		connection from a brick of size new_size to old_node.
		
		Args:
		    old_node (int): Integer representing the node index for the node we want to find valid
		    connections for.
		    new_size (tuple): Size of the new brick we want to connect (i.e (4, 2))
		
		Returns:
		    list: a list of (x_shfit, z_shift) pairs that can be used to form a valid
			connection from a brick of size new_size to old_node.
		"""
		self.assert_already_converted_to_LDraw()

		if self.is_disjoint() == False:
			return self.lego_assembly.get_open_bottom_connections_for(old_node, new_size)

		else:
			for subgraph in self.lego_subgraphs:
				if old_node in list(subgraph.g_undirected.nodes):
					return subgraph.lego_assembly.get_open_bottom_connections_for(old_node, new_size)



	def assert_already_converted_to_LDraw(self):
		# Checks that we have already converted to LDraw and converts if we haven't
		if len(self.lego_assembly) != self.number_of_nodes():
			self.convert_to_LDraw()

		assert len(self.lego_assembly) == self.number_of_nodes(), 'Issue converting to LDraw: got lens {} and {}'.format(len(self.lego_assembly), self.number_of_nodes())



	def is_disjoint(self):
		# Returns if graph is disjoint
		if len(list(self.g_undirected.nodes)) <= 1:
			return False
		return not nx.is_connected(self.g_undirected)



	def set_invalid(self):
		self.valid_graph = False



	def update_nx_graphs(self):
		# It's useful to carry around up-to-date networkx versions of the graph so this updates them
		self.g_directed = nx.DiGraph(self.to_networkx())
		self.g_undirected = self.g_directed.to_undirected()



	def convert_to_LDraw_and_verify(self, filename, start_node = 0):
		"""Summary: Convert the graph to LDraw and verify it is a proper LEGO build. Saves it to the
		given file.

		Args:
		    filename (str): The filename to save to. The '.ldr' ending is not added automatically.
		    If filename == None then validity is checked and returned but the file is not saved.
		    start_node (int, optional): The node to start the conversion with (i.e the node/brick that is
		    placed at the origin)
		
		Returns:
		    bool: Validity of the LEGO graph
		"""
		return self.LDraw_conversion.convert_to_LDraw_and_verify(self, filename, start_node = start_node)



	def convert_to_LDraw(self, *args, start_node = 0):
		"""Summary: Convert the graph to LDraw.

		Args:
		    filename (str): The filename to save to. The '.ldr' ending is not added automatically
		    start_node (int, optional): The node to start the conversion with (i.e the node/brick that is
		    placed at the origin)
		"""
		self.LDraw_conversion.convert_to_LDraw(self, *args, start_node = start_node)



	def save_graph(self, filename):
		#For pickling/storing the graph itself
		import pickle
		with open(filename, 'wb') as f:
			pickle.dump(self, f)



	def write_to_file(self, filename):
		#For outputting the graph to LDraw to visualize as a LEGO structure
		self.assert_already_converted_to_LDraw()
		if self.is_disjoint():
			self.__write_subgraphs_to_file(filename)

		else:
			utils.write_to_file(self, filename)



	def get_LDraw_representation(self):
		# Get the LDraw representation for writing to file
		self.assert_already_converted_to_LDraw()

		LDraw_representation = []
		for brick in list(self.lego_assembly.values()):
			LDraw_representation.append(brick.LDraw_representation)
		return LDraw_representation

	def add_generated_node(self, type):
		"""Summary: Adds a node of the given type to the graph and updates networkx representations.
		See LegoGraphToActionSequence class in pyFiles/utils.py to see/change the type of each node.
		
		Args:
		    type (int): The node type to be added. Used to index a dictionary in a utils.LegoGraphToActionSequence
		    class.
		"""
		assert 0 < type <= 2, 'Node type to add must be 0 < type <= 2 (for Brick(4,2), Brick(2, 4)), got {}'.format(type)
		
		self.add_nodes(1)
		self.node_labels[len(self.nodes) - 1] = self.graph_to_action_sequence.node_action_to_type[type] # Convert int to node label
		
		# Update networkx representations
		self.g_directed.add_node(self.number_of_nodes() - 1)
		self.g_undirected.add_node(self.number_of_nodes() - 1)

		if len(self.nodes) == 1:
			self.LDraw_conversion.initialize_conversion(self) # Initialize conversion to LDraw



	def add_generated_edge(self, src, dest, x_shift, z_shift):
		"""Summary: Adds a generated edge between the given nodes with the given type. Updates LDraw and
		networkx representations, as well as the validity of the graph.
		
		Args:
		    src (int): The node index for the source of the edge.
		    dest (int): The node index for the destination of the edge.
		    x_shift (int): The x_shift of the edge.
		    z_shift (int): The z_shift of the edge.
		"""
		assert abs(x_shift) <= 3 and abs(z_shift) <= 3, 'Shift out of range, got ({} and {})'.format(x_shift, z_shift)
		assert not g.has_edge_between(src, dest), 'Edge already present between {} and {}'.format(src, dest)
		self.add_edges([src], [dest])
		self.edge_labels[src, dest] = '({}, {})'.format(x_shift, z_shift) # Set edge label
		
		# Update networkx representations
		self.g_undirected.add_edges_from([(src, dest), (dest, src)])
		self.g_directed.add_edge(src, dest)

		self.graph_validation.check_if_shift_invalid(self, src, dest)
		self.convert_to_LDraw(src, dest) # Update LDraw representation



	def __write_subgraphs_to_file(self, filename):
			subgraph_counter = 0
			filename = filename[:-4]
			for subgraph in self.lego_subgraphs:
				if len(list(subgraph.g_undirected.nodes)) > 1:
					utils.write_to_file(subgraph, '{}_{}.ldr'.format(filename, subgraph_counter))
					subgraph_counter += 1



	def initialize_conversion(self):
		# Initialize conversion to LDraw
		self.LDraw_conversion.initialize_conversion(self)



	def set_class(self, class_num):
		self.class_num = class_num
		


	def set_brick_list(self, brick_list):
		for ix, brick in enumerate(brick_list):
			self.lego_assembly.add_brick_to_assembly(brick, ix)



#-------------------------------------------------------------------------------------------



class GeneratedLegoGraph(LegoGraph):
	"""Summary: Class for a generated LEGO graph. Methods for adding generated nodes, edges.
	"""

	def __init__(self, *args):
		super().__init__(*args)
		#For determining which class from the dataset this graph is supposed to be (for generation)
		self.class_num = 0



	def is_valid(self):
		if self.valid_graph == False or self.number_of_edges() == 0:
			return False
		return True



	def add_generated_node(self, type):
		"""Summary: Adds a node of the given type to the graph and updates networkx representations.
		See LegoGraphToActionSequence class in pyFiles/utils.py to see/change the type of each node.
		
		Args:
		    type (int): The node type to be added. Used to index a dictionary in a utils.LegoGraphToActionSequence
		    class.
		"""
		assert 0 < type <= 2, 'Node type to add must be 0 < type <= 2 (for Brick(4,2), Brick(2, 4)), got {}'.format(type)
		
		self.add_nodes(1)
		self.node_labels[len(self.nodes) - 1] = self.graph_to_action_sequence.node_action_to_type[type] # Convert int to node label
		
		# Update networkx representations
		self.g_directed.add_node(self.number_of_nodes() - 1)
		self.g_undirected.add_node(self.number_of_nodes() - 1)

		if len(self.nodes) == 1:
			self.LDraw_conversion.initialize_conversion(self) # Initialize conversion to LDraw



	def add_generated_edge(self, src, dest, x_shift, z_shift):
		"""Summary: Adds a generated edge between the given nodes with the given type. Updates LDraw and
		networkx representations, as well as the validity of the graph.
		
		Args:
		    src (int): The node index for the source of the edge.
		    dest (int): The node index for the destination of the edge.
		    x_shift (int): The x_shift of the edge.
		    z_shift (int): The z_shift of the edge.
		"""
		assert abs(x_shift) <= 3 and abs(z_shift) <= 3, 'Shift out of range, got ({} and {})'.format(x_shift, z_shift)
		self.add_edges([src], [dest])
		self.edge_labels[src, dest] = '({}, {})'.format(x_shift, z_shift) # Set edge label
		
		# Update networkx representations
		self.g_undirected.add_edges_from([(src, dest), (dest, src)])
		self.g_directed.add_edge(src, dest)

		self.graph_validation.check_if_shift_invalid(self, src, dest)
		self.convert_to_LDraw(src, dest) # Update LDraw representation



	def convert_to_LDraw_and_verify(self, filename, start_node = 0):
		"""Summary: Convert the graph to LDraw and verify it is a proper LEGO build. Saves it to the
		given file.

		Args:
		    filename (str): The filename to save to. The '.ldr' ending is not added automatically.
		    If filename == None then validity is checked and returned but the file is not saved.
		    start_node (int, optional): The node to start the conversion with (i.e the node/brick that is
		    placed at the origin)
		
		Returns:
		    bool: Validity of the LEGO graph
		"""
		self.update_nx_graphs()
		return self.LDraw_conversion.convert_to_LDraw_and_verify(self, filename, start_node = start_node)



#-------------------------------------------------------------------------------------------



class GeneratedLegoGraphSubgraph(GeneratedLegoGraph):

	"""Summary: Used for keeping track of subgraphs in a LEGO graph.
	"""

	def __init__(self, subgraph, parent_lego_graph):
		super().__init__()
		self.g_undirected = subgraph
		self.g_directed = subgraph.to_directed()
		self.parent_lego_graph = parent_lego_graph

		self.__add_data_from_parent(subgraph, parent_lego_graph) # Copy nodes & edges & labels
		
		self.LDraw_conversion.initialize_conversion(self, start_node = list(self.g_directed.nodes)[0])



	def __add_data_from_parent(self, subgraph, parent):
		self.__add_node_labels_from_parent(subgraph, parent)
		
		self.__add_edge_labels_from_parent(subgraph, parent)
		


	def __add_node_labels_from_parent(self, subgraph, parent):
		for node in subgraph.nodes:
			self.node_labels[node] = parent.node_labels[node]
			self.add_nodes(1)



	def __add_edge_labels_from_parent(self, subgraph, parent):
		for edge in subgraph.edges:
			try:
				self.edge_labels[edge[0], edge[1]] = parent.edge_labels[edge[0], edge[1]]
				self.g_directed.remove_edge(edge[1], edge[0])
			except:
				self.edge_labels[edge[1], edge[0]] = parent.edge_labels[edge[1], edge[0]]
				self.g_directed.remove_edge(edge[0], edge[1])



#-------------------------------------------------------------------------------------------



class LegoGraphToLDraw():
	rebrick_database = utils.RebrickableDatabase()
	graph_validation = utils.LegoGraphValidation()
	def __init__(self):
		self.pending = False



	def convert_to_LDraw_and_verify(self, g, fileName, start_node = 0):
		"""
		Summary: Converts the graph to an LDraw file, and creates and 
		saves it to a new file with the given file name if it is valid (ie no
		overconstrained bricks, no merged bricks).
		Params: 
			fileName (str): The name of the file to be converted to LegoGraph. If it is None then
			no file is saved.
		Returns:
			bool: True if if the graph is a valid lego build, False otherwise. If False then no file is saved.
		"""
		
		if self.graph_validation.check_if_graph_has_invalid_shift(g): # Check for invalid shifts
			g.invalid_shift = True
			g.valid_graph = False
			return False

		self.convert_to_LDraw(g, start_node = start_node) # Check for overconstrained bricks
		if g.overconstrained_brick:
			g.valid_graph = False
			return False

		self.graph_validation.check_if_bricks_merged(g) # Check for merged bricks
		if g.merged_brick:
			g.valid_graph = False

		if g.valid_graph and fileName is not None: # Write graph to LDraw file
			g.write_to_file(fileName)

		return g.valid_graph



	def convert_to_LDraw(self, g, *args, start_node = 0):
		"""Summary: Converts to LDraw without saving a file. Checks for overconstrained bricks,
		and sets g.overconstrained_brick and g.valid_graph as required. Also optionally takes 2 node
		indices. One of these should already be present in a pre-existing LDraw representation.
		If possible, this function will simply add the new node to the LDraw representation
		without doing a full conversion (and iterating through all nodes in the graph).
		
		Args:
		    g (LegoGraph): The LegoGraph to convert
		    *args: len(args) must be 0 or 2. If 0, then a full conversion to LDraw is done (i.e iterate through
		    all nodes and convert to LDraw), if 2 then this function attempts to only add the new node to the
		    pre-existing LDraw representation.
		    start_node (int, optional): The node/brick that the conversion should be started from (i.e the brick
		    that is placed at the origin).
		
		Raises:
		    Exception: If the code breaks
		"""
		assert len(args) == 0 or len(args) == 2, 'num optional args passed to convert_to_ldraw must be 0 or 2'

		if len(args) == 0 and not g.is_disjoint(): # If not disjoint
			self.__convert_to_LDraw_connected(g, start_node = start_node)
		
		elif g.is_disjoint():
			# If the graph is disjoint, we need to do a full conversion to LDraw once it is no longer
			# disjoint or the code will break. Self.pending keeps track of this.
			self.pending = True
			self.__convert_to_LDraw_disjoint(g)

		elif len(args) == 2:
			src = args[0]; dest = args[1]
			already_added_src = src in self.visited_list # If src is already in the LDraw rep.
			already_added_dest = dest in self.visited_list # If dest is already in the LDraw rep.

			if already_added_src and not self.pending:
				cur_brick = dest
				prev_brick = src
				self.add_brick_to_LDraw(g, cur_brick, prev_brick) # Add single brick to LDraw rep.

			elif already_added_dest and not self.pending:
				cur_brick = src
				prev_brick = dest
				self.add_brick_to_LDraw(g, cur_brick, prev_brick) # Add single brick to LDraw rep.

			elif not already_added_dest and not already_added_src or self.pending: 
				self.pending = False
				self.__convert_to_LDraw_connected(g) # Reconvert entire graph to LDraw

			else:
				raise Exception('Not supposed to be able to get here')



	def __convert_to_LDraw_connected(self, g, start_node = 0):
		assert start_node in list(g.g_undirected.nodes), 'start_node must already be added to the graph'
		self.initialize_conversion(g, start_node = start_node)

		visited_list = {}
		cur_node = [start_node]
		while len(cur_node) > 0:
			temp = cur_node.copy()
			for node in temp:
				visited_list[node] = 1
				for neighbor in g.g_undirected.neighbors(node): #g.neighbors returns a list of ints representing the nodes neighbors
					#Function recursively adds all nodes to the .ldr, throws an exception when a lego brick is over constrained
					self.add_brick_to_LDraw(g, neighbor, node)
					if neighbor not in visited_list:
						cur_node.append(neighbor)
						visited_list[neighbor] = 1
				del cur_node[0]



	def __convert_to_LDraw_disjoint(self, g):
		self.initialize_conversion(g)
		subgraphs = (g.g_undirected.subgraph(c) for c in nx.connected_components(g.g_undirected))
		try:
			del g.lego_subgraphs
		except:
			pass
		g.lego_subgraphs = []
		for i, subgraph in enumerate(subgraphs):
			g.lego_subgraphs.append(GeneratedLegoGraphSubgraph(subgraph, g))
			start_node = list(g.lego_subgraphs[i].g_directed.nodes)[0]
			g.lego_subgraphs[i].convert_to_LDraw(start_node = start_node)

		for subgraph in g.lego_subgraphs:
			for key, val in subgraph.lego_assembly.items():
				g.lego_assembly[key] = val
			for key, val in subgraph.lego_assembly.bricks_indexed_by_height.items():
				g.lego_assembly.bricks_indexed_by_height[key] = val
				for brick in val:
					if brick['ix'] not in g.lego_assembly.bricks_indexed_by_height_list:
						g.lego_assembly.bricks_indexed_by_height_list.append(brick['ix'])



	def initialize_conversion(self, g, start_node = 0):
		# Initializes the conversion to LDraw
		try:
			#Initialize the recursion by writing the first brick to the origin.
			node_ix = start_node
			#Retrieve the size of the brick
			brick_size = g.node_labels[brick1]
		except:
			node_ix = list(g.g_undirected.nodes)[0]
			brick_size = g.node_labels[node_ix]
		#Get the required transformation matrix and part number for the given brick size
		transformation_matrix, brick_name = self.rebrick_database.get_transformation_and_brick_name(brick_size)
		
		#Add it to list of bricks in the graph
		del g.lego_assembly
		g.lego_assembly = Brick.LEGOAssembly()
		new_brick = Brick.LegoPiece(0, 0, 0, brick_name, transformation_matrix, colour=14)
		g.lego_assembly.add_brick_to_assembly(new_brick, node_ix)
		
		#To keep track of which nodes/bricks have already been added to the file
		self.visited_list = {}
		#Update visited_list - used to determine when to stop recursion
		self.visited_list[node_ix] = 1


		#Converting to .ldr is done recursively by iterating through each nodes neighbors, using the
		#edge level embedding to create the correct coordinates.
		#To verify that a brick isn't overconstrained, we check that a node has the same coordinates
		#regardless of what neighbor the coordinates are generated from.
		#This dictionary keeps track of the neighbors that have already been used to generate coordinates
		#for a given node.
		self.visited_list_neighbors = {}



	def add_brick_to_LDraw(self, g, cur_brick, prev_brick):
		if prev_brick == cur_brick or (cur_brick, prev_brick) in self.visited_list_neighbors:
			return

		#Move coordinates from top middle of previous brick to top middle of current brick
		cur_x, cur_y, cur_z = self.calculate_position_from_previous_brick(g, cur_brick, prev_brick)
		self.visited_list_neighbors[cur_brick, prev_brick] = 1
		
		#Verify that the coordinates generated for this brick through this node pair is the same as previous pairs
		if cur_brick in self.visited_list:
			self.__check_if_brick_is_overconstrained(cur_brick, g, cur_x, cur_y, cur_z)
		
		#Write to the .ldr file if we haven't visited this node yet
		else:
			self.__add_brick_to_file(cur_brick, g, cur_x, cur_y, cur_z)
		
		try:
			g.lego_assembly.add_connections(cur_brick, prev_brick) # Update the open connections for each brick.
		except utils.MergedBrick:
			pass
		#Update visited list
		self.visited_list[cur_brick] = 1



	def __check_if_brick_is_overconstrained(self, cur_brick, g, x, y, z):
		# If the new position that we calcualted for this brick is different from a 
		# previously calculated position then the brick is overconstrained.
		if g.lego_assembly[cur_brick].positions_are_different(x, y, z):
			g.overconstrained_brick = True
			g.valid_graph = False



	def __add_brick_to_file(self, cur_brick, g, cur_x, cur_y, cur_z):
		#Get brick size from node label
		brick_size = g.node_labels[cur_brick]
		#Get transformation matrix and part number from database
		transformation_matrix, brick_name = self.rebrick_database.get_transformation_and_brick_name(brick_size)
		#Add to list of bricks in the graph
		new_brick = Brick.LegoPiece(cur_x, cur_y, cur_z, brick_name, transformation_matrix)
		g.lego_assembly.add_brick_to_assembly(new_brick, cur_brick)



	def calculate_position_from_previous_brick(self, g, cur_brick, prev_brick):
		"""
		Summary: Moves the current coordinates from the top middle of the previous brick to the top middle of the current brick.
		Used when converting to .ldr to generate the position of each brick.
		Params:
			cur_brick (int): The node number of the current brick (whose coordinates need to be obtained)
			prev_brick (LegoPiece object): The LegoPiece object whose coordinates are being used to calculate the coordinates of the current brick
			g (networkx.DiGraph): The graph used to determine the connection of cur_brick and prev_brick (ie which is on top)
		returns: 
			cur_x, cur_y, cur_z (int): The coordinates of the current brick
		"""

		cur_x = g.lego_assembly[prev_brick].x
		cur_y = g.lego_assembly[prev_brick].y
		cur_z = g.lego_assembly[prev_brick].z

		#Obtain the size of each brick as a tuple (ie (4, 2))
		cur_brick_size = utils.get_brick_size_tuple(g, cur_brick)
		prev_brick_size = utils.get_brick_size_tuple(g, prev_brick)
		
		#If the current brick is connected on top of the previous brick
		if utils.first_is_on_top_of_second(g, first = cur_brick, second = prev_brick):
			#Retrieve the edge embedding
			edge_shifts = ast.literal_eval(g.edge_labels[prev_brick, cur_brick])
			
			#Change coordinates from middle of previous brick to middle of current brick
			cur_x += (edge_shifts[0] * LDR_UNITS_PER_STUD)
			cur_z += (edge_shifts[1] * LDR_UNITS_PER_STUD)

			#Shift height by size of the current brick
			cur_y -= (cur_brick_size[2] * LDR_UNITS_PER_PLATE)

		#Current brick is connected underneath the previous brick
		else:
			#Retrieve the edge embedding
			edge_shifts = ast.literal_eval(g.edge_labels[cur_brick, prev_brick])
			
			#Change coordinates from middle of previous brick to middle of current brick
			cur_x -= (edge_shifts[0] * LDR_UNITS_PER_STUD)
			cur_z -= (edge_shifts[1] * LDR_UNITS_PER_STUD)

			#Shift hiehgt by size of the previous brick
			cur_y += (prev_brick_size[2] * LDR_UNITS_PER_PLATE)
		
		return int(cur_x), int(cur_y), int(cur_z)



#-------------------------------------------------------------------------------------------



class OverconstrainedBrick(Exception):
	pass