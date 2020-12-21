import networkx as nx
import numpy as np
import importlib

from pyFiles import Brick, LegoGraph

importlib.reload(Brick)
importlib.reload(LegoGraph)


def LDraw_to_graph(fileName):
	"""
	Summary: Converts a file with the given file name to a lego graph representation.
	Params: 
		fileName (str): The name of the .ldr file to convert to lego graph representation
	Returns: 
		g (LegoGraph): The lego graph representation of the .ldr file

	Notes: .LDR (lego drawing) files are basically a text file with a bunch of lines, with each line
	representing something different. If the line begins with the number 1, then the line represents a lego brick.
	Several space-separated numbers follow the number 1, and these represent the x, y, z coordinates of the brick, the
	colour of the brick, and a transformation matrix for changing the orientation of the brick. The line ends with a
	file name/part number, which can be used to obtain information about the part from Rebrickable.
	"""
	
	#Break the file into an array where each element is a line in the file
	lines = get_file_lines(fileName)
	#Iterate through the file and create new brick objects for each brick in the file
	brick_list = make_brick_list(lines)
	# Make the lego graph
	g = make_graph(brick_list)
	g.update_nx_graphs() #Updates networkx version of the graph - useful to carry around

	return g



def get_file_lines(fileName):
	"""Summary: Breaks the given file into a list where each element is a line in the file
	
	Args:
		fileName (str): The name of the file
	
	Returns:
		lines (list): A list where each element is a line in the given file
	"""
	with open(fileName) as file:
		#Split the file by new line characters
		lines = [line.rstrip('\n').split() for line in file]
   
	return lines



def make_brick_list(lines):
	"""
	Summary: Iterates through a list representation of the file, and creates a brick object
	for each brick in the file.
	Params: 
		lines (list): A list where each element is a string containing a line of a .ldr file.
	Returns:
		brick_list (list of Brick): A list of LegoPiece objects representing each brick in a .ldr file generated
		from the lines parameter.
	"""
	brick_list = []
	for line in lines:
		#If the line doesn't represent a brick
		if line[0] != '1':
			continue

		#Extract the x, y, and z coordinates of the brick
		x = line[2]
		y = line[3]
		z = line[4]
		#Extract the transformation matrix (to determine the orientation of the brick)
		transformation_matrix = ' '.join(line[5:14])
		#The part number/filename (to determine the size of the brick)
		brick_name = line[14].split('.')[0]

		#Create a new brick object and append it to the list
		brick = Brick.LegoPiece(x, y, z, brick_name, transformation_matrix)
		brick_list.append(brick)

	return brick_list
		


def print_brick_list(fileName):
	lines = get_file_lines(fileName)
	brick_list = make_brick_list(lines)
	
	for brick in brick_list:
		print(brick.get_brick_info())



def make_graph(brick_list):
	"""
	Summary: Iterates through the list created by make_brick_list, adds each brick as a node to a graph,
	and creates connections between each brick if necessary.
	Params: 
		brick_list (list of LegoPiece): A list of LegoPiece objects representing each brick in a .ldr file.
	Returns: 
		g (LegoGraph): The lego graph representation of the .ldr file
	"""

	g = LegoGraph.LegoGraph()
	g.set_brick_list(brick_list)
	#Add required amount of vertices/nodes to graph
	g.add_nodes(len(brick_list))

	#Iterate through each brick in the file
	for i, brick1 in enumerate(brick_list):
		g.node_labels[i] = brick1.get_node_label()

		#For each brick in the file, check if it can connect to every other brick in the file
		for j, brick2 in enumerate(brick_list):
			if brick1 == brick2: #Skip because a brick can't connect to itself
				continue

			#Only true if brick1 is above brick2, ie directed edge from brick2 to brick1
			if brick1.can_connect_on_top_of(brick2):
				add_edge(g, j, i, brick_list)

	return g



def add_edge(g, src, dest, bricks):
	#Add the edge to the networkx graph
	g.add_edges([src], [dest])
	#Get the edge embedding (ie a string like '((2, 2), (2, 1))' representing which studs are connected)
	connection_points = bricks[dest].get_connection_points(bricks[src])

	#Edge labels dictionary for lego graph
	g.edge_labels[src, dest] = str(connection_points)



def verify_LDraw_from_file_lines(file_lines):
	brick_list = make_brick_list(file_lines)
	return verify_LDraw(brick_list)



def verify_LDraw_from_graph(g):
	"""
	Summary: Checks that an .ldr file doesn't have bricks merged together.
	Params: The name of the file to check
	Returns: True if the .ldr is valid, False otherwise
	"""
	brick_list = g.lego_assembly.bricks.values()
	return verify_LDraw(brick_list)



def verify_LDraw(brick_list):
	#For each brick, check that all other bricks are not merged with it
	for i, brick1 in enumerate(brick_list):
		for j, brick2 in enumerate(brick_list):
			if i == j: #Don't check if a brick is merged with itself
				continue

			if brick1.bricks_merged_together(brick2):
				return False

	return True