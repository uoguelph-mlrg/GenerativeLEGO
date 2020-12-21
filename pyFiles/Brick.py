import numpy as np
import json
import importlib
import rebrick
from pyFiles import utils
from pyFiles import LegoGraph

importlib.reload(utils)
importlib.reload(LegoGraph)

LDR_UNITS_PER_STUD = 20 #width/length of stud in LDR units
LDR_UNITS_PER_PLATE = 8 #height of a lego plate

#For rebrickable api
API_KEY = "a352b3831dfad1f813bc645e5e480569"
rebrick.init(API_KEY)

global DEBUG
DEBUG = utils.DEBUG


class Rebrickable():
	def get_dimensions(self, brick, part_num):
		"""
		Function: get_dimensions
		Summary: Retrieves the dimensions of a lego piece given a Lego part number.
		Parameters:
			part_num (str): A string containing the Lego part number
		Returns:
			dims: (tuple of int): A tuple containing the dimensions of the Lego part
			in # of studs (ie (4, 2))"""
		try:
			dims = self._get_dimensions_from_db(brick, part_num)
		except:
			dims = self._get_dimensions_from_rebrick(brick, part_num)
		return dims
		


	def _get_dimensions_from_db(self, brick, part_num):
		#Try and load the brick information from the database (this line throws an exception if not already present)
		size = tuple(brick.rebrick_database.brick_sizes[brick.transformation_string + ' ' + part_num]) #Returns a tuple of (width, length) in #studs x #studs
		
		return size



	def _get_dimensions_from_rebrick(self, brick, part_num):
		#Rebrickable API to get brick info
		response = rebrick.lego.get_part(part_num)
		#Get string of the form 'Brick width x length' or 'Plate w x l', dimensions are #studs
		brick_description = json.loads(response.read())['name'] 

		if self.brick_type != 'Slope':
			#Strip non-numerical characters, ie the 'Brick' or 'Plate'
			size = ''.join(c for c in brick_description if c.isdigit())
			
		else:
			#Need to extract first two words for sloped bricks (can be 'Slope' or 'Slope Inverted')
			temp = brick_description.split('x')
			size = (temp[0][-2], temp[1][1])
 
		#Apply transformation matrix
		size = brick.transformation_matrix.dot(np.vstack((int(size[0]), brick.height, int(size[1]))))
		#Create tuple (width, length) in #studs x #studs
		brick.size = (abs(int(size[2][0])), abs(int(size[0][0])))
			
		#Add to database for next time
		brick.rebrick_database.update_brick_sizes(brick.transformation_string + ' ' + part_num, brick.size)
		brick.rebrick_database.update_brick_sizes(brick.brick_type + str(brick.size), brick.transformation_string + ' ' + part_num)
			
		return brick.size



class LegoPiece():
	"""
	Class: LegoPiece
	Summary: Implements many of the methods for helping to convert from LDraw to graph representation.
	The most important being determining if two bricks can connect, and which studs can be used to connect
	two bricks together."""
	rebrick_database = utils.RebrickableDatabase()
	rebrick = Rebrickable()



	class LegoStuds():

		"""Summary: Representation of the LEGO studs on a LEGO brick. Used for determining
		if/where connections are available on a brick. The useful parts of this class will probably
		break if we use different sizes of bricks (i.e a 4x2 and 2x2)
		
		Attributes:
			length (int): Length of the LEGO brick
			occupied_studs (np.array): An array of length x width indicating if each stud is occupied.
			A 1 indicates it is occupied, 0 indicates it is free. 
			width (int): Width of the LEGO brick
			buffer (np.array): I represent bricks as a np array with some buffer space. I.e a 1x1 brick might be represented as:
				# [[0, 0, 0],
				#  [0, 1, 0],
				#  [0, 0, 0]]. I use builtin numpy functions to apply shifts to the buffer/brick/array and determine 
				# if two bricks can connect after applying these shifts.
		"""
	
		def __init__(self, width, length):
			self.occupied_studs = np.zeros((length, width))
			self.width = width
			self.length = length
			self.buffer = self.__make_buffer(width, length)



		def has_open_connections_for(self, size):
			"""Summary: Returns if these LEGO studs have space to connect to a brick of the given size.
			Note that this method does not check if any of these connections are valid in the context of
			the entire LEGO assembly. It is possible for connections to be available on this brick, but
			adding a brick to fill these connections results in it occupying the same space as an adjacent
			brick in the assembly.
			
			Args:
				size (tuple of ints): Returns if a brick of this size can connect to these LEGO studs.
			
			Returns:
				bool: If there is space available for the given brick size on these studs.
			"""
			return len(self.get_open_connections_for(size)) > 0



		def get_open_connections_for(self, size):
			"""Summary: Returns the shifts that are available for a brick of a given size to connect to
			these LEGO studs.
			Note that this method does not check if any of these connections are valid in the context of
			the entire LEGO assembly. It is possible for connections to be available on this brick, but
			adding a brick to fill these connections results in it occupying the same space as an adjacent
			brick in the assembly.
			
			Args:
				size (tuple of ints): Returns possible connections for a brick of this size to these studsW.
			
			Returns:
				list of lists: A list of lists where each sublist is a possible connection as [x shift, z_shift]
			"""
			if (self.occupied_studs == 1).all(): # No connections available if all studs are full
				return []

			new_brick_width = size[0]; new_brick_length = size[1]

			# Buffer for the new brick
			new_brick_buffer = self.__make_buffer(new_brick_width, new_brick_length)

			possible_conns = []
			for x_shift in range(new_brick_width + self.width): # Iterate over all possible x shifts
				x_shift -= int((new_brick_width + self.width) / 2)
				for z_shift in range(new_brick_length + self.length): # Iterate over all possible z shifts
					z_shift -= int((new_brick_length + self.length) / 2)

					# Apply the x shift, z shift combo to the new brick we want to connect
					new_brick_shifted = self.__shift_buffer(new_brick_buffer, x_shift, -z_shift)
					if self.__bricks_can_connect(new_brick_shifted):
						possible_conns.append([x_shift, z_shift])

			return possible_conns



		def __bricks_can_connect(self, new_brick_shifted):
			"""Summary: Determines if the new brick with a shift applied can connect to the LEGO studs.
			
			Args:
				new_brick_shifted (np.array): A buffer/array representing the space the brick occupies 
			
			Returns:
				bool: Whether the brick can connect to these studs.
			"""
			# Create a buffer representing the already occupied studs
			connection_buffer = self.__make_buffer(self.width, self.length, fill = self.occupied_studs)

			# Apply the xor (odd) function - returns 1 where new_brick_shifted != connection_buffer
			xor_res = np.logical_xor(new_brick_shifted, connection_buffer)
			# Apply the or function - returns 1 where new_brick_shifted == 1 or connection_buffer == 1
			or_res = np.logical_or(new_brick_shifted, connection_buffer)

			if (xor_res ==  or_res).all(): # They're equal when the new brick doesn't have any overlaps with already occupied studs
				# No connection is possible when the bricks themselves have no overlap after the shift is applied
				if (np.logical_and(new_brick_shifted, self.buffer) == 0).all():
					return False
				else:
					return True

			return False



		def add_connections(self, new_brick_buffer, x_shift, z_shift):
			"""Summary: Updates the current studs to show that it has been connected to the new brick.
			
			Args:
				new_brick_buffer (np.array): Buffer representing the new brick to connect.
				x_shift (int): The x shift to apply to the new brick
				z_shift (int): The z shift to apply to the new brick
			"""
			new_brick_shifted = self.__shift_buffer(new_brick_buffer, x_shift, -z_shift)
			self.__update_connections(new_brick_shifted)



		def __make_buffer(self, width, length, fill = None):
			dims = max(width, length) * 2
			buff = np.zeros((dims, dims))

			start_x = int((buff.shape[0] - width) / 2)
			start_z = int((buff.shape[1] - length) / 2)

			if fill is None:
				buff[start_z: start_z + length, start_x: start_x + width] = 1
			else:
				buff[start_z: start_z + length, start_x: start_x + width] = fill

			return buff



		def __shift_buffer(self, buff, x_shift, z_shift):
			temp = np.roll(buff, x_shift, axis = 1)
			return np.roll(temp, z_shift, axis = 0)



		def __update_connections(self, new_brick_shifted):
			start_x = int((self.buffer.shape[0] - self.width) / 2)
			start_z = int((self.buffer.shape[1] - self.length) / 2)

			new_connections = np.logical_and(new_brick_shifted, self.buffer).astype(int)
			new_connections = new_connections[start_z: start_z + self.length, start_x: start_x + self.width]
			
			self.occupied_studs = np.logical_or(new_connections, self.occupied_studs).astype(int)



	def __init__(self, x, y, z, part_num, transformation, colour=None):
		"""
		Initializer for LegoPiece class
		Summary: Sets the x, y, and z coordinates of the brick. This represents the top center of the brick.
		Determine the size and type of the brick, and the coordinates which are occupied by this brick.
		"""
		self.colour = np.random.randint(0, 10) if colour is None else colour
		self.brick_type = 'Brick'
		self.height = 3 # LEGO bricks are 3 LEGO plates tall
		self._set_xyz(x, y, z)
		
		#Transformation is string of numbers; convert to proper matrix form
		self._get_transformation_matrix(transformation)
		self.transformation_string = transformation

		self.LDraw_representation = '1 {} {} {} {} {} {}.dat'.format(self.colour, x, y, z, transformation, part_num)
		
		#Determine size and orientation of the brick
		self.size = self.rebrick.get_dimensions(self, part_num)
		self.width = self.size[0]
		self.length = self.size[1]
		
		#Get coordinate ranges for space this brick occupies
		self.bottom_coord_range = self._get_bottom_coordinate_range()
		self.top_coord_range = self._get_top_coordinate_range()

		self.bottom_studs = self.LegoStuds(self._get_bottom_width(), self._get_bottom_length())
		self.top_studs = self.LegoStuds(self._get_top_width(), self._get_top_length())



	def has_bottom_open_connections_for(self, size):
		# See LEGOStuds.has_open_connections_for for documentation
		assert size == (4, 2) or size == (2, 4), 'Got size {}, only (4, 2) and (2, 4) supported'.format(size)

		return self.bottom_studs.has_open_connections_for(size)



	def has_top_open_connections_for(self, size):
		# See LEGOStuds.has_open_connections_for for documentation
		assert size == (4, 2) or size == (2, 4), 'Got size {}, only (4, 2) and (2, 4) supported'.format(size)

		return self.top_studs.has_open_connections_for(size)



	def get_bottom_open_connections_for(self, size):
		# See LEGOStuds.get_open_connections_for for documentation
		assert size == (4, 2) or size == (2, 4), 'Got size {}, only (4, 2) and (2, 4) supported'.format(size)

		temp = self.bottom_studs.get_open_connections_for(size)
		for i, shifts in enumerate(temp):
			for j, shift in enumerate(shifts):
				temp[i][j] = -shift
		return temp



	def get_top_open_connections_for(self, size):
		# See LEGOStuds.get_open_connections_for for documentation
		assert size == (4, 2) or size == (2, 4), 'Got size {}, only (4, 2) and (2, 4) supported'.format(size)

		return self.top_studs.get_open_connections_for(size)



	def add_connections(self, brick):
		"""Summary: Fills the studs occupied after a connection between these two bricks has been formed.
		Fills the studs for calling object and the passed object (i.e no need to call it twice for the
		same connection).
		
		Args:
			brick (LegoPiece): The brick that we are connecting to this one
		"""
		if self.can_connect_on_top_of(brick):
			# Get shifts
			x_shift, z_shift = self.get_connection_points(brick)
			brick_buffer = brick.top_studs.buffer
			# Add connections to this brick
			self.bottom_studs.add_connections(brick_buffer, -x_shift, -z_shift)
			# Add connections to passed brick
			brick.top_studs.add_connections(self.bottom_studs.buffer, x_shift, z_shift)

		elif brick.can_connect_on_top_of(self):
			x_shift, z_shift = brick.get_connection_points(self)
			brick_buffer = brick.bottom_studs.buffer
			self.top_studs.add_connections(brick_buffer, x_shift, z_shift)
			brick.bottom_studs.add_connections(self.top_studs.buffer, -x_shift, -z_shift)



	def get_bottom_connections(self):
		return self.bottom_studs.occupied_studs



	def get_top_connections(self):
		return self.top_studs.occupied_studs



	def positions_are_different(self, x, y, z):
		return self.x != x or self.y != y or self.z != z



	def can_connect_on_top_of(self, brick):
		"""
		Function: can_connect_on_top_of
		Summary: Determines if the calling brick can connect on top of the given brick.
		Params:
			brick (LegoPiece): Checks if a connection is possible with this brick.
		Returns:
			True if this brick can connect on top of the given brick, False otherwise.
		Raises:
			utils.MergedBrick if the two bricks occupy the same space
		"""
		if brick == self: # Brick can't connect to itself
			return False
		
		#The height required for a brick to connect on top of passed brick
		brick_connection_height = brick._get_top_connection_height()
		#The height required for a brick to connect on the bottom of this brick
		self_connection_height = self._get_bottom_connection_height()
		
		#Determine if the x and z coordinates of the bricks intersect
		x_intersect = self.__x_coords_intersect(self.bottom_coord_range, brick.top_coord_range)
		z_intersect = self.__z_coords_intersect(self.bottom_coord_range, brick.top_coord_range)
		
		#If the heights match and the x and z coordinates intersect (ie the bricks can connect)
		if brick_connection_height == self_connection_height and x_intersect and z_intersect:
			return True
		
		#If the bricks are merged together
		elif self.bricks_merged_together(brick):
			raise utils.MergedBrick('Bricks merged together')
		
		#Bricks can't connect
		return False
	


	def x_coords_intersect_test(self, brick):
		return self.__x_coords_intersect(self.bottom_coord_range, brick.top_coord_range)



	def z_coords_intersect_test(self, brick):
		return self.__z_coords_intersect(self.bottom_coord_range, brick.top_coord_range)



	def __x_coords_intersect(self, self_bottom_range, brick_top_range):
		#Returns True if the given ranges of coordinates intersect
		if (self_bottom_range['lowest_coord']['x'] >= brick_top_range['highest_coord']['x'] or
			self_bottom_range['highest_coord']['x'] <= brick_top_range['lowest_coord']['x']):
				return False

		return True
	


	def __z_coords_intersect(self, self_bottom_range, brick_top_range):
		#Returns True if the given ranges of coordinates intersect
		if (self_bottom_range['lowest_coord']['z'] >= brick_top_range['highest_coord']['z'] or
			self_bottom_range['highest_coord']['z'] <= brick_top_range['lowest_coord']['z']):
				return False

		return True
	


	def bricks_merged_together(self, brick):
		"""Summary: Determines if the calling object and the passed brick
		are occupying the same space.
		
		Args:
			brick (LegoPiece): Check if this brick is merged with the calling object
		
		Returns:
			bool: Whether the two bricks occupy the same space. Don't need the entire
			bricks to be overlapping, will return True if only a portion of the bricks
			occupy the same space.
		"""
		if brick == self: # Brick can't be merged with itself
			return False

		brick_top_of_brick = -brick._get_top_connection_height()
		self_bottom_of_brick = -self._get_bottom_connection_height()
		brick_bottom_of_brick = -brick._get_bottom_connection_height()
		self_top_of_brick = -self._get_top_connection_height()

		
		# Determine if the x and z coordinates overlap
		x_intersect = self.__x_coords_intersect(self.bottom_coord_range, brick.top_coord_range)
		z_intersect = self.__z_coords_intersect(self.bottom_coord_range, brick.top_coord_range)

		if not x_intersect or not z_intersect: # Can't be merged if x and z don't intersect
			return False

		if ((self_bottom_of_brick >= brick_top_of_brick) 
			  or (self_top_of_brick <= brick_bottom_of_brick)): # If the x and z intersect but they are at different heights
			return False

		return True    



	def get_connection_points(self, brick):
		"""
		Function: get_connection_points
		Summary: Generates the edge level embedding (or x shift, zshift) for two connected nodes
		Parameters:
			brick (LegoPiece): A brick that can be connected on top of the current brick
		Returns: 
			tuple of ints: Tuple of (x_shift, z_shift)
		"""
		assert self.can_connect_on_top_of(brick), 'Calling object should be able to connect on top of passed object'
		x = (self.x - brick.x) / LDR_UNITS_PER_STUD
		z = (self.z - brick.z) / LDR_UNITS_PER_STUD

		return int(x), int(z)



# Getters/Setters --------------------------------------------------------------------------------------
	def _set_xyz(self, x, y, z):
		#In LDR units/coordinates
		self.x = int(float(x))
		self.y = int(float(y))
		self.z = int(float(z))



	def get_brick_info(self):
		#For debugging mostly
		#return 'size: {}, x: {}, y: {}, z: {}'.format(self.size, self.x, self.y, self.z)
		return 'type: {}, x: {}, y: {}, z: {}, size: {}, range: {}'.format(self.brick_type, self.x, self.y, self.z, self.size, self.bottom_coord_range)



	def get_connection_heights(self):
		return [self._get_bottom_connection_height(), self._get_top_connection_height()]    



	def _get_coordinate_range(self, width, length):
		"""
		Function: _get_coordinate_range
		Summary: Almost identical to _get_connection_range, except this function returns the entire
		coordinate range occupied by the brick, rather than where it can make connections. Used later to
		determine if two bricks are merged together. 
		Params:
			width (int): The width of the lego piece
			length (int): The length of the lego piece
		Returns:
			coord_range (dict of dict): The range of coordinates that a brick occupies.
		"""
		coord_range = {}
		coord_range['lowest_coord'] = {}
		coord_range['highest_coord'] = {}
		
		#Lowest x-coord for the brick
		coord_range['lowest_coord']['x'] = self.x - ((width / 2) * LDR_UNITS_PER_STUD)
		#Lowest z-coord for the brick
		coord_range['lowest_coord']['z'] = self.z - ((length / 2) * LDR_UNITS_PER_STUD)
		#Highest x-coord for the brick
		coord_range['highest_coord']['x'] = self.x + ((width / 2) * LDR_UNITS_PER_STUD)
		#Highest z-coord for the brick
		coord_range['highest_coord']['z'] = self.z + ((length / 2) * LDR_UNITS_PER_STUD)
		
		return coord_range
	


	def _get_bottom_coordinate_range(self):
		"""
		Function: _get_bottom_coordinate_range
		Summary: Determines the range of coordinates that the bottom of a brick occupies.
		"""
		
		width = self._get_bottom_width()
		length = self._get_bottom_length()
		
		return self._get_coordinate_range(width, length)
	


	def _get_top_coordinate_range(self):
		"""
		Function: _get_top_coordinate_range
		Summary: Determines the range of coordinates that the top of a brick occupies.
		"""
		return self._get_bottom_coordinate_range()
	


	def _get_bottom_width(self):
		#Retrieve the width of the bottom of the brick
		#Valid only for Brick and Plate subclasses - override for Slope subclass
		return self.width
	


	def _get_bottom_length(self):
		#Retrieve the length of the bottom of the brick
		#Valid only for Brick and Plate subclasses - override for Slope subclass
		return self.length
	


	def _get_top_width(self):
		#Retrieve the width of the top of the brick
		#Valid only for Brick and Plate subclasses - override for Slope subclass
		return self.width
	


	def _get_top_length(self):
		#Retrieve the length of the top of the brick
		#Valid only for Brick and Plate subclasses - override for Slope subclass
		return self.length
	


	def _get_bottom_connection_height(self):
		#Retrieve the height that a brick must be at to connect to the bottom of this brick
		return self.y + (self.height * LDR_UNITS_PER_PLATE)
	


	def _get_top_connection_height(self):
		#Retrieve the height that a brick must be at to connect to the top of this brick
		return self.y



	def get_node_label(self):
		#Label (as a string) of the form 'brick type(width, length)', ie 'Brick(2,4)'
		return self.brick_type + str(self.size)

	

	def _get_transformation_matrix(self, transformation):
		#Takes in the transformation as a string of ints and converts it to a proper matrix.
		#Also checks that it is a proper transformation (ie orthonormal, and I blocked transformations that flip
		#the brick upside down)
		a = [int(i) for i in transformation.split()[0:3]]
		b = [int(i) for i in transformation.split()[3:6]]
		c = [int(i) for i in transformation.split()[6:9]]
		self.transformation_matrix = np.vstack((a, b, c))
		if self._bad_transformation_matrix():
			raise Exception('Bad transformation matrix')
	


	def _bad_transformation_matrix(self):
		#Returns true if the matrix isn't orthonormal or rotates the brick outide of the x-z plane
		return (np.eye(3) != self.transformation_matrix.dot(self.transformation_matrix.T)).any() or (self.transformation_matrix[1] != [0, 1, 0]).any()
	


#------------------------------------------------------------------------------------------



class LEGOAssembly():
	database = utils.RebrickableDatabase()
	def __init__(self):
		self.bricks = {}

		# Dictionary of bricks where the key is the height (y coordinate) of bricks
		# Used to speed up certain algorithms where we need to find bricks at a certain height
		self.bricks_indexed_by_height = {} 
		self.bricks_indexed_by_height_list = []


	def __len__(self):
		return len(self.bricks)

	def __getitem__(self, node_ix):
		return self.bricks[node_ix]

	def __setitem__(self, node_ix, val):
		self.bricks[node_ix] = val

	def values(self):
		return self.bricks.values()

	def keys(self):
		return self.bricks.keys()

	def items(self):
		return self.bricks.items()



	def add_brick_to_assembly(self, brick, node_ix):
		self.bricks[node_ix] = brick

		for height in brick.get_connection_heights():
			height_present = height in self.bricks_indexed_by_height
			if height_present == True and {'data': brick, 'ix': node_ix} not in self.bricks_indexed_by_height[height]:
				self.bricks_indexed_by_height[height].append({'data': brick, 'ix': node_ix})
			elif height_present == True: 
				continue
			else:
				self.bricks_indexed_by_height[height] = [{'data': brick, 'ix': node_ix}]

		if node_ix not in self.bricks_indexed_by_height_list:
			self.bricks_indexed_by_height_list.append(node_ix)



	def add_connections(self, cur_brick, prev_brick):
		self.bricks[cur_brick].add_connections(self.bricks[prev_brick])



	def brick_is_in_valid_position(self, brick):
		# for old_brick in self.bricks.values():
		try:
			for old_brick in self.bricks_indexed_by_height[brick.y]: 
				if old_brick['data'].bricks_merged_together(brick):
				# if old_brick.bricks_merged_together(brick):
					return False
		except:
			pass
			
		return True



	def get_bricks_with_open_bottom_connections_for(self, size):
		bricks = []
		for node_ix, brick in self.bricks.items():
			connection_func = brick.get_bottom_open_connections_for
			if self.__brick_has_open_connections_for(size, brick, connection_func, new_brick_is_on_top = False):
				bricks.append(node_ix)

		return bricks



	def get_bricks_with_open_top_connections_for(self, size):
		bricks = []
		for node_ix, brick in self.bricks.items():
			connection_func = brick.get_top_open_connections_for
			if self.__brick_has_open_connections_for(size, brick, connection_func, new_brick_is_on_top = True):
				bricks.append(node_ix)

		return bricks



	def __brick_has_open_connections_for(self, size, brick, connection_func, new_brick_is_on_top):
		open_connections = connection_func(size)
		if len(open_connections) == 0:
			return False

		for connection in open_connections:
			test_brick = self.__make_brick_from_connection(connection, brick, size, new_brick_is_on_top)
			if self.brick_is_in_valid_position(test_brick):
				return True
	
		return False



	def get_open_top_connections_for(self, old_brick_ix, new_size):
		old_brick = self.bricks[old_brick_ix]
		connections = old_brick.get_top_open_connections_for(new_size)
		valid_connections = []
		for connection in connections:
			test_brick = self.__make_brick_from_connection(connection, old_brick, new_size, True)
			if self.brick_is_in_valid_position(test_brick):
				valid_connections.append(connection)

		return valid_connections



	def get_open_bottom_connections_for(self, old_brick_ix, new_size):
		old_brick = self.bricks[old_brick_ix]
		connections = old_brick.get_bottom_open_connections_for(new_size)
		valid_connections = []
		for connection in connections:
			test_brick = self.__make_brick_from_connection(connection, old_brick, new_size, False)
			if self.brick_is_in_valid_position(test_brick):
				valid_connections.append(connection)

		return valid_connections



	def __get_open_connections_for(self, size, bricks_dict, brick, node_ix, open_connection_func, new_brick_is_on_top):
		open_connections = open_connection_func(size)
		if len(open_connections) == 0:
			return

		for connection in open_connections:
			test_brick = self.__make_brick_from_connection(connection, brick, size, new_brick_is_on_top)
			if self.brick_is_in_valid_position(test_brick):
				if node_ix not in bricks_dict:
					bricks_dict[node_ix] = {}
					bricks_dict[node_ix]['brick'] = brick
					bricks_dict[node_ix]['open_connections'] = []
				bricks_dict[node_ix]['open_connections'].append(connection)



	def __make_brick_from_connection(self, connection, brick, size, new_brick_is_on_top):
		cur_x = brick.x; cur_y = brick.y; cur_z = brick.z
		brick_type = 'Brick{}'.format(size)
		transformation_matrix, brick_name = self.database.get_transformation_and_brick_name(brick_type)

		if new_brick_is_on_top:
			cur_x += (connection[0] * LDR_UNITS_PER_STUD)
			cur_z += (connection[1] * LDR_UNITS_PER_STUD)

			cur_y -= (3 * LDR_UNITS_PER_PLATE)

		else:
			cur_x -= (connection[0] * LDR_UNITS_PER_STUD)
			cur_z -= (connection[1] * LDR_UNITS_PER_STUD)

			cur_y += (3 * LDR_UNITS_PER_PLATE)

		return LegoPiece(cur_x, cur_y, cur_z, brick_name, transformation_matrix)