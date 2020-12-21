import os
import numpy as np


try:
	clone_dir = os.path.join(os.getcwd(), 'bo_temp')
	os.mkdir(clone_dir)
	os.system('pip install geometric_primitives')
	os.system('git clone https://github.com/POSTECH-CVLab/Combinatorial-3D-Shape-Generation {}'.format(clone_dir))
except:
	pass

res_dir = os.path.join(os.getcwd(), 'ldr_files')
try:
	os.mkdir(res_dir)
except:
	pass
res_dir = os.path.join(res_dir, 'dataset')
try:
	os.mkdir(res_dir)
except:
	pass

dataset_dir = os.path.join(clone_dir, 'dataset')


classes = {}
classes['label00'] = '2blocks'
classes['label01'] = '2blocks-perpendicular'
classes['label11'] = 'tower'
classes['label12'] = 'line'
classes['label13'] = 'flat-block'
classes['label14'] = 'wall'
classes['label15'] = 'tall-block'
classes['label16'] = 'pyramid'
classes['label21'] = 'chair'
classes['label22'] = 'couch'
classes['label23'] = 'cup'
classes['label24'] = 'hollow-cylinder'
classes['label25'] = 'table'
classes['label26'] = 'car'
classes['random'] = 'random'

def get_class_name(filename):
	return classes[filename.split('_')[0]]

LDR_UNITS_PER_STUD = 20
LDR_UNITS_PER_PLATE = 8
PLATES_PER_BRICK = 3
def npy2ldr(in_filename, out_filename):
	bricks = np.load(in_filename, allow_pickle = True)
	with open(out_filename, 'w') as file:
		for brick in bricks[()].bricks:
			if brick.get_direction() == 1:
				transformation_string = "1 0 0 0 1 0 0 0 1"
			else:
				transformation_string = "0 0 -1 0 1 0 1 0 0"
			coords = brick.get_position()
			x_coord = coords[0] * LDR_UNITS_PER_STUD
			y_coord = -coords[2] * LDR_UNITS_PER_PLATE * PLATES_PER_BRICK
			z_coord = coords[1] * LDR_UNITS_PER_STUD
			file.write('1 4 {} {} {} {} 3001.dat\n'.format(x_coord, y_coord, z_coord, transformation_string))



rotate_90 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
def npy2ldr_augmented_90(in_filename, out_filename):
	bricks = np.load(in_filename, allow_pickle = True)
	with open(out_filename, 'w') as file:
		for brick in bricks[()].bricks:
			if brick.get_direction() == 1:
				transformation_string = "0 0 -1 0 1 0 1 0 0"
			else:
				transformation_string = "1 0 0 0 1 0 0 0 1"
			coords = brick.get_position().dot(rotate_90)
			x_coord = coords[0] * LDR_UNITS_PER_STUD
			y_coord = -coords[2] * LDR_UNITS_PER_PLATE * PLATES_PER_BRICK
			z_coord = coords[1] * LDR_UNITS_PER_STUD
			file.write('1 4 {} {} {} {} 3001.dat\n'.format(x_coord, y_coord, z_coord, transformation_string))



rotate_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
def npy2ldr_augmented_180(in_filename, out_filename):
	bricks = np.load(in_filename, allow_pickle = True)
	with open(out_filename, 'w') as file:
		for brick in bricks[()].bricks:
			if brick.get_direction() == 1:
				transformation_string = "1 0 0 0 1 0 0 0 1"
			else:
				transformation_string = "0 0 -1 0 1 0 1 0 0"
			coords = brick.get_position().dot(rotate_180)
			x_coord = coords[0] * LDR_UNITS_PER_STUD
			y_coord = -coords[2] * LDR_UNITS_PER_PLATE * PLATES_PER_BRICK
			z_coord = coords[1] * LDR_UNITS_PER_STUD
			file.write('1 4 {} {} {} {} 3001.dat\n'.format(x_coord, y_coord, z_coord, transformation_string))



rotate_270 = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])
def npy2ldr_augmented_270(in_filename, out_filename):
	bricks = np.load(in_filename, allow_pickle = True)
	with open(out_filename, 'w') as file:
		for brick in bricks[()].bricks:
			if brick.get_direction() == 1:
				transformation_string = "0 0 -1 0 1 0 1 0 0"
			else:
				transformation_string = "1 0 0 0 1 0 0 0 1"
			coords = brick.get_position().dot(rotate_270)
			x_coord = coords[0] * LDR_UNITS_PER_STUD
			y_coord = -coords[2] * LDR_UNITS_PER_PLATE * PLATES_PER_BRICK
			z_coord = coords[1] * LDR_UNITS_PER_STUD
			file.write('1 4 {} {} {} {} 3001.dat\n'.format(x_coord, y_coord, z_coord, transformation_string))



for file in os.listdir(dataset_dir):
	filename = os.fsdecode(file)
	if filename.endswith(".npy"):
		if 'label02' in filename:
			continue
		class_name = get_class_name(filename)
		sample_num = filename.split('_')[1].split('.')[0]
		
		out_file = os.path.join(res_dir, class_name + '_augmented90_' + sample_num + '.ldr')
		npy2ldr_augmented_90(os.path.join(dataset_dir, filename), out_file)
		
		out_file = os.path.join(res_dir, class_name + '_augmented180_' + sample_num + '.ldr')
		npy2ldr_augmented_180(os.path.join(dataset_dir, filename), out_file)
		
		out_file = os.path.join(res_dir, class_name + '_augmented270_' + sample_num + '.ldr')
		npy2ldr_augmented_270(os.path.join(dataset_dir, filename), out_file)
		
		out_file = os.path.join(res_dir, class_name + '_' + sample_num + '.ldr')
		npy2ldr(os.path.join(dataset_dir, filename), out_file)
		
	else:
		continue

os.system('rm {}/bo_temp/ -r -f'.format(os.getcwd()))
os.system('pip uninstall geometric_primitives -y')