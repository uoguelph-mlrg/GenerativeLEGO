import torch
from torch.utils.data import Dataset
import numpy as np
from pyFiles import LDraw, LegoGraph, utils
from DGL_GIN.gin import GIN
import os
import ast
import random
import time
import pickle
import json



class FileToTarget():
    classes = {}
    def get_target(self, filename):
        class_name = filename.split('_')[0]
        if class_name not in self.classes:
            print(class_name)
            self.classes[class_name] = len(self.classes)

        return self.classes[class_name]

    def get_class_name(self, class_num):
        return list(self.classes.keys())[class_num]

    def num_classes(self):
        return len(self.classes)

    def get_all_classes(self):
        return list({k: v for k, v in sorted(self.classes.items(), key=lambda item: item[1])}.keys())

def include_in_dataset(filename, config = None):
    is_ldr = filename.endswith('.ldr')
    splits = filename.split('_')
    class_name = splits[0]
    augmented = splits[1]
    if is_ldr and (class_name != '2blocks-perpendicular' \
        and class_name != '2blocks' and class_name != 'random'):
        if 'augmented' not in augmented or config['include_augmented']:
            return True

    return False


class LegoDataset(Dataset):
    def __init__(self, config = None):
        super().__init__()
        self.dataset = []
        self.file_to_target = FileToTarget()
        try:
            dataset_with_filenames = self.load_dataset_from_file(config)
            print('loaded dataset from file')
        except FileNotFoundError:
            print('making dataset')
            dataset_with_filenames = self.make_dataset_from_LDraw(config)
            self.write_dataset_to_file(config, dataset_with_filenames)

        self.add_dataset(dataset_with_filenames)
        random.shuffle(self.dataset)

    def load_dataset_from_file(self, config):
        pickle_file = 'dataset/augmented-{}.dat'.format(config['include_augmented'])
        with open(pickle_file, 'rb') as f:
            dataset_with_filenames = pickle.load(f)

        return dataset_with_filenames


    def add_dataset(self, dataset_with_filenames):
        to_sequence = utils.LegoGraphToActionSequence()
        for graph_with_filename in dataset_with_filenames:
            filename = graph_with_filename[0]
            graph = graph_with_filename[1]
            self.add_graph_to_dataset(graph, filename, to_sequence)


    def make_dataset_from_LDraw(self, config):
        dataset_with_filenames = []
        to_sequence = utils.LegoGraphToActionSequence()
        directory = os.fsencode(os.path.join(os.getcwd() + '/ldr_files/dataset'))
        for file in sorted(os.listdir(directory)):
            filename = os.fsdecode(file)
            directory_str = os.fsdecode(directory) + '/'

            if include_in_dataset(filename, config = config):
                lg = LDraw.LDraw_to_graph(directory_str + filename)
                dataset_with_filenames.append([filename, lg])

        return dataset_with_filenames


    def write_dataset_to_file(self, config, dataset_with_filenames):
        try:
            os.mkdir(os.path.join(os.getcwd(), 'dataset'))
        except FileExistsError:
            pass

        pickle_file = 'dataset/augmented-{}.dat'.format(config['include_augmented'])
        with open(pickle_file, 'wb') as f:
            pickle.dump(dataset_with_filenames, f)


    def __len__(self):
        return len(self.dataset)

    def collate_single(self, batch):
        assert len(batch) == 1, 'Currently we do not support batched training'
        return batch[0]

    def collate_batch(self, batch):
        return batch


class GetDGMGDataset(LegoDataset):
    def __init__(self, config = None):
        super().__init__(config = config)


    def add_graph_to_dataset(self, graph, filename, graph_to_sequence):
        class_num = self.file_to_target.get_target(filename)
        actions = graph_to_sequence.to_action_sequence(graph, class_num)
        actions.insert(0, filename)
        self.dataset.append(actions)



    def __getitem__(self, index):
        return self.dataset[index]



class GINDataset(LegoDataset):
    def __init__(self, list_of_graphs = None, with_edge_types = False, config = None):
        self.with_edge_types = with_edge_types
        
        if list_of_graphs is None:
            super().__init__(config = config)
        elif list_of_graphs is not None:
            self.dataset = list_of_graphs
        
        self.gclasses = len(FileToTarget().classes)
        
    def add_graph_to_dataset(self, graph, filename, *args):
        try:
            del graph.ndata['hg']
        except:
            pass
        if 'attr' not in graph.ndata:
            add_node_attributes(graph)
            if self.with_edge_types:
                add_edge_attributes(graph)
        class_num = self.file_to_target.get_target(filename)
        if class_num >= 8:
            class_num += 1
        self.dataset.append({'graph': graph, 'target': class_num, 'filename': filename})


    def __getitem__(self, index):
        return self.dataset[index]['graph'], self.dataset[index]['target']


def add_edge_attributes(lg):
    eye = torch.eye(49)
    embed_edge = lambda x_shift, z_shift: eye[x_shift + 3 + ((z_shift + 3) * 7)]
    init_attr_graph = torch.Tensor([])
    for dest in range(lg.number_of_nodes()):
        init_attr_node = torch.zeros(49)
        for src in lg.in_edges(dest)[0].detach().numpy():
            shift = ast.literal_eval(lg.edge_labels[src, dest])
            init_attr_node += embed_edge(shift[0], shift[1])
        init_attr_node = torch.cat([lg.ndata['attr'][dest], init_attr_node]).view(-1, 51)
        init_attr_graph = torch.cat([init_attr_graph, init_attr_node])
    lg.ndata['attr'] = init_attr_graph


def add_node_attributes(lg):
    eye = np.eye(2)
    embed_node = lambda x: eye[1] if x == 'Brick(4, 2)' else eye[0]
    init_attrs = torch.tensor([embed_node(lg.node_labels[x]) for x in range(lg.number_of_nodes())], dtype = torch.float32)
    lg.ndata['attr'] = init_attrs

def save_model(model, epoch, dir, optimizer, scheduler = None):
    filename = os.path.join(dir, 'models', 'epoch_{:04d}.h5'.format(epoch))
    to_save = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    if scheduler is not None:
        to_save['scheduler_state_dict'] = scheduler.state_dict()
        
    torch.save(to_save, filename)



def load_gin():
    file = torch.load('pretrained_GIN.h5')
    config = file['gin_config']
    gin = GIN(config['num_layers'], config['num_mlp_layers'], config['input_dim'],
        config['hidden_dim'], config['output_dim'], config['final_dropout'], 
        config['learn_eps'], config['graph_pooling_type'], config['neighbor_pooling_type'])
    gin.load_state_dict(file['gin_state_dict'])

    return gin
