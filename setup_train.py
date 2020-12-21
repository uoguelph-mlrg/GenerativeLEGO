import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import DGL_DGMG
from DGL_DGMG import dgmg_helpers
from DGL_DGMG.dgmg_helpers import LegoPrinting, LegoModelEvaluation
from DGL_DGMG.model_batch import DGMG
import ast
import helpers
import pickle



def get_data_loaders(config, batch_size):
    dataset = helpers.GetDGMGDataset(config=vars(config))
    collate = dataset.collate_batch
    validation_size = int(config.valid_split * len(dataset))
    train_size = len(dataset) - validation_size
    config.ds_size = train_size
    print('DGMG validation, train sizes: {}, {}'.format(validation_size, train_size))
    train_set, valid_set = torch.utils.data.random_split(dataset, (train_size, validation_size))
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 0,
                                collate_fn = collate)
    valid_loader = DataLoader(valid_set, batch_size = validation_size, shuffle = True, num_workers = 0,
                                collate_fn = collate)
    return train_loader, valid_loader



def initialize_training(args, batch_size):
    config = args
    config.clip_grad = True
    config.clip_bound = 0.25
    config.num_samples = 200 # Num samples to generate at each epoch

    # For the Kim-et-al data
    config.num_shifts = 7
    config.num_node_types = 2

    torch.manual_seed(config.seed)

    #Setup model evaluator
    evaluator = dgmg_helpers.LegoModelEvaluation(v_max = config.max_generated_graph_size,
        edge_max = config.max_edges_per_node)
    printer = dgmg_helpers.LegoPrinting(num_epochs = config.epochs) # For printing results after each epoch

    # Setup dataset and data loader
    train_loader, valid_loader = get_data_loaders(args, batch_size)

    # Initialize_model
    model = DGMG(**vars(config))

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr = config.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = config.lr_step_size, gamma = config.lr_decay_rate)

    return evaluator, printer, train_loader, valid_loader, model, optimizer, scheduler



def parse_args():
    parser = argparse.ArgumentParser(description='DGMG')

    # configure
    parser.add_argument('--seed', type=int, default=42, 
        help='random seed')

    parser.add_argument('--batch_size', type=int, default = 10, 
        help = 'batch size to use for training')

    parser.add_argument('--auto_implied_edges', default = 'False', choices = ['True', 'False'], 
        help='Whether we want to manually add all implied edges for the model')
    
    parser.add_argument('--max_generated_graph_size', default = 235, type = int, 
        help='Max generated graph size')
    
    parser.add_argument('--max_edges_per_node', default = 12, type = int, choices = [1000, 12], 
        help='Max edges per node. 1000 is basically unlimited, 12 is the maximum number of edges \
        we can add to a 4x2 LEGO brick and still have a valid graph')

    parser.add_argument('--edge_generation', type = str, choices = ['ordinal', 'softmax'], default = 'softmax', 
        help = 'method for generating edges')
    
    parser.add_argument('--edge_embedding', type = str, default = 'embedding', choices = ['one-hot', 'embedding', 'ordinal'], 
        help = 'method for embedding/encoding edge types in the graph')

    parser.add_argument('--class_conditioning', default = 'embedding', choices = ['None', 'one-hot', 'embedding'],
        help = 'Which type of class-conditioning to use', type = str)

    parser.add_argument('--class_conditioning_size', default = 25, type = int,
        help = 'The size of the class condition embedding (if class-condition == embedding)')

    parser.add_argument('--epochs', default = 200, type = int, 
        help = 'the number of epochs to train for')

    parser.add_argument('--lr', default = 5e-4, type = float, 
        help = 'The learning rate')

    parser.add_argument('--lr_decay_rate', default = 0.85, type = float, 
        help = 'The learning rate decay rate')

    parser.add_argument('--lr_step_size', default = 50, type = int, 
        help = 'How often to decay the lr by lr_decay_rate')

    parser.add_argument('--node_hidden_size', default = 80, type = int, 
        help = 'The hidden dimensionality of each node')

    parser.add_argument('--num_prop_rounds', default = 2, type = int, 
        help = 'The number of graph propagation rounds to do')

    parser.add_argument('--edge_hidden_size', default = 80, type = int, 
        help = 'The edge hidden size to use')

    parser.add_argument('--num_decision_layers', default = 4, type = int, 
        help = 'The number of layers to use in the decision modules')

    parser.add_argument('--decision_layer_hidden_size', default = 40, type = int, 
        help = 'The number of neurons in each decision hidden layer')

    parser.add_argument('--num_propagation_mlp_layers', default = 4, type = int,
     help = 'The number of layers in each MLP in graph prop')

    parser.add_argument('--prop_mlp_hidden_size', default = 40, type = int, 
        help = 'The hidden size of the mlp in graph prop')

    parser.add_argument('--include_augmented', default = 'True', choices = ['True', 'False'], 
        help = 'Whether to include all dataset augmentations (90, 180, 270 deg rotations)')
    
    parser.add_argument('--missing_implied_edges_isnt_error', default = 'True', choices = ['True', 'False'], 
        help = 'Whether a model not adding implied edges counts as an error or not')

    parser.add_argument('--valid_split', default = 0.15, type = float, 
        help='The proportion of the dataset to be used for validation')

    parser.add_argument('--force_valid', default = 'False', choices = ['False'])

    parser.add_argument('--stop_generation', default = 'all_errors', choices = ['all_errors', 'one_error', 'None'],
        help='When to stop generating a graph. Can stop once a graph contains all errors, a single or error, or when the model \
        decides to stop regardless of any errors')

    args = parser.parse_args()

    args.auto_implied_edges = ast.literal_eval(args.auto_implied_edges)
    args.missing_implied_edges_isnt_error = ast.literal_eval(args.missing_implied_edges_isnt_error)
    args.force_valid = ast.literal_eval(args.force_valid)
    args.include_augmented = ast.literal_eval(args.include_augmented)
     
    if args.edge_hidden_size % 2 != 0:
        raise Exception('Edge hidden size % 2 != 0, {}'.format(args.edge_hidden_size))

    args.dataset = 'Kim-et-al'
    args.num_classes = 12

    if args.class_conditioning == 'one-hot':
        args.class_conditioning_size = args.num_classes
    
    elif args.class_conditioning == 'None':
        args.class_conditioning_size = 0

    return args