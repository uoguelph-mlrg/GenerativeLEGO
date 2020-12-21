import torch
from torch.distributions import Bernoulli, Categorical
import torch.nn as nn
import torch.nn.functional as F
from pyFiles import LegoGraph, LDraw, utils
from DGL_DGMG import common
import numpy as np
import ast

class ChooseEdgeType(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_shifts = kwargs['num_shifts']; edge_generation = kwargs['edge_generation']
        class_conditioning_size = kwargs['class_conditioning_size']; num_layers = kwargs['num_decision_layers']
        hidden_size = kwargs['decision_layer_hidden_size']; node_hidden_size = kwargs['node_hidden_size']
        edge_embedding = kwargs['edge_embedding']
        edge_hidden_size = kwargs['edge_hidden_size']
        self.force_valid = kwargs['force_valid']


        # For Kim et al. dataset our shifts are in the range [-3, 3] when added to the graph, but we
        # convert it to be in the range [0, 6] to index lists/tensors pretty frequently,
        # so it's nice to store this variable if we move away from Kim-et-al data
        self.half_num_shifts = self.num_shifts // 2 # or 3 for the Kim-et-al dataset

        # Generate edges using an ordinal encoding or softmax (one-hot)
        self.__init_edge_generation(edge_generation, node_hidden_size, class_conditioning_size,
            num_layers, hidden_size)

        # Setting edge embeddings as one-hot, learned embedding, etc.
        self.__init_edge_embedding_method(edge_embedding, edge_hidden_size)

        # if self.force_valid is True then we remove the possibility of generating invalid
        # structures by altering the softmax.
        if self.force_valid:
            self.get_possible_connections = self.get_valid_connections
            self.apply_softmax_temperature = self.remove_invalid_edge_types
        else:
            self.get_possible_connections = lambda *args, **kwargs: np.zeros((2, 2))
            self.apply_softmax_temperature = lambda scores,  *args, **kwargs: scores

        self.get_actions = lambda batch_prob, *args, **kwargs: Categorical(batch_prob).sample()

        self.counter = 0

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        


    def __init_edge_generation(self, edge_generation, *args):
        self.edge_generation = edge_generation
        if edge_generation == 'softmax':
            self.__init_softmax_generation(*args)

        elif edge_generation == 'ordinal-coral':
            self.__init_ordinal_coral_generation(*args)

        elif edge_generation == 'ordinal':
            self.__init_ordinal_generation(*args)

        elif edge_generation == 'scalar':
            self.__init_scalar_generation(*args)

        else:
            raise Exception('unsupported edge generation {}'.format(edge_generation))



    def __init_softmax_generation(self, node_hidden_size, class_conditioning_size, 
        num_layers, hidden_size):
        # Function that sets up the class to use softmax generation for edge types
        # when forward is called.

        input_size = (2 * node_hidden_size) + class_conditioning_size
        self.choose_x_shift = common.MLP(num_layers, hidden_size, input_size, output_size = self.num_shifts)
        self.choose_z_shift =  common.MLP(num_layers, hidden_size, input_size, output_size = self.num_shifts)

        # Uniform weighting across each shift
        self.x_class_weights = torch.ones(self.num_shifts)
        self.z_class_weights = torch.ones(self.num_shifts)

        self.generate_edge_type = self.generate_edge_type_softmax



    def __init_edge_embedding_method(self, edge_embedding, edge_hidden_size):
        # Function that sets up the class to use appropriate edge embedding function
        # based on the arguments
        self.edge_hidden_size = edge_hidden_size
        self.edge_embedding = edge_embedding

        if self.edge_embedding == 'embedding':
            # Use an embedding module
            self.x_edge_type_embed = nn.Embedding(self.num_shifts, self.edge_hidden_size // 2)
            self.z_edge_type_embed = nn.Embedding(self.num_shifts, self.edge_hidden_size // 2)

            self.edge_embed_module = lambda x_shift, z_shift: torch.cat([self.x_edge_type_embed(torch.LongTensor([x_shift]).to(self.device)), 
                                                            self.z_edge_type_embed(torch.LongTensor([z_shift]).to(self.device))]).view(1, -1)

        elif self.edge_embedding == 'one-hot':
            # One-hot encode each shift (i.e two 7 dimensional vectors for Kim-et-al)
            self.onehot_embed = lambda shift: torch.eye(self.num_shifts)[shift]

            self.edge_embed_module = lambda x_shift, z_shift: torch.cat([self.onehot_embed(x_shift), 
                                                            self.onehot_embed(z_shift)]).view(1, -1)

        elif edge_embedding == 'one-hot-big':
            # One-hot encode the edge shifts together (i.e 49 dimensional vector for Kim-et-al)
            self.edge_embed_module = lambda x_shift, z_shift: \
                (torch.eye(self.num_shifts ** 2)[x_shift + ((z_shift) * self.num_shifts)]).view(1, -1)

        elif edge_embedding == 'ordinal':
            self.edge_embed_module = lambda x_shift, z_shift: \
                torch.cat([self.__get_ordinal_target_single_example(x_shift - self.half_num_shifts), 
                    self.__get_ordinal_target_single_example(z_shift - self.half_num_shifts)]).view(1, -1)

        elif edge_embedding == 'scalar':
            self.edge_embed_module = lambda x_shift, z_shift: torch.tensor([[x_shift, z_shift]], dtype = torch.float32)
        
        else:
            raise Exception('unsupported edge embedding {}'.format(edge_embedding))



    def forward(self, g_list, srcs, dests, class_conditioning, edge_type):
        # print('edge type')
        """
        For each g in g_list (and src/dest in srcs/dests), choose the edge type/shift
        for each new edge. 

        During training, edge_type is passed rather than chosen and the
        log P of the action is recorded.

        During inference, edge_type is sampled from a distribution modeled.

        Parameters
        ----------
        g_list : list
            A list of LegoGraph.GeneratedLegoGraph objects

        srcs/dests : lists
            Lists containing the src/dest of each new edge to be added


        edge_type : None or list
            - During training, edge_type is a list of tuples specifying the shifts
              for each graph in g_list.
            - During inference, edge_type is None.
        """
        if self.training:
            assert len(edge_type) == len(srcs) and len(edge_type) == len(g_list), "Lengths dont match in choose edge type"
        
        if self.training:
            # Extract the shifts for each graph
            x_shifts = torch.tensor(edge_type)[:, 0]
            z_shifts = torch.tensor(edge_type)[:, 1]

        else:
            x_shifts = None
            z_shifts = None

        src_embed_batch = []
        dest_embed_batch = []
        class_conditioning_batch = []
        for i, g in enumerate(g_list):
            #Get node embeddings
            src_embed = g.nodes[srcs[i]].data['hv']
            src_embed_batch.append(src_embed.view(1, -1))
            dest_embed = g.nodes[dests[i]].data['hv']
            dest_embed_batch.append(dest_embed.view(1, -1))
            

        # Convert lists to tensors
        src_embed_batch = torch.cat(src_embed_batch, dim = 0)
        dest_embed_batch = torch.cat(dest_embed_batch, dim = 0)
        # print(class_conditioning.shape, src_embed_batch.shape, dest_embed_batch.shape, len(g_list))

        x_shift_scores = self.choose_x_shift(
                    torch.cat([src_embed_batch, dest_embed_batch, 
                        class_conditioning], dim = 1))
        z_shift_scores = self.choose_z_shift(
                    torch.cat([src_embed_batch, dest_embed_batch, 
                        class_conditioning], dim = 1))

        self.generate_edge_type(g_list, srcs, dests, x_shifts, z_shifts, x_shift_scores, z_shift_scores)



    def initialize_edge_representation(self, g_list, srcs, dests, x_shifts, z_shifts):
        """
        For each g in g_list, initialize the edge representation for the new edge to add

        Parameters
        ----------
        g_list : list
            A list of LegoGraph.GeneratedLegoGraph objects

        srcs/dests : lists
            Lists containing the src/dest of each new edge to be added

        x_shifts/z_shifts : torch.tensors
            Tensors containing the x and z shift of each new edge to add.
        """

        #Add edge to graph if not already present and initialize the edge representation
        for i, g in enumerate(g_list):
            src = srcs[i]
            dest = dests[i]
            x_shift = int(x_shifts[i].detach().item())
            z_shift = int(z_shifts[i].detach().item())
            if not g.has_edge_between(src, dest):
                g.add_generated_edge(src, dest, x_shift, z_shift)
                
                # Get initial edge representation
                # And convert shifts from the range [-3, 3] to [0, 6] for indexing tensor
                edge_repr = self.edge_embed_module(x_shift + self.half_num_shifts, z_shift + self.half_num_shifts)
                edge_repr = ((edge_repr - torch.mean(edge_repr)) / (torch.max(edge_repr) - torch.min(edge_repr)))
                g.edges[src, dest].data['he'] = edge_repr



    def prepare_training(self):
        self.log_prob = []



    def generate_edge_type_ordinal(self, g, src, dest, x_shifts, z_shifts, x_logits, z_logits, *args):
        # Function for generating the edge types using ordinal regression

        # Ordinal coral has separate bias term
        if self.edge_generation == 'ordinal-coral':
            x_logits = x_logits + self.x_bias
            z_logits = z_logits + self.z_bias

        x_probs = torch.sigmoid(x_logits)
        z_probs = torch.sigmoid(z_logits)

        if not self.training:
            x_result = Bernoulli(x_probs).sample()
            z_result = Bernoulli(z_probs).sample()
            x_shifts, z_shifts = self.__get_edge_shifts_ordinal(x_result, z_result)

        self.initialize_edge_representation(g, src, dest, x_shifts, z_shifts)

        if self.training:
            x_targets = self.__get_ordinal_target(x_shifts)
            z_targets = self.__get_ordinal_target(z_shifts)

            x_loss = nn.BCELoss(weight = self.x_class_weights, reduction = 'sum')
            z_loss = nn.BCELoss(weight = self.z_class_weights, reduction = 'sum')
            self.log_prob.append(x_loss(x_probs, x_targets).view(-1, 1))
            self.log_prob.append(z_loss(z_probs, z_targets).view(-1, 1))



    def __get_edge_shifts_ordinal(self, x_result, z_result):
        x_shift = self.__get_shift_from_result(x_result)
        z_shift = self.__get_shift_from_result(z_result)

        return x_shift, z_shift



    def __get_shift_from_result(self, results):
        # Result is a vector of the form similar to [1, 1, 0, 0, 0, 0]
        # The vector [1, 1, 0, 0, 0, 0] indicates a shift of -1
        shifts = []
        for result in results:
            for ix, res in enumerate(result):
                if res == 0:
                    shift = ix - self.half_num_shifts
                    break
            shift = len(result) - self.half_num_shifts if res != 0 else shift
            shifts.append(shift)

        return torch.tensor(shifts)



    def __get_ordinal_target(self, shifts):
        result = []
        for shift in shifts:
            result.append(list(self.__get_ordinal_target_single_example(shift)))

        return torch.tensor(result)



    def __get_ordinal_target_single_example(self, shift):
        return torch.cat([torch.ones(shift + self.half_num_shifts), 
                    torch.zeros(self.num_shifts - 1 - shift - self.half_num_shifts)])


       
    def generate_edge_type_softmax(self, g_list, srcs, dests, x_shifts, z_shifts, x_shift_scores, z_shift_scores):
        # Function for generating the edge type using a softmax layer. This is the only edge type
        # generation function that has been adapted to the batched implementation. 

        #Sample from probability if not training and update dest
        if not self.training:
            # Get valid connections (or do nothing if force_valid == False)
            possible_connections = self.get_possible_connections(g_list, srcs, dests)
            # Remove invalid x_shift options from x_shift_scores (or do nothing)
            self.apply_softmax_temperature(x_shift_scores, possible_connections)
            # Determine each x_shift
            x_shift_probs = F.softmax(x_shift_scores / self.softmax_temperature, dim = 1)
            x_shifts = self.get_actions(x_shift_probs, g_list) - self.half_num_shifts
            # Remove invalid z_shift options from z_shift_scores (or do nothing)
            self.apply_softmax_temperature(z_shift_scores, possible_connections, x_shifts = x_shifts)
            #Determine each z_shift
            z_shift_probs = F.softmax(z_shift_scores / self.softmax_temperature, dim = 1)
            z_shifts = self.get_actions(z_shift_probs, g_list) - self.half_num_shifts


        self.initialize_edge_representation(g_list, srcs, dests, x_shifts, z_shifts)

        #Get log probability
        if self.training:
            x_loss = nn.CrossEntropyLoss(weight = self.x_class_weights, reduction = 'none')
            z_loss = nn.CrossEntropyLoss(weight = self.z_class_weights, reduction = 'none')
            x_targets = x_shifts + self.half_num_shifts
            z_targets = z_shifts + self.half_num_shifts

            x_loss_res = x_loss(x_shift_scores, x_targets)
            z_loss_res = z_loss(z_shift_scores, z_targets)

            if x_shift_scores.nelement() > 1:
                self.log_prob.append(x_loss_res)
            if z_shift_scores.nelement() > 1:
                self.log_prob.append(z_loss_res)



    def get_valid_connections(self, g_list, srcs, dests):
        assert len(g_list) == len(dests), 'len(node_types), srcs, g_list: {} {} {}'.format(len(dests), len(g_list))
        valid_connections = []
        for ix, g in enumerate(g_list):
            size = g.node_labels[g.number_of_nodes() - 1]
            size = ''.join(c for c in size if not c.isalpha())
            size = ast.literal_eval(size)
            if srcs[ix] < dests[ix]:
                old_node = srcs[ix]
                conns = g.get_valid_connections_old_underneath(old_node, size)
                assert len(conns) != 0, 'No possible connections'
                valid_connections.append(conns)
            else:
                old_node = dests[ix]
                conns = g.get_valid_connections_old_on_top(old_node, size)
                assert len(conns) != 0, 'No possible connections'
                valid_connections.append(conns)

        return valid_connections



    def remove_invalid_edge_types(self, scores, all_connections, x_shifts = None):
        for i, connections in enumerate(all_connections):
            if x_shifts is None:
                # print(connections)
                valid_shifts = np.unique(np.array(connections)[:, 0])
            else:
                valid_shifts = [shift[1] for shift in connections if shift[0] == x_shifts[i].item()]

            assert len(valid_shifts) != 0, 'No valid shifts, x_shifts: {}, connections: {}'.format(x_shifts[i], connections)
            invalid_shifts = [shift for shift in range(self.num_shifts) if (shift - self.half_num_shifts) not in valid_shifts]
            scores[i][invalid_shifts] = -float('inf')

