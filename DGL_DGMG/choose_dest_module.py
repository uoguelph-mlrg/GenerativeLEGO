import torch
from torch.distributions import Bernoulli, Categorical
import torch.nn as nn
import torch.nn.functional as F
from pyFiles import LegoGraph, LDraw, utils
import ast
from DGL_DGMG import common

class ChooseDest(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        class_conditioning_size = kwargs['class_conditioning_size']; num_layers = kwargs['num_decision_layers']
        hidden_size = kwargs['decision_layer_hidden_size']; node_hidden_size = kwargs['node_hidden_size']
        force_valid = kwargs['force_valid']

        input_size = (2 * node_hidden_size) + class_conditioning_size
        self.choose_dest = common.MLP(num_layers, hidden_size, input_size, output_size = 1)

        if force_valid:
            self.apply_softmax_temperature = self.remove_invalid_decisions
        else:
            self.apply_softmax_temperature = lambda *args, **kwargs: None

        self.get_actions = lambda batch_prob, *args, **kwargs: Categorical(torch.squeeze(batch_prob, dim = 2)).sample().flatten()

        self.total_sum_dist = torch.zeros(1)
        self.class_sum_dist = torch.zeros(12, 1)



    def prepare_training(self):
        """
        This function will only be called during training.
        It stores all log probabilities for ChooseDest actions.
        Each element is a tensor of shape [1, 1].
        """
        self.log_prob = []



    def forward(self, g_list, dirns, class_conditioning, dests):
        """
        For each g in g_list, choose a node to connect the newly added
        node to.

        During training, dests is passed rather than chosen and the
        log P of the action is recorded.

        During inference, dests is sampled from a Categorical
        distribution modeled.

        Parameters
        ----------
        g_list : list
            A list of LegoGraph.GeneratedLegoGraph objects

        dirns: list
            A list specifying which direction the added edge will be.
            If dirn == 1: edge is from newly added node -> dest (old node).
            If dirn != 1: edge is from dest (old node) -> newly added node.
            dirns is the output from the add_edge module

            I assume that the choose_dest module/function is variant relative to the ordering of
            the nodes, i.e choose_dest(node_1, node_2) != choose_dest(node_2, node_1). Rather
            than giving choose_dest the direction of each edge and making it learn that it
            can be used to determine which node is src and dest, I use dirns to ensure that 
            the nodes passed to choose_dest is always like choose_dest(src, dest).


        dests : None or list
            - During training, dests is a list of integers specifying dest for
              each graph in g_list.
            - During inference, d is None.

        Returns
        -------
        bottom_node : list
            list of nodes that are the source of a new edge (i.e outgoing directed edge)

        top_node : list
            list of nodes that are the destination of a new edge (i.e incoming directed edge)
        """
        if self.training:
            assert len(dests) == len(g_list) and len(dests) == len(dirns), "Lengths dont match in choose dest"
        
        src_embed_expand_batch = []
        dest_embed_batch = []
        class_conditioning_batch = []
        srcs = []
        for i, g in enumerate(g_list):
            src = g.number_of_nodes() - 1
            srcs.append(src)
            possible_dests = range(src)

            # Get edge embeddings
            src_embed_expand = g.nodes[src].data['hv'].expand(len(possible_dests), -1)
            possible_dests_embed = g.nodes[possible_dests].data['hv']

            class_conditioning_temp = class_conditioning[i].expand(1, len(possible_dests), -1)
            class_conditioning_batch.append(class_conditioning_temp.view(1, len(possible_dests), -1))

            if dirns[i] == 1:
                dest_embed_batch.append(src_embed_expand.view(1, len(possible_dests), -1))
                src_embed_expand_batch.append(possible_dests_embed.view(1, len(possible_dests), -1))
            else:
                src_embed_expand_batch.append(src_embed_expand.view(1, len(possible_dests), -1))
                dest_embed_batch.append(possible_dests_embed.view(1, len(possible_dests), -1))

        # Convert lists to tensors
        src_embed_expand_batch = torch.cat(src_embed_expand_batch, dim = 0)
        dest_embed_batch = torch.cat(dest_embed_batch, dim = 0)
        class_conditioning_batch = torch.cat(class_conditioning_batch, dim = 0)
        dests_scores = self.choose_dest(torch.cat([src_embed_expand_batch, 
                                dest_embed_batch, class_conditioning_batch], dim = 2))


        if not self.training:
            self.apply_softmax_temperature(g_list, dirns, dests_scores)
            dests_probs = F.softmax(dests_scores / self.softmax_temperature, dim=1)
            dests = self.get_actions(dests_probs, g_list).tolist()

        # The next modules require which node is src and dest, so make a list of them
        bottom_node = []
        top_node = []
        for i in range(len(g_list)):
            if dirns[i] == 1:
                bottom_node.append(srcs[i]) 
                top_node.append(dests[i])
            else:
                bottom_node.append(dests[i])
                top_node.append(srcs[i])

        if self.training:
            target = torch.tensor(dests, dtype = torch.int64)
            loss = nn.CrossEntropyLoss(reduction = 'none')
            loss_res = loss(dests_scores, target.view(-1, 1))
            self.log_prob.append(loss_res)

        return bottom_node, top_node



    def remove_invalid_decisions(self, g_list, dirns, scores):
        for i, g in enumerate(g_list):
            dirn = dirns[i];
            node_label = g.node_labels[g.number_of_nodes() - 1]
            size = ast.literal_eval(''.join(c for c in node_label if not c.isalpha()))
            invalid_nodes = self.get_invalid_nodes(g, dirn, size)
            scores[i][invalid_nodes] = -float('inf')



    def get_invalid_nodes(self, g, dirn, size):
        if dirn == 1: # Old node is on top
            possible_nodes = g.get_nodes_that_can_connect_on_top_of(size)
            invalid_nodes = [node for node in range(g.number_of_nodes()) if node not in possible_nodes]

        else:
            possible_nodes = g.get_nodes_that_can_connect_underneath_of(size)
            invalid_nodes = [node for node in range(g.number_of_nodes()) if node not in possible_nodes]

        return invalid_nodes