import torch
from torch.distributions import Bernoulli, Categorical
import torch.nn as nn
import torch.nn.functional as F
from pyFiles import LegoGraph, LDraw, utils
from DGL_DGMG import common

class AddEdge(nn.Module):
    def __init__(self, g_ops, **kwargs):
        super(AddEdge, self).__init__()
        self.graph_op = {'embed': g_ops.graph_embed}
        graph_hidden_size = g_ops.graph_hidden_size

        class_conditioning_size = kwargs['class_conditioning_size']; num_layers = kwargs['num_decision_layers']
        hidden_size = kwargs['decision_layer_hidden_size']; node_hidden_size = kwargs['node_hidden_size']

        # Output is a 3D vector - decision to not add an edge, and decisions
        # for an incoming/outgoing edge from the newly added node
        input_size = graph_hidden_size + node_hidden_size + class_conditioning_size
        self.add_edge = common.MLP(num_layers, hidden_size, input_size, output_size = 3)

        self.get_actions = lambda batch_prob, *args, **kwargs: Categorical(batch_prob).sample()



    def prepare_training(self):
        """
        This function will only be called during training.
        It stores all log probabilities for AddEdge actions.
        Each element is a tensor of shape [batch_size, 1].
        """
        self.log_prob = []



    def forward(self, g_list, class_conditioning, a=None):
        """
        Decide if a new edge should be added for each graph in
        the `g_list`, as well as the direction of each edge. 
        Record graphs for which a new edge is to be added.

        During training, the action is passed rather than made
        and the log P of the action is recorded.

        During inference, the action is sampled from a Bernoulli
        distribution modeled.

        Parameters
        ----------
        g_list : list
            A list of LegoGraph.GeneratedLegoGraph objects
        a : None or list
            - During training, a is a list of integers specifying
              whether a new edge should be added/direction of added edges.
            - During inference, a is None.

        Returns
        -------
        g_to_add_edge : list
            list of graphs that need a new edge to be added

        a: list
            list of actions specifying the direction of each edge that is to be added
        """

        if self.training:
            assert len(a) == len(g_list), "Lengths dont match in add edge"
        g_to_add_edge = []

        self.stop = 0

        # Graph embeddings
        batch_graph_embed = self.graph_op['embed'](g_list)
        # Node embeddings
        batch_src_embed = torch.cat([g.nodes[g.number_of_nodes() - 1].data['hv']
                                     for g in g_list], dim=0)

        batch_logit = self.add_edge(torch.cat([batch_graph_embed,
                                               batch_src_embed, class_conditioning], dim=1))
        
        batch_prob = F.softmax(batch_logit / self.softmax_temperature, dim = 1)
        if not self.training:
            a = self.get_actions(batch_prob, g_list).tolist()

        # Add graphs to list if the action wasn't to stop
        actions_to_add_edge = []
        for i, g in enumerate(g_list):
            action = a[i]
            if action != self.stop:
                actions_to_add_edge.append(action)
                g_to_add_edge.append(g)

        if self.training:
            loss = nn.CrossEntropyLoss(reduction = 'none')
            targets = torch.tensor([a], dtype = torch.int64).flatten()
            loss_res = loss(batch_logit, targets).view(-1, 1)
            self.log_prob.append(loss_res)

        # Return list of graphs that we need to add an edge to, and the direction of
        # each edge for future modules.
        return g_to_add_edge, actions_to_add_edge