import torch
from torch.distributions import Bernoulli, Categorical
import torch.nn as nn
import torch.nn.functional as F
from pyFiles import LegoGraph, LDraw, utils
from DGL_DGMG import common



class AddNode(nn.Module):
    def __init__(self, g_ops, **kwargs):
        super(AddNode, self).__init__()
        self.graph_op = {'embed': g_ops.graph_embed, 'prop': g_ops.graph_prop}
        graph_hidden_size = g_ops.graph_hidden_size

        class_conditioning_size = kwargs['class_conditioning_size']; num_layers = kwargs['num_decision_layers']
        hidden_size = kwargs['decision_layer_hidden_size']; num_node_types = kwargs['num_node_types']
        self.node_hidden_size = kwargs['node_hidden_size']

        self.stop = 0
        # Output is num_node_types + 1 - decisions for each node type, as well as decision to not add
        # a node of any type
        input_size = graph_hidden_size + class_conditioning_size
        self.add_node = common.MLP(num_layers, hidden_size, 
            input_size, output_size = num_node_types + 1)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # If to add a node, initialize its hv
        self.node_type_embed = nn.Embedding(num_node_types + 1, self.node_hidden_size)
        self.initialize_hv = nn.Linear(self.node_hidden_size + \
                                       graph_hidden_size,
                                       self.node_hidden_size)
        self.node_embed_func = lambda node_type, graph_embed: torch.cat([
                self.node_type_embed(torch.LongTensor([node_type - 1]).to(self.device)),
                graph_embed], dim=1)
        self._initialize_node_repr = self.initialize_node_DGMG


        self.get_actions = lambda batch_prob, *args, **kwargs: Categorical(batch_prob).sample()

        self.init_node_activation = torch.zeros(1, 2 * self.node_hidden_size)



    def initialize_node_DGMG(self, g, node_type, graph_embed):
        num_nodes = g.number_of_nodes()
        embed = self.node_embed_func(node_type, graph_embed)
        hv_init = self.initialize_hv(embed)
        g.nodes[num_nodes - 1].data['type'] = torch.tensor([node_type])

        hv_init = ((hv_init - torch.mean(hv_init)) / (torch.max(hv_init) - torch.min(hv_init)))
        g.nodes[num_nodes - 1].data['hv'] = hv_init
        g.nodes[num_nodes - 1].data['a'] = self.init_node_activation



    def prepare_training(self):
        """
        This function will only be called during training.
        It stores all log probabilities for AddNode actions.
        Each element is a tensor of shape [batch_size, 1].
        """
        self.log_prob = []



    def forward(self, g_list, class_conditioning, a = None):
        """
        Decide if a new node should be added for each graph in
        the `g_list`, as well as the type of the new node. 
        If a new node is added, initialize its node representations. 
        Record graphs for which a new node is added.

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
              whether a new node should be added.
            - During inference, a is None.

        Returns
        -------
        g_non_stop : list
            list of graphs that have a new node added
        a : list
            list of actions specifying the type of each node added
        """

        if self.training:
            assert len(a) == len(g_list), "Lengths dont match in add node"
        g_non_stop = []

        # Graph embeddings
        batch_graph_embed = self.graph_op['embed'](g_list)
        
        # TODO: implement class conditioning properly
        batch_logit = self.add_node(torch.cat([batch_graph_embed, class_conditioning], dim = 1))
        batch_prob = F.softmax(batch_logit / self.softmax_temperature, dim = 1)

        if not self.training:
            a = self.get_actions(batch_prob, g_list).tolist()

        # Get a list of graphs that we're adding a node to, and initialize node embeddings
        actions_non_stop = []
        for i, g in enumerate(g_list):
            action = a[i]
            if action != self.stop:
                actions_non_stop.append(action)
                g_non_stop.append(g)
                g.add_generated_node(action)
                self._initialize_node_repr(g, action,
                                           batch_graph_embed[i, :].view(1, -1))

        if self.training:
            loss = nn.CrossEntropyLoss(reduction = 'none')
            targets = torch.tensor([a], dtype = torch.int64).flatten()
            loss_res = loss(batch_logit, targets).view(-1, 1)
            self.log_prob.append(loss_res)

        # Return list of graphs we added a node to
        return g_non_stop, actions_non_stop