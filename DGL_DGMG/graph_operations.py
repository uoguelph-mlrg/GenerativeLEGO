import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import helpers
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
import dgl.function as fn
from dgl.utils import expand_as_pair
from DGL_DGMG import common



class DGMGGraphEmbed(nn.Module):
    def __init__(self, node_hidden_size):
        super(DGMGGraphEmbed, self).__init__()

        # Setting from the paper
        self.graph_hidden_size = 2 * node_hidden_size

        # Embed graphs
        self.node_gating = nn.Sequential(
            nn.Linear(node_hidden_size, 1),
            nn.Sigmoid()
        )
        self.node_to_graph = nn.Sequential(nn.Linear(node_hidden_size,
                                       self.graph_hidden_size))


    def forward(self, g_list):
        # With our current batched implementation of DGMG, new nodes
        # are not added for any graph until all graphs are done with
        # adding edges starting from the last node. Therefore all graphs
        # in the graph_list should have the same number of nodes.
        if g_list[0].number_of_nodes() == 0:
            return torch.zeros(len(g_list), self.graph_hidden_size)

        bg = dgl.batch(g_list, node_attrs = ['hv', 'a'])
        bhv = bg.ndata['hv']
        bg.ndata['hg'] = self.node_gating(bhv) * self.node_to_graph(bhv)

        return dgl.sum_nodes(bg, 'hg')



class DGMGGraphProp(nn.Module):
    def __init__(self, num_prop_rounds, num_mlp_layers, mlp_hidden_size, node_hidden_size, 
        edge_hidden_size, num_shifts, edge_embedding):
        super(DGMGGraphProp, self).__init__()
        self.num_prop_rounds = num_prop_rounds

        # Setting from the paper
        self.node_activation_hidden_size = 2 * node_hidden_size

        if edge_embedding == 'one-hot':
            # Overwrite edge_hidden_size - passed value only used when edge_embedding == 'embedding'
            edge_hidden_size = num_shifts * 2
        elif edge_embedding == 'ordinal':
            edge_hidden_size = (num_shifts - 1) * 2
        elif edge_embedding == 'one-hot-big':
            # Overwrite edge_hidden_size - passed value only used when edge_embedding == 'embedding'
            edge_hidden_size = num_shifts ** 2
        elif edge_embedding == 'scalar':
            edge_hidden_size = 2

        message_funcs = []
        node_update_funcs = []
        self.reduce_funcs = []
        message_func_input_size = (2 * node_hidden_size) + edge_hidden_size

        for t in range(num_prop_rounds):
            # input being [hv, hu, xuv]
            message_funcs.append(common.MLP(num_mlp_layers, mlp_hidden_size, message_func_input_size,
                        output_size = self.node_activation_hidden_size))
            self.reduce_funcs.append(partial(self.dgmg_reduce, round=t))

            node_update_funcs.append(
                nn.GRUCell(self.node_activation_hidden_size,
                    node_hidden_size))
            self.update_node = lambda round, message_vector, node_embed: \
                self.node_update_funcs[round](message_vector, node_embed)

        self.message_funcs = nn.ModuleList(message_funcs)
        self.node_update_funcs = nn.ModuleList(node_update_funcs)



    def dgmg_msg(self, edges):
        """
        For an edge u->v, return concat([h_u, x_uv])
        """
        return {'m': torch.cat([edges.src['hv'],
                                edges.data['he']],
                               dim=1)}



    def dgmg_reduce(self, nodes, round):
        hv_old = nodes.data['hv']
        m = nodes.mailbox['m']
        message = torch.cat([
            hv_old.unsqueeze(1).expand(-1, m.size(1), -1), m], dim=2)
        node_activation = (self.message_funcs[round](message)).sum(1)
        return {'a': node_activation}



    def forward(self, g_list):
        # Merge small graphs into a large graph.
        bg = dgl.batch(g_list, node_attrs = ['hv', 'a'])

        if bg.number_of_edges() == 0:
            return
        else:
            for t in range(self.num_prop_rounds):
                bg.send_and_recv(bg.edges(), message_func=self.dgmg_msg,
                              reduce_func=self.reduce_funcs[t])
                bg.ndata['hv'] = self.update_node(t, bg.ndata['a'], bg.ndata['hv'])

        # The graphs in g_list are my custom class GeneratedLegoGraph, and dgl.unbatch returns
        # dgl.DGLGraphs, so we just copy the important data from those graphs so we can keep using
        # our GeneratedLegoGraph object
        for i, g in enumerate(dgl.unbatch(bg)):
            g_list[i].ndata['hv'] = g.ndata['hv']
            g_list[i].ndata['a'] = g.ndata['a']
        return g_list

        

class GraphOps(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        node_hidden_size = kwargs.get('node_hidden_size'); hidden_size = kwargs.get('graph_hidden_size')
        self.graph_hidden_size = node_hidden_size * 2
        self.graph_embed = DGMGGraphEmbed(node_hidden_size)

        num_prop_rounds = kwargs.get('num_prop_rounds'); num_propagation_mlp_layers = kwargs.get('num_propagation_mlp_layers')
        prop_mlp_hidden_size = kwargs.get('prop_mlp_hidden_size'); edge_hidden_size = kwargs.get('edge_hidden_size')
        num_shifts = kwargs.get('num_shifts'); edge_embedding = kwargs.get('edge_embedding')
        self.graph_prop = DGMGGraphProp(num_prop_rounds, num_propagation_mlp_layers,
           prop_mlp_hidden_size, node_hidden_size, edge_hidden_size, num_shifts, 
           edge_embedding)