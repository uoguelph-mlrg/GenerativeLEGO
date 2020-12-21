import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pyFiles import LegoGraph, LDraw, utils
from DGL_DGMG.dgmg_setup import weights_init, dgmg_message_weight_init
from DGL_DGMG import common, graph_operations, add_node_module, add_edge_module, choose_dest_module, choose_edge_type_module, dgmg_helpers
import helpers
from functools import partial



class DGMG(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.validate_graph = utils.LegoGraphValidation()
        self.implied_edges = utils.ImpliedEdgesUtil()
        self.stop_generation = kwargs['stop_generation']; self.class_conditioning = kwargs['class_conditioning']
        self.class_conditioning_size = kwargs['class_conditioning_size']; self.auto_implied_edges = kwargs['auto_implied_edges']
        self.num_classes = kwargs['num_classes']; self.missing_implied_edges_isnt_error = kwargs['missing_implied_edges_isnt_error']


        # Different criteria for when to stop generating a graph:
        # a) When the graph contains a single error that means it isn't physically realizable
        # b) When the graph contains all errors that we keep track of
        # c) When the model decides to stop, regardless of any errors in the graph
        # option a) is the fastest (sometimes by quite a bit), especially early on in training
        self.__init_stop_generation_criteria(self.stop_generation)

        # Use one-hot encoding of classes, learned embedding, or no class-conditioning
        self.__init_class_conditioning_method(self.class_conditioning, self.class_conditioning_size)

        # Whether we want to automatically add implied edges
        self.__init_auto_implied_edges(self.auto_implied_edges)

        # Graph embedding module
        g_ops = graph_operations.GraphOps(**kwargs)
        self.graph_embed = g_ops.graph_embed; self.graph_prop = g_ops.graph_prop

        # Decision modules
        self.add_node_agent = add_node_module.AddNode(g_ops, **kwargs)
        self.add_edge_agent = add_edge_module.AddEdge(g_ops, **kwargs)
        self.choose_dest_agent = choose_dest_module.ChooseDest(**kwargs)
        self.choose_edge_type_agent = choose_edge_type_module.ChooseEdgeType(**kwargs)

        # Weight initialization
        self.init_weights()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            self.device = torch.device('cpu')

        self.to(self.device)

        self.step_count = 0



    def __init_stop_generation_criteria(self, stop_generation):
        # Set the self.stopping_criteria function
        if stop_generation == 'all_errors':
            self.stopping_criteria = self.classify_all
        elif stop_generation == 'one_error':
            self.stopping_criteria = self.single_error
        elif stop_generation == 'None':
            self.stopping_criteria = lambda *args, **kwargs: False



    def classify_all(self, g):
        # Stop graph generation once all LEGO errors have been encountered or max size is reached
        return self.validate_graph.invalid_shift(g) \
            and self.validate_graph.check_if_bricks_merged(g) \
            and (self.auto_implied_edges or self.validate_graph.check_if_brick_overconstrained(g)) \
            and (self.missing_implied_edges_isnt_error or self.auto_implied_edges or self.validate_graph.check_if_converted_graph_is_different(g))
    


    def single_error(self, g):
        # Stop graph generation at first LEGO error or max size is reached
        return self.validate_graph.invalid_shift(g) \
            or self.validate_graph.check_if_bricks_merged(g) \
            or self.validate_graph.check_if_brick_overconstrained(g) \
            or (self.validate_graph.check_if_converted_graph_is_different(g) and not self.missing_implied_edges_isnt_error)



    def __init_class_conditioning_method(self, class_conditioning, class_conditioning_size):
        # Set-up how we get class-conditioning vectors from a given class
        self.class_conditioning = class_conditioning

        if class_conditioning == 'embedding':
            self.class_embedding = nn.Embedding(self.num_classes, class_conditioning_size)
            self.class_conditioning_module = lambda cur_class: self.class_embedding(torch.LongTensor([cur_class]).to(self.device))

        elif class_conditioning == 'one-hot':
            self.class_conditioning_module = lambda cur_class: torch.eye(self.num_classes)[cur_class].to(self.device)

        else: #No class conditioning
            self.class_conditioning_module = lambda cur_class: torch.Tensor([]).to(self.device)

        self.class_conditioning_func = lambda classes: torch.cat([self.class_conditioning_module(class_num).view(1, -1) for class_num in classes])



    def __init_auto_implied_edges(self, auto_implied_edges):
        # Set-up function to automatically add implied edges (or do nothing)
        self.auto_implied_edges = auto_implied_edges
        if auto_implied_edges:
            self.auto_implied_edges_func = self.add_auto_implied_edges
        else:
            self.auto_implied_edges_func = lambda *args, **kwargs: None



    def init_weights(self):
        self.graph_embed.apply(weights_init)
        self.graph_prop.apply(weights_init)
        self.graph_prop.message_funcs.apply(dgmg_message_weight_init)

        self.add_node_agent.apply(weights_init)
        self.add_edge_agent.apply(weights_init)
        self.choose_dest_agent.apply(weights_init)



    def forward(self, batch_size = 200, v_max = 450, edge_max = 500, 
        class_to_generate = None, actions=None, softmax_temperature=1):
        if self.training:
            batch_size = len(actions)
        self.prepare(batch_size, class_to_generate)

        if self.training:
            self.softmax_temperature = 1
            return self.forward_train(actions)
        else:
            self.softmax_temperature = softmax_temperature
            return self.forward_inference(v_max, edge_max, class_to_generate)



    def forward_train(self, actions):
        """
        Go through all decisions in actions and record their
        log probabilities for calculating the loss.

        Parameters
        ----------
        actions : list
            list of decisions extracted for generating a graph using DGMG

        Returns
        -------
        tensor of shape torch.Size([])
            log P(Generate a batch of graphs using DGMG)
        """
        self.actions = actions
        stop = 0

        # First action in the list is the filename (for debugging mostly)
        self.filenames = self.get_actions('node')
        # Second action in the list is the class number (for class-conditioning)
        class_nums = self.get_actions('node')

        for i, class_num in enumerate(class_nums):
            self.g_active[i].set_class(class_num)

        self.class_conditioning_tensor = self.class_conditioning_func(class_nums)
        
        add_node_action, _ = self.add_node_decision(a = self.get_actions('node'))
        while add_node_action != stop:
            add_edge_action, dirns = self.add_edge_decision(a = self.get_actions('edge'))

            # If we want to add edges to the node we just inserted.
            while add_edge_action != stop:
                srcs, dests, edge_class_conditioning = self.choose_dest(dirns = dirns, a = self.get_actions('edge'))
                self.choose_edge_type(srcs = srcs, dests = dests, edge_type = self.get_actions('edge'), class_conditioning = edge_class_conditioning)
                
                self.perform_graph_prop()
                self.auto_implied_edges_func() # Does nothing if self.auto_implied_edges is False

                add_edge_action, dirns = self.add_edge_decision(a = self.get_actions('edge'))

            add_node_action, _ = self.add_node_decision(a = self.get_actions('node'))


        # This is just kind of a sanity check that I use when I'm testing
        # It just verifies that each graph we made using the actions is the same as the one
        # we get when we convert from LEGO build -> graph.
        # Also that when we convert our generated graph to actions, that we get the same actions
        # that we followed to generate said graph. 

        helper = utils.TestingHelpers()
        for i, g in enumerate(self.g_list):
            filename = self.filenames[i]
            g_test = LDraw.LDraw_to_graph('ldr_files/dataset/' + filename)
            if helper.is_same_lego_build(g, g_test) == False:
                print('{} is different lego build'.format(filename))
                g.write_to_file('test/{}_from_actions.ldr'.format(filename[:-4]))
                g_test.write_to_file('test/{}_from_ldr.ldr'.format(filename[:-4]))

            to_sequence = utils.LegoGraphToActionSequence()
            actions_test = to_sequence.to_action_sequence(g, helpers.FileToTarget().get_target(filename))
            actions_test.insert(0, filename)
            if actions_test != self.actions[i]:
                print('{} is different actions'.format(filename))
                print('test (from generation): ', actions_test)
                print('actual (from ldr): ', self.actions[i])

        return self.get_log_prob()



    def forward_inference(self, v_max, edge_max, class_to_generate):
        """
        Generate graph(s) on the fly.

        Returns
        -------
        self.g_list : list
            A list of LegoGraph.GeneratedLegoGraph objects.
        """
        stop = 0

        self.class_conditioning_tensor = self.class_conditioning_func(class_to_generate)

        add_node_action, _ = self.add_node_decision()
        while add_node_action != stop:
            num_trials = 0
            add_edge_action, dirns = self.add_edge_decision()

            # Some graphs need more edges to be added for the latest node and
            # the number of trials does not exceed the number of maximum possible
            # edges. Note that this limit on the number of edges eliminate the
            # possibility of multi-graph and one may want to remove it.
            while (add_edge_action != stop) and (num_trials < self.g_active[0].number_of_nodes() - 1) \
                and (num_trials < edge_max):
                srcs, dests, edge_class_conditioning = self.choose_dest(dirns)
                self.choose_edge_type(srcs, dests, edge_class_conditioning)
                self.update_active_list()
                self.perform_graph_prop()

                self.auto_implied_edges_func(edge_max) # Does nothing if self.auto_implied_edges is False
                
                num_trials += 1
                add_edge_action, dirns = self.add_edge_decision()

            if len(self.g_active) > 0 and self.g_active[0].number_of_nodes() < v_max: 
                add_node_action, _ = self.add_node_decision()
            else:
                add_node_action = stop
                
        return self.g_list



    def prepare(self, batch_size, class_to_generate):
        # Track how many actions have been taken for each graph.
        self.step_count = [0] * batch_size
        # self.g_list is a list of all the graphs in the batch
        self.g_list = []
        # self.g_active is a list of all the graphs in the batch that are active (i.e haven't made the decision to stop yet)
        self.g_active = []

        # Set the class we want to generate for inference
        # Class conditioning during training is done using the list of actions
        if not self.training and self.class_conditioning != 'None' and class_to_generate == None:
            # Generate random classes
            class_to_generate = np.random.choice(self.num_classes, batch_size)

        elif self.class_conditioning == 'None':
            # Dummy variable when class conditioning is None
            class_to_generate = [0] * batch_size

        elif self.training:
            # Dummy variable, class conditioning during training done somewhere else
            class_to_generate = [0] * batch_size

        assert batch_size == len(class_to_generate), 'Batch size != size of class to generate list ({} and {})'.format(batch_size, len(class_to_generate))


        def create_empty_graph(i):
            # Custom LegoGraph class that inherits from dgl.DGLGraph that has some LEGO specific methods
            g = LegoGraph.GeneratedLegoGraph()
            g.ndata['attr'] = torch.tensor([])

            g.set_class(class_to_generate[i])

            g.index = i

            # If there are some features for nodes and edges,
            # zero tensors will be set for those of new nodes and edges.
            g.set_n_initializer(dgl.frame.zero_initializer)
            g.set_e_initializer(dgl.frame.zero_initializer)
            return g


        for i in range(batch_size):
            g = create_empty_graph(i)
            self.g_list.append(g)
            self.g_active.append(g)

        if self.training:
            self.add_node_agent.prepare_training()
            self.add_edge_agent.prepare_training()
            self.choose_dest_agent.prepare_training()
            self.choose_edge_type_agent.prepare_training()



    def get_action_step(self, g_list):
        """
        This function should only be called during training.

        Collect the number of actions taken for each graph
        in the given list. After collecting
        the number of actions, increment it by 1.
        """

        old_step_count = []

        for g in g_list:
            old_step_count.append(self.step_count[g.index])
            self.step_count[g.index] += 1

        return old_step_count



    def get_actions(self, mode):
        """
        This function should only be called during training.

        Decide which graphs are related with the next batched
        decision and extract the actions to take for each of
        the graph.
        """

        if mode == 'node':
            # Graphs that are still active (i.e haven't made the decision to stop yet)
            g_list = self.g_active
        elif mode == 'edge':
            # Graphs that we are still adding edges to
            g_list = self.g_to_add_edge
        else:
            raise ValueError("Expected mode to be in ['node', 'edge'], "
                             "got {}".format(mode))

        # Get which action number/step we are on for each graph in the list
        action_indices = self.get_action_step(g_list)
        # Actions for all graphs indexed by indices at timestep t
        actions_t = []

        # Get the actions corresponding to each action number/step
        for i, g in enumerate(g_list):
            actions_t.append(self.actions[g.index][action_indices[i]])

        return actions_t



    def add_node_decision(self, a = None):
        """
        Decide if to add a new node for each graph being generated.

        The action(s) a are passed during training and
        sampled (hence None) during inference.
        """
        # Each decision module returns a list of the decisions we made (not including the stop decision),
        # and T/F indicating whether there any graphs that we need to make the next decision for. I.e if we add
        # a node to any graph, it would be True indicating we need to make edge decisions for at least one graph.
        if len(self.g_active) == 0:
            return False, []

        #g_active is a list of graphs we added a node to
        class_conditioning = self.class_conditioning_tensor[[g.index for g in self.g_active]]
        g_active, actions = self.add_node_agent(self.g_active, class_conditioning, a)

        self.g_active = g_active.copy()
        # For all newly added nodes we need to decide
        # if an edge is to be added for each of them.
        self.g_to_add_edge = g_active

        # Return whether we need to make edge decisions for any graph
        return len(self.g_active) > 0, actions



    def add_edge_decision(self, a = None):
        """
        For each graph, decide whether or not to add an edge,
        and the direction of the edge.

        The action(s) a are passed during training and
        sampled (hence None) during inference.
        """
        # Each decision module returns a list of the decisions we made (not including the stop decision),
        # and T/F indicating whether there any graphs that we need to make the next decision for. I.e if we add
        # an edge to any graph, it would be True indicating we need to make direction/type decisions for at least one graph.
        if len(self.g_to_add_edge) == 0:
            return False, []

        # self.g_to_add_edge is a list of graphs that we need to make a direction/type decision for
        class_conditioning = self.class_conditioning_tensor[[g.index for g in self.g_to_add_edge]]
        self.g_to_add_edge, dirns = self.add_edge_agent(self.g_to_add_edge, class_conditioning, a)
        assert len(self.g_to_add_edge) == len(dirns), 'number of graphs != number of actions in add edge ({} and {})'.format(len(self.g_to_add_edge), len(dirns))

        # Return if we need to make another decision and the directions of each edge
        return len(self.g_to_add_edge) > 0, dirns



    def choose_dest(self, dirns, a = None):
        """
        Choose the destination of each edge we decided to add

        The action(s) a are passed during training and
        sampled (hence None) during inference.
        """

        # This module just chooses the destination/direction of the edges we are adding, so 
        # self.g_to_add_edge is unchanged and we just return the actions we made

        class_conditioning = self.class_conditioning_tensor[[g.index for g in self.g_to_add_edge]]
        srcs, dests = self.choose_dest_agent(self.g_to_add_edge, dirns = dirns, class_conditioning = class_conditioning, dests = a)
        assert len(srcs) == len(self.g_to_add_edge), 'number of graphs != number of srcs in add node ({} and {})'.format(len(self.g_to_add_edge), len(srcs))
        assert len(dests) == len(self.g_to_add_edge), 'number of graphs != number of dests in add node ({} and {})'.format(len(self.g_to_add_edge), len(dests))

        return srcs, dests, class_conditioning



    def choose_edge_type(self, srcs, dests, class_conditioning, edge_type = None):
        """
        Choose the edge type (i.e positional shifts) of each edge
        we decided to add

        The actions (edge_type) are passed during training and sampled
        (hence None) during inference.
        """

        # Don't need the result of this decision for any other module so don't need to return anything
        # Choose the shifts for each edge we're adding
        self.choose_edge_type_agent(self.g_to_add_edge, srcs = srcs, dests = dests, class_conditioning = class_conditioning,
            edge_type = edge_type)



    def perform_graph_prop(self):
        if len(self.g_to_add_edge) == 0:
            return

        self.g_to_add_edge = self.graph_prop(self.g_to_add_edge)



    def update_active_list(self):
        # Update g_to_add_edge_list
        self.g_to_add_edge = [g for g in self.g_to_add_edge if not self.stopping_criteria(g)]

        # Update active list
        self.g_active = [g for g in self.g_active if not self.stopping_criteria(g)]



    def add_auto_implied_edges(self, *args):
        if self.training:
            # The next decision will be to add edge or not (during training), can skip it
            self.get_actions('edge')

        implied_edges = self.__get_implied_edges(*args)

        if len(implied_edges) > 0:
            self.__add_implied_edges(implied_edges)



    def __get_implied_edges(self, *args):
        num_graphs = len(self.g_to_add_edge)
        implied_edges = []
        # Reversed so that deleting from the list doesnt mess up our for loop/list
        for i, g in enumerate(reversed(self.g_to_add_edge)):
            ix = num_graphs - i - 1
            # Get implied edges missing from the node we just added
            implied_edges_g = self.implied_edges.get_edges_implied_by_node(g, g.number_of_nodes() - 1)
            
            if len(implied_edges_g) == 0: #If there's no implied edges to add
                del self.g_to_add_edge[ix]
            else:
                implied_edges.append(implied_edges_g)

        return implied_edges



    def __add_implied_edges(self, implied_edges):        
        # Implied edges is a list of lists, where each sublist contains all the implied edges
        # to be added to a single graph. The sublists could be of varying lengths.
        max_num_edges = len(max(implied_edges, key=len))
        for i in range(max_num_edges):
            assert len(implied_edges) == len(self.g_to_add_edge), 'Num graphs missing implied != num graphs to add edge, got ({} and {})'.format(len(implied_edges), len(self.g_to_add_edge))

            g_list = []; srcs = []; dests = []; x_shifts = []; z_shifts = []
            done_adding_edges = {}
            del_ix = []
            for ix, edges in enumerate(implied_edges):
                cur_edge = edges[i]
                g_list.append(cur_edge.g); srcs.append(cur_edge.src); dests.append(cur_edge.dest)
                x_shifts.append(torch.tensor([cur_edge.x_shift])); z_shifts.append(torch.tensor([cur_edge.z_shift]))

                if i + 1 >= len(edges):
                    del_ix.append(ix)
                    done_adding_edges[cur_edge.g.index] = 1

            # Add each implied edge to the graph and perform graph prop
            self.choose_edge_type_agent.initialize_edge_representation(g_list, srcs, dests, x_shifts, z_shifts)
            self.perform_graph_prop()

            if self.training:
                for i in range(3):
                    # For these graphs, we skipped the actions for adding a new edge, choosing the destination, and choosing the shifts
                    skipped_actions = self.get_actions('edge')
                    assert len(skipped_actions) == len(implied_edges), 'len skipped actions != num graphs adding edge to, got ({} and {})'.format(len(skipped_actions), len(edges))

            # Remove the graphs we're done adding implied edges to
            for ix in reversed(del_ix):
                del implied_edges[ix]
            self.g_to_add_edge = [g for g in self.g_to_add_edge if g.index not in done_adding_edges]

        # Should always be empty after this function (since we perform all valid edge decisions automatically)
        # Being empty ensures that the following add_edge_decision call returns the decision to stop
        assert len(self.g_to_add_edge) == 0, 'self.g_to_add_edge isnt empty after auto implied, {}'.format(len(self.g_to_add_edge))



    def get_log_prob(self):
        return torch.cat(self.add_node_agent.log_prob).sum()\
               + torch.cat(self.add_edge_agent.log_prob).sum()\
               + torch.cat(self.choose_dest_agent.log_prob).sum()\
               + torch.cat(self.choose_edge_type_agent.log_prob).sum()



    @property
    def softmax_temperature(self):
        return self._softmax_temperature
    
    @softmax_temperature.setter
    def softmax_temperature(self, new_temp):
        self.choose_dest_agent.softmax_temperature = new_temp
        self.choose_edge_type_agent.softmax_temperature = new_temp
        self.add_node_agent.softmax_temperature = new_temp
        self.add_edge_agent.softmax_temperature = new_temp
