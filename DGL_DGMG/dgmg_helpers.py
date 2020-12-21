import networkx as nx
import os
import pickle
import random
import torch
import torch.nn as nn
import numpy as np
import helpers
from pyFiles import utils, LDraw
import numpy as np
# import tensorflow as tf
from time import time
import helpers
from helpers import GINDataset
from DGL_DGMG.dataloader import GraphDataLoader, collate
from scipy import linalg
import sklearn
from sklearn.metrics.pairwise import polynomial_kernel
import tensorflow as tf
import tensorflow_gan as tfgan
import datetime
from scipy.linalg import toeplitz
import pyemd
import time
import concurrent.futures
from functools import partial
from pyFiles import utils
from scipy.linalg import eigvalsh


def get_embed_func(GIN, embed_func):
    if embed_func == 'get_graph_embed_sum':
        embed_function = GIN.get_graph_embed_sum
    elif embed_func == 'get_graph_embed_concat':
        embed_function = GIN.get_graph_embed_concat
    elif embed_func == 'forward':
        embed_function = GIN.forward
    else:
        embed_function = lambda g, h: torch.nn.functional.softmax(GIN.forward(g, h), dim = 1)
    return embed_function

def get_activations(GIN, embed_function, dataset):
    dataloader, _ = get_dataloader(dataset)

    for (graphs, labels) in dataloader:
        feat = graphs.ndata['attr']
        graph_embeds = embed_function(graphs, feat)

    return graph_embeds.detach().numpy()

def get_dataloader(dataset):
    if not isinstance(dataset, GINDataset) and not isinstance(dataset, GraphDataLoader):
        dataset = GINDataset(list_of_graphs = dataset)
        num_samples = len(dataset.dataset)
        dataloader, _ = GraphDataLoader(dataset, batch_size = num_samples,
                device = torch.device('cpu'), collate_fn = collate, shuffle = True, split_name = 'rand',
                split_ratio = 1.0).train_valid_loader()
    elif isinstance(dataset, GINDataset):
        num_samples = len(dataset.dataset)
        dataloader, _ = GraphDataLoader(dataset, batch_size = num_samples,
                device = torch.device('cpu'), collate_fn = collate, shuffle = True, split_name = 'rand',
                split_ratio = 1.0).train_valid_loader()

    return dataloader, num_samples

#----------------------------------------------------------------------------

class DGMGEvaluation():
    def __init__(self, GIN_dataset, GIN, embed_func = 'get_graph_embed_concat', knn = False, eval_mmd = False):
        self.eval_with_GIN = DGMGEvaluationWithGIN(GIN_dataset, GIN, embed_func = embed_func, knn = knn)
        self.eval_mmd = eval_mmd
        if self.eval_mmd:
            self.mmd_eval = MMDEval(GIN_dataset.dataset)

    def evaluate_all(self, dataset, calculate_accuracy = False):
        res = self.eval_with_GIN.evaluate_all(dataset, calculate_accuracy = calculate_accuracy)
        if self.eval_mmd:
            mmd_metrics = self.mmd_eval.evaluate_all(dataset)
            for key, val in mmd_metrics.items():
                res[key] = val
        return res


#----------------------------------------------------------------------------

class MMDEval():
    #From the GraphRNN github: https://github.com/JiaxuanYou/graph-generation
    class MMD():
        def __init__(self, is_parallel):
            self.is_parallel = is_parallel

        def emd(self, x, y, distance_scaling=1.0):
            support_size = max(len(x), len(y))
            d_mat = toeplitz(range(support_size)).astype(np.float)
            distance_mat = d_mat / distance_scaling

            # convert histogram values x and y to float, and make them equal len
            x = x.astype(np.float)
            y = y.astype(np.float)
            if len(x) < len(y):
                x = np.hstack((x, [0.0] * (support_size - len(x))))
            elif len(y) < len(x):
                y = np.hstack((y, [0.0] * (support_size - len(y))))

            emd = pyemd.emd(x, y, distance_mat)
            return emd

        def l2(self, x, y):
            dist = np.linalg.norm(x - y, 2)
            return dist

        def gaussian_tv(x, y, sigma=1.0):  
            support_size = max(len(x), len(y))
            # convert histogram values x and y to float, and make them equal len
            x = x.astype(np.float)
            y = y.astype(np.float)
            if len(x) < len(y):
                x = np.hstack((x, [0.0] * (support_size - len(x))))
            elif len(y) < len(x):
                y = np.hstack((y, [0.0] * (support_size - len(y))))

            dist = np.abs(x - y).sum() / 2.0
            return np.exp(-dist * dist / (2 * sigma * sigma))


        def gaussian_emd(self, x, y, sigma=1.0, distance_scaling=1.0):
            ''' Gaussian kernel with squared distance in exponential term replaced by EMD
            Args:
              x, y: 1D pmf of two distributions with the same support
              sigma: standard deviation
            '''
            support_size = max(len(x), len(y))
            d_mat = toeplitz(range(support_size)).astype(np.float)
            distance_mat = d_mat / distance_scaling

            # convert histogram values x and y to float, and make them equal len
            x = x.astype(np.float)
            y = y.astype(np.float)
            if len(x) < len(y):
                x = np.hstack((x, [0.0] * (support_size - len(x))))
            elif len(y) < len(x):
                y = np.hstack((y, [0.0] * (support_size - len(y))))

            emd = pyemd.emd(x, y, distance_mat)
            return np.exp(-emd * emd / (2 * sigma * sigma))

        def gaussian(self, x, y, sigma=1.0):
            dist = np.linalg.norm(x - y, 2)
            return np.exp(-dist * dist / (2 * sigma * sigma))

        def kernel_parallel_unpacked(self, x, samples2, kernel):
            d = 0
            for s2 in samples2:
                d += kernel(x, s2)
            return d

        def kernel_parallel_worker(self, t):
            return self.kernel_parallel_unpacked(*t)

        def disc(self, samples1, samples2, kernel, *args, **kwargs):
            ''' Discrepancy between 2 samples
            '''
            d = 0
            if not self.is_parallel:
                for s1 in samples1:
                    for s2 in samples2:
                        d += kernel(s1, s2, *args, **kwargs)
            else:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for dist in executor.map(self.kernel_parallel_worker, 
                            [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1]):
                        d += dist
            d /= len(samples1) * len(samples2)
            return d


        def compute_mmd(self, samples1, samples2, kernel, is_hist=True, *args, **kwargs):
            ''' MMD between two samples
            '''
            # normalize histograms into pmf
            if is_hist:
                samples1 = [s1 / np.sum(s1) for s1 in samples1]
                samples2 = [s2 / np.sum(s2) for s2 in samples2]
            # print('===============================')
            # print('s1: ', disc(samples1, samples1, kernel, *args, **kwargs))
            # print('--------------------------')
            # print('s2: ', disc(samples2, samples2, kernel, *args, **kwargs))
            # print('--------------------------')
            # print('cross: ', disc(samples1, samples2, kernel, *args, **kwargs))
            # print('===============================')
            return self.disc(samples1, samples1, kernel, *args, **kwargs) + \
                    self.disc(samples2, samples2, kernel, *args, **kwargs) - \
                    2 * self.disc(samples1, samples2, kernel, *args, **kwargs)

        def compute_emd(self, samples1, samples2, kernel, is_hist=True, *args, **kwargs):
            ''' EMD between average of two samples
            '''
            # normalize histograms into pmf
            if is_hist:
                samples1 = [np.mean(samples1)]
                samples2 = [np.mean(samples2)]
            # print('===============================')
            # print('s1: ', disc(samples1, samples1, kernel, *args, **kwargs))
            # print('--------------------------')
            # print('s2: ', disc(samples2, samples2, kernel, *args, **kwargs))
            # print('--------------------------')
            # print('cross: ', disc(samples1, samples2, kernel, *args, **kwargs))
            # print('===============================')
            return self.disc(samples1, samples2, kernel, *args, **kwargs),[samples1[0],samples2[0]]

    # maps motif/orbit name string to its corresponding list of indices from orca output
    motif_to_indices = {
            '3path' : [1, 2],
            '4cycle' : [8],
    }
    COUNT_START_STR = 'orbit counts: \n'
    

    def __init__(self, dataset, is_parallel=False, bins=100):
        self.dataset = [g['graph'].g_undirected for g in dataset]
        self.bins = bins
        self.is_parallel = is_parallel
        self.mmd = self.MMD(is_parallel)
        self.init_dataset_stats()
        
        
    def init_dataset_stats(self):
        self.sample_ref_deg = self.get_deg_stats(self.dataset)
        #self.sample_ref_clust = self.get_clust_stats(self.dataset)
        #self.total_counts_ref_orbit = self.get_orbit_stats(self.dataset)
        #self.spectral_ref = self.get_spectral_stats(self.dataset)


    def get_deg_stats(self, dataset):
        res = []
        if self.is_parallel:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for deg_hist in executor.map(self.degree_worker, dataset):
                    res.append(deg_hist)
        else:
            for g in dataset:
                degree_temp = np.array(nx.degree_histogram(g))
                res.append(degree_temp)
        return res


    def get_clust_stats(self, dataset):
        res = []
        if self.is_parallel:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for clustering_hist in executor.map(self.clustering_worker, 
                    [(G, self.bins) for G in dataset]):
                    res.append(clustering_hist)
        else:
            for g in dataset:
                clustering_coeffs_list = list(nx.clustering(g).values())
                hist, _ = np.histogram(
                        clustering_coeffs_list, bins=self.bins, range=(0.0, 1.0), density=False)
                res.append(hist)
        return res

    
    def get_orbit_stats(self, dataset):
        res = []
        for G in dataset:
            try:
                orbit_counts = self.orca(G)
            except:
                continue
            orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
            res.append(orbit_counts_graph)
        return np.array(res)

    def get_spectral_stats(self, dataset):
        res = []
        if self.is_parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for spectral_density in executor.map(self.spectral_worker, dataset):
                    res.append(spectral_density)
        else:
            for g in dataset:
                spectral_temp = self.spectral_worker(g)
                res.append(spectral_temp)
        return res


    def evaluate_all(self, dataset):
        dataset = [g['graph'].g_undirected for g in dataset if g['graph'].g_undirected.number_of_nodes() != 0]

        mmd_deg = self.degree_stats(dataset)
        # mmd_clus = self.clustering_stats(dataset)
        # mmd_orbit = self.orbit_stats_all(dataset)
        # mmd_spectral = self.spectral_stats(dataset)
        return dict(degree=mmd_deg)#, clustering=mmd_clus, orbit=mmd_orbit, spectral=mmd_spectral)


    def degree_worker(self, G):
        return np.array(nx.degree_histogram(G))


    def add_tensor(self, x,y):
        support_size = max(len(x), len(y))
        if len(x) < len(y):
            x = np.hstack((x, [0.0] * (support_size - len(x))))
        elif len(y) < len(x):
            y = np.hstack((y, [0.0] * (support_size - len(y))))
        return x+y

    def degree_stats(self, graph_pred_list):
        ''' Compute the distance between the degree distributions of two unordered sets of graphs.
        Args:
          graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
        '''
        sample_pred = self.get_deg_stats(graph_pred_list)

        mmd_dist = self.mmd.compute_mmd(self.sample_ref_deg, sample_pred, kernel=self.mmd.gaussian_emd)

        return mmd_dist

    def clustering_worker(self, param):
        G, bins = param
        clustering_coeffs_list = list(nx.clustering(G).values())
        hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
        return hist

    def clustering_stats(self, graph_pred_list):
        sample_pred = self.get_clust_stats(graph_pred_list)
        
        mmd_dist = self.mmd.compute_mmd(self.sample_ref_clust, sample_pred, kernel=self.mmd.gaussian_emd,
                                   sigma=1.0/10, distance_scaling=self.bins)

        return mmd_dist

    def edge_list_reindexed(self, G):
        idx = 0
        id2idx = dict()
        for u in G.nodes():
            id2idx[str(u)] = idx
            idx += 1

        edges = []
        for (u, v) in G.edges():
            edges.append((id2idx[str(u)], id2idx[str(v)]))
        return edges

    def orca(self, graph):
        tmp_fname = 'eval/orca/tmp.txt'
        f = open(tmp_fname, 'w')
        f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
        for (u, v) in self.edge_list_reindexed(graph):
            f.write(str(u) + ' ' + str(v) + '\n')
        f.close()

        output = sp.check_output(['./eval/orca/orca', 'node', '4', 'eval/orca/tmp.txt', 'std'])
        output = output.decode('utf8').strip()
        
        idx = output.find(self.COUNT_START_STR) + len(self.COUNT_START_STR)
        output = output[idx:]
        node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ') ))
              for node_cnts in output.strip('\n').split('\n')])

        try:
            os.remove(tmp_fname)
        except OSError:
            pass

        return node_orbit_counts
        

    def orbit_stats_all(self, graph_pred_list):
        total_counts_pred = self.get_orbit_stats(graph_pred_list)
        if len(self.total_counts_ref_orbit) == 0 or len(total_counts_pred) == 0:
            print('Cant compute orbits')
            return -1
        else:
            mmd_dist = self.mmd.compute_mmd(self.total_counts_ref_orbit, total_counts_pred, kernel=self.mmd.gaussian,
                is_hist=False, sigma=30.0)

            return mmd_dist

    def spectral_worker(self, G):
        # eigs = nx.laplacian_spectrum(G)
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())  
        spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
        spectral_pmf = spectral_pmf / spectral_pmf.sum()
        # from scipy import stats  
        # kernel = stats.gaussian_kde(eigs)
        # positions = np.arange(0.0, 2.0, 0.1)
        # spectral_density = kernel(positions)

        # import pdb; pdb.set_trace()
        return spectral_pmf

    def spectral_stats(self, graph_pred_list):
        ''' Compute the distance between the degree distributions of two unordered sets of graphs.
        Args:
          graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
        '''
        sample_pred = self.get_spectral_stats(graph_pred_list)

        # print(len(sample_ref), len(sample_pred))

        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        print(len(self.spectral_ref), len(sample_pred))
        mmd_dist = self.mmd.compute_mmd(self.spectral_ref, sample_pred, kernel=self.mmd.gaussian_tv)
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)
        return mmd_dist


#----------------------------------------------------------------------------

class DGMGEvaluationWithGIN():
    def __init__(self, GIN_dataset, GIN, embed_func = 'get_graph_embed_concat', knn = False):
        GIN.eval()
        self.GIN = GIN
        self.embed_function = self._init_embed_func(embed_func)
        if knn:
            self.GIN_dataset = GIN_dataset.dataset
            self.activations = np.array([self._get_activations([g]) for g in self.GIN_dataset]).squeeze()

        self.fid_eval = FIDEvaluation(GIN_dataset, GIN, embed_func)
        self.gin_acc = GINAccuracy(GIN)
        self.kid_eval = KIDEvaluation(GIN_dataset, GIN, embed_func)
        self.prdc_eval = prdcEvaluation(GIN_dataset, GIN, embed_func)

        print('GIN reference dataset size: ', len(GIN_dataset.dataset))


    def evaluate_all(self, dataset, calculate_accuracy = False):
        activations = self._get_activations(dataset)
        fid = self.fid_eval.calculate_FID_from_activations(activations)
        kid = self.kid_eval.calculate_KID_from_activations(activations)
        prdc = self.prdc_eval.calculate_prdc_from_activations(activations)
        if calculate_accuracy:
            gin_acc = self.calculate_GIN_accuracy(dataset)
        else:
            gin_acc = 0

        return dict(fid=fid, kid=kid, GIN_accuracy=gin_acc, precision=prdc['precision'],
            recall=prdc['recall'], density=prdc['density'], coverage=prdc['coverage'])

    def calculate_FID(self, dataset):
        return self.fid_eval.calculate_FID(dataset)

    def calculate_GIN_accuracy(self, dataset):
        return self.gin_acc.calculate_accuracy(dataset)

    def calculate_KID(self, dataset):
        return self.kid_eval.calculate_KID(dataset)

    def calculate_prdc(self, dataset):
        return self.prdc_eval.calculate_prdc(dataset)

    def get_nearest_neighbor(self, generated_sample, k = 1):
        from sklearn.neighbors import NearestNeighbors
        g = LDraw.LDraw_to_graph(generated_sample)
        helpers.add_node_attributes(g)
        helpers.add_edge_attributes(g)

        for attr in list(g.ndata.keys()):
            if attr != 'attr':
                del g.ndata[attr]

        g = [{'graph': g, 'target': 0}]
        g_activation = self._get_activations(g)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(self.activations)
        dist, indices = nbrs.kneighbors(g_activation)
        return [self.GIN_dataset[ix]['filename'] for i in indices for ix in i]


    def _init_embed_func(self, embed_func):
        return get_embed_func(self.GIN, embed_func)
    
    def _get_activations(self, dataset):
        return get_activations(self.GIN, self.embed_function, dataset)

    def _get_dataloader(self, dataset):
        return get_dataloader(dataset)

#----------------------------------------------------------------------------

class DGMGMetric():
    def __init__(self, GIN_dataset, GIN, embed_func = 'get_graph_embed_concat'):
        GIN.eval()
        self.GIN = GIN
        self.embed_function = self._init_embed_func(embed_func)

        self.activations = self._get_activations(GIN_dataset)

    def _init_embed_func(self, embed_func):
        return get_embed_func(self.GIN, embed_func)
    
    def _get_activations(self, dataset):
        return get_activations(self.GIN, self.embed_function, dataset)

    def _get_dataloader(self, dataset):
        return get_dataloader(dataset)

#----------------------------------------------------------------------------

class GINAccuracy(DGMGMetric):
    def __init__(self, GIN):
        self.GIN = GIN


    def calculate_accuracy(self, dataset):
        dataloader, _ = self._get_dataloader(dataset)
        
        for (graphs, labels) in dataloader:
            feat = graphs.ndata['attr']
            total = len(labels)
            outputs = self.GIN(graphs, feat)
            _, predicted = torch.max(outputs.data, 1)
            total_correct = (predicted == labels.data).sum().item()

        return (total_correct / total) * 100

#----------------------------------------------------------------------------

class KIDEvaluation(DGMGMetric):
    # From tensorflow github: https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/eval/classifier_metrics.py#L1036
    def __init__(self, GIN_dataset, GIN, embed_func = 'get_graph_embed_concat'):
        super().__init__(GIN_dataset, GIN, embed_func = embed_func)
        self.activations = tf.convert_to_tensor(self.activations)


    def calculate_KID_from_activations(self, test_activations):
        test_activations = tf.convert_to_tensor(test_activations)
        return tfgan.eval.kernel_classifier_distance_and_std_from_activations(self.activations, test_activations)[0].numpy()


    def calculate_KID(self, dataset):
        test_activations = self._get_activations(dataset)
        return self.calculate_KID_from_activations(test_activations)


    def kernel_classifier_distance_and_std_from_activations(self, real_activations,
                                                        generated_activations,
                                                        max_block_size=10,
                                                        dtype=None):
        """Kernel "classifier" distance for evaluating a generative model.
        This methods computes the kernel classifier distance from activations of
        real images and generated images. This can be used independently of the
        kernel_classifier_distance() method, especially in the case of using large
        batches during evaluation where we would like to precompute all of the
        activations before computing the classifier distance, or if we want to
        compute multiple metrics based on the same images. It also returns a rough
        estimate of the standard error of the estimator.
        This technique is described in detail in https://arxiv.org/abs/1801.01401.
        Given two distributions P and Q of activations, this function calculates
            E_{X, X' ~ P}[k(X, X')] + E_{Y, Y' ~ Q}[k(Y, Y')]
              - 2 E_{X ~ P, Y ~ Q}[k(X, Y)]
        where k is the polynomial kernel
            k(x, y) = ( x^T y / dimension + 1 )^3.
        This captures how different the distributions of real and generated images'
        visual features are. Like the Frechet distance (and unlike the Inception
        score), this is a true distance and incorporates information about the
        target images. Unlike the Frechet score, this function computes an
        *unbiased* and asymptotically normal estimator, which makes comparing
        estimates across models much more intuitive.
        The estimator used takes time quadratic in max_block_size. Larger values of
        max_block_size will decrease the variance of the estimator but increase the
        computational cost. This differs slightly from the estimator used by the
        original paper; it is the block estimator of https://arxiv.org/abs/1307.1954.
        The estimate of the standard error will also be more reliable when there are
        more blocks, i.e. when max_block_size is smaller.
        NOTE: the blocking code assumes that real_activations and
        generated_activations are both in random order. If either is sorted in a
        meaningful order, the estimator will behave poorly.
        Args:
          real_activations: 2D Tensor containing activations of real data. Shape is
            [batch_size, activation_size].
          generated_activations: 2D Tensor containing activations of generated data.
            Shape is [batch_size, activation_size].
          max_block_size: integer, default 1024. The distance estimator splits samples
            into blocks for computational efficiency. Larger values are more
            computationally expensive but decrease the variance of the distance
            estimate. Having a smaller block size also gives a better estimate of the
            standard error.
          dtype: if not None, coerce activations to this dtype before computations.
        Returns:
         The Kernel Inception Distance. A floating-point scalar of the same type
           as the output of the activations.
         An estimate of the standard error of the distance estimator (a scalar of
           the same type).
        """

        real_activations.shape.assert_has_rank(2)
        generated_activations.shape.assert_has_rank(2)
        real_activations.shape[1].assert_is_compatible_with(
            generated_activations.shape[1])

        if dtype is None:
            dtype = real_activations.dtype
            assert generated_activations.dtype == dtype
        else:
            real_activations = math_ops.cast(real_activations, dtype)
            generated_activations = math_ops.cast(generated_activations, dtype)

        # Figure out how to split the activations into blocks of approximately
        # equal size, with none larger than max_block_size.
        n_r = array_ops.shape(real_activations)[0]
        n_g = array_ops.shape(generated_activations)[0]

        n_bigger = math_ops.maximum(n_r, n_g)
        n_blocks = math_ops.to_int32(math_ops.ceil(n_bigger / max_block_size))

        v_r = n_r // n_blocks
        v_g = n_g // n_blocks

        n_plusone_r = n_r - v_r * n_blocks
        n_plusone_g = n_g - v_g * n_blocks

        sizes_r = array_ops.concat([
            array_ops.fill([n_blocks - n_plusone_r], v_r),
            array_ops.fill([n_plusone_r], v_r + 1),
        ], 0)
        sizes_g = array_ops.concat([
            array_ops.fill([n_blocks - n_plusone_g], v_g),
            array_ops.fill([n_plusone_g], v_g + 1),
        ], 0)

        zero = array_ops.zeros([1], dtype=dtypes.int32)
        inds_r = array_ops.concat([zero, math_ops.cumsum(sizes_r)], 0)
        inds_g = array_ops.concat([zero, math_ops.cumsum(sizes_g)], 0)

        dim = math_ops.cast(tf.shape(real_activations)[1], dtype)

        ests = functional_ops.map_fn(
            self.compute_kid_block, math_ops.range(n_blocks), dtype=dtype, back_prop=False)

        mn = math_ops.reduce_mean(ests)

        # nn_impl.moments doesn't use the Bessel correction, which we want here
        n_blocks_ = math_ops.cast(n_blocks, dtype)
        var = control_flow_ops.cond(
            math_ops.less_equal(n_blocks, 1),
            lambda: array_ops.constant(float('nan'), dtype=dtype),
            lambda: math_ops.reduce_sum(math_ops.square(ests - mn)) / (n_blocks_ - 1))

        return mn, math_ops.sqrt(var / n_blocks_)

    def compute_kid_block(self, i):
        'Compute the ith block of the KID estimate.'
        r_s = inds_r[i]
        r_e = inds_r[i + 1]
        r = real_activations[r_s:r_e]
        m = math_ops.cast(r_e - r_s, dtype)

        g_s = inds_g[i]
        g_e = inds_g[i + 1]
        g = generated_activations[g_s:g_e]
        n = math_ops.cast(g_e - g_s, dtype)

        k_rr = (math_ops.matmul(r, r, transpose_b=True) / dim + 1)**3
        k_rg = (math_ops.matmul(r, g, transpose_b=True) / dim + 1)**3
        k_gg = (math_ops.matmul(g, g, transpose_b=True) / dim + 1)**3
        return (-2 * math_ops.reduce_mean(k_rg) +
                (math_ops.reduce_sum(k_rr) - math_ops.trace(k_rr)) / (m * (m - 1)) +
                (math_ops.reduce_sum(k_gg) - math_ops.trace(k_gg)) / (n * (n - 1)))


#----------------------------------------------------------------------------

class FIDEvaluation(DGMGMetric):
    def __init__(self, GIN_dataset, GIN, embed_func = 'get_graph_embed_concat'):
        super().__init__(GIN_dataset, GIN, embed_func = embed_func)

        self.mu, self.cov = self.__calculate_dataset_stats(self.activations)


    def __calculate_dataset_stats(self, activations):
        mu = np.mean(activations, axis = 0)
        cov = np.cov(activations, rowvar = False)

        return mu, cov


    def calculate_FID_from_activations(self, activations):
        mu_generated, cov_generated = self.__calculate_dataset_stats(activations)
        return self.compute_FID(self.mu, mu_generated, self.cov, cov_generated)


    def calculate_FID(self, dataset):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """
        activations = self._get_activations(dataset)
        return self.calculate_FID_from_activations(activations)


    def compute_FID(self, mu1, mu2, cov1, cov2, eps = 1e-6):
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert cov1.shape == cov2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(cov1.shape[0]) * eps
            covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                #raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(cov1) +
                np.trace(cov2) - 2 * tr_covmean)


#----------------------------------------------------------------------------

class prdcEvaluation(DGMGMetric):
    # From PRDC github: https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py#L54
    def calculate_prdc(self, dataset, k = 5):
        test_activations = self._get_activations(dataset)
        return self.calculate_prdc_from_activations(test_activations, nearest_k = k)


    def __compute_pairwise_distance(self, data_x, data_y=None):
        """
        Args:
            data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
            data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
        Returns:
            numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
        """
        if data_y is None:
            data_y = data_x
        dists = sklearn.metrics.pairwise_distances(
            data_x, data_y, metric='euclidean', n_jobs=8)
        return dists


    def __get_kth_value(self, unsorted, k, axis=-1):
        """
        Args:
            unsorted: numpy.ndarray of any dimensionality.
            k: int
        Returns:
            kth values along the designated axis.
        """
        indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
        kth_values = k_smallests.max(axis=axis)
        return kth_values


    def __compute_nearest_neighbour_distances(self, input_features, nearest_k):
        """
        Args:
            input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int
        Returns:
            Distances to kth nearest neighbours.
        """
        distances = self.__compute_pairwise_distance(input_features)
        radii = self.__get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii


    def calculate_prdc_from_activations(self, fake_features, nearest_k = 5):
        """
        Computes precision, recall, density, and coverage given two manifolds.
        Args:
            real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int.
        Returns:
            dict of precision, recall, density, and coverage.
        """

        # print('Num real: {} Num fake: {}'
              # .format(real_features.shape[0], fake_features.shape[0]))
        real_features = self.activations

        real_nearest_neighbour_distances = self.__compute_nearest_neighbour_distances(
            real_features, nearest_k)
        fake_nearest_neighbour_distances = self.__compute_nearest_neighbour_distances(
            fake_features, nearest_k)
        distance_real_fake = self.__compute_pairwise_distance(
            real_features, fake_features)

        precision = (
                distance_real_fake <
                np.expand_dims(real_nearest_neighbour_distances, axis=1)
        ).any(axis=0).mean()

        recall = (
                distance_real_fake <
                np.expand_dims(fake_nearest_neighbour_distances, axis=0)
        ).any(axis=1).mean()

        density = (1. / float(nearest_k)) * (
                distance_real_fake <
                np.expand_dims(real_nearest_neighbour_distances, axis=1)
        ).sum(axis=0).mean()

        coverage = (
                distance_real_fake.min(axis=1) <
                real_nearest_neighbour_distances
        ).mean()

        return dict(precision=precision, recall=recall,
            density=density, coverage=coverage)

#----------------------------------------------------------------------------

class LegoModelEvaluation(object):
    def __init__(self, v_max, edge_max):
        super(LegoModelEvaluation, self).__init__()

        self.v_max = v_max
        self.edge_max = edge_max

        self.__initialize_eval_metrics_with_GIN()
        self.class_name_mapping = helpers.FileToTarget()

        self.graph_validation = utils.LegoGraphValidation()
        self.helpers = utils.TestingHelpers()
        

    def __initialize_eval_metrics_with_GIN(self):
        include_edge_types = True
        gin = helpers.load_gin()
        config = {'include_augmented': True}
        
        GIN_dataset = helpers.GINDataset(with_edge_types = include_edge_types, config = config)
        self.gin_dataset = GIN_dataset.dataset
        self.dgmg_eval = DGMGEvaluation(GIN_dataset, gin)


    def evaluate_model(self, model, num_samples, dir, classes_to_generate = None, softmax_temperature=1):        
        self.__prepare_evaluation(model, num_samples, dir, classes_to_generate)
        sampled_graphs = self.__generate_graphs(model, num_samples, softmax_temperature)
        with open(os.path.join(self.graph_dir, 'graphs.h5'), 'wb') as f:
            pickle.dump(sampled_graphs, f)
        gin_and_mmd_metrics, lego_metrics = self.__evaluate_graphs(sampled_graphs)
        os.system('zip -r -j {}/ldrs.zip {}/*.ldr'.format(self.path, self.path))
        os.system('rm {}/*.ldr'.format(self.path))

        novel_percentage, novel_and_valid = self.__get_novel_percentage(sampled_graphs)
        lego_metrics['novel %'] = novel_percentage
        lego_metrics['novel and valid %'] = novel_and_valid

        if self.invalid_counter[self.num_classes] == num_samples:
            try:
                os.rmdir(self.path)
            except:
                pass

        return gin_and_mmd_metrics, lego_metrics

    def __get_novel_percentage(self, sampled_graphs):
        novel = 0
        novel_and_valid = 0
        for g_generated in sampled_graphs:
            novel_res = True
            novel_and_valid_res = True
            for g in self.gin_dataset:
                g = g['graph']
                is_same_graph = self.helpers.is_same_graph(g, g_generated)
                if is_same_graph:
                    novel_res = False
                    novel_and_valid_res = False
                elif self.graph_validation.is_valid(g_generated) == False:
                    novel_and_valid_res = False
            novel += novel_res
            novel_and_valid += novel_and_valid_res
        novel /= len(sampled_graphs)
        novel_and_valid /= len(sampled_graphs)
        return novel * 100, novel_and_valid * 100


    def __prepare_evaluation(self, model, num_samples, dir, classes_to_generate):
        assert classes_to_generate is None or len(classes_to_generate) == num_samples, 'len(classes_to_generate) != num_samples, {} and {}'.format(len(classes_to_generate), num_samples)
        model.eval()

        self.__set_num_classes(model)
        self.__set_classes_to_generate(num_samples, classes_to_generate)
        self.__make_results_dir(dir)
        self.missing_implied_edges_isnt_error = model.missing_implied_edges_isnt_error


    def __set_num_classes(self, model):
        if model.class_conditioning == 'None':
            self.num_classes = 0
        else:
            self.num_classes = 12


    def __set_classes_to_generate(self, num_samples, classes_to_generate):
        if classes_to_generate is None:
            classes_to_generate = []
            for i in range(num_samples):
                classes_to_generate.append(i % max(1, self.num_classes))
        self.classes_to_generate = classes_to_generate


    def __make_results_dir(self, dir):
        ldr_dir = os.path.join(dir, 'ldr_files')
        try:
            os.makedirs(ldr_dir)
        except:
            pass
        self.path = ldr_dir
        graph_dir = os.path.join(dir, 'graphs')
        try:
            os.makedirs(graph_dir)
        except:
            pass
        self.graph_dir = graph_dir

    def __generate_graphs(self, model, num_samples, softmax_temperature):
        return model(batch_size = num_samples, v_max = self.v_max, 
            edge_max = self.edge_max, class_to_generate = self.classes_to_generate,
            softmax_temperature=softmax_temperature)
        

    def __evaluate_graphs(self, sampled_graphs):
        self.__initialize_counters()
        list_of_graphs = []
        for i, graph in enumerate(sampled_graphs):
            self.__evaluate_graph(graph, list_of_graphs, self.classes_to_generate[i], i)
        self.__average_counters()
        calculate_GIN_accuracy = self.num_classes > 0
        gin_and_mmd_metrics = self.dgmg_eval.evaluate_all(list_of_graphs, calculate_accuracy = calculate_GIN_accuracy)
        lego_metrics = self.__get_res_dict()
        return gin_and_mmd_metrics, lego_metrics


    def __evaluate_graph(self, g, list_of_graphs, class_num, sample_num):
        helpers.add_node_attributes(g)
        helpers.add_edge_attributes(g)

        for attr in list(g.ndata.keys()):
            if attr != 'attr':
                del g.ndata[attr]
        if class_num >= 8:
            class_num += 1
        list_of_graphs.append({'graph': g, 'target': class_num})
        self.__increment_counters(g, sample_num)


    def __increment_counters(self, sampled_graph, sample_num):
        class_num = sampled_graph.class_num if self.num_classes > 0 else 0

        #Increment total counters (regardless of validity)
        graph_size = sampled_graph.number_of_nodes()
        self.__increment_counter(self.avg_size, graph_size, class_num)

        graph_num_edges = sampled_graph.number_of_edges()
        self.__increment_counter(self.avg_num_edges, graph_num_edges, class_num)

        self.__increment_counter(self.num_samples_per_class, 1, class_num)


        if self.graph_validation.invalid_shift(sampled_graph) == True:
            self.invalid_shifts_ratio += 1

        if self.graph_validation.check_if_brick_overconstrained(sampled_graph) == True:
            self.overconstrained_ratio += 1

        if self.graph_validation.check_if_bricks_merged(sampled_graph) == True:
            self.merged_ratio += 1
            
        if self.graph_validation.check_if_missing_implied_edges(sampled_graph) == True:
            self.implied_edges_error_ratio += 1
            if self.missing_implied_edges_isnt_error == False:
                sampled_graph.set_invalid()

        if graph_size == 0:
            self.zero_nodes_ratio += 1

        if graph_size > 1 and graph_num_edges == 0:
            self.zero_edges_ratio += 1

        if self.graph_validation.is_disjoint(sampled_graph) or (graph_size > 1 and graph_num_edges == 0):
            self.disjoint_ratio += 1

        if 0 < graph_size < 5:
            self.too_small_ratio += 1

        if self.graph_validation.is_valid(sampled_graph) == True:
            self.avg_valid_total_edges += graph_num_edges
            self.avg_valid_total_size += graph_size

            class_name = self.class_name_mapping.get_class_name(class_num) if self.num_classes > 0 else 'sample'
            fileName = '{}/{}_{:04d}.ldr'.format(self.path, class_name, sample_num)
            sampled_graph.write_to_file(fileName)

            if self.graph_validation.check_if_missing_implied_edges(sampled_graph) == True:
                self.valid_but_missing_implied += 1

        else:
            self.__increment_counter(self.invalid_counter, 1, class_num)

    
    def __increment_counter(self, counter, amount, class_num):
        counter[class_num] += amount
        counter[self.num_classes] += amount if self.num_classes > 0 else 0 


    def __average_counters(self):
        for i in range(self.num_classes + 1):
            if self.num_samples_per_class[i] > 0:
                self.avg_size[i]  /= self.num_samples_per_class[i]
                self.avg_num_edges[i] /= self.num_samples_per_class[i]
                self.invalid_ratio[i] = (self.invalid_counter[i] / self.num_samples_per_class[i]) * 100

        self.avg_valid_total_size /= (self.num_samples_per_class[self.num_classes] - self.invalid_ratio[self.num_classes])
        self.avg_valid_total_edges /= (self.num_samples_per_class[self.num_classes] - self.invalid_ratio[self.num_classes])
        self.overconstrained_ratio /= (self.num_samples_per_class[self.num_classes] * 0.01)
        self.merged_ratio /= (self.num_samples_per_class[self.num_classes] * 0.01)
        self.invalid_shifts_ratio /= (self.num_samples_per_class[self.num_classes] * 0.01)
        self.implied_edges_error_ratio /= (self.num_samples_per_class[self.num_classes] * 0.01)
        self.valid_but_missing_implied /= (self.num_samples_per_class[self.num_classes] * 0.01)
        self.zero_nodes_ratio /= (self.num_samples_per_class[self.num_classes] * 0.01)
        self.too_small_ratio /= (self.num_samples_per_class[self.num_classes] * 0.01)
        self.disjoint_ratio /= (self.num_samples_per_class[self.num_classes] * 0.01)
        self.valid_but_too_small /= (self.num_samples_per_class[self.num_classes] * 0.01)
        self.zero_edges_ratio /= (self.num_samples_per_class[self.num_classes] * 0.01)


    def __get_res_dict(self):
        res_dict = {}
        for i in range(self.num_classes + 1):
            if self.num_classes > 0 and i != self.num_classes:
                class_name = self.class_name_mapping.get_class_name(i)
            else:
                class_name = 'Overall'

            res_dict['{} invalid lego build (%)'.format(class_name)] = self.invalid_ratio[i]
            res_dict['{} average number of lego bricks'.format(class_name)] = self.avg_size[i]
            res_dict['{} average number of edges'.format(class_name)] = self.avg_num_edges[i]

        class_name = 'Overall'
        res_dict['{} average valid number of lego bricks'.format(class_name)] = self.avg_valid_total_size
        res_dict['{} average valid number of edge'.format(class_name)] = self.avg_valid_total_edges
        res_dict[ '{} overconstrained brick %'.format(class_name)] = self.overconstrained_ratio
        res_dict['{} merged brick %'.format(class_name)] = self.merged_ratio
        res_dict['{} invalid shift %'.format(class_name)] = self.invalid_shifts_ratio
        res_dict['{} missing implied edges %'.format(class_name)] = self.implied_edges_error_ratio
        res_dict['{} valid but missing implied edges error %'.format(class_name)] = self.valid_but_missing_implied
        res_dict['{} zero nodes %'.format(class_name)] = self.zero_nodes_ratio
        res_dict['{} zero edges %'.format(class_name)] = self.zero_edges_ratio
        res_dict['{} build too small %'.format(class_name)] = self.too_small_ratio
        res_dict['{} disjoint graph %'.format(class_name)] = self.disjoint_ratio

        return res_dict


    def __initialize_counters(self):
        #Extra index is for keeping track of overall total
        self.avg_valid_total_size = 0
        self.avg_valid_total_edges = 0

        self.avg_size = np.zeros(self.num_classes + 1)
        self.avg_num_edges = np.zeros(self.num_classes + 1)
        self.num_samples_per_class = np.zeros(self.num_classes + 1)


        self.invalid_counter = np.zeros(self.num_classes + 1)
        self.invalid_ratio = np.zeros(self.num_classes + 1)
        self.overconstrained_ratio = 0
        self.merged_ratio = 0
        self.invalid_shifts_ratio = 0
        self.implied_edges_error_ratio = 0
        self.valid_but_missing_implied = 0
        self.zero_nodes_ratio = 0
        self.too_small_ratio = 0
        self.disjoint_ratio = 0
        self.valid_but_too_small = 0
        self.zero_edges_ratio = 0

        self.fid = 0
        self.accuracy = 0

class LegoPrinting(object):
    def __init__(self, num_epochs):
        super(LegoPrinting, self).__init__()

        self.num_epochs = num_epochs

    def update(self, epoch, metrics):

        msg = 'epoch {:d}/{:d}'.format(epoch, self.num_epochs)
        for key, value in metrics.items():
            msg += ', {}: {:4f}'.format(key, value)
        print(msg)

