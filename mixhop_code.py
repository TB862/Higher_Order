"""Data reading tools."""

import json
import torch
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from texttable import Texttable

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    return graph

def feature_reader(features):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param path: Path to the JSON file.
    :return out_features: Dict with index and value tensor.
    """
    index_1 = [int(k) for k, v in features.items() for fet in v]
    index_2 = [int(fet) for k, v in features.items() for fet in v]
    values = [1.0]*len(index_1)
    nodes = [int(k) for k, v in features.items()]
    node_count = max(nodes)+1
    feature_count = max(index_2)+1
    features = sparse.coo_matrix((values, (index_1, index_2)),
                                 shape=(node_count, feature_count),
                                 dtype=np.float32)
    out_features = dict()
    ind = np.concatenate([features.row.reshape(-1, 1), features.col.reshape(-1, 1)], axis=1)
    out_features["indices"] = torch.LongTensor(ind.T)
    out_features["values"] = torch.FloatTensor(features.data)
    out_features["dimensions"] = features.shape
    return out_features

def target_reader(path):
    """
    Reading the target vector from disk.
    :param path: Path to the target.
    :return target: Target vector.
    """
    target = torch.LongTensor(np.array(pd.read_csv(path)["target"]))
    return target

def create_adjacency_matrix(graph):
    """
    Creating a sparse adjacency matrix.
    :param graph: NetworkX object.
    :return A: Adjacency matrix.
    """
    index_1 = [edge[0] for edge in graph.edges()] + [edge[1] for edge in graph.edges()]
    index_2 = [edge[1] for edge in graph.edges()] + [edge[0] for edge in graph.edges()]
    values = [1 for index in index_1]
    node_count = max(max(index_1)+1, max(index_2)+1)
    A = sparse.coo_matrix((values, (index_1, index_2)),
                          shape=(node_count, node_count),
                          dtype=np.float32)
    return A

def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sparse.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat

def create_propagator_matrix(graph):
    """
    Creating a propagator matrix.
    :param graph: NetworkX graph.
    :return propagator: Dictionary of matrix indices and values.
    """
    A = create_adjacency_matrix(graph)
    I = sparse.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    propagator = dict()
    A_tilde_hat = sparse.coo_matrix(A_tilde_hat)
    ind = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1)
    propagator["indices"] = torch.LongTensor(ind.T)
    propagator["values"] = torch.FloatTensor(A_tilde_hat.data)
    return propagator

import math
import torch
from torch_sparse import spmm

def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

class SparseNGCNLayer(torch.nn.Module):
    """
    Multi-scale Sparse Feature Matrix GCN layer.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param iterations: Adjacency matrix power order.
    :param dropout_rate: Dropout value.
    """
    def __init__(self, in_channels, out_channels, iterations, dropout_rate):
        super(SparseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Doing a forward pass.
        :param normalized_adjacency_matrix: Normalized adjacency matrix.
        :param features: Feature matrix.
        :return base_features: Convolved features.
        """
        feature_count, _ = torch.max(features["indices"],dim=1)
        feature_count = feature_count + 1

        base_features = spmm(features["indices"], features["values"], feature_count[0],
                             feature_count[1], self.weight_matrix)

        base_features = base_features + self.bias

        base_features = torch.nn.functional.dropout(base_features,
                                                    p=self.dropout_rate,
                                                    training=self.training)

        base_features = torch.nn.functional.relu(base_features)
        for _ in range(self.iterations-1):
            base_features = spmm(normalized_adjacency_matrix["indices"],
                                 normalized_adjacency_matrix["values"],
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)
        return base_features

class DenseNGCNLayer(torch.nn.Module):
    """
    Multi-scale Dense Feature Matrix GCN layer.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param iterations: Adjacency matrix power order.
    :param dropout_rate: Dropout value.
    """
    def __init__(self, in_channels, out_channels, iterations, dropout_rate):
        super(DenseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Doing a forward pass.
        :param normalized_adjacency_matrix: Normalized adjacency matrix.
        :param features: Feature matrix.
        :return base_features: Convolved features.
        """
        base_features = torch.mm(features, self.weight_matrix)
        base_features = torch.nn.functional.dropout(base_features,
                                                    p=self.dropout_rate,
                                                    training=self.training)
        for _ in range(self.iterations-1):
            base_features = spmm(normalized_adjacency_matrix["indices"],
                                 normalized_adjacency_matrix["values"],
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)
        base_features = base_features + self.bias
        return base_features

class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)

class MixHopNetwork(torch.nn.Module):
    """
    MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing.
    :param args: Arguments object.
    :param feature_number: Feature input number.
    :param class_number: Target class number.
    """
    def __init__(self, args, feature_number, class_number):
        super(MixHopNetwork, self).__init__()
        self.args = args
        self.feature_number = feature_number
        self.class_number = class_number
        self.calculate_layer_sizes()
        self.setup_layer_structure()
        self.to(torch.device('cuda:0'))

    def calculate_layer_sizes(self):
        self.abstract_feature_number_1 = sum(self.args.layers_1)
        self.abstract_feature_number_2 = sum(self.args.layers_2)
        self.order_1 = len(self.args.layers_1)
        self.order_2 = len(self.args.layers_2)

    def setup_layer_structure(self):
        """
        Creating the layer structure (3 convolutional upper layers, 3 bottom layers) and dense final.
        """
        self.upper_layers = [SparseNGCNLayer(self.feature_number, self.args.layers_1[i-1], i, self.args.dropout) for i in range(1, self.order_1+1)]
        self.upper_layers = ListModule(*self.upper_layers)
        self.bottom_layers = [DenseNGCNLayer(self.abstract_feature_number_1, self.args.layers_2[i-1], i, self.args.dropout) for i in range(1, self.order_2+1)]
        self.bottom_layers = ListModule(*self.bottom_layers)
        self.fully_connected = torch.nn.Linear(self.abstract_feature_number_2, self.class_number)

    def calculate_group_loss(self):
        """
        Calculating the column losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            upper_column_loss = torch.norm(self.upper_layers[i].weight_matrix, dim=0)
            loss_upper = torch.sum(upper_column_loss)
            weight_loss = weight_loss + self.args.lambd*loss_upper
        for i in range(self.order_2):
            bottom_column_loss = torch.norm(self.bottom_layers[i].weight_matrix, dim=0)
            loss_bottom = torch.sum(bottom_column_loss)
            weight_loss = weight_loss + self.args.lambd*loss_bottom
        return weight_loss

    def calculate_loss(self):
        """
        Calculating the losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            loss_upper = torch.norm(self.upper_layers[i].weight_matrix)
            weight_loss = weight_loss + self.args.lambd*loss_upper
        for i in range(self.order_2):
            loss_bottom = torch.norm(self.bottom_layers[i].weight_matrix)
            weight_loss = weight_loss + self.args.lambd*loss_bottom
        return weight_loss
            
    def forward(self, features, norm_adj):
        normalized_adjacency_matrix = norm_adj
        """
        Forward pass.
        :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
        :param features: Feature matrix.
        :return predictions: Label predictions.
        """

        abstract_features_1 = torch.cat([self.upper_layers[i](normalized_adjacency_matrix, features) for i in range(self.order_1)], dim=1)
        abstract_features_2 = torch.cat([self.bottom_layers[i](normalized_adjacency_matrix, abstract_features_1) for i in range(self.order_2)], dim=1)
        predictions = torch.nn.functional.log_softmax(self.fully_connected(abstract_features_2), dim=1)

        return predictions
