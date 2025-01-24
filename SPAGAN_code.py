import numpy as np
import scipy.sparse as sp
import torch
import pickle
import time
import os
#import graph_tool.topology as g_topo
#import graph_tool.generation as g_gen
import networkx as nx
import sys

import math
import numpy as np
import time
import os

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_sparse import spmm
from torch_scatter import scatter_add, scatter_max

import scipy
from torch_geometric.utils import convert
import json 
#from torch_sparse import spspmm
#from torch_scatter import scatter_max
#from torch_geometric.utils import softmax

# torch10py37
# using export CUDA_LAUNCH_BLOCKING=1 will decreace the speed from 0.03->0.045
    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, layerN=''):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.layerN = layerN

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        
        self.bias = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.bias.data, gain=1.414)

        # attention param for path
        self.pathW = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.pathW.data, gain=1.414)
        
        self.pathbias = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.pathbias.data, gain=1.414)
        
        self.patha = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.patha.data, gain=1.414)

        self.patha_2 = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.patha_2.data, gain=1.414)

        self.patha_3 = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.patha_3.data, gain=1.414)
        
        self.pathMerge = nn.Parameter(torch.zeros(size=(2*out_features, out_features)))
        nn.init.xavier_normal_(self.pathMerge.data, gain=1.414)
        
        self.lenAtt = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.lenAtt.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)


    def gat_layer(self, input, adj, genPath=False, eluF=True):
        N = input.size()[0]
        edge = adj._indices()

        h = torch.mm(input, self.W)
        h = h+self.bias                # h: N x out

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()     # edge_h: 2*D x E
        edge_att = self.a.mm(edge_h).squeeze()
        edge_e_a = self.leakyrelu(edge_att)     # edge_e_a: E   attetion score for each edge
        if genPath:
            with torch.no_grad():
                edge_weight = edge_e_a
                p_a_e = edge_weight - scatter_max(edge_weight, edge[0,:], dim=0, dim_size=N)[0][edge[0,:]]
                p_a_e = p_a_e.exp()
                p_a_e = p_a_e / (scatter_add(p_a_e, edge[0,:], dim=0, dim_size=N)[edge[0,:]]\
                                    +torch.Tensor([9e-15]).cuda())
                
                scisp = convert.to_scipy_sparse_matrix(edge, p_a_e, N)
                scipy.sparse.save_npz(os.path.join(genPath, 'attmat_{:s}.npz'.format(self.layerN)), scisp)

        edge_e = torch.exp(edge_e_a - torch.max(edge_e_a))                  # edge_e: E
        e_rowsum = spmm(edge, edge_e, N, N, torch.ones((N, 1), device='cuda'))
        edge_e = self.dropout(edge_e)       # add dropout improve from 82.4 to 83.8
        # edge_e: E
        
        h_prime = spmm(edge, edge_e, N, N, h)
        h_prime = h_prime.div(e_rowsum+torch.Tensor([9e-15]).cuda())        # h_prime: N x out
        
        if self.concat and eluF:
            return F.elu(h_prime)
        else:
            return h_prime


    def pathat_layer(self, input, pathM, pathlens, eluF=True):
        N = input.size()[0]
        pathh = torch.mm(input, self.pathW)
        pathh = pathh+self.pathbias                # h: N x out
        
        if not self.concat:  # if the last layer
            pathlens = [2]
        
        pathfeat_all = None
        for pathlen_iter in pathlens:
            i = pathM[ pathlen_iter ]['indices']
            v = pathM[ pathlen_iter ]['values']
            featlen = pathh.shape[1]
            pathlen = v.shape[1]
            pathfeat = tuple( (pathh[v[:,i], :] for i in range(1,pathlen)) )
            pathfeat = torch.cat(pathfeat, dim=1)
            pathfeat = pathfeat.view(-1,pathlen-1,featlen)
            pathfeat, _ = torch.max(pathfeat, dim=1)    # seems max is better?
            #pathfeat = torch.mean(pathfeat, dim=1)     #
            att_feat = torch.cat( (pathfeat, pathh[i[0,:],:]), dim=1 ).t()
            if pathlen_iter==2:
                path_att = self.leakyrelu(self.patha_2.mm(att_feat).squeeze())
            else:
                path_att = self.leakyrelu(self.patha_3.mm(att_feat).squeeze())    
            # softmax of p_a -> p_a_e
            path_att = path_att - scatter_max(path_att, i[0,:], dim=0, dim_size=N)[0][i[0,:]]
            path_att = path_att.exp()
            path_att = path_att / (scatter_add(path_att, i[0,:], dim=0, dim_size=N)[i[0,:]] \
                                    + torch.Tensor([9e-15]).cuda())
            path_att = path_att.view(-1,1)
            path_att = self.dropout(path_att)         # add dropout here of p_a_e
            w_pathfeat = torch.mul(pathfeat, path_att)
            h_path_prime = scatter_add(w_pathfeat, i[0,:], dim=0)
            # h_path_prime is the feature embedded from paths  N*feat
            
            if pathfeat_all is None:
                pathfeat_all = h_path_prime
            else:
                pathfeat_all = torch.cat((pathfeat_all, h_path_prime), dim=0)

        if len(pathlens)==2:
            leni = torch.tensor(np.array(list(range(N))+list(range(N)))).cuda()
            
            att_feat = torch.cat( (pathfeat_all, pathh[leni,:]), dim=1 ).t()
            path_att = self.leakyrelu(self.lenAtt.mm(att_feat).squeeze())
            # softmax of p_a -> p_a_e
            path_att = path_att - scatter_max(path_att, leni, dim=0, dim_size=N)[0][leni]
            path_att = path_att.exp()
            path_att = path_att / (scatter_add(path_att, leni, dim=0, dim_size=N)[leni] \
                                    + torch.Tensor([9e-15]).cuda())
            path_att = path_att.view(-1,1)
            # path_att = self.dropout(path_att)         # add dropout here of p_a_e
            w_pathfeat = torch.mul(pathfeat_all, path_att)
            h_path_prime = scatter_add(w_pathfeat, leni, dim=0)

        if self.concat and eluF:
            return F.elu( h_path_prime )
        else:
            return h_path_prime

    def forward(self, input, adj, pathM, pathlens=[2], genPath=False, mode='GAT'): 
        if not self.concat:     # if the last layer
            pathM = {}
            pathM[2] = {}
            pathM[2]['indices'] = adj._indices()
            pathM[2]['values'] = adj._indices().transpose(1,0)
        
        if mode=="GAT":
            return self.gat_layer(input, adj, genPath=genPath)
        elif mode=="SPAGAN":
            return self.pathat_layer(input, pathM=pathM, pathlens=pathlens)

class SpaGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpaGAT, self).__init__()
        self.dropout = dropout
        
        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True, layerN='1_'+str(i)) for i in range(nheads[0])]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        self.out_atts = [SpGraphAttentionLayer(nhid * nheads[0], 
                                             nclass,
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False, layerN='2_'+str(i))  for i in range(nheads[1])]

        for i, out_att in enumerate(self.out_atts):
            self.add_module('out_att_{}'.format(i), out_att)

    def forward(self, x, adj, pathM=None, pathlens=[2,3], genPath=False, mode='GAT', ret_attn=False):
        # mode can be GAT, PathAT, Combine
        # pathM: layer, head, pathLen
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, (None if pathM is None else pathM[0][0]), pathlens, genPath, mode) \
                    for head, att in enumerate(self.attentions)], dim=1)
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.stack([out_att(x, adj, (None if pathM is None else pathM[0][0]), pathlens, genPath, mode) \
                    for head, out_att in enumerate(self.out_atts)], dim=2)
        
        x = torch.mean(x, dim=2)
        x = F.elu(x)
        p = F.log_softmax(x, dim=1)
        
        return p

    def reset_parameters(self):
        pass 


def single_gen_path(pathRes, shape, pathLen=3, Nratio=0.6, Ndeg=2):
    cols = []
    rows = []
    pathValues = []

    for row in pathRes.keys():
        degree = 0
        tmppath = []; tmplen = []; tmpcols = []; tmprows = []
        dists = pathRes[row][0]
        paths = pathRes[row][1]
        for col in paths.keys():
            path = paths[col]
            if len(path)==pathLen:
                tmpcols.append(col)
                tmprows.append(row)
                tmplen.append(dists[col])
                tmppath.append(path)
            if len(path)==2:
                degree += 1
        if pathLen==2:
            ratio = 1.0
            maxRange = int(ratio*len(tmplen))
        else:
            ratio = Nratio
            maxRange = int( min([degree*Ndeg ,int(ratio*len(tmplen))]) )
            
        topk = sorted( range(len(tmplen)), key=lambda i:tmplen[i])[ 0: maxRange]
        
        pathValues = pathValues +  [tmppath[i] for i in topk]
        
        cols = cols + [tmpcols[i] for i in topk]
        rows = rows + [tmprows[i] for i in topk]

    cols = cols + [i for i in range(shape[0])]
    rows = rows + [i for i in range(shape[0])]
    pathValues = pathValues + [[j]*pathLen for j in range(shape[0])]

    cols = np.array(cols).reshape(1,-1)
    rows = np.array(rows).reshape(1,-1)
    pathValues = np.array(pathValues)
    i = np.concatenate((rows, cols), axis=0)
    i = torch.LongTensor(i)
    
    pathV = torch.FloatTensor(pathValues)
    
    pathM = torch.sparse.FloatTensor(i, pathV, torch.Size([shape[0], shape[1], pathLen]))
    
    return pathM, 0


def gen_pathm(nheads=[8], ds_name=None, Nratio=0.6, Ndeg=2):     
    # nheads=[8]: does not generate for the last layer
    # generate path matrix for each attention head
    # pathM is a dict of dict of dict, each element is a pytorch sparse matrix
    #   the first key is which 'layer' (start from 0)
    #   the second key is which 'head'  (start from 0)
    #   the third key is which 'path length' (start from 2)

    pathM = {}
    layerPathM = {}
    for layer, nhead in enumerate(nheads):
        #if layer>0:
        #    continue
        headPathM = {}
        headMatrix = None
        for head in range(nhead):
            spmatrix = sp.load_npz(f'{ds_name}_adj.npz')

            #if not matpath:                                                                    
            #    spmatrix = sp.load_npz('')                                                     
            #else:                                                                              
            #    spmatrix = sp.load_npz(matpath+'/attmat_{:d}_{:d}.npz'.format(layer+1, head))   

            if headMatrix is None:
                headMatrix = spmatrix
            else:
                headMatrix.data += spmatrix.data

        headMatrix.data = headMatrix.data / nhead
        headMatrix.data = -headMatrix.data
        headMatrix.setdiag(0)       # not include self attention weight

        attMin = min(headMatrix.data)
        attMax = max(headMatrix.data)
        headMatrix.data = (headMatrix.data - attMin)/(attMax-attMin)
        
        print(min(headMatrix.data), max(headMatrix.data))
        
        G = nx.from_scipy_sparse_array(headMatrix)
        t = time.time()
        pathRes = dict(nx.all_pairs_dijkstra(G, cutoff=3))
        #pathRes = graph_tool_apsp(headMatrix, cutoff=3)
        print('shortest path time:', time.time()-t)
        
        # single_path: sparse matrix N*N*pathLen
        t = time.time()
        lenPathM = {}
        for pathLen in range(2,4):  # generate path 2,3,...
            indexValue = {}
            single_path, _ = single_gen_path(pathRes, headMatrix.shape, pathLen=pathLen, Nratio=Nratio, Ndeg=Ndeg)
            indexValue['indices'] = single_path._indices()
            indexValue['values'] = single_path._values()
            print("pathlen:{:d}, #{:d}".format(pathLen, indexValue['indices'].shape[1]))
            lenPathM[pathLen] = indexValue
        headPathM[0] = lenPathM
        print('generate path time:', time.time()-t)
        
        layerPathM[layer] = headPathM
        
    pathM = layerPathM

    return pathM