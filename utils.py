import os 
import sys 

sys.path.append(os.path.abspath(os.path.join('grand_src')))

import torch_geometric.nn.models as models
from torch_geometric.nn.models import GCN, GraphSAGE, GIN, GAT
import torch_geometric.datasets as datasets
from torch_geometric.loader import DataLoader
from ml_collections import ConfigDict
from typing import *
import numpy as np 
import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.typing import Tensor, Adj
from torchmetrics import Accuracy, AUROC
from hop_utils import get_K_adjs, get_HO_laplacian
from grand_src.GNN import GNN
from grand_src.best_params import best_params_dict
from grand_src.run_GNN import merge_cmd_args
import torch_geometric.transforms as T
from torch_geometric.data import Data
from itertools import product 
from torch_geometric.transforms import Compose, NormalizeFeatures, RandomNodeSplit
from torch_geometric.data import InMemoryDataset, Data 
from functools import partial 
from HiGCN.node_classify.utils.param_utils import get_net
from ml_collections import ConfigDict

from torch_geometric.utils import add_self_loops, remove_self_loops

#from spmpnn.src.utils.model_loader import get_model

#from PathNNs_expressive.benchmarks.model_ogb import EdgePathNN
#from PathNNs_expressive.benchmarks.preprocess_data import PathTransform

#from gnn_magna.codes.MAGNA import MAGNA
#from gnn_magna.graphUtils.gutils import reorginize_self_loop_edges
#from spmpnn.src.utils.shortest_paths import ShortestPathTransform

from nodeformer import NodeFormer  
#from model_gtn import GTN 
from models import get_model 

from gnns import * 

from multi_hop import GenericGAT
from copy import copy
import torch

from SPAGAN_code import * 
from mixhop_code import * 


best_params_dict['Actor'] = best_params_dict['Citeseer']
best_params_dict['Physics'] = best_params_dict['Pubmed']
best_params_dict['CS'] = best_params_dict['CoauthorCS']
best_params_dict['Photo'] = best_params_dict['Pubmed']
best_params_dict['Computers'] = best_params_dict['Pubmed']
best_params_dict['Wisconsin'] = best_params_dict['CoauthorCS']
best_params_dict['Texas'] = best_params_dict['CoauthorCS']
best_params_dict['Chameleon'] = best_params_dict['CoauthorCS']
best_params_dict['Squirrel'] = best_params_dict['CoauthorCS']


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False)

    def forward(self, x, edge_index, return_attention_weights=False):
        
        if return_attention_weights:
            x, attn_weights1 = self.conv1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x, attn_weights2 = self.conv2(x, edge_index, return_attention_weights=True)
            return x, attn_weights2
        else:
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
            return x
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


class CombinedLoader:
    def __init__(self, *loaders):
        self.length = np.Inf
        self.iters = []
        self.loaders = loaders[0]
        self.dataset = self.loaders[0].dataset 

        for loader in self.loaders:
            self.length = min(self.length, len(loader))

    def gen(self, loader, edge_set):
        for item in loader:
            item.edge_index = edge_set      # this was broken before 
            yield item

    def __iter__(self):
        self.iters = [] 
        for loader in self.loaders:
            gen = self.gen(loader, loader.dataset.edge_index)
            self.iters.append(gen)

        return self

    def __next__(self):
        items = [] 
        for n, it in enumerate(self.iters):
            next_item = next(it)
            items.append(next_item)

        return self.length
        

def to_numpy(x: Tensor):
    return x.cpu().detach().numpy()


def get_ds_config(config: ConfigDict, ds_name: str) -> ConfigDict:
    if ds_name == 'Pubmed':
        ds_config = config.pubmed
    elif ds_name == 'Cora':
        ds_config = config.cora 
    elif ds_name == 'Citeseer':
        ds_config = config.citeseer
    elif ds_name == 'CS':
        ds_config = config.cs
    elif ds_name == 'Physics':
        ds_config = config.physics
    elif ds_name == 'Computers':
        ds_config = config.computers
    elif ds_name == 'Photo':
        ds_config = config.photo 
    elif ds_name == 'Actor':
        ds_config = config.actor 
    elif ds_name == 'Wisconsin':
        ds_config = config.wisconsin 
    elif ds_name == 'Texas':
        ds_config = config.texas
    elif ds_name == 'Chameleon':
        ds_config = config.chameleon 
    elif ds_name == 'Squirrel':
        ds_config = config.squirrel
    else:
        raise ValueError('Invalid dataset name')

    return ds_config


def get_metric_functions(ds_config, device):
    metric_names = ds_config.experiments.metrics
    callables = {}
    for mn in metric_names:
        if mn == 'Accuracy':
            callables[mn] = Accuracy(task="multiclass", num_classes=ds_config.num_classes).to(device)
        elif mn == 'AUC':
            callables[mn] = AUROC(task="multiclass", num_classes=ds_config.num_classes).to(device)
        else:
            raise ValueError()

    return callables


def load_from_checkpoint(config: ConfigDict, ds_name: str, nrepeats: int) -> Dict[AnyStr, Any]:   
    """
        Loads in the metrics of models from checkpoint 
    """

    model_names = config.baselines.names
    metric_names = config[ds_name.lower()].experiments.metrics
    metric_dict = {mn: {key: [] for key in metric_names} for mn in model_names}

    for mn, mi in product(model_names, range(nrepeats)):                                
        full_path = os.path.join('checkpoints', ds_name, mn+str(mi), 'metrics.txt')
        if not os.path.exists(full_path):
            print(f'Metric file path {full_path} not found')
            continue
        # read into metric dict 
        with open(full_path, 'r') as reader:
            for mline in reader.readlines():
                mline = mline.split(' ')
                key = str(mline[0])
                val = float(mline[1])
                metric_dict[mn][key].append(val) 

    return metric_dict 


def save_to_checkpoint(model: nn.Module, save_dir: str):
    """
        Saves the model
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model, os.path.join(save_dir, 'model.pt'))


def get_args(batch, device):
    features, labels = batch.x.to(device), batch.y.to(device)
    masks = {'train': batch.train_mask, 'val': batch.val_mask, 'test': batch.test_mask}
    call_args = {'x': features, 'edge_index': batch.edge_index}

    if hasattr(batch, 'edge_weights'):
        call_args['edge_weights'] = batch.edge_weights.to(device)
    if hasattr(batch, 'HL'):
        call_args['HL'] = batch.HL
    if hasattr(batch, 'pathM'):
        call_args['adj'] = batch.adj 
        call_args['pathM'] = batch.pathM 
        del call_args['edge_index']
    if hasattr(batch, 'adjs'):
        call_args['adjs'] = batch.adjs 
        del call_args['edge_index']
        # assign from train loop 
    if hasattr(batch, 'norm_adj'):
        batch.norm_adj['indices'] = batch.norm_adj['indices'].to(device)
        batch.norm_adj['values'] = batch.norm_adj['values'].to(device)
        batch.feature_dict['indices'] = batch.feature_dict['indices'].to(device)
        batch.feature_dict['values'] = batch.feature_dict['values'].to(device)

        call_args['norm_adj'] = batch.norm_adj
        call_args['features'] = batch.feature_dict
        del call_args['edge_index']
        del call_args['x']
    
    return call_args, labels, masks 


def collect_metrics(model, data, callables: Dict[AnyStr, Callable], device) -> Dict[AnyStr, float]:
    """
        Collects metrics from a series of callable metric functions
    """
    
    metrics = {kw: 0 for kw in callables}
    model.eval()

    call_args, labels, masks = get_args(data, device)
    preds = model(**call_args)
    for it_nm, item_func in callables.items():
        stat = to_numpy(item_func(preds[masks['test']], labels[masks['test']]))
        metrics[it_nm] = stat 

    return metrics
    

def join_metrics(existing_dict: Dict[AnyStr, List], new_dict: Dict[AnyStr, float], model_name: str):
    """
        Adds the new metric dictionary to a collection of metrics of all previous runs
    """

    for sn in new_dict:
        if sn not in existing_dict[model_name]:
            existing_dict[model_name][sn] = []
        existing_dict[model_name][sn].append(new_dict[sn])

    return existing_dict


def assign_to_config(config, opt, training=True):
    for key in opt:
        best_param = opt[key]
        if isinstance(best_param, ConfigDict):
            print(key)
            continue 
        if key in config and config[key] is not None:
            del config[key]

        config[key] = best_param 

    if hasattr(opt, 'decay') and training:
        config.training.weight_decay = opt['decay'] 
        config.training.lr = opt['lr']
        config.dropout = opt['dropout']
    
    
def create_models(config: ConfigDict, ds_config: ConfigDict, data_dict) -> Dict[AnyStr, List[nn.Module]]:
    """
        Given some dataset, will create number and type of models specified in the config file, packed into a dictionary
    """

    in_chnls = ds_config.in_channels        # data
    hdn_chnls = ds_config.hidden_channels
    num_classes = ds_config.num_classes
    ds_name = ds_config.name
    dataset = data_dict['dataset']
    num_nodes = data_dict['num_nodes']
    device = config.device

    # create model
    baselines = {}
    for mn in config.baselines.names:
        if mn == 'GIN':
            model_config = config.baselines.GIN
            dropout = model_config.drop_out
            num_layers = model_config.num_layers

            model = GIN(in_chnls, hdn_chnls, num_layers, num_classes, dropout=dropout)

        elif mn == 'Graph Sage':
            model = GraphSAGE(in_chnls, hdn_chnls, num_layers, num_classes)
        
        elif mn == 'GAT':
            model_config = config.baselines.GAT
            dropout = model_config.drop_out
            num_layers = model_config.num_layers
            num_heads = model_config.num_heads 

            #model =  GenericGAT(config.baselines.GAT, ds_config, device=device, layer_type='normal')
            model = GAT(in_chnls, hdn_chnls, out_channels=num_classes, heads=8)

        elif mn == 'GCN':
            model_config = config.baselines.GCN
            dropout = model_config.drop_out 
            
            model = GCN(in_chnls, hdn_chnls, num_layers=model_config.num_layers, out_channels=num_classes, dropout=dropout)

        elif mn == 'GRAND':   
            best_opt = best_params_dict[ds_config.name]
            model_config = config.baselines.GRAND
            assign_to_config(model_config, best_opt)

            model = lambda: GNN(model_config, dataset, device)
        
        elif mn == 'MultiHop_GRAND':
            best_opt = best_params_dict[ds_config.name]
            model_config = config.baselines.MultiHop_GRAND

            assign_to_config(model_config, config.baselines.GRAND, training=False)
            assign_to_config(model_config, best_opt)

            model_config['block'] = 'attention'
            dataset = data_dict['multihop_dataset']

            dataset.edge_index = dataset.data.edge_index = dataset.edge_index[0][0]

            model = lambda: GNN(model_config, dataset, device)

        elif mn in ['HiGCN', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN', 'BernNet']:
            model_config = config.baselines.HiGCN
            net_init = get_net(mn)
            if mn == 'HiGCN':
                model = lambda: net_init(dataset, model_config)
            else:
                model = net_init(dataset, model_config)

        elif mn == 'SPAGAN':
            model_config = config.baselines.SPAGAN
            model =  lambda: SpaGAT(
                                        nfeat=ds_config.in_channels,
                                        nhid=model_config.hidden,
                                        nclass=ds_config.num_classes,
                                        dropout=model_config.dropout,
                                        nheads=model_config.nheads,
                                        alpha=model_config.alpha
            )

        elif mn == 'SP-MPNN':
            pass

            #model = get_model(config.baselines.SP_MPNN, ds_name, device, num_classes=num_classes, num_features=in_chnls)
        #elif mn.startswith('PathNN'):
        #    if mn.endswith('SP'):
        #        model = get_path_nn(config.baselines.Path_NN, device, dataset, 'SP')
        #    elif mn.endswith('SP+'):
        #        model = get_path_nn(config.baselines.Path_NN, device, dataset, 'SP+')
        #    elif mn.endswith('AP'):
        #        model = get_path_nn(config.baselines.Path_NN, device, dataset, 'AP')
        #    else:
        #        raise ValueError('Unsupported PathNN model')


        elif mn == 'NodeFormer':
            args = config.baselines.NodeFormer
            model = NodeFormer(data_dict['num_features'], args.hidden_channels, data_dict['num_classes'], num_layers=args.num_layers, dropout=args.dropout,
                    num_heads=args.num_heads, use_bn=args.use_bn, nb_random_features=args.M,
                    use_gumbel=args.use_gumbel, use_residual=args.use_residual, use_act=args.use_act, use_jk=args.use_jk,
                    nb_gumbel_sample=args.K, rb_order=args.rb_order, rb_trans=args.rb_trans).to(device)

        elif mn =='GraphTransformer':
            model = lambda: GTN(        
                                num_edge=len(dataset.data.edge_index),
                                num_channels=64,
                                w_in=64,
                                w_out=ds_config.num_nodes,
                                num_class=num_classes,
                                num_layers=2,
                                num_nodes=num_nodes
                        )

        elif mn == 'MultiHop_GAT':
            num_nodes = data_dict['num_nodes']
            model = GenericGAT(config.baselines.MultiHop_GAT, ds_config, device=device, num_nodes=num_nodes, layer_type='multi_hop')

        elif mn == 'lp':
            args = config.baselines.NodeFormer
            mult_bin = False
            model = MultiLP(data_dict['num_classes'], args.lp_alpha, args.hops, mult_bin=mult_bin)

        elif mn == 'MixHop':
            args = config.baselines.MixHop
            model = lambda: MixHopNetwork(args, ds_config.in_channels, ds_config.num_classes)

        elif mn == 'mixhop':
            args = config.baselines.NodeFormer
            model = MixHop(ds_config.in_channels, 8,  ds_config.num_classes, num_layers=args.num_layers,
                        dropout=args.dropout, hops=3).to(device)
            
        elif mn == 'gcnjk':
            args = config.baselines.NodeFormer
            model = GCNJK(data_dict['num_features'], args.hidden_channels, data_dict['num_classes'], num_layers=args.num_layers,
                            dropout=args.dropout, jk_type=args.jk_type).to(device)
            
        elif mn == 'gatjk':
            args = config.baselines.NodeFormer
            model = GATJK(data_dict['num_features'], args.hidden_channels, data_dict['num_classes'], num_layers=args.num_layers,
                            dropout=args.dropout, heads=args.gat_heads,
                            jk_type=args.jk_type).to(device)
            
        elif mn == 'h2gcn':
            args = config.baselines.NodeFormer
            model = H2GCN(data_dict['num_features'], args.hidden_channels, data_dict['num_classes'], data_dict['graph'],
                            data_dict['num_nodes'],
                            num_layers=args.num_layers, dropout=args.dropout,
                            num_mlp_layers=args.num_mlp_layers).to(device)

        else:
            raise ValueError('Invalid/unsupported model name')

        baselines[mn] = model

    return baselines
    

def gen(num_layers, num_heads):
    for layer_idx in range(num_layers):
        heads = num_heads[layer_idx]
        for head_idx in range(heads):
            yield layer_idx, head_idx 


def mask_transform(data, **masks):
    data.train_mask = masks['train']
    data.val_mask = masks['val']
    data.test_mask = masks['test']

    return data
    

### Node former code 

def adj_mul(adj_i, adj, N):
    adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
    adj_j = adj_j.coalesce().indices()

    return adj_j

def adj_mats(edge_index, num_nodes, order):
    adjs = []
    adj, _ = remove_self_loops(edge_index)
    adj, _ = add_self_loops(adj, num_nodes=num_nodes)
    adjs.append(adj)
    for i in range(order - 1): 
        adj = adj_mul(adj, adj, num_nodes)
        adjs.append(adj)
    
    return adjs 

### End 

def spcoo_to_sptensor(cmat, device):
    rows = torch.tensor(cmat.row, dtype=torch.long)
    cols = torch.tensor(cmat.col, dtype=torch.long)
    values = torch.tensor(cmat.data, dtype=torch.float32) 

    indices = torch.stack([rows, cols])
    torch_sparse_tensor = torch.sparse_coo_tensor(indices, values, cmat.shape)
    torch_sparse_tensor = torch_sparse_tensor.to(device)

    return torch_sparse_tensor

def edge_index_to_nx(edge_index):
    graph = nx.Graph()
    edges = edge_index.t().tolist()
    graph.add_edges_from(edges)

    return graph

def index_mask(ds, i=0):
    ds.train_mask = ds[0].train_mask[:,i]
    ds[0].train_mask = ds[0].train_mask[:,i]
    ds.val_mask = ds[0].val_mask[:,i]
    ds[0].val_mask = ds[0].val_mask[:,i]
    ds.test_mask = ds[0].test_mask[:,i]
    ds[0].test_mask = ds[0].test_mask[:,i]

def fetch_dataset(config: ConfigDict, ds_name: str, unpack: bool = False, add_args=dict({})) -> Union[Dict[AnyStr, Any], List]:
    """
        Downloads (if not present at path) a particular dataset, and returns unpacked dictionary
    """

    device = config.device
    ds_path = os.path.join('Datasets', ds_name) 

    train_path, val_path, test_path = os.path.join(ds_path, 'train_mask'), os.path.join(ds_path, 'val_mask'), os.path.join(ds_path, 'test_mask')
    create_masks = (np.sum([os.path.exists(train_path), os.path.exists(val_path), os.path.exists(test_path)]) != 3)
    
    transform = [NormalizeFeatures()]
    if not create_masks:        
        train_mask, val_mask, test_mask = torch.load(train_path), torch.load(val_path), torch.load(test_path)
        transform.append(partial(mask_transform, train=train_mask, val=val_mask, test=test_mask))
    transform = Compose(transform)                  

    if ds_name in ['Pubmed', 'Cora', 'Citeseer']:   # fixed split datasets 
        ds = datasets.Planetoid(root=ds_path, name=ds_name, transform=NormalizeFeatures()).to(device)
    else:
        if ds_name in ['CS', 'Physics']:
            ds = datasets.Coauthor(root=ds_path, name=ds_name, transform=transform).to(device)
        elif ds_name in ['Photo', 'Computers']:
            ds = datasets.Amazon(root=ds_path, name=ds_name, transform=transform).to(device)
        elif ds_name in ['Actor']:
            ds = datasets.Actor(root=ds_path, transform=transform).to(device)
        elif ds_name in ['Texas', 'Wisconsin']:
            ds = datasets.WebKB(root=ds_path, name=ds_name).to(device)
            index_mask(ds)
        elif ds_name in ['Squirrel', 'Chameleon']:
            ds = datasets.WikipediaNetwork(root=ds_path, name=ds_name).to(device)
            index_mask(ds)
        else:   
            raise ValueError('Invalid dataset name')  

    if ds_name not in ['Texas', 'Wisconsin', 'Squirrel', 'Chameleon']:
        ds.train_mask = ds[0].train_mask
        ds.val_mask = ds[0].val_mask
        ds.test_mask = ds[0].test_mask

    torch.save(ds[0].train_mask, train_path)
    torch.save(ds[0].val_mask, val_path)
    torch.save(ds[0].test_mask, test_path)

    ds_config = get_ds_config(config, ds_name)
    batch_size = ds_config.training.batch_size
    max_cutoff = ds_config.path.max_cutoff
    num_classes = ds_config.num_classes
    model_names = config.baselines.names

    data = {
                'dataset': ds,
                'graph': ds.edge_index,
                'train': ds[0].train_mask,
                'test': ds[0].test_mask,
                'val': ds[0].val_mask,
                'num_classes': num_classes,  
                'num_features': ds[0].num_features, 
                'num_nodes': ds_config.num_nodes,
                'name': ds_config.name 
            }
    
    if 'MixHop' in model_names:
        graph = edge_index_to_nx(data['graph'])
        norm_adj = create_propagator_matrix(graph)
        feature_dict = {str(i): torch.nonzero(ds.x[i]).squeeze(1).tolist() for i in range(ds.x.shape[0])}
        feature_dict = feature_reader(feature_dict)
        data['dataset'].norm_adj = data['dataset'].data.norm_adj = norm_adj 
        data['dataset'].feature_dict = data['dataset'].data.feature_dict = feature_dict 
    if 'HiGCN' in model_names:
        hl_path = os.path.join(ds_path, 'HiGCN_Lap.pt')
        if not os.path.exists(hl_path):
            ho_lap = get_HO_laplacian(data['graph'], data['num_nodes'])
            torch.save(ho_lap, hl_path)
        else:
            ho_lap = torch.load(hl_path)
        
        data['dataset'].HL = data['dataset'].data.HL = ho_lap

    if 'SPAGAN' in model_names:
        sp_coo_mat = sp.load_npz(f'{ds_name}_adj.npz')
        sp_coo_tensor = spcoo_to_sptensor(sp_coo_mat, device)
        pathm = gen_pathm(nheads=[1], ds_name=ds_name)     
        data['dataset'].adj = data['dataset'].data.adj = sp_coo_tensor 
        data['dataset'].pathM = data['dataset'].data.pathM = pathm 

    if 'MultiHop_GAT' in model_names or 'MultiHop_GRAND' in model_names:        # should reference model dict here 
        model_config = config.baselines.MultiHop_GAT
        K_hop = model_config.K_hops
        load_samples = model_config.load_samples

        num_heads = model_config.num_heads 
        num_layers = model_config.num_layers 

        data['multihop_dataset'] = [[] for _ in range(num_layers)] 
        for layer_idx, head_idx in gen(num_layers, num_heads):
            if layer_idx >= 1:                                                 # hard code since I know only ony adj list is needed 
                continue 
            nlayer_heads = num_heads[layer_idx]
            save_path = os.path.join(ds_config.save_path, model_config.select_method, str(layer_idx), str(head_idx))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if load_samples:
                edges = [] 
            else:
                edges = get_K_adjs(
                                    ds.edge_index,
                                    model_config, 
                                    ds_config, 
                                    feature_set=ds.x.to(device), 
                                    device=device
                ) 

            # populate for embedding component of model
            for k in range(K_hop):
                if k == 0:
                    if load_samples:
                        edges.append(ds.edge_index.to(torch.int64))
                    continue 
                path = os.path.join(os.getcwd(), save_path, str(k))
                print('save path:', path)
                if load_samples:
                    edges.append(torch.load(path).to(device).to(torch.int64))
                else:
                    torch.save(edges[k].to(torch.int64), os.path.abspath(path))

            multihop_data, num_nodes = [], []
            for k, edge_set in enumerate(edges):    
                multihop_data.append(edge_set[:,0:edges[0].shape[-1]])        # hard code for extra dimension in odd numbered edge index 

            data['multihop_dataset'][layer_idx].append(multihop_data)

        for l in range(len(data['multihop_dataset'])):
            if l == 0:
                continue 
            headl = min(num_heads[l], num_heads[0])
            data['multihop_dataset'][l] = data['multihop_dataset'][0][0:headl]
        
        cp_dataset = copy(data['dataset'])
        cp_dataset.edge_index = cp_dataset.data.edge_index = data['multihop_dataset']
        data['multihop_dataset'] = cp_dataset

    if 'NodeFormer' in model_names:
        rb_order = config.baselines.NodeFormer.rb_order
        data['dataset'].data.adjs = adj_mats(data['graph'], data['num_nodes'], rb_order)
    data = ConfigDict(data)

    if unpack:
        return [data[key] for key in data]
    else:
        return data

