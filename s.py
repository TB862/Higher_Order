def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    ##############################################################################################

    subp = os.path.join(os.path.expanduser('~'), 'GRAND_project', 'Datasets', 'Cora')
    dpath = os.path.join(subp, 'processed', 'data.pt')
    data = torch.load(dpath)[0]

    x, y, graph = data['x'], data['y'], data['edge_index']
    if os.path.exists(os.path.join(subp, 'train_mask')):
        train_mask = torch.load(os.path.join(subp, 'train_mask')).cpu()
        val_mask = torch.load(os.path.join(subp, 'val_mask')).cpu()
        test_mask = torch.load(os.path.join(subp, 'test_mask')).cpu()
        print(test_mask.device, val_mask.device)
    elif 'train_mask' in data:
        train_mask, val_mask, test_mask = data['train_mask'], data['val_mask'], data['test_mask']

    idx_train = torch.nonzero(train_mask)
    idx_val = torch.nonzero(val_mask)
    test_idx_reorder = torch.nonzero(test_mask)

    ##############################################################################################
    
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    features = x 

    data = Data(edge_index=graph, num_nodes=torch.max(graph))
    adj = nx.adjacency_matrix(to_networkx(data, to_undirected=True))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = y

    idx_test = test_idx_range.tolist()
    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(torch.from_numpy(features)).float()
    labels = torch.LongTensor(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test