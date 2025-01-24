import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, n_heads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.conv1 = GATConv(n_feat, n_hid, heads=n_heads, dropout=dropout)
        self.conv2 = GATConv(n_hid * n_heads, n_class, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        # Apply dropout to input features
        x = F.dropout(x, self.dropout, training=self.training)
        # Apply first GAT layer and activation function
        x = F.elu(self.conv1(x, edge_index))
        # Apply dropout to hidden features
        x = F.dropout(x, self.dropout, training=self.training)
        # Apply second GAT layer
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Parameters from the original paper
n_feat = 1433  # Number of input features
n_hid = 8  # Number of hidden units per head
n_class = 7  # Number of output classes
dropout = 0.6  # Dropout rate
n_heads = 8  # Number of attention heads

# Instantiate the model
model = GAT(n_feat, n_hid, n_class, dropout, n_heads)
print(model)