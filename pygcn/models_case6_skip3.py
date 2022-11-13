import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

# gcn layer varies from 2 to 7 for HW3-1
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nhid)
        self.gc7 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))+x
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))+x
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc5(x, adj))+x
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc6(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc7(x, adj)
        return F.log_softmax(x, dim=1)