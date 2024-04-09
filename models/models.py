import torch
from torch import nn

class BaseFM(nn.Module):
    def __init__(self, input_dim, embed_dim=8):
        super(BaseFM, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.linear = nn.Linear(self.input_dim, 1, bias=True)
        self.v = nn.Parameter(torch.tensor(self.input_dim, self.embed_dim), requires_grad=True)
        self.v.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        linear_part = self.linear(x)
        inter_part1 = torch.pow(torch.mm(x, self.v), 2)
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        pair_interactions = 0.5 * torch.sum(torch.sub(inter_part1 - inter_part2), dim=1, keepdim=True)
        output = linear_part + pair_interactions
        return output
    
class FM(nn.Module):
    def __init__(self, sparse_dims, dense_dim, embed_dim=8):
        super(FM, self).__init__()
        self.sparse_dims = sparse_dims
        self.dense_dim = dense_dim
        self.embed_dim = embed_dim
            
        self.linear_sparse = nn.ModuleList([nn.Embedding(voc_size, 1) for voc_size in self.sparse_dims])
        self.order2_sparse = nn.ModuleList([nn.Embedding(voc_size, embed_dim) for voc_size in self.sparse_dims])

        if self.dense_dim is not None:
            self.linear_dense = nn.Linear(self.dense_dim, 1)
            self.order2_dense = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(self.dense_dim)])
        

    def forward(self, x_sparse, x_dense):
        linear_part = [emb(x_sparse[:, i]).unsqueeze(1) for i, emb in enumerate(self.linear_sparse)]
        linear_part = torch.cat(linear_part, dim=1)
        linear_part = torch.sum(linear_part, dim=1, keepdim=True).squeeze(-1)

        if self.dense_dim is not None:
            linear_dense = self.linear_dense(x_dense)
            linear_part = linear_part + linear_dense

        order2_part = [emb(x_sparse[:, i]).unsqueeze(1) for i, emb in enumerate(self.order2_sparse)]
        order2_part = torch.cat(order2_part, dim=1)

        if self.dense_dim is not None:
            order2_dense = [emb(x_dense[:, i].unsqueeze(1)).unsqueeze(1) for i, emb in enumerate(self.order2_dense)]
            order2_dense = torch.cat(order2_dense, dim=1)
            order2_part = torch.cat([order2_part, order2_dense], dim=1)

        square_of_sum = torch.pow(torch.sum(order2_part, dim=1), 2)
        sum_of_square = torch.sum(torch.pow(order2_part, 2), dim=1)

        cross_part = torch.sub(square_of_sum, sum_of_square)
        cross_part = 0.5 * torch.sum(cross_part, dim=1, keepdim=True)

        return linear_part + cross_part
    
class FMV(nn.Module):
    def __init__(self, sparse_dims, dense_dim, embed_dim=8, units=64, dropout=0.2):
        super(FMV, self).__init__()
        self.sparse_dims = sparse_dims
        self.dense_dim = dense_dim
        self.embed_dim = embed_dim

        self.linear_sparse = nn.ModuleList([nn.Embedding(voc_size, 1) for voc_size in self.sparse_dims])
        self.order2_sparse = nn.ModuleList([
                nn.Sequential(
                nn.Embedding(voc_size, units),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(units, embed_dim)) for voc_size in self.sparse_dims])

        if self.dense_dim is not None:
            self.linear_dense = nn.Linear(self.dense_dim, 1)
            self.order2_dense = nn.ModuleList([
                nn.Sequential(
                nn.Linear(1, units),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(units, embed_dim)) for _ in range(self.dense_dim)])
            

    def forward(self, x_sparse, x_dense):
        linear_part = [emb(x_sparse[:, i]).unsqueeze(1) for i, emb in enumerate(self.linear_sparse)]
        linear_part = torch.cat(linear_part, dim=1)
        linear_part = torch.sum(linear_part, dim=1, keepdim=True).squeeze(-1)

        if self.dense_dim is not None:
            linear_dense = self.linear_dense(x_dense)
            linear_part = linear_part + linear_dense

        order2_part = [emb(x_sparse[:, i]).unsqueeze(1) for i, emb in enumerate(self.order2_sparse)]
        order2_part = torch.cat(order2_part, dim=1)

        if self.dense_dim is not None:
            order2_dense = [emb(x_dense[:, i].unsqueeze(1)).unsqueeze(1) for i, emb in enumerate(self.order2_dense)]
            order2_dense = torch.cat(order2_dense, dim=1)
            order2_part = torch.cat([order2_part, order2_dense], dim=1)

        square_of_sum = torch.pow(torch.sum(order2_part, dim=1), 2)
        sum_of_square = torch.sum(torch.pow(order2_part, 2), dim=1)

        cross_part = torch.sub(square_of_sum, sum_of_square)
        cross_part = 0.5 * torch.sum(cross_part, dim=1, keepdim=True)

        return linear_part + cross_part
    
class DeepFM(nn.Module):
    def __init__(self, sparse_dims, dense_dim, embed_dim=8, hidden_units=[128, 64], units=64, dnn_weight = 0.5, dropout=0.2):
        super(DeepFM, self).__init__()
        self.sparse_dims = sparse_dims
        self.dense_dim = dense_dim
        self.embed_dim = embed_dim
        self.dnn_weight = dnn_weight
            
        self.linear_sparse = nn.ModuleList([nn.Embedding(voc_size, 1) for voc_size in self.sparse_dims])
        # self.order2_sparse = nn.ModuleList([nn.Embedding(voc_size, embed_dim) for voc_size in self.sparse_dims])
        self.order2_sparse = nn.ModuleList([
                nn.Sequential(
                nn.Embedding(voc_size, units),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(units, embed_dim)) for voc_size in self.sparse_dims])

        if self.dense_dim is not None:
            self.linear_dense = nn.Linear(self.dense_dim, 1)
            # self.order2_dense = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(self.dense_dim)])
            self.order2_dense = nn.ModuleList([
                nn.Sequential(
                nn.Linear(1, units),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(units, embed_dim)) for _ in range(self.dense_dim)])

        self.deep_input_dim = len(self.sparse_dims) * embed_dim if self.dense_dim is None \
            else (len(self.sparse_dims) + self.dense_dim) * embed_dim
        
        layers = []
        for i in range(len(hidden_units)):
            layers.append(nn.Linear(self.deep_input_dim, hidden_units[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            self.deep_input_dim = hidden_units[i]
        self.deep = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_units[-1], 1)

    def forward(self, x_sparse, x_dense):
        linear_part = [emb(x_sparse[:, i]).unsqueeze(1) for i, emb in enumerate(self.linear_sparse)]
        linear_part = torch.cat(linear_part, dim=1)
        linear_part = torch.sum(linear_part, dim=1, keepdim=True).squeeze(-1)

        if self.dense_dim is not None:
            linear_dense = self.linear_dense(x_dense)
            linear_part = linear_part + linear_dense

        order2_part = [emb(x_sparse[:, i]).unsqueeze(1) for i, emb in enumerate(self.order2_sparse)]
        order2_part = torch.cat(order2_part, dim=1)

        if self.dense_dim is not None:
            order2_dense = [emb(x_dense[:, i].unsqueeze(1)).unsqueeze(1) for i, emb in enumerate(self.order2_dense)]
            order2_dense = torch.cat(order2_dense, dim=1)
            order2_part = torch.cat([order2_part, order2_dense], dim=1)

        deep_part = torch.flatten(order2_part, start_dim=1)
        deep_part = self.deep(deep_part)
        deep_part = self.fc(deep_part)

        square_of_sum = torch.pow(torch.sum(order2_part, dim=1), 2)
        sum_of_square = torch.sum(torch.pow(order2_part, 2), dim=1)

        cross_part = torch.sub(square_of_sum, sum_of_square)
        cross_part = 0.5 * torch.sum(cross_part, dim=1, keepdim=True)        

        return linear_part + cross_part + self.dnn_weight * deep_part