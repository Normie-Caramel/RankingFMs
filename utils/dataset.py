import torch
from torch.utils.data import Dataset
import numpy as np

class BPRData(Dataset):
    def __init__(self, uir_df, uir_mat, user_features=None, item_features=None, add_weight=True, num_neg=1, device='cpu'):
        super(BPRData, self).__init__()
        """
        uir_df: user-item-rating dataframe in the form of (user_id, item_id, weight)
        user_features: user features in the form of (num_users, num_user_features)
        item_features: item features in the form of (num_items, num_item_features)
        num_neg: number of negative samples for each positive sample
        """
        self.uir_df = uir_df
        self.num_neg = num_neg
        self.uir_mat = uir_mat
        self.add_weight = add_weight
        self.num_users = uir_mat.shape[0]
        self.num_items = uir_mat.shape[1]
        self.sparse_dims = [self.num_users, self.num_items] if user_features is None \
            else [self.num_users, self.num_items] + [user_features[col].nunique() for col in user_features.columns]
        self.dense_dim = None if item_features is None \
            else item_features.shape[1]
        self.user_features = None if user_features is None \
            else torch.LongTensor(user_features.values).to(device)
        self.item_features = None if item_features is None \
            else torch.FloatTensor(item_features.values).to(device)
        self.empty_long = torch.tensor([], dtype=torch.long).to(device)
        self.empty_float = torch.tensor([], dtype=torch.float32).to(device)
        self.device = device

    def neg_sample(self):
        upn = []
        wts = []
        for user, item, weight in self.uir_df.itertuples(index=False):
            for _ in range(self.num_neg):
                neg_item = np.random.randint(self.num_items)
                while self.uir_mat[user, neg_item] > 0:
                    neg_item = np.random.randint(self.num_items)
                upn.append([user, item, user, neg_item])
                wts.append([weight])
        self.sampled_uir = torch.LongTensor(upn).to(self.device)
        self.sampled_wts = torch.FloatTensor(wts).to(self.device) if self.add_weight \
            else torch.ones(len(upn), 1, dtype=torch.float32).to(self.device)
    
    def get_dims(self):
        return self.sparse_dims, self.dense_dim

    def __len__(self):
        return self.num_neg * self.uir_df.shape[0]

    def __getitem__(self, idx):
        u = self.sampled_uir[idx][0]
        i = self.sampled_uir[idx][1]
        n = self.sampled_uir[idx][3]
        weight = self.sampled_wts[idx]
        user_features = self.empty_long if self.user_features is None \
            else self.user_features[u]
        x_pos_sparse = torch.cat([self.sampled_uir[idx][:2], user_features])
        x_neg_sparse = torch.cat([self.sampled_uir[idx][2:], user_features])
        x_pos_dense = self.empty_float if self.item_features is None else self.item_features[i]
        x_neg_dense = self.empty_float if self.item_features is None else self.item_features[n]
        return x_pos_sparse, x_pos_dense, x_neg_sparse, x_neg_dense, weight