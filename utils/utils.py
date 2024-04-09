import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_data(root):
    events = pd.read_csv(root + 'events_mini.csv')
    users = pd.read_csv(root + 'users_mini.csv')
    items = pd.read_csv(root + 'tracks_mini.csv')

    events.drop(columns=['album_id'], inplace=True)
    users.drop(columns=['creation_time'], inplace=True)
    events = events.groupby(['user_id', 'track_id']).sum().reset_index()

    index_to_user = pd.Series(np.sort(np.unique(events['user_id'])))
    index_to_item = pd.Series(np.sort(np.unique(events['track_id'])))
    user_to_index = pd.Series(index_to_user.index, index=index_to_user)
    item_to_index = pd.Series(index_to_item.index, index=index_to_item)

    items['track_id'] = items['track_id'].map(item_to_index)
    items = items.set_index('track_id').sort_index()
    items.index.name = None
    scaler = MinMaxScaler()
    items = pd.DataFrame(scaler.fit_transform(items), columns=items.columns, index=items.index)

    users['user_id'] = users['user_id'].map(user_to_index)
    users = users.set_index('user_id').sort_index()
    users.index.name = None
    encoder = LabelEncoder()
    for c in users.columns:
        users[c] = encoder.fit_transform(users[c])

    events['user_id'] = events['user_id'].map(user_to_index)
    events['item_id'] = events['track_id'].map(item_to_index)
    events['weight'] = np.log2(events['count'] + 1)
    events = events[['user_id', 'item_id', 'weight']]

    events['user_id'] = events['user_id'].astype(int)
    events['item_id'] = events['item_id'].astype(int)
    users = users.astype(int)

    return events, users, items

def get_csr_mat(events, num_row, num_col):
    return sp.csr_matrix(
        (events['weight'], (events['user_id'], events['item_id'])),
        shape=(num_row, num_col)
    )

def get_np_mat(events, num_row, num_col):
    return np.array(
        sp.csr_matrix(
            (events['weight'], (events['user_id'], events['item_id'])),
            shape=(num_row, num_col)
        ).todense()
    )

def predict(model, user_features, item_features, device='cpu'):
    model.eval()
    user_features = torch.LongTensor(user_features.values).to(device)
    item_features = torch.FloatTensor(item_features.values).to(device)
    scores = []
    for u in range(user_features.shape[0]):
        x_sparse = torch.cat([
            torch.tensor(u).repeat(item_features.shape[0], 1).to(device),
            torch.arange(item_features.shape[0]).unsqueeze(1).to(device),
            user_features[u].repeat(item_features.shape[0], 1)
        ], dim=1)
        scores.append(model(x_sparse, item_features).squeeze().detach().cpu().numpy())
    return np.array(scores)

def metrics_at_k(model, valid_data, train_data, user_features, item_features, k=10, device='cpu', precision=True, roc_auc=True):
    model.eval()
    precision_train, precision_valid, roc_auc_train, roc_auc_valid = np.nan, np.nan, np.nan, np.nan
    
    num_users = user_features.shape[0]
    num_items = item_features.shape[0]
    valid = get_np_mat(valid_data, num_users, num_items)
    train = get_np_mat(train_data, num_users, num_items)
    valid[valid > 0] = 1
    train[train > 0] = 1

    scores = predict(model, user_features, item_features, device)

    # calculate roc_auc
    if roc_auc:
        negs = 1 - (valid + train)
        diff_sum_train = 0
        base_sum_train = 0
        diff_sum_valid = 0
        base_sum_valid = 0
        for u in range(num_users):
            train_diff = scores[u][train[u] > 0]
            train_diff = train_diff[:, np.newaxis] - scores[u][negs[u] > 0]
            valid_diff = scores[u][valid[u] > 0]
            valid_diff = valid_diff[:, np.newaxis] - scores[u][negs[u] > 0]
            train_diff = train_diff > 0
            valid_diff = valid_diff > 0
            diff_sum_train += train_diff.sum()
            base_sum_train += np.sum(negs[u]) * np.sum(train[u])
            diff_sum_valid += valid_diff.sum()
            base_sum_valid += np.sum(negs[u]) * np.sum(valid[u])
        roc_auc_train = diff_sum_train / base_sum_train
        roc_auc_valid = diff_sum_valid / base_sum_valid

    # calculate precision
    if precision:
        scores_train = scores - valid * 9999
        scores_valid = scores - train * 9999
        scores_train = num_items - np.argsort(np.argsort(scores_train, axis=1), axis=1)
        scores_valid = num_items - np.argsort(np.argsort(scores_valid, axis=1), axis=1)

        scores_train = scores_train * train
        scores_valid = scores_valid * valid
        scores_train = (scores_train > 0) & (scores_train < k + 1)
        scores_valid = (scores_valid > 0) & (scores_valid < k + 1)
        precision_train = np.sum(scores_train, axis=1) / k
        precision_valid = np.sum(scores_valid, axis=1) / k
        precision_train = precision_train.mean()
        precision_valid = precision_valid.mean()

    return precision_train, precision_valid, roc_auc_train, roc_auc_valid