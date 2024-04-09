import numpy as np
import torch
import torch.optim as optim
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models.models import *
from utils.utils import *
from utils.dataset import *


def train(args):
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    events, users, items = load_data(args.data_path, miss_ratio=args.miss_ratio)
    train_df, valid_df = train_test_split(events, test_size=args.split, random_state=args.seed)
    train_mat = get_csr_mat(train_df, users.shape[0], items.shape[0])
    valid_mat = get_csr_mat(events, users.shape[0], items.shape[0])
    train_data = BPRData(train_df, train_mat,
                            user_features=users if args.user_features else None, 
                            item_features=items if args.item_features else None, 
                             num_neg=args.num_neg, device=args.device, add_weight=True)
    valid_data = BPRData(valid_df, valid_mat,
                            user_features=users if args.user_features else None,
                            item_features=items if args.item_features else None,
                            num_neg=args.num_neg, device=args.device, add_weight=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)

    # Load model
    sparse_dims, dense_dim = train_data.get_dims()
    model = None
    if args.model == 'fm':
        model = FM(sparse_dims, dense_dim, embed_dim=args.embed_dim).to(args.device)
    elif args.model == 'fmv':
        model = FMV(sparse_dims, dense_dim, embed_dim=args.embed_dim).to(args.device)
    elif args.model == 'deepfm':
        model = DeepFM(sparse_dims, dense_dim, embed_dim=args.embed_dim).to(args.device)
    else:
        raise ValueError('Invalid model')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    with open(args.log_path + args.task + '.csv', 'w') as f:
        f.write('train_loss,valid_loss,train_precision,valid_precision,roc_auc_train,roc_auc_valid\n')
    p_best = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_data.neg_sample()
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}', disable=not args.verbose)
        for x_pos_sparse, x_pos_dense, x_neg_sparse, x_neg_dense, weight in progress:
            optimizer.zero_grad()
            pos_score = model(x_pos_sparse, x_pos_dense)
            neg_score = model(x_neg_sparse, x_neg_dense)
            loss = -(torch.log(torch.sigmoid(pos_score - neg_score)) * weight).mean()
            if torch.isinf(loss): continue
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0
        valid_data.neg_sample()
        for x_pos_sparse, x_pos_dense, x_neg_sparse, x_neg_dense, weight in valid_loader:
            pos_score = model(x_pos_sparse, x_pos_dense)
            neg_score = model(x_neg_sparse, x_neg_dense)
            loss = -(torch.log(torch.sigmoid(pos_score - neg_score)) * weight).mean()
            valid_loss += loss.item()
        valid_loss /= len(valid_loader)

        p_train, p_valid, roc_train, roc_valid = \
            metrics_at_k(model, valid_df, train_df, users, items, k=args.topk, device=args.device, precision=True, roc_auc=True)
        if p_valid > p_best:
            p_best = p_valid
            torch.save(model.state_dict(), args.model_path + args.task + '.pth')
        with open(args.log_path + args.task + '.csv', 'a') as f:
            f.write(f'{train_loss:.4f},{valid_loss:.4f},{p_train:.4f},{p_valid:.4f},{roc_train:.4f},{roc_valid:.4f}\n')
            
        print(f'Train Loss: {train_loss:.4f}, ' +
            f'Valid Loss: {valid_loss:.4f}, ' +
            f'Precision@10 (Train): {p_train:.4f}, ' +
            f'Precision@10 (Valid): {p_valid:.4f}, ' +
            f'ROC AUC (Train): {roc_train:.4f},' +
            f'ROC AUC (Valid): {roc_valid:.4f}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--model_path', type=str, default='best/')
    parser.add_argument('--log_path', type=str, default='log/')
    parser.add_argument('--task', type=str, default='fm')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--split', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--num_neg', type=int, default=1)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--model', type=str, default='fm')
    parser.add_argument('--user_features', type=bool, default=False)
    parser.add_argument('--item_features', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--miss_ratio', type=float, default=0.0)
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        raise ValueError('Invalid mode')
    