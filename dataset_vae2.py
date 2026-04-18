import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import json
import torch

from vae_model2 import CategoricalVAE, train_vae_for_column

DATA_DIR = 'datasets'


def load_dataset(dataname, idx=0, mask_type='MCAR', ratio='30', 
                 embedding_dim=8, vae_epochs=500, device='cpu'):
    """
    Load dataset dengan VAE embedding untuk categorical features.
    
    Args:
        dataname: nama dataset
        idx: split index
        mask_type: tipe masking
        ratio: masking ratio
        embedding_dim: dimensi embedding dari VAE
        vae_epochs: epochs untuk training VAE
        device: torch device
    
    Returns:
        train_X, test_X, train_mask, test_mask, train_num, test_num, 
        train_cat_idx, test_cat_idx, extend_train_mask, extend_test_mask, 
        cat_emb_dims, vae_models (dict of VAE models per column)
    """
    data_dir = f'datasets/{dataname}'
    info_path = f'datasets/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    data_path = f'{data_dir}/data.csv'
    train_path = f'{data_dir}/train.csv'
    test_path = f'{data_dir}/test.csv'

    train_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
    test_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy'

    data_df = pd.read_csv(data_path)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_mask = np.load(train_mask_path)
    test_mask = np.load(test_mask_path)

    cols = train_df.columns

    data_num = data_df[cols[num_col_idx]].values.astype(np.float32)
    data_cat = data_df[cols[cat_col_idx]].astype(str)
    data_y = data_df[cols[target_col_idx]]

    train_num = train_df[cols[num_col_idx]].values.astype(np.float32)
    train_cat = train_df[cols[cat_col_idx]].astype(str)
    train_y = train_df[cols[target_col_idx]]

    test_num = test_df[cols[num_col_idx]].values.astype(np.float32)
    test_cat = test_df[cols[cat_col_idx]].astype(str)
    test_y = test_df[cols[target_col_idx]]
    
    cat_columns = data_cat.columns

    train_cat_idx, test_cat_idx = None, None
    extend_train_mask = None
    extend_test_mask = None
    cat_emb_dims = None
    vae_models = {}

    # Only contain numerical features
    if len(cat_col_idx) == 0:
        train_X = train_num
        test_X = test_num

        extend_train_mask = train_mask[:, num_col_idx]
        extend_test_mask = test_mask[:, num_col_idx]

    # Contain both numerical and categorical features
    else:
        # Create or load category mappings
        if not os.path.exists(f'{data_dir}/{cat_columns[0]}_map_idx.json'):
            for column in cat_columns:
                map_path_idx = f'{data_dir}/{column}_map_idx.json'
                categories = data_cat[column].unique()
                category_to_idx = {category: index for index, category in enumerate(categories)}
                
                with open(map_path_idx, 'w') as f:
                    json.dump(category_to_idx, f)

        train_cat_idx_list = []
        test_cat_idx_list = []
        cat_emb_dims = []
        
        # VAE embeddings untuk setiap categorical column
        vae_ckpt_dir = f'{data_dir}/vae_embeddings/emb{embedding_dim}'
        os.makedirs(vae_ckpt_dir, exist_ok=True)
        
        train_cat_embeddings = []
        test_cat_embeddings = []
        
        for col_i, column in enumerate(cat_columns):
            print(f"Processing categorical column: {column}")
            
            map_path_idx = f'{data_dir}/{column}_map_idx.json'
            
            with open(map_path_idx, 'r') as f:
                category_to_idx = json.load(f)
            
            num_categories = len(category_to_idx)
            
            # Get category indices
            train_cat_idx_i = train_cat[column].map(category_to_idx).to_numpy().astype(np.int64)
            test_cat_idx_i = test_cat[column].map(category_to_idx).to_numpy().astype(np.int64)
            
            train_cat_idx_list.append(train_cat_idx_i)
            test_cat_idx_list.append(test_cat_idx_i)
            
            # One-hot encode untuk training VAE
            train_onehot = np.eye(num_categories)[train_cat_idx_i].astype(np.float32)
            test_onehot = np.eye(num_categories)[test_cat_idx_i].astype(np.float32)
            
            # Train or load VAE
            vae_path = f'{vae_ckpt_dir}/{column}_vae.pt'
            
            if os.path.exists(vae_path):
                print(f"  Loading VAE from {vae_path}")
                vae = CategoricalVAE(num_categories, embedding_dim).to(device)
                vae.load_state_dict(torch.load(vae_path, map_location=device))
                vae.eval()
            else:
                print(f"  Training VAE for {column} ({num_categories} categories)...")
                vae = train_vae_for_column(
                    train_onehot, 
                    num_categories,
                    embedding_dim=embedding_dim,
                    epochs=vae_epochs,
                    device=device,
                    verbose=True
                )
                torch.save(vae.state_dict(), vae_path)
                print(f"  Saved VAE to {vae_path}")
            
            vae_models[column] = vae
            
            # Get embeddings
            with torch.no_grad():
                train_onehot_t = torch.from_numpy(train_onehot).float().to(device)
                test_onehot_t = torch.from_numpy(test_onehot).float().to(device)
                
                train_emb = vae.get_embedding(train_onehot_t).cpu().numpy()
                test_emb = vae.get_embedding(test_onehot_t).cpu().numpy()
            
            train_cat_embeddings.append(train_emb)
            test_cat_embeddings.append(test_emb)
            cat_emb_dims.append(embedding_dim)
        
        # Concatenate embeddings
        train_cat_emb = np.concatenate(train_cat_embeddings, axis=1).astype(np.float32)
        test_cat_emb = np.concatenate(test_cat_embeddings, axis=1).astype(np.float32)
        
        train_cat_idx = np.stack(train_cat_idx_list, axis=1)
        test_cat_idx = np.stack(test_cat_idx_list, axis=1)
        
        cat_emb_dims = np.array(cat_emb_dims)
        
        # Concatenate numerical and categorical embeddings
        train_X = np.concatenate([train_num, train_cat_emb], axis=1)
        test_X = np.concatenate([test_num, test_cat_emb], axis=1)
        
        # Extend masks for embeddings
        train_num_mask = train_mask[:, num_col_idx]
        train_cat_mask = train_mask[:, cat_col_idx]
        test_num_mask = test_mask[:, num_col_idx]
        test_cat_mask = test_mask[:, cat_col_idx]
        
        def extend_mask(mask, emb_dims):
            """Extend mask untuk embedding dimensions"""
            num_rows, num_cols = mask.shape
            cum_sum = emb_dims.cumsum()
            cum_sum = np.insert(cum_sum, 0, 0)
            result = np.zeros((num_rows, emb_dims.sum()), dtype=bool)
            
            for idx in range(num_cols):
                res = np.tile(mask[:, idx][:, np.newaxis], emb_dims[idx])
                result[:, cum_sum[idx]:cum_sum[idx + 1]] = res
            
            return result
        
        train_cat_mask_ext = extend_mask(train_cat_mask, cat_emb_dims)
        test_cat_mask_ext = extend_mask(test_cat_mask, cat_emb_dims)
        
        extend_train_mask = np.concatenate([train_num_mask, train_cat_mask_ext], axis=1)
        extend_test_mask = np.concatenate([test_num_mask, test_cat_mask_ext], axis=1)

    return (train_X, test_X, train_mask, test_mask, train_num, test_num, 
            train_cat_idx, test_cat_idx, extend_train_mask, extend_test_mask, 
            cat_emb_dims, vae_models)


def mean_std(data, mask):
    """
    Calculate mean and std for non-missing values with robust normalization
    
    Args:
        data: data array [N, features]
        mask: mask array (True = missing, False = observed)
    
    Returns:
        mean, std: normalization parameters
    """
    # Inverse mask: True = observed, False = missing
    obs_mask = ~mask
    obs_mask = obs_mask.astype(np.float32)
    
    # Count observed values per feature
    obs_count = obs_mask.sum(0)
    obs_count[obs_count == 0] = 1  # Prevent division by zero
    
    # Calculate mean from observed values
    mean = (data * obs_mask).sum(0) / obs_count
    
    # Calculate variance from observed values
    var = ((data - mean) ** 2 * obs_mask).sum(0) / obs_count
    std = np.sqrt(var)
    
    # Robust handling of std
    # If std is very small, it means feature is nearly constant
    # Use 1.0 to avoid numerical issues but preserve the constant nature
    std[std < 1e-6] = 1.0
    
    return mean, std


def robust_normalize(data, mask, method='standard'):
    """
    Robust normalization for mixed numerical and categorical embeddings
    
    Args:
        data: data array [N, features]
        mask: mask array (True = missing, False = observed)
        method: 'standard' (z-score) or 'minmax' or 'robust'
    
    Returns:
        normalized_data, norm_params (dict dengan mean, std, method)
    """
    obs_mask = ~mask
    obs_mask = obs_mask.astype(np.float32)
    obs_count = obs_mask.sum(0)
    obs_count[obs_count == 0] = 1
    
    if method == 'standard':
        # Z-score normalization (recommended for diffusion models)
        mean = (data * obs_mask).sum(0) / obs_count
        var = ((data - mean) ** 2 * obs_mask).sum(0) / obs_count
        std = np.sqrt(var)
        std[std < 1e-6] = 1.0
        
        normalized = (data - mean) / std
        norm_params = {'mean': mean, 'std': std, 'method': 'standard'}
        
    elif method == 'minmax':
        # Min-Max normalization to [0, 1]
        # Better for preserving data distribution but may have outlier issues
        data_obs = data.copy()
        data_obs[~obs_mask.astype(bool)] = np.nan
        
        min_val = np.nanmin(data_obs, axis=0)
        max_val = np.nanmax(data_obs, axis=0)
        
        # Handle constant features
        range_val = max_val - min_val
        range_val[range_val < 1e-6] = 1.0
        
        normalized = (data - min_val) / range_val
        norm_params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
        
    elif method == 'robust':
        # Robust normalization using median and IQR
        # Less sensitive to outliers
        data_obs = data.copy()
        data_obs[~obs_mask.astype(bool)] = np.nan
        
        median = np.nanmedian(data_obs, axis=0)
        q25 = np.nanpercentile(data_obs, 25, axis=0)
        q75 = np.nanpercentile(data_obs, 75, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-6] = 1.0
        
        normalized = (data - median) / iqr
        norm_params = {'median': median, 'iqr': iqr, 'method': 'robust'}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, norm_params


def denormalize(data, norm_params):
    """
    Denormalize data back to original scale
    
    Args:
        data: normalized data
        norm_params: normalization parameters from robust_normalize
    
    Returns:
        denormalized data
    """
    method = norm_params['method']
    
    if method == 'standard':
        return data * norm_params['std'] + norm_params['mean']
    elif method == 'minmax':
        range_val = norm_params['max'] - norm_params['min']
        return data * range_val + norm_params['min']
    elif method == 'robust':
        return data * norm_params['iqr'] + norm_params['median']
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize_for_diffusion(train_X, test_X, train_mask, test_mask, num_num, 
                           norm_method='standard', clip_range=(-5, 5)):
    """
    Unified normalization untuk diffusion model dengan handling special untuk categorical embeddings
    
    Strategy:
    1. Normalize ALL features dengan method yang sama (unified scale)
    2. Tapi simpan info numerical vs categorical untuk evaluation
    3. Clip extreme values untuk stability
    
    Args:
        train_X: training data [N, num_features + cat_embeddings]
        test_X: test data
        train_mask: training mask (True = missing)
        test_mask: test mask
        num_num: number of numerical features
        norm_method: 'standard' or 'robust'
        clip_range: clip normalized values to this range
    
    Returns:
        train_X_norm, test_X_norm, norm_params_dict
    """
    # PENTING: Normalize SEMUA features dengan parameter yang SAMA
    # Ini memastikan range yang consistent untuk diffusion model
    
    if norm_method == 'standard':
        # Calculate global mean and std dari SEMUA features
        obs_mask = ~train_mask
        obs_mask = obs_mask.astype(np.float32)
        obs_count = obs_mask.sum(0)
        obs_count[obs_count == 0] = 1
        
        mean = (train_X * obs_mask).sum(0) / obs_count
        var = ((train_X - mean) ** 2 * obs_mask).sum(0) / obs_count
        std = np.sqrt(var)
        std[std < 1e-6] = 1.0
        
        # Normalize dengan parameter yang sama
        train_X_norm = (train_X - mean) / std
        test_X_norm = (test_X - mean) / std
        
        # Clip untuk numerical stability
        train_X_norm = np.clip(train_X_norm, clip_range[0], clip_range[1])
        test_X_norm = np.clip(test_X_norm, clip_range[0], clip_range[1])
        
        norm_params = {
            'mean': mean,
            'std': std,
            'method': 'standard',
            'clip_range': clip_range
        }
        
    elif norm_method == 'robust':
        # Robust normalization dengan median dan IQR
        data_obs = train_X.copy()
        data_obs[train_mask] = np.nan
        
        median = np.nanmedian(data_obs, axis=0)
        q25 = np.nanpercentile(data_obs, 25, axis=0)
        q75 = np.nanpercentile(data_obs, 75, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-6] = 1.0
        
        train_X_norm = (train_X - median) / iqr
        test_X_norm = (test_X - median) / iqr
        
        # Clip untuk stability
        train_X_norm = np.clip(train_X_norm, clip_range[0], clip_range[1])
        test_X_norm = np.clip(test_X_norm, clip_range[0], clip_range[1])
        
        norm_params = {
            'median': median,
            'iqr': iqr,
            'method': 'robust',
            'clip_range': clip_range
        }
    
    else:
        raise ValueError(f"Unknown normalization method: {norm_method}")
    
    # Save info tentang numerical vs categorical untuk evaluation
    norm_params['num_num'] = num_num
    
    return train_X_norm, test_X_norm, norm_params


def denormalize_all(data, norm_params):
    """
    Denormalize SEMUA features kembali ke skala original
    
    Strategy yang LEBIH SIMPLE:
    1. Denormalize semua (numerical + categorical embeddings)
    2. Baru split untuk evaluation:
       - Numerical: RMSE/MAE di original scale
       - Categorical: VAE decoder di original embedding scale
    
    Args:
        data: normalized data [N, num_features + cat_embeddings]
        norm_params: normalization parameters
    
    Returns:
        denormalized data (all features)
    """
    method = norm_params['method']
    
    if method == 'standard':
        # Inverse z-score: X = X_norm * std + mean
        return data * norm_params['std'] + norm_params['mean']
    
    elif method == 'robust':
        # Inverse robust: X = X_norm * IQR + median
        return data * norm_params['iqr'] + norm_params['median']
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_eval(dataname, X_recon, X_true, truth_cat_idx, num_num, cat_emb_dims, 
             mask, vae_models, cat_columns, device='cpu', oos=False):
    """
    Evaluation dengan VAE decoding untuk categorical features
    
    Args:
        dataname: nama dataset
        X_recon: reconstructed data (num + cat embeddings)
        X_true: true data (num + cat embeddings)
        truth_cat_idx: ground truth category indices
        num_num: jumlah numerical features
        cat_emb_dims: dimensi embedding untuk tiap categorical column
        mask: original mask
        vae_models: dict of trained VAE models
        cat_columns: list of categorical column names
        device: torch device
        oos: out-of-sample flag
    """
    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    # mask di repo ini: True(1) = missing, False(0) = observed
    num_mask = mask[:, num_col_idx].astype(bool)
    cat_mask = mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else None

    num_pred = X_recon[:, :num_num]
    cat_pred_emb = X_recon[:, num_num:]  # embeddings

    num_true = X_true[:, :num_num]

    # special-case dari repo: buang 1 baris di news oos (biar align)
    if dataname == 'news' and oos is True:
        drop = 6265
        num_mask = np.delete(num_mask, drop, axis=0)
        num_pred = np.delete(num_pred, drop, axis=0)
        num_true = np.delete(num_true, drop, axis=0)
        if cat_mask is not None:
            cat_mask = np.delete(cat_mask, drop, axis=0)
        if truth_cat_idx is not None:
            truth_cat_idx = np.delete(truth_cat_idx, drop, axis=0)
        cat_pred_emb = np.delete(cat_pred_emb, drop, axis=0)

    # ===== Continuous metrics (hanya di posisi missing) =====
    div = num_pred[num_mask] - num_true[num_mask]
    mae = np.abs(div).mean()
    rmse = np.sqrt((div ** 2).mean())

    # ===== Discrete metric: Accuracy (menggunakan VAE decoder) =====
    acc = np.nan
    if (truth_cat_idx is not None) and (len(cat_col_idx) > 0) and (cat_emb_dims is not None):
        cat_emb_dims = np.array(cat_emb_dims).astype(int)
        ends = np.cumsum(cat_emb_dims)
        starts = np.concatenate(([0], ends[:-1]))

        correct_total = 0
        total_missing = 0

        for j, (s, e) in enumerate(zip(starts, ends)):
            rows_miss = cat_mask[:, j]
            if rows_miss.sum() == 0:
                continue

            # Get embedding untuk column ini
            emb_j = cat_pred_emb[:, s:e]
            
            # Decode menggunakan VAE
            column_name = cat_columns[j]
            vae = vae_models[column_name]
            vae.eval()
            
            with torch.no_grad():
                emb_j_t = torch.from_numpy(emb_j).float().to(device)
                pred_idx = vae.embedding_to_category_idx(emb_j_t).cpu().numpy()
            
            true_idx = truth_cat_idx[:, j].astype(np.int64)

            # Jumlah kelas valid
            nclass = int(true_idx.max()) + 1
            valid = pred_idx < nclass

            correct = ((pred_idx == true_idx) & valid & rows_miss).sum()
            total = rows_miss.sum()

            correct_total += int(correct)
            total_missing += int(total)

        if total_missing > 0:
            acc = correct_total / total_missing

    return mae, rmse, acc