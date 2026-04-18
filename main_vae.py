import os
import torch

import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
from tqdm import tqdm

from model import MLPDiffusion, Model
from dataset_vae import load_dataset, get_eval, mean_std
from diffusion_utils import sample_step, impute_mask

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Missing Value Imputation with VAE Categorical Encoding')

parser.add_argument('--dataname', type=str, default='california', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--split_idx', type=int, default=0, help='Split idx.')
parser.add_argument('--max_iter', type=int, default=10, help='Maximum iteration.')
parser.add_argument('--ratio', type=str, default=30, help='Masking ratio.')
parser.add_argument('--hid_dim', type=int, default=1024, help='Hidden dimension.')
parser.add_argument('--mask', type=str, default='MCAR', help='Masking mechanisms.')
parser.add_argument('--num_trials', type=int, default=10, help='Number of sampling times.')
parser.add_argument('--num_steps', type=int, default=50, help='Number of diffusion steps.')
parser.add_argument('--vae_embedding_dim', type=int, default=8, help='VAE embedding dimension per categorical column.')
parser.add_argument('--vae_epochs', type=int, default=500, help='VAE training epochs.')

args = parser.parse_args()

# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = f'cuda:{args.gpu}'
else:
    args.device = 'cpu'


def _sync_if_cuda(device: str):
    """Synchronize CUDA for accurate timing (GPU ops are async)."""
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


if __name__ == '__main__':

    dataname = args.dataname
    split_idx = args.split_idx
    device = args.device
    hid_dim = args.hid_dim
    mask_type = args.mask
    ratio = args.ratio
    num_trials = args.num_trials
    num_steps = args.num_steps
    vae_embedding_dim = args.vae_embedding_dim
    vae_epochs = args.vae_epochs

    if mask_type == 'MNAR':
        mask_type = 'MNAR_logistic_T2'

    # === Create result path ===
    result_save_path = f'results/{dataname}/rate{ratio}/{mask_type}/vae_emb{vae_embedding_dim}/{split_idx}/{num_trials}_{num_steps}'
    os.makedirs(result_save_path, exist_ok=True)

    # === Total runtime timer ===
    _sync_if_cuda(device)
    run_t0 = time.perf_counter()

    print(f"Loading dataset with VAE encoding (embedding_dim={vae_embedding_dim})...")
    
    # Load dataset dengan VAE embedding
    (train_X, test_X, ori_train_mask, ori_test_mask, train_num, test_num, 
     train_cat_idx, test_cat_idx, train_mask, test_mask, 
     cat_emb_dims, vae_models) = load_dataset(
        dataname, split_idx, mask_type, ratio, 
        embedding_dim=vae_embedding_dim,
        vae_epochs=vae_epochs,
        device=device
    )
    # ini rangkuman cek train data hasil load dataset sudah termasuk embedding, yang dicek data kategorikalnya saja
    # Get categorical column names
    import json
    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)
    import pandas as pd
    data_path = f'datasets/{dataname}/data.csv'
    data_df = pd.read_csv(data_path)
    cols = data_df.columns
    cat_col_idx = info['cat_col_idx']
    cat_columns = cols[cat_col_idx].tolist() if len(cat_col_idx) > 0 else []
    
    print(f"Data loaded. Train shape: {train_X.shape}, Test shape: {test_X.shape}")
    if len(cat_columns) > 0:
        print(f"VAE models trained for {len(cat_columns)} categorical columns: {cat_columns}")
        print(f"Categorical embedding dimensions: {cat_emb_dims}")
    
    #ini standarisasi atau normalisasi hasil load dataset, cek lagi bisa jadi ini yang jadi masalah!!
    #ini pakai Z-score
    mean_X, std_X = mean_std(train_X, train_mask)    
    in_dim = train_X.shape[1]

    X = (train_X - mean_X) / std_X / 2 #kenapa ini dibagi 2???
    X = torch.tensor(X)

    X_test = (test_X - mean_X) / std_X / 2
    X_test = torch.tensor(X_test)

    mask_train = torch.tensor(train_mask)
    mask_test = torch.tensor(test_mask)

    MAEs = []
    RMSEs = []
    ACCs = []

    MAEs_out = []
    RMSEs_out = []
    ACCs_out = []

    start_time = time.time()

    for iteration in range(args.max_iter):

        # === Iteration timer ===
        _sync_if_cuda(device)
        iter_t0 = time.perf_counter()

        ## M-Step: Density Estimation
     
        ckpt_dir = f'ckpt/{dataname}/rate{ratio}/{mask_type}/vae_emb{vae_embedding_dim}/{split_idx}/{num_trials}_{num_steps}'
        os.makedirs(f'{ckpt_dir}/{iteration}', exist_ok=True)

        print(f'\n{"="*60}')
        print(f'Iteration: {iteration}')
        print(f'Checkpoint directory: {ckpt_dir}')
        print(f'{"="*60}')

        if iteration == 0: #ini membentuk initial state untuk imputasi dimana 1 untuk missing,0 untuk non-missing
            X_miss = (1. - mask_train.float()) * X
            train_data = X_miss.numpy()
            print(f'[INFO] Loaded X_miss shape: {train_data.shape}, range: [{train_data.min():.4f}, {train_data.max():.4f}]')
        else: #untuk iteasi selanjutnya tidak perlu bentuk lagi initian state tinggal pakai initial state dari iterasi sebelumnya
            print(f'Loading X_miss from {ckpt_dir}/iter_{iteration}.npy')
            X_miss = np.load(f'{ckpt_dir}/iter_{iteration}.npy') / 2 #pertanyaannya INI KENAPA DIBAGI 2 SEDANGKAN ITERASI 0 TIDAK DIBAGI 2
            train_data = X_miss                                     #bisa jadi ini ada indikasi nilainya melebar mangkanya meledak
            print(f'[INFO] Loaded X_miss shape: {train_data.shape}, range: [{train_data.min():.4f}, {train_data.max():.4f}]')
 
        batch_size = 4096

        if not torch.is_tensor(train_data):
            train_data = torch.as_tensor(train_data, dtype=torch.float32)

        use_cuda = (str(device).startswith('cuda') and torch.cuda.is_available())
        num_workers = 0 if os.name == 'nt' else 4

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0),
        )

        num_epochs = 10000 + 1

        denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)

        if iteration == 0:
            print(denoise_fn)

        model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=50)

        model.train()

        best_loss = float('inf')
        patience = 0

        # === M-step timer ===
        _sync_if_cuda(device)
        m_t0 = time.perf_counter()

        pbar = tqdm(range(num_epochs), desc='Training')
        for epoch in pbar:

            batch_loss = 0.0
            len_input = 0
 
            for batch in train_loader:
                inputs = batch.to(device, non_blocking=True)

                loss = model(inputs).mean()

                if not torch.is_tensor(batch_loss):
                    batch_loss = torch.zeros((), device=device)
                batch_loss += loss.detach() * inputs.size(0)
                len_input += inputs.size(0)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            curr_loss = (batch_loss/len_input).item() if torch.is_tensor(batch_loss) else (batch_loss/len_input)
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
                best_loss = curr_loss
                patience = 0
                torch.save(model.state_dict(), f'{ckpt_dir}/{iteration}/model.pt')
            else:
                patience += 1
                if patience == 500:
                    print('Early stopping')
                    break
            
            pbar.set_postfix(loss=curr_loss)

            if epoch % 1000 == 0:
                torch.save(model.state_dict(), f'{ckpt_dir}/{iteration}/model_{epoch}.pt')

        _sync_if_cuda(device)
        m_sec = time.perf_counter() - m_t0
        print(f"[TIME] iter {iteration}: M_step_sec = {m_sec:.3f}s")

        end_time = time.time()

        ## E-Step: Missing Value Imputation

        # === E-step timer ===
        _sync_if_cuda(device)
        e_t0 = time.perf_counter()

        # In-sample imputation
        rec_Xs = []

        X = torch.tensor(X, dtype=torch.float32, device=device)
        X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
        mask_train_f = mask_train.to(device=device, dtype=torch.float32)
        impute_X = ((1. - mask_train_f) * X).to(device)

        in_dim = X.shape[1]
        denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)
        model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
        model.load_state_dict(torch.load(f'{ckpt_dir}/{iteration}/model.pt', map_location=device))
        model.eval()
        net = model.denoise_fn_D

        num_samples, dim = X.shape[0], X.shape[1]

        with torch.no_grad():
            for trial in tqdm(range(num_trials), desc='In-sample imputation'):
                rec_X = impute_mask(net, impute_X, mask_train, num_samples, dim, num_steps, device)
                rec_X = rec_X * mask_train_f + impute_X * (1 - mask_train_f)
                rec_Xs.append(rec_X)
        rec_X = torch.stack(rec_Xs, dim=0).mean(0) 

        rec_X = rec_X.cpu().numpy() * 2 #ini dikali 2 karena denormalisasi?
        X_true = X.cpu().numpy() * 2

        np.save(f'{ckpt_dir}/iter_{iteration+1}.npy', rec_X)

        pred_X = rec_X[:]
        len_num = train_num.shape[1]

        # De-standardize categorical embeddings
        res = pred_X[:, len_num:] * std_X[len_num:] + mean_X[len_num:]
        pred_X[:, len_num:] = res

        mae, rmse, acc = get_eval(
            dataname, pred_X, X_true,
            train_cat_idx, train_num.shape[1],
            cat_emb_dims, ori_train_mask,
            vae_models, cat_columns, device
        )
        MAEs.append(mae)
        RMSEs.append(rmse)
        ACCs.append(acc)

        print(f'In-sample - MAE: {mae:.4f}, RMSE: {rmse:.4f}, ACC: {acc:.4f}')

        # Out-of-sample imputation
        rec_Xs = []

        mask_test_f = mask_test.to(device=device, dtype=torch.float32)
        impute_X = ((1. - mask_test_f) * X_test).to(device)

        in_dim = X_test.shape[1]
        denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)
        model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
        model.load_state_dict(torch.load(f'{ckpt_dir}/{iteration}/model.pt', map_location=device))
        model.eval()
        net = model.denoise_fn_D

        num_samples, dim = X_test.shape[0], X_test.shape[1]

        with torch.no_grad():
            for trial in tqdm(range(num_trials), desc='Out-of-sample imputation'):
                rec_X = impute_mask(net, impute_X, mask_test, num_samples, dim, num_steps, device)
                rec_X = rec_X * mask_test_f + impute_X * (1 - mask_test_f)
                rec_Xs.append(rec_X)
        rec_X = torch.stack(rec_Xs, dim=0).mean(0) 

        rec_X = rec_X.cpu().numpy() * 2
        X_true = X_test.cpu().numpy() * 2

        pred_X = rec_X[:]
        len_num = train_num.shape[1]
        
        # De-standardize categorical embeddings
        res = pred_X[:, len_num:] * std_X[len_num:] + mean_X[len_num:]
        pred_X[:, len_num:] = res

        mae_out, rmse_out, acc_out = get_eval(
            dataname, pred_X, X_true,
            test_cat_idx, test_num.shape[1],
            cat_emb_dims, ori_test_mask, 
            vae_models, cat_columns, device, oos=True
        )
        MAEs_out.append(mae_out)
        RMSEs_out.append(rmse_out)
        ACCs_out.append(acc_out)

        _sync_if_cuda(device)
        e_sec = time.perf_counter() - e_t0
        print(f"[TIME] iter {iteration}: E_step_sec = {e_sec:.3f}s")

        # === Iteration total time ===
        _sync_if_cuda(device)
        iter_sec = time.perf_counter() - iter_t0
        print(f"[TIME] iter {iteration}: TOTAL_iter_sec = {iter_sec:.3f}s")

        with open(f'{result_save_path}/result.txt', 'a+', encoding='utf-8') as f:
            f.write(f'iteration {iteration}, MAE: in-sample: {mae}, out-of-sample: {mae_out}\n')
            f.write(f'iteration {iteration}: RMSE: in-sample: {rmse}, out-of-sample: {rmse_out}\n')
            f.write(f'iteration {iteration}: ACC: in-sample: {acc}, out-of-sample: {acc_out}\n')
            f.write(f'iteration {iteration}: TIME_M_SEC: {m_sec:.6f}, TIME_E_SEC: {e_sec:.6f}, TIME_TOTAL_SEC: {iter_sec:.6f}\n')

        print(f'Out-of-sample - MAE: {mae_out:.4f}, RMSE: {rmse_out:.4f}, ACC: {acc_out:.4f}')
        print(f'Results saved to {result_save_path}')

    # === Total runtime end ===
    _sync_if_cuda(device)
    run_total_sec = time.perf_counter() - run_t0
    print(f"\n[TIME] total_run_sec = {run_total_sec:.3f}s")

    with open(f'{result_save_path}/time_log.txt', 'a+', encoding='utf-8') as f:
        f.write(f"TOTAL_RUN_SEC: {run_total_sec:.6f}\n")
    
    print(f"\nFinal Results Summary:")
    print(f"{'='*60}")
    print(f"VAE Embedding Dimension: {vae_embedding_dim}")
    print(f"In-sample  - MAE: {MAEs[-1]:.4f}, RMSE: {RMSEs[-1]:.4f}, ACC: {ACCs[-1]:.4f}")
    print(f"Out-sample - MAE: {MAEs_out[-1]:.4f}, RMSE: {RMSEs_out[-1]:.4f}, ACC: {ACCs_out[-1]:.4f}")
    print(f"{'='*60}")
