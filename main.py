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
from dataset import load_dataset, get_eval, mean_std
from diffusion_utils import sample_step, impute_mask

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='california', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--split_idx', type=int, default=0, help='Split idx.')
parser.add_argument('--max_iter', type=int, default=5, help='Maximum iteration.')
parser.add_argument('--ratio', type=str, default=30, help='Masking ratio.')
parser.add_argument('--hid_dim', type=int, default=1024, help='Hidden dimension.')
parser.add_argument('--mask', type=str, default='MCAR', help='Masking machenisms.')
parser.add_argument('--num_trials', type=int, default=10, help='Number of sampling times.')
parser.add_argument('--num_steps', type=int, default=50, help='Number of diffusion steps.')

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

    if mask_type == 'MNAR':
        mask_type = 'MNAR_logistic_T2'

    # === Create result path ONCE (so we can log total time safely) ===
    result_save_path = f'results/{dataname}/rate{ratio}/{mask_type}/{split_idx}/{num_trials}_{num_steps}'
    os.makedirs(result_save_path, exist_ok=True)

    # === Total runtime timer (1 run) ===
    _sync_if_cuda(device)
    run_t0 = time.perf_counter()

    train_X, test_X, ori_train_mask, ori_test_mask, train_num, test_num, train_cat_idx, test_cat_idx, train_mask, test_mask, cat_bin_num = load_dataset(dataname, split_idx, mask_type, ratio)
    
    mean_X, std_X = mean_std(train_X, train_mask)    
    in_dim = train_X.shape[1]

    X = (train_X - mean_X) / std_X / 2
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

    # NOTE: Keep original start_time if you still want it, but we use perf_counter timing.
    start_time = time.time()

    for iteration in range(args.max_iter):

        # === Iteration (EM) timer: includes M-step + E-step + eval + saving ===
        _sync_if_cuda(device)
        iter_t0 = time.perf_counter()

        ## M-Step: Density Estimation
     
        ckpt_dir = f'ckpt/{dataname}/rate{ratio}/{mask_type}/{split_idx}/{num_trials}_{num_steps}'
        os.makedirs(f'{ckpt_dir}/{iteration}') if not os.path.exists(f'{ckpt_dir}/{iteration}') else None

        print(f'iteration: {iteration}')
        print(ckpt_dir)

        if iteration == 0:
            X_miss = (1. - mask_train.float()) * X
            train_data = X_miss.numpy()
        else:
            print(f'Loading X_miss from {ckpt_dir}/iter_{iteration}.npy')
            X_miss = np.load(f'{ckpt_dir}/iter_{iteration}.npy') / 2
            train_data = X_miss
 
        batch_size = 4096

        # NOTE: keep algorithm unchanged; only reduce CPU↔GPU overhead.
        # Convert to float32 tensor once (DataLoader will yield torch tensors on CPU).
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

        model = Model(denoise_fn = denoise_fn, hid_dim = in_dim).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=50)

        model.train()

        best_loss = float('inf')
        patience = 0

        # === M-step timer (training only) ===
        _sync_if_cuda(device)
        m_t0 = time.perf_counter()

        # progress bar
        pbar = tqdm(range(num_epochs), desc='Training')
        for epoch in pbar:

            batch_loss = 0.0
            len_input = 0
 
            for batch in train_loader:
                # non_blocking transfer works with pin_memory=True
                inputs = batch.to(device, non_blocking=True)

                loss = model(inputs).mean()

                # Avoid per-batch GPU sync from loss.item(); aggregate on GPU then sync once/epoch
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

        # === E-step timer (imputation + eval only) ===
        _sync_if_cuda(device)
        e_t0 = time.perf_counter()

        # In-sample imputation

        rec_Xs = []

        # Pre-compute masked input once (same across trials) and keep tensors on GPU
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
        rec_X = torch.stack(rec_Xs, dim = 0).mean(0) 

        rec_X = rec_X.cpu().numpy() * 2
        X_true = X.cpu().numpy() * 2

        np.save(f'{ckpt_dir}/iter_{iteration+1}.npy', rec_X)

        pred_X = rec_X[:]
        len_num = train_num.shape[1]

        res = pred_X[:, len_num:] * std_X[len_num:] + mean_X[len_num:]
        pred_X[:, len_num:] = res

        mae, rmse, acc = get_eval(
            dataname, pred_X, X_true,
            train_cat_idx, train_num.shape[1],
            cat_bin_num, ori_train_mask
        )
        MAEs.append(mae)
        RMSEs.append(rmse)
        ACCs.append(acc)

        print('in-sample', mae, rmse, 'ACC', acc)

        # out-of_sample_imputation

        rec_Xs = []

        # Pre-compute masked input once (same across trials) and keep tensors on GPU
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
                # For out-of-sample imputation, no results from previous iterations are used
                rec_X = impute_mask(net, impute_X, mask_test, num_samples, dim, num_steps, device)
                rec_X = rec_X * mask_test_f + impute_X * (1 - mask_test_f)
                rec_Xs.append(rec_X)
        rec_X = torch.stack(rec_Xs, dim = 0).mean(0) 

        rec_X = rec_X.cpu().numpy() * 2
        X_true = X_test.cpu().numpy() * 2

        pred_X = rec_X[:]
        len_num = train_num.shape[1]
        res = pred_X[:, len_num:] * std_X[len_num:] + mean_X[len_num:]
        pred_X[:, len_num:] = res

        mae_out, rmse_out, acc_out = get_eval(
            dataname, pred_X, X_true,
            test_cat_idx, test_num.shape[1],
            cat_bin_num, ori_test_mask, oos=True
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

        with open (f'{result_save_path}/result.txt', 'a+', encoding='utf-8') as f:
            f.write(f'iteration {iteration}, MAE: in-sample: {mae}, out-of-sample: {mae_out} \n')
            f.write(f'iteration {iteration}: RMSE: in-sample: {rmse}, out-of-sample: {rmse_out} \n')
            f.write(f'iteration {iteration}: ACC: in-sample: {acc}, out-of-sample: {acc_out} \n')
            f.write(f'iteration {iteration}: TIME_M_SEC: {m_sec:.6f}, TIME_E_SEC: {e_sec:.6f}, TIME_TOTAL_SEC: {iter_sec:.6f}\n')

        print('out-of-sample', mae_out, rmse_out)
        print(f'saving results to {result_save_path}')

    # === Total runtime end ===
    _sync_if_cuda(device)
    run_total_sec = time.perf_counter() - run_t0
    print(f"[TIME] total_run_sec = {run_total_sec:.3f}s")

    with open(f'{result_save_path}/time_log.txt', 'a+', encoding='utf-8') as f:
        f.write(f"TOTAL_RUN_SEC: {run_total_sec:.6f}\n")