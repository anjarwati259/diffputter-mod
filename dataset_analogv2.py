import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import json

DATA_DIR = 'datasets'

# =============================================================================
# ANALOG BITS ENCODING  (sesuai paper Section 3.2 & Chen et al. 2022)
# -----------------------------------------------------------------------------
# Perbedaan utama dari one-hot encoding biasa:
#   - Kategori direpresentasikan dalam binary bits (seperti sebelumnya)
#   - Nilai 0 dikonversi ke -1  →  range menjadi {-1, +1}
#   - Range {-1, +1} membuat data lebih distinguishable untuk diffusion model
#     karena lebih simetris di sekitar nol (seperti skor noise normal)
#
# Saat decode:
#   - output diffusion > 0  →  bit = 1
#   - output diffusion ≤ 0  →  bit = 0   (karena 0 = -1 dalam analog bits)
#   - bit string → integer → predicted category index
# =============================================================================

def _binary_to_analog_bits(binary_array: np.ndarray) -> np.ndarray:
    """
    Konversi array binary {0, 1} ke analog bits {-1, +1}.

    Paper (Chen et al. 2022, "Analog Bits"):
        "we further convert 0 to −1 in ... analog bits encoding"

    Parameter:
        binary_array : np.ndarray, dtype int/float, nilai {0, 1}

    Return:
        np.ndarray dtype float32, nilai {-1.0, +1.0}
    """
    return (binary_array.astype(np.float32) * 2.0) - 1.0   # 0→-1, 1→+1


def load_dataset(dataname, idx = 0, mask_type = 'MCAR', ratio = '30'):
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
    cat_bin_num = None

    # -------------------------------------------------------------------------
    # Hanya fitur numerik
    # -------------------------------------------------------------------------
    if len(cat_col_idx) == 0:
        train_X = train_num
        test_X = test_num

        extend_train_mask = train_mask[:, num_col_idx]
        extend_test_mask = test_mask[:, num_col_idx]

    # -------------------------------------------------------------------------
    # Fitur numerik + kategorik  →  encode kategorik dengan ANALOG BITS
    # -------------------------------------------------------------------------
    else:
        # Buat/load mapping binary string & integer index per kolom kategorik
        if not os.path.exists(f'{data_dir}/{cat_columns[0]}_map.json'):

            for column in cat_columns:
                map_path_bin = f'{data_dir}/{column}_map_bin.json'
                map_path_idx = f'{data_dir}/{column}_map_idx.json'
                categories = data_cat[column].unique()
                num_categories = len(categories)

                num_bits = (num_categories - 1).bit_length()

                # Map kategori → binary string (e.g. "011") dan → integer index
                category_to_binary = {
                    category: format(index, '0' + str(num_bits) + 'b')
                    for index, category in enumerate(categories)
                }
                category_to_idx = {
                    category: index for index, category in enumerate(categories)
                }

                with open(map_path_bin, 'w') as f:
                    json.dump(category_to_binary, f)
                with open(map_path_idx, 'w') as f:
                    json.dump(category_to_idx, f)

        train_cat_analog = []   # ← analog bits {-1, +1}  (bukan {0, 1})
        test_cat_analog  = []

        train_cat_idx = []
        test_cat_idx  = []
        cat_bin_num   = []

        for column in cat_columns:
            map_path_bin = f'{data_dir}/{column}_map_bin.json'
            map_path_idx = f'{data_dir}/{column}_map_idx.json'

            with open(map_path_bin, 'r') as f:
                category_to_binary = json.load(f)
            with open(map_path_idx, 'r') as f:
                category_to_idx = json.load(f)

            # --- Train ---
            train_cat_enc_i  = train_cat[column].map(category_to_binary).to_numpy()
            train_cat_idx_i  = train_cat[column].map(category_to_idx).to_numpy().astype(np.int64)
            # Binary {0,1}
            train_cat_bin_i  = np.array([list(map(int, binary)) for binary in train_cat_enc_i])
            # ANALOG BITS: konversi 0 → -1
            train_cat_analog_i = _binary_to_analog_bits(train_cat_bin_i)

            # --- Test ---
            test_cat_enc_i   = test_cat[column].map(category_to_binary).to_numpy()
            test_cat_idx_i   = test_cat[column].map(category_to_idx).to_numpy().astype(np.int64)
            test_cat_bin_i   = np.array([list(map(int, binary)) for binary in test_cat_enc_i])
            test_cat_analog_i  = _binary_to_analog_bits(test_cat_bin_i)

            train_cat_analog.append(train_cat_analog_i)
            test_cat_analog.append(test_cat_analog_i)

            train_cat_idx.append(train_cat_idx_i)
            test_cat_idx.append(test_cat_idx_i)
            cat_bin_num.append(train_cat_bin_i.shape[1])

        # Gabungkan semua kolom kategorik yang sudah di-encode analog bits
        train_cat_analog = np.concatenate(train_cat_analog, axis=1).astype(np.float32)
        test_cat_analog  = np.concatenate(test_cat_analog,  axis=1).astype(np.float32)

        train_cat_idx = np.stack(train_cat_idx, axis=1)
        test_cat_idx  = np.stack(test_cat_idx,  axis=1)

        cat_bin_num = np.array(cat_bin_num)

        # Gabungkan fitur numerik + analog bits kategorik
        # Layout: [num_features | analog_bits_cat_features]
        train_X = np.concatenate([train_num, train_cat_analog], axis=1)
        test_X  = np.concatenate([test_num,  test_cat_analog],  axis=1)

        # Extend mask (mask per kolom asli → mask per bit analog)
        train_num_mask = train_mask[:, num_col_idx]
        train_cat_mask = train_mask[:, cat_col_idx]
        test_num_mask  = test_mask[:, num_col_idx]
        test_cat_mask  = test_mask[:, cat_col_idx]

        def extend_mask(mask, bin_num):
            """
            Memperluas mask dari dimensi kolom asli ke dimensi bit analog.
            Setiap kolom kategorik dengan b bits → b kolom mask yang sama.
            """
            num_rows, num_cols = mask.shape
            cum_sum = bin_num.cumsum()
            cum_sum = np.insert(cum_sum, 0, 0)
            result  = np.zeros((num_rows, bin_num.sum()), dtype=bool)

            for idx in range(num_cols):
                res = np.tile(mask[:, idx][:, np.newaxis], bin_num[idx])
                result[:, cum_sum[idx]:cum_sum[idx + 1]] = res

            return result

        train_cat_mask = extend_mask(train_cat_mask, cat_bin_num)
        test_cat_mask  = extend_mask(test_cat_mask,  cat_bin_num)

        extend_train_mask = np.concatenate([train_num_mask, train_cat_mask], axis=1)
        extend_test_mask  = np.concatenate([test_num_mask,  test_cat_mask],  axis=1)

    return (train_X, test_X,
            train_mask, test_mask,
            train_num, test_num,
            train_cat_idx, test_cat_idx,
            extend_train_mask, extend_test_mask,
            cat_bin_num)


def mean_std(data, mask):
    mask = ~mask
    mask = mask.astype(np.float32)
    mask_sum = mask.sum(0)
    mask_sum[mask_sum == 0] = 1
    mean = (data * mask).sum(0) / mask_sum
    var  = ((data - mean) ** 2 * mask).sum(0) / mask_sum
    std  = np.sqrt(var)
    return mean, std


def _analog_bits_to_int(analog_bits: np.ndarray) -> np.ndarray:
    """
    Decode analog bits {-1, +1} → integer category index.

    Sesuai paper Section 3.2 (analog bits recovery):
        "we convert every output element to 1 if the output element is larger
         than 0, otherwise we convert it to −1"

    Langkah:
        1. output diffusion > 0  →  bit = 1
           output diffusion ≤ 0  →  bit = 0  (representasi internal 0)
        2. bit string → integer index via bobot posisi binary

    Parameter:
        analog_bits : np.ndarray, shape (N, b)
                      Nilai kontinu hasil prediksi diffusion model.
                      Range bebas; hanya tanda (positif/negatif) yang dipakai.

    Return:
        idx : np.ndarray, shape (N,) — integer label hasil decoding
    """
    b = analog_bits.shape[1]

    # Threshold 0 (bukan 0.5!) karena encoding simetris di sekitar nol
    bits_rounded = (analog_bits > 0).astype(np.int32)   # 1 jika >0, else 0

    # Bobot posisi: [2^(b-1), 2^(b-2), ..., 2^0]
    powers = (2 ** np.arange(b - 1, -1, -1)).astype(np.int32)

    # Dot product → integer index
    idx = bits_rounded.dot(powers)   # shape (N,)
    return idx


# Alias untuk kompatibilitas backward (nama lama masih dapat dipakai)
_bits_to_int = _analog_bits_to_int


def get_eval(dataname, X_recon, X_true, truth_cat_idx, num_num, cat_bin_num, mask, oos=False, pred_cat_idx=None):
    """
    Menghitung MAE, RMSE (kolom numerik) dan Accuracy (kolom kategorik)
    hanya pada posisi MISSING (mask == True).

    Logika Accuracy — dua mode:
    ----------------------------------------
    Mode A (pred_cat_idx=None): decode dari nilai kontinu analog bits output diffusion.
        1. output > 0  →  bit = 1  |  output ≤ 0  →  bit = 0  (threshold = 0)
        2. bit string → integer index
        3. bandingkan dengan ground-truth label index (truth_cat_idx)

    Mode B (pred_cat_idx tersedia): langsung pakai integer index hasil majority voting.
        - Bypass decode kontinu sama sekali
        - Lebih akurat karena keputusan diambil per-trial sebelum digabung
    """

    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    # True(1) = missing, False(0) = observed
    num_mask = mask[:, num_col_idx].astype(bool)
    cat_mask = mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else None

    num_pred     = X_recon[:, :num_num]
    cat_pred_analog = X_recon[:, num_num:]   # analog bits output dari diffusion

    num_true = X_true[:, :num_num]

    # Special-case: buang 1 baris di news oos agar dimensi align
    if dataname == 'news' and oos is True:
        drop = 6265
        num_mask       = np.delete(num_mask,       drop, axis=0)
        num_pred       = np.delete(num_pred,       drop, axis=0)
        num_true       = np.delete(num_true,       drop, axis=0)
        if cat_mask is not None:
            cat_mask   = np.delete(cat_mask,       drop, axis=0)
        if truth_cat_idx is not None:
            truth_cat_idx = np.delete(truth_cat_idx, drop, axis=0)
        cat_pred_analog = np.delete(cat_pred_analog, drop, axis=0)

    # ===== Metrik numerik: MAE & RMSE hanya pada posisi missing =====
    div  = num_pred[num_mask] - num_true[num_mask]
    mae  = np.abs(div).mean()
    rmse = np.sqrt((div ** 2).mean())

    # ===== Metrik kategorik: Accuracy hanya pada posisi missing =====
    acc = np.nan
    if (truth_cat_idx is not None) and (len(cat_col_idx) > 0) and (cat_bin_num is not None):

        cat_bin_num = np.array(cat_bin_num).astype(int)
        ends   = np.cumsum(cat_bin_num)
        starts = np.concatenate(([0], ends[:-1]))

        correct_total = 0
        total_missing = 0

        for j, (s, e) in enumerate(zip(starts, ends)):

            rows_miss = cat_mask[:, j]   # baris yang missing pada kolom kat ke-j
            if rows_miss.sum() == 0:
                continue

            # Ground-truth integer index
            true_idx = truth_cat_idx[:, j].astype(int)  # shape (N,)

            # ---------------------------------------------------------------
            # Mode B: langsung pakai integer index hasil majority voting
            # Mode A: decode dari nilai kontinu analog bits (threshold = 0)
            # ---------------------------------------------------------------
            if pred_cat_idx is not None:
                # Mode B — bypass decode kontinu, langsung pakai hasil voting
                pred_idx = pred_cat_idx[:, j].astype(int)  # shape (N,)
            else:
                # Mode A — decode dari analog bits kontinu
                pred_analog = cat_pred_analog[:, s:e]       # shape (N, b)
                pred_idx = _analog_bits_to_int(pred_analog) # shape (N,)

            # Clamp: cegah index melebihi jumlah kelas yang valid
            nclass   = int(true_idx.max()) + 1
            pred_idx = np.clip(pred_idx, 0, nclass - 1)

            # Hitung correct hanya pada baris yang missing
            correct = ((pred_idx == true_idx) & rows_miss).sum()
            total   = rows_miss.sum()

            correct_total += int(correct)
            total_missing += int(total)

        if total_missing > 0:
            acc = correct_total / total_missing

    return mae, rmse, acc