import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import json

DATA_DIR = 'datasets'

# ===========================================================================
#  Supervised Learnable Embedding Model dengan Label
#  Arsitektur: nn.Embedding per kolom kategorikal + MLP Classifier untuk prediksi label
#  Konsep: Neural Network Embedding yang dilatih secara supervised
# ===========================================================================

def compute_embedding_size(n_categories: int) -> int:
    """
    Hitung ukuran embedding optimal berdasarkan jumlah kategori.
    Rumus: min(600, round(1.6 * n_categories^0.56))
    Referensi: Guo & Berkhahn (2016)
    """
    return min(600, round(1.6 * n_categories ** 0.56))
    #return min(16, max(2, round(n_categories ** 0.5)))


class SupervisedLearnableEmbeddingModel(nn.Module):
    """
    Model Supervised Learnable Embedding untuk fitur kategorikal tabular.
    
    Alur Supervised Learning:
      cat_idx [batch, n_cat_cols]
        → nn.Embedding per kolom → concat → [batch, total_emb_dim]   (encode)
        → MLP Classifier → [batch, n_classes]                        (predict label)
        → Linear Decoder per kolom (reconstruction)                  (decode → logits per kolom)
    
    Parameter
    ---------
    cat_dims    : list[int]   jumlah kategori unik per kolom
    emb_sizes   : list[int]   dimensi embedding per kolom
    n_classes   : int         jumlah kelas untuk klasifikasi (dari target)
    dropout     : float       dropout setelah concat embedding
    hidden_dim  : int         dimensi hidden layer untuk classifier
    """

    def __init__(self, cat_dims: list, emb_sizes: list, n_classes: int, 
                 dropout: float = 0.1, hidden_dim: int = 256):
        super().__init__()

        # Satu nn.Embedding per kolom kategorikal
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=n_cat, embedding_dim=emb_dim)
            for n_cat, emb_dim in zip(cat_dims, emb_sizes)
        ])

        self.dropout       = nn.Dropout(dropout)
        self.total_emb_dim = sum(emb_sizes)
        self.n_cols        = len(cat_dims)
        self.cat_dims      = cat_dims
        self.emb_sizes     = emb_sizes
        self.n_classes     = n_classes

        # MLP Classifier untuk supervised learning (prediksi label)
        self.classifier = nn.Sequential(
            nn.Linear(self.total_emb_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim), # ini tambahan
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            #nn.BatchNorm1d(hidden_dim // 2), #ini tambahan
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

        # Satu Linear Decoder per kolom kategorikal (untuk reconstruction)
        # PERBAIKAN: Input per decoder hanya emb_size kolom terkait (bukan total_emb_dim)
        # Agar simetris dengan encoder yang memisahkan embedding per kolom
        self.decoders = nn.ModuleList([
            nn.Linear(emb_size, n_cat)
            for n_cat, emb_size in zip(cat_dims, emb_sizes)
        ])

    def encode(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Encode integer index kategorikal → vektor embedding dense.

        x_cat  : [batch, n_cat_cols]  — integer index tiap kolom
        return : [batch, total_emb_dim]
        """
        embedded = [
            self.embeddings[i](x_cat[:, i])   # [batch, emb_dim_i]
            for i in range(self.n_cols)
        ]
        z = torch.cat(embedded, dim=1)         # [batch, total_emb_dim]
        z = self.dropout(z)
        return z

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """
        Classifier: embedding → logit untuk prediksi label.

        z      : [batch, total_emb_dim]
        return : [batch, n_classes]
        """
        return self.classifier(z)

    def decode(self, z: torch.Tensor) -> list:
        """
        Linear Decoder: embedding → logit tiap kolom kategorikal.
        
        PERBAIKAN: Split concatenated embedding kembali ke per-column,
        lalu decode setiap kolom menggunakan embedding-nya sendiri.
        Ini membuat decoder simetris dengan encoder.

        z      : [batch, total_emb_dim]
        return : list[n_cat_cols] of [batch, vocab_size_i]
        """
        # Split embedding yang sudah di-concat kembali ke per-column
        embeddings = torch.split(z, self.emb_sizes, dim=1)
        
        # Decode setiap kolom dari embedding-nya sendiri
        return [
            self.decoders[i](embeddings[i])
            for i in range(self.n_cols)
        ]

    def forward(self, x_cat: torch.Tensor):
        """
        Full forward: encode → classify + decode.

        return : (z, class_logits, recon_logits)
            z             [batch, total_emb_dim]
            class_logits  [batch, n_classes]
            recon_logits  list[n_cat_cols] of [batch, vocab_i]
        """
        z             = self.encode(x_cat)
        class_logits  = self.classify(z)
        recon_logits  = self.decode(z)
        return z, class_logits, recon_logits


def train_supervised_embedding_model(cat_idx_array: np.ndarray,
                                     labels: np.ndarray,
                                     cat_dims: list,
                                     emb_sizes: list,
                                     n_classes: int,
                                     device: str,
                                     n_epochs: int = 50,
                                     batch_size: int = 1024,
                                     lr: float = 1e-3,
                                     dropout: float = 0.1,
                                     hidden_dim: int = 256,
                                     patience: int = 30) -> SupervisedLearnableEmbeddingModel:
    """
    Latih SupervisedLearnableEmbeddingModel menggunakan supervised learning.
    
    Loss: alpha * classification_loss + beta * reconstruction_loss
    
    Parameter
    ---------
    cat_idx_array : np.ndarray [N, n_cat_cols]  — label-encoded integer
    labels        : np.ndarray [N]               — target labels untuk supervised learning
    cat_dims      : list[int]  vocab size per kolom
    emb_sizes     : list[int]  dimensi embedding per kolom
    n_classes     : int        jumlah kelas target
    device        : str        device target ('cuda:0', 'cpu', dst.)
    n_epochs      : int        jumlah epoch training maksimum
    batch_size    : int        ukuran batch
    lr            : float      learning rate Adam
    dropout       : float      dropout rate
    hidden_dim    : int        hidden dimension untuk classifier
    patience      : int        early stopping patience

    Return
    ------
    model : SupervisedLearnableEmbeddingModel  (sudah di-train, mode eval)
    """
    model = SupervisedLearnableEmbeddingModel(
        cat_dims, emb_sizes, n_classes, dropout, hidden_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss   = nn.CrossEntropyLoss()
    
    # Tensor data di GPU, tapi generator shuffle HARUS di CPU
    cat_tensor   = torch.tensor(cat_idx_array, dtype=torch.long, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    dataset      = torch.utils.data.TensorDataset(cat_tensor, label_tensor)
    cpu_gen      = torch.Generator(device='cpu')
    loader       = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = False,
        generator   = cpu_gen,
    )

    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Loss weights: reconstruction dinaikkan agar decoder lebih terlatih untuk imputasi.
    # alpha diturunkan sedikit; beta dinaikkan ke 1.0 karena akurasi imputasi bergantung
    # langsung pada kualitas rekonstruksi embedding, bukan hanya klasifikasi.
    alpha = 0.7  # classification loss weight (diturunkan dari 1.0)
    beta  = 1.0  # reconstruction loss weight (dinaikkan dari 0.5)
    
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        total_class_loss = 0.0
        total_recon_loss = 0.0
        n_batches = 0
        
        for batch_cat, batch_labels in loader:
            optimizer.zero_grad()
            z, class_logits, recon_logits = model(batch_cat)

            # Classification loss (supervised learning - UTAMA)
            class_loss = ce_loss(class_logits, batch_labels)
            
            # Reconstruction loss (auxiliary task untuk embedding quality)
            recon_loss = sum(
                ce_loss(recon_logits[i], batch_cat[:, i])
                for i in range(model.n_cols)
            ) / model.n_cols

            # Combined loss
            loss = alpha * class_loss + beta * recon_loss

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_recon_loss += recon_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_class_loss = total_class_loss / n_batches
        avg_recon_loss = total_recon_loss / n_batches
        
        # Print progress setiap 10 epoch
        if (epoch + 1) % 10 == 0:
            print(f'[Embedding] Epoch {epoch+1}/{n_epochs} - '
                  f'Loss: {avg_loss:.4f} (Class: {avg_class_loss:.4f}, '
                  f'Recon: {avg_recon_loss:.4f})')
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'[Embedding] Early stopping triggered at epoch {epoch+1}')
            print(f'[Embedding] Best loss: {best_loss:.4f}')
            break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f'[Embedding] Loaded best model from epoch {epoch + 1 - patience_counter}')
    
    model.eval()
    return model


def encode_with_embedding(model: SupervisedLearnableEmbeddingModel,
                          cat_idx_array: np.ndarray,
                          device: str,
                          batch_size: int = 4096) -> np.ndarray:
    """
    Encode seluruh data kategorikal → embedding numpy array.

    Return : np.ndarray [N, total_emb_dim]
    """
    model.eval()
    cat_tensor = torch.tensor(cat_idx_array, dtype=torch.long, device=device)
    dataset    = torch.utils.data.TensorDataset(cat_tensor)
    loader     = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )

    all_z = []
    with torch.no_grad():
        for (batch,) in loader:
            z = model.encode(batch)   # hanya encode, hindari compute classify+decode
            all_z.append(z.cpu().numpy())

    return np.concatenate(all_z, axis=0).astype(np.float32)


def decode_cat_from_embedding(model: SupervisedLearnableEmbeddingModel,
                              emb_array: np.ndarray,
                              device: str,
                              batch_size: int = 4096) -> np.ndarray:
    """
    Decode embedding → prediksi kelas kategorikal (argmax logits).

    emb_array : np.ndarray [N, total_emb_dim]
    Return    : np.ndarray [N, n_cat_cols]  — predicted integer index
    """
    model.eval()
    emb_tensor = torch.tensor(emb_array, dtype=torch.float32, device=device)
    dataset    = torch.utils.data.TensorDataset(emb_tensor)
    loader     = torch.utils.data.DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )

    all_pred = []
    with torch.no_grad():
        for (batch,) in loader:
            recon_logits = model.decode(batch)
            # argmax per kolom
            pred_idx = torch.stack([
                torch.argmax(logits, dim=1)
                for logits in recon_logits
            ], dim=1)  # [batch, n_cat_cols]
            all_pred.append(pred_idx.cpu().numpy())

    return np.concatenate(all_pred, axis=0).astype(np.int64)


# ===========================================================================
#  Load Dataset
# ===========================================================================

def load_dataset(dataname, idx=0, mask_type='MCAR', ratio='30'):
    """
    Load dataset dengan supervised embedding learning.
    
    Perubahan utama:
    - Membaca label dari target_col_idx di info.json
    - Melatih embedding model secara supervised dengan label
    - Tetap menggunakan struktur output yang sama untuk kompatibilitas
    
    Parameters
    ----------
    dataname : str
        Nama dataset (e.g., 'adult')
    idx : int
        Split index (default 0). Bisa juga dipanggil sebagai 'split_idx' untuk kompatibilitas
    mask_type : str
        Tipe masking ('MCAR', 'MNAR_logistic_T2', dll)
    ratio : str or int
        Rasio missing (e.g., '30', 30)
    
    Return
    ------
    train_X           : [N_train, num_num + total_emb_dim]  float32
    test_X            : [N_test,  num_num + total_emb_dim]  float32
    ori_train_mask    : mask asli train [N_train, total_cols]
    ori_test_mask     : mask asli test  [N_test,  total_cols]
    train_num         : [N_train, num_num]  — hanya numerik
    test_num          : [N_test,  num_num]
    train_cat_idx     : [N_train, n_cat_cols] integer index  (atau None)
    test_cat_idx      : [N_test,  n_cat_cols] integer index  (atau None)
    extend_train_mask : mask yang sudah diperluas ke dimensi X
    extend_test_mask  : mask yang sudah diperluas ke dimensi X
    cat_bin_num       : None  (tidak digunakan lagi, digantikan emb_sizes)
    emb_model         : SupervisedLearnableEmbeddingModel  (atau None jika no cat)
    emb_sizes         : list[int] dimensi embedding per kolom (atau None)
    """
    # Convert ratio to string if needed
    ratio = str(ratio)
    
    # ── Paths sama persis dengan original ────────────────────────────────
    data_dir  = f'datasets/{dataname}'
    info_path = f'datasets/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx    = info['num_col_idx']
    cat_col_idx    = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    data_path       = f'{data_dir}/data.csv'
    train_path      = f'{data_dir}/train.csv'
    test_path       = f'{data_dir}/test.csv'
    train_mask_path = f'{data_dir}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy'
    test_mask_path  = f'{data_dir}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy'

    # ── Load files sama persis dengan original ───────────────────────────
    data_df  = pd.read_csv(data_path)
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    train_mask = np.load(train_mask_path)
    test_mask  = np.load(test_mask_path)

    cols = train_df.columns

    # ── Fitur numerik ────────────────────────────────────────────────────
    data_num  = data_df[cols[num_col_idx]].values.astype(np.float32)
    train_num = train_df[cols[num_col_idx]].values.astype(np.float32)
    test_num  = test_df[cols[num_col_idx]].values.astype(np.float32)

    # ── Extract labels untuk supervised learning ─────────────────────────
    train_y = train_df[cols[target_col_idx]]
    test_y  = test_df[cols[target_col_idx]]
    
    # Label encoding untuk supervised learning
    # PERBAIKAN: fit pada data.csv (seluruh data) agar konsisten dengan encoder fitur,
    # dan memastikan semua kelas tercakup meski jarang muncul di split tertentu.
    label_encoder = LabelEncoder()
    data_y = data_df[cols[target_col_idx]]
    label_encoder.fit(data_y.values.ravel().astype(str))
    
    train_labels = label_encoder.transform(train_y.values.ravel().astype(str))
    test_labels  = label_encoder.transform(test_y.values.ravel().astype(str))
    n_classes    = len(label_encoder.classes_)
    
    print(f'[Dataset] Detected {n_classes} classes for supervised learning')
    print(f'[Dataset] Classes: {label_encoder.classes_}')

    # ── Kasus: hanya fitur numerik ───────────────────────────────────────
    if len(cat_col_idx) == 0:
        train_X = train_num
        test_X  = test_num

        extend_train_mask = train_mask[:, num_col_idx]
        extend_test_mask  = test_mask[:, num_col_idx]

        return (train_X, test_X,
                train_mask, test_mask,
                train_num, test_num,
                None, None,
                extend_train_mask, extend_test_mask,
                None,   # cat_bin_num (legacy, tidak dipakai)
                None,   # emb_model
                None)   # emb_sizes

    # ── Kasus: ada fitur kategorikal → Supervised Learnable Embedding ───
    cat_columns = cols[cat_col_idx]

    data_cat  = data_df[cat_columns].astype(str)
    train_cat = train_df[cat_columns].astype(str)
    test_cat  = test_df[cat_columns].astype(str)

    # Label encoding: fit pada seluruh data (data.csv) agar konsisten
    encoders   = {}
    cat_dims   = []
    train_cat_idx_list = []
    test_cat_idx_list  = []

    for col in cat_columns:
        le = LabelEncoder()
        le.fit(data_cat[col])             # fit pada semua data
        encoders[col] = le
        cat_dims.append(len(le.classes_))

        train_cat_idx_list.append(
            le.transform(train_cat[col]).astype(np.int64)
        )
        test_cat_idx_list.append(
            le.transform(test_cat[col]).astype(np.int64)
        )

    train_cat_idx = np.stack(train_cat_idx_list, axis=1)  # [N_train, n_cat]
    test_cat_idx  = np.stack(test_cat_idx_list,  axis=1)  # [N_test,  n_cat]

    # Hitung ukuran embedding sesuai rumus Guo & Berkhahn (2016)
    emb_sizes = [compute_embedding_size(n) for n in cat_dims]

    print(f'[Embedding] cat_dims={cat_dims}, emb_sizes={emb_sizes}, '
          f'total_emb_dim={sum(emb_sizes)}')

    # Tentukan device (gunakan CUDA jika tersedia; main.py akan overwrite)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Latih supervised embedding model menggunakan data TRAIN + LABELS
    print('[Embedding] Melatih SupervisedLearnableEmbeddingModel ...')
    emb_model = train_supervised_embedding_model(
        cat_idx_array = train_cat_idx,
        labels        = train_labels,
        cat_dims      = cat_dims,
        emb_sizes     = emb_sizes,
        n_classes     = n_classes,
        device        = device,
        n_epochs      = 1000,      # maksimum epochs
        batch_size    = 1024,
        lr            = 1e-3,
        dropout       = 0.1,
        hidden_dim    = 256,
        patience      = 40,       # early stopping patience (dinaikkan dari 10)
    )
    print('[Embedding] Training selesai.')

    # Encode: integer index → embedding vector
    train_cat_emb = encode_with_embedding(emb_model, train_cat_idx, device)
    test_cat_emb  = encode_with_embedding(emb_model, test_cat_idx,  device)
    # shape: [N, total_emb_dim]

    # Gabungkan numerik + embedding
    train_X = np.concatenate([train_num, train_cat_emb], axis=1)
    test_X  = np.concatenate([test_num,  test_cat_emb],  axis=1)

    # ── Buat extended mask ───────────────────────────────────────────────
    # Setiap kolom kategorikal dalam mask asli → diperluas ke emb_size kolom
    # (seluruh dimensi embedding kolom tsb dianggap missing/observed bersama)
    train_num_mask = train_mask[:, num_col_idx]
    train_cat_mask = train_mask[:, cat_col_idx]
    test_num_mask  = test_mask[:, num_col_idx]
    test_cat_mask  = test_mask[:, cat_col_idx]

    emb_sizes_arr = np.array(emb_sizes, dtype=int)

    def extend_mask_emb(mask: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        """
        Perluas mask dari [N, n_cat_cols] → [N, total_emb_dim].
        Setiap kolom kategorikal ke-j diperluas ke sizes[j] kolom.
        """
        N       = mask.shape[0]
        cum     = np.concatenate(([0], sizes.cumsum()))
        result  = np.zeros((N, sizes.sum()), dtype=bool)
        for j in range(len(sizes)):
            col_mask = mask[:, j][:, np.newaxis]           # [N, 1]
            result[:, cum[j]:cum[j + 1]] = np.tile(col_mask, sizes[j])
        return result

    ext_train_cat_mask = extend_mask_emb(train_cat_mask, emb_sizes_arr)
    ext_test_cat_mask  = extend_mask_emb(test_cat_mask,  emb_sizes_arr)

    extend_train_mask = np.concatenate([train_num_mask, ext_train_cat_mask], axis=1)
    extend_test_mask  = np.concatenate([test_num_mask,  ext_test_cat_mask],  axis=1)

    return (train_X, test_X,
            train_mask, test_mask,
            train_num, test_num,
            train_cat_idx, test_cat_idx,
            extend_train_mask, extend_test_mask,
            None,       # cat_bin_num (legacy, tidak dipakai)
            emb_model,  # SupervisedLearnableEmbeddingModel
            emb_sizes)  # list[int]


def mean_std(data, mask):
    mask      = (~mask).astype(np.float32)
    mask_sum  = mask.sum(0)
    mask_sum[mask_sum == 0] = 1
    mean      = (data * mask).sum(0) / mask_sum
    var       = ((data - mean) ** 2 * mask).sum(0) / mask_sum
    std       = np.sqrt(var)
    return mean, std


# ===========================================================================
#  Evaluasi
# ===========================================================================

def get_eval(dataname, X_recon, X_true, truth_cat_idx,
             num_num, emb_model, emb_sizes, mask,
             device='cpu', oos=False):
    """
    Hitung MAE, RMSE (numerik) dan Accuracy (kategorikal).

    Konvensi input — mengikuti pola asli DiffPutter:
    ------------------------------------------------
    X_recon[:, :num_num]   → fitur numerik dalam skala TERNORMALISASI (X-mean)/std
                             (belum di-denorm ke skala asli, konsisten dengan paper)
    X_recon[:, num_num:]   → embedding kategorikal dalam skala ASLI
                             (sudah di-invers-norm: × std_emb + mean_emb)
                             → siap dikirim ke Linear Decoder emb_model

    Mengapa numerik TIDAK di-denorm?
    Kode asli DiffPutter (binary encoding) juga menghitung MAE/RMSE pada skala
    (X-mean)/std, bukan skala asli. Kita ikuti konvensi yang sama agar hasil
    bisa dibandingkan secara apple-to-apple.
    """
    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    # mask: True(1) = missing, False(0) = observed
    num_mask = mask[:, num_col_idx].astype(bool)
    cat_mask = mask[:, cat_col_idx].astype(bool) if len(cat_col_idx) > 0 else None

    num_pred = X_recon[:, :num_num]
    num_true = X_true[:, :num_num]

    # Bagian embedding dari rekonstruksi
    cat_emb_pred = X_recon[:, num_num:]

    # Special case: news dataset
    if dataname == 'news' and oos:
        drop = 6265
        num_mask  = np.delete(num_mask, drop, axis=0)
        num_pred  = np.delete(num_pred, drop, axis=0)
        num_true  = np.delete(num_true, drop, axis=0)
        if cat_mask is not None:
            cat_mask = np.delete(cat_mask, drop, axis=0)
        if truth_cat_idx is not None:
            truth_cat_idx = np.delete(truth_cat_idx, drop, axis=0)
        cat_emb_pred = np.delete(cat_emb_pred, drop, axis=0)

    # ── Numerik: MAE & RMSE (hanya pada posisi missing) ─────────────────
    div  = num_pred[num_mask] - num_true[num_mask]
    mae  = np.abs(div).mean()
    rmse = np.sqrt((div ** 2).mean())

    # ── Kategorikal: Akurasi via Linear Decoder ──────────────────────────
    acc = np.nan
    if (truth_cat_idx is not None
            and len(cat_col_idx) > 0
            and emb_model is not None
            and emb_sizes is not None):

        # Decode embedding → prediksi kelas per kolom
        pred_cat_idx = decode_cat_from_embedding(
            emb_model, cat_emb_pred, device
        )  # [N, n_cat_cols]

        correct_total  = 0
        total_missing  = 0
        emb_sizes_arr  = np.array(emb_sizes, dtype=int)

        for j in range(len(cat_col_idx)):
            rows_miss = cat_mask[:, j]
            if rows_miss.sum() == 0:
                continue

            pred_j  = pred_cat_idx[:, j]
            true_j  = truth_cat_idx[:, j].astype(int)

            # Hitung akurasi hanya pada posisi missing
            correct = (pred_j[rows_miss] == true_j[rows_miss]).sum()
            correct_total += int(correct)
            total_missing += int(rows_miss.sum())

        if total_missing > 0:
            acc = correct_total / total_missing

    return mae, rmse, acc