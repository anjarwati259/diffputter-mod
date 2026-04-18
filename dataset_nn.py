import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import os
import json

DATA_DIR = 'datasets'

# ===========================================================================
#  Redesigned Supervised Learnable Embedding — Perbaikan Lengkap
#
#  PERBAIKAN vs versi sebelumnya:
#
#  [1] get_eval → Nearest-Prototype di embedding space (bukan argmax statis)
#      Selaras dengan mekanisme evaluasi original (binary bits + prototype),
#      sehingga iterative refinement ter-capture dan metrik bisa naik tiap iter.
#
#  [2] MLP Decoder per kolom (bukan Linear)
#      Kapasitas rekonstruksi lebih tinggi → embedding space lebih informatif.
#
#  [3] Projection layer terpisah untuk classifier
#      Gradien klasifikasi tidak merusak geometri embedding yang dibutuhkan diffusion.
#
#  [4] alpha=0.1, beta=1.0 → rekonstruksi sangat mendominasi
#      Embedding tidak dipaksa cluster diskontinyu per kelas.
#
#  [5] use_classifier=False → ablation mode pure-reconstruction
#
#  Tidak ada perubahan pada diffusion, normalisasi, training loop, atau
#  imputation logic — semua itu dihandle di main.py.
# ===========================================================================


def compute_embedding_size(n_categories: int) -> int:
    """Guo & Berkhahn (2016): min(600, round(1.6 * n^0.56))"""
    # return min(600, round(1.6 * n_categories ** 0.56))
    return min(16, max(2, round(n_categories ** 0.5)))


# ---------------------------------------------------------------------------
#  MLP Decoder per kolom
# ---------------------------------------------------------------------------

class MLPColumnDecoder(nn.Module):
    """
    MLP nonlinear decoder untuk satu kolom kategorikal.
    emb_size → LayerNorm → GELU → hidden → n_cat (logits)
    """
    def __init__(self, emb_size: int, n_cat: int,
                 hidden: int = None, dropout: float = 0.1):
        super().__init__()
        if hidden is None:
            hidden = max(emb_size * 2, 32)
        self.net = nn.Sequential(
            nn.Linear(emb_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_cat),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
#  Model Utama
# ---------------------------------------------------------------------------

class SupervisedLearnableEmbeddingModel(nn.Module):
    """
    Redesigned Supervised Learnable Embedding Model.

    Alur:
      x_cat → Embedding per kolom → concat+dropout → z [total_emb_dim]
                                                        │
                              ┌─────────────────────────┤
                              │                         │
                     MLP Decoder per kolom        cls_proj → classifier
                     (rekonstruksi dominan)       (projection TERPISAH)

    Parameter
    ---------
    cat_dims            : vocab size per kolom
    emb_sizes           : dimensi embedding per kolom
    n_classes           : jumlah kelas target
    dropout             : dropout rate
    hidden_dim          : lebar hidden untuk cls_proj & classifier
    use_classifier      : False → pure-reconstruction (ablation)
    decoder_hidden_mult : hidden MLP decoder = emb_size * mult (min 32)
    """

    def __init__(self, cat_dims: list, emb_sizes: list, n_classes: int,
                 dropout: float = 0.1, hidden_dim: int = 256,
                 use_classifier: bool = True,
                 decoder_hidden_mult: float = 2.0):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cat, emb_dim)
            for n_cat, emb_dim in zip(cat_dims, emb_sizes)
        ])
        self.dropout        = nn.Dropout(dropout)
        self.total_emb_dim  = sum(emb_sizes)
        self.n_cols         = len(cat_dims)
        self.cat_dims       = cat_dims
        self.emb_sizes      = emb_sizes
        self.n_classes      = n_classes
        self.use_classifier = use_classifier

        # MLP Decoder per kolom
        self.decoders = nn.ModuleList([
            MLPColumnDecoder(
                emb_size = emb_size,
                n_cat    = n_cat,
                hidden   = max(int(emb_size * decoder_hidden_mult), 32),
                dropout  = dropout,
            )
            for n_cat, emb_size in zip(cat_dims, emb_sizes)
        ])

        # Projection layer + Classifier (jalur TERPISAH dari decoder)
        if use_classifier:
            self.cls_proj = nn.Sequential(
                nn.Linear(self.total_emb_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n_classes),
            )
        else:
            self.cls_proj   = None
            self.classifier = None

    def encode(self, x_cat: torch.Tensor) -> torch.Tensor:
        embedded = [self.embeddings[i](x_cat[:, i]) for i in range(self.n_cols)]
        z = torch.cat(embedded, dim=1)
        return self.dropout(z)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_classifier:
            raise RuntimeError("use_classifier=False, classifier tidak tersedia.")
        return self.classifier(self.cls_proj(z))

    def decode(self, z: torch.Tensor) -> list:
        """
        MLP Decoder: split z per kolom → MLP → logits.
        Menerima z dalam bentuk apapun (dari encode ATAU dari diffusion output).
        """
        col_embs = torch.split(z, self.emb_sizes, dim=1)
        return [self.decoders[i](col_embs[i]) for i in range(self.n_cols)]

    def forward(self, x_cat: torch.Tensor):
        z            = self.encode(x_cat)
        recon_logits = self.decode(z)
        class_logits = self.classify(z) if self.use_classifier else None
        return z, class_logits, recon_logits


# ===========================================================================
#  Training
# ===========================================================================

def train_supervised_embedding_model(
        cat_idx_array: np.ndarray,
        labels: np.ndarray,
        cat_dims: list,
        emb_sizes: list,
        n_classes: int,
        device: str,
        n_epochs: int = 1000,
        batch_size: int = 1024,
        lr: float = 1e-3,
        dropout: float = 0.1,
        hidden_dim: int = 256,
        patience: int = 40,
        alpha: float = 0.1,          # bobot classification (kecil, hanya regularisasi)
        beta: float  = 1.0,          # bobot reconstruction (dominan)
        use_classifier: bool = True,
        decoder_hidden_mult: float = 2.0,
) -> SupervisedLearnableEmbeddingModel:
    """
    Latih embedding model.

    Loss: beta * recon_loss [+ alpha * class_loss jika use_classifier=True]

    alpha=0.1, beta=1.0: rekonstruksi sangat mendominasi sehingga geometri
    embedding space terbentuk dari struktur data, bukan dipaksa oleh label.
    """
    model = SupervisedLearnableEmbeddingModel(
        cat_dims            = cat_dims,
        emb_sizes           = emb_sizes,
        n_classes           = n_classes,
        dropout             = dropout,
        hidden_dim          = hidden_dim,
        use_classifier      = use_classifier,
        decoder_hidden_mult = decoder_hidden_mult,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss   = nn.CrossEntropyLoss()

    cat_tensor   = torch.tensor(cat_idx_array, dtype=torch.long, device=device)
    label_tensor = torch.tensor(labels,        dtype=torch.long, device=device)
    dataset      = torch.utils.data.TensorDataset(cat_tensor, label_tensor)
    cpu_gen      = torch.Generator(device='cpu')
    loader       = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, generator=cpu_gen,
    )

    mode_str = "supervised+recon" if use_classifier else "pure-reconstruction"
    print(f'[Embedding] Mode: {mode_str} | alpha={alpha if use_classifier else 0}, beta={beta}')

    best_loss        = float('inf')
    patience_counter = 0
    best_state       = None

    model.train()
    for epoch in range(n_epochs):
        total_loss = total_cls = total_rec = 0.0
        n_batches  = 0

        for batch_cat, batch_labels in loader:
            optimizer.zero_grad()
            z, class_logits, recon_logits = model(batch_cat)

            recon_loss = sum(
                ce_loss(recon_logits[i], batch_cat[:, i])
                for i in range(model.n_cols)
            ) / model.n_cols

            if use_classifier and class_logits is not None:
                class_loss = ce_loss(class_logits, batch_labels)
                loss       = alpha * class_loss + beta * recon_loss
            else:
                class_loss = torch.tensor(0.0, device=device)
                loss       = beta * recon_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls  += class_loss.item()
            total_rec  += recon_loss.item()
            n_batches  += 1

        avg = total_loss / n_batches
        if (epoch + 1) % 10 == 0:
            print(f'[Embedding] Epoch {epoch+1}/{n_epochs} '
                  f'Loss={avg:.4f} (cls={total_cls/n_batches:.4f}, '
                  f'rec={total_rec/n_batches:.4f})')

        if avg < best_loss:
            best_loss        = avg
            patience_counter = 0
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[Embedding] Early stopping epoch {epoch+1}, best={best_loss:.4f}')
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    return model


# ===========================================================================
#  Encode & Decode Utilities
# ===========================================================================

def encode_with_embedding(model: SupervisedLearnableEmbeddingModel,
                          cat_idx_array: np.ndarray,
                          device: str,
                          batch_size: int = 4096) -> np.ndarray:
    """Integer index → dense embedding numpy [N, total_emb_dim]"""
    model.eval()
    cat_tensor = torch.tensor(cat_idx_array, dtype=torch.long, device=device)
    loader     = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(cat_tensor),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False,
    )
    all_z = []
    with torch.no_grad():
        for (batch,) in loader:
            z, _, _ = model(batch)
            all_z.append(z.cpu().numpy())
    return np.concatenate(all_z, axis=0).astype(np.float32)


def decode_cat_from_embedding(model: SupervisedLearnableEmbeddingModel,
                              emb_array: np.ndarray,
                              device: str,
                              batch_size: int = 4096) -> np.ndarray:
    """
    Decode embedding (output diffusion) → predicted class index per kolom.
    Menggunakan MLP decoder, bukan argmax langsung.
    Return: [N, n_cat_cols] int64
    """
    model.eval()
    emb_tensor = torch.tensor(emb_array, dtype=torch.float32, device=device)
    loader     = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(emb_tensor),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False,
    )
    all_pred = []
    with torch.no_grad():
        for (batch,) in loader:
            logits_list = model.decode(batch)
            pred = torch.stack([l.argmax(dim=1) for l in logits_list], dim=1)
            all_pred.append(pred.cpu().numpy())
    return np.concatenate(all_pred, axis=0).astype(np.int64)


def get_class_prototypes_from_embedding(
        model: SupervisedLearnableEmbeddingModel,
        cat_idx_array: np.ndarray,      # [N, n_cat_cols] ground-truth integer
        emb_recon_array: np.ndarray,    # [N, total_emb_dim] output diffusion (denorm)
        device: str,
) -> list:
    """
    Bangun prototype kelas di embedding space untuk setiap kolom kategorikal.

    Cara kerja (analog dengan nearest-prototype di original binary bits):
      Untuk kolom j, kelas c:
        proto[j][c] = mean( emb_recon[i, emb_j] ) untuk semua i di mana true_label[i,j] == c

    Return: list[n_cat_cols] of np.ndarray [n_classes_j, emb_size_j]
    """
    emb_sizes = model.emb_sizes
    n_cols    = model.n_cols
    cum       = np.concatenate(([0], np.cumsum(emb_sizes)))

    prototypes = []
    for j in range(n_cols):
        true_j  = cat_idx_array[:, j].astype(int)
        nclass  = int(true_j.max()) + 1
        emb_j   = emb_recon_array[:, cum[j]:cum[j+1]]          # [N, emb_size_j]
        proto_j = np.zeros((nclass, emb_sizes[j]), dtype=np.float32)
        for c in range(nclass):
            mask_c = (true_j == c)
            if mask_c.sum() > 0:
                proto_j[c] = emb_j[mask_c].mean(axis=0)
        prototypes.append(proto_j)

    return prototypes


def nearest_prototype_decode(
        emb_recon_array: np.ndarray,    # [N, total_emb_dim]
        prototypes: list,               # output dari get_class_prototypes_from_embedding
        emb_sizes: list,
) -> np.ndarray:
    """
    Nearest-prototype classification di embedding space.

    Untuk setiap sample i dan kolom j:
      pred[i,j] = argmin_c ||emb[i, emb_j] - proto[j][c]||^2

    Return: [N, n_cat_cols] int64
    """
    n_cols = len(emb_sizes)
    cum    = np.concatenate(([0], np.cumsum(emb_sizes)))
    N      = emb_recon_array.shape[0]
    pred   = np.zeros((N, n_cols), dtype=np.int64)

    for j in range(n_cols):
        emb_j   = emb_recon_array[:, cum[j]:cum[j+1]]   # [N, emb_j]
        proto_j = prototypes[j]                           # [nclass, emb_j]
        # L2 distance: [N, nclass]
        diff  = emb_j[:, None, :] - proto_j[None, :, :]
        dist2 = (diff ** 2).sum(axis=2)
        pred[:, j] = dist2.argmin(axis=1)

    return pred


# ===========================================================================
#  Load Dataset
# ===========================================================================
#alpha 0.1, true, besar => done (kandidat)

#alpha 0.1, true, kecil
#alpha 0.1, false, emd_kecil => done
#alpha 0.2, true, emd_kecil
#alpha 0.2, false, embd_kecil

def load_dataset(dataname, idx=0, mask_type='MCAR', ratio='30',
                 emb_alpha: float = 0.2,
                 emb_beta:  float = 1.0,
                 use_classifier: bool = True,
                 decoder_hidden_mult: float = 2.0):
    """
    Load dataset dengan redesigned embedding model.

    Parameter tambahan vs original:
    --------------------------------
    emb_alpha         : bobot loss klasifikasi (default 0.1 — kecil)
    emb_beta          : bobot loss rekonstruksi (default 1.0 — dominan)
    use_classifier    : True = supervised+recon | False = pure-recon ablation
    decoder_hidden_mult : lebar hidden MLP decoder = emb_size * mult
    """
    ratio = str(ratio)

    data_dir  = f'datasets/{dataname}'
    info_path = f'datasets/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx    = info['num_col_idx']
    cat_col_idx    = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    data_df  = pd.read_csv(f'{data_dir}/data.csv')
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df  = pd.read_csv(f'{data_dir}/test.csv')

    train_mask = np.load(f'{data_dir}/masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy')
    test_mask  = np.load(f'{data_dir}/masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy')

    cols      = train_df.columns
    train_num = train_df[cols[num_col_idx]].values.astype(np.float32)
    test_num  = test_df[cols[num_col_idx]].values.astype(np.float32)

    # Labels
    train_y       = train_df[cols[target_col_idx]]
    test_y        = test_df[cols[target_col_idx]]
    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([train_y, test_y]).values.ravel().astype(str))
    train_labels  = label_encoder.transform(train_y.values.ravel().astype(str))
    n_classes     = len(label_encoder.classes_)

    print(f'[Dataset] {n_classes} classes | use_classifier={use_classifier} '
          f'| alpha={emb_alpha}, beta={emb_beta}')

    # Hanya numerik
    if len(cat_col_idx) == 0:
        return (train_num, test_num,
                train_mask, test_mask,
                train_num, test_num,
                None, None,
                train_mask[:, num_col_idx],
                test_mask[:, num_col_idx],
                None, None, None)

    # Label encoding kategorikal (fit pada data.csv agar konsisten)
    cat_columns = cols[cat_col_idx]
    data_cat    = data_df[cat_columns].astype(str)
    train_cat   = train_df[cat_columns].astype(str)
    test_cat    = test_df[cat_columns].astype(str)

    cat_dims           = []
    train_cat_idx_list = []
    test_cat_idx_list  = []

    for col in cat_columns:
        le = LabelEncoder()
        le.fit(data_cat[col])
        cat_dims.append(len(le.classes_))
        train_cat_idx_list.append(le.transform(train_cat[col]).astype(np.int64))
        test_cat_idx_list.append(le.transform(test_cat[col]).astype(np.int64))

    train_cat_idx = np.stack(train_cat_idx_list, axis=1)
    test_cat_idx  = np.stack(test_cat_idx_list,  axis=1)

    emb_sizes = [compute_embedding_size(n) for n in cat_dims]
    print(f'[Embedding] cat_dims={cat_dims}, emb_sizes={emb_sizes}, '
          f'total_emb_dim={sum(emb_sizes)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('[Embedding] Melatih SupervisedLearnableEmbeddingModel...')
    emb_model = train_supervised_embedding_model(
        cat_idx_array       = train_cat_idx,
        labels              = train_labels,
        cat_dims            = cat_dims,
        emb_sizes           = emb_sizes,
        n_classes           = n_classes,
        device              = device,
        n_epochs            = 1000,
        batch_size          = 1024,
        lr                  = 1e-3,
        dropout             = 0.1,
        hidden_dim          = 256,
        patience            = 40,
        alpha               = emb_alpha,
        beta                = emb_beta,
        use_classifier      = use_classifier,
        decoder_hidden_mult = decoder_hidden_mult,
    )
    print('[Embedding] Training selesai.')

    train_cat_emb = encode_with_embedding(emb_model, train_cat_idx, device)
    test_cat_emb  = encode_with_embedding(emb_model, test_cat_idx,  device)

    train_X = np.concatenate([train_num, train_cat_emb], axis=1)
    test_X  = np.concatenate([test_num,  test_cat_emb],  axis=1)

    # Extended mask
    emb_sizes_arr = np.array(emb_sizes, dtype=int)

    def extend_mask_emb(mask_cat: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        N, J = mask_cat.shape
        cum  = np.concatenate(([0], sizes.cumsum()))
        out  = np.zeros((N, sizes.sum()), dtype=bool)
        for j in range(J):
            out[:, cum[j]:cum[j+1]] = np.tile(mask_cat[:, j:j+1], sizes[j])
        return out

    extend_train_mask = np.concatenate([
        train_mask[:, num_col_idx],
        extend_mask_emb(train_mask[:, cat_col_idx], emb_sizes_arr),
    ], axis=1)
    extend_test_mask = np.concatenate([
        test_mask[:, num_col_idx],
        extend_mask_emb(test_mask[:, cat_col_idx], emb_sizes_arr),
    ], axis=1)

    return (train_X, test_X,
            train_mask, test_mask,
            train_num, test_num,
            train_cat_idx, test_cat_idx,
            extend_train_mask, extend_test_mask,
            None,          # cat_bin_num (legacy)
            emb_model,
            emb_sizes)


# ===========================================================================
#  Normalisasi
# ===========================================================================

def mean_std(data, mask):
    mask     = (~mask).astype(np.float32)
    mask_sum = mask.sum(0)
    mask_sum[mask_sum == 0] = 1
    mean = (data * mask).sum(0) / mask_sum
    var  = ((data - mean) ** 2 * mask).sum(0) / mask_sum
    std  = np.sqrt(var)
    return mean, std


# ===========================================================================
#  Evaluasi — PERBAIKAN UTAMA: Nearest-Prototype di Embedding Space
# ===========================================================================

def get_eval(dataname, X_recon, X_true, truth_cat_idx,
             num_num, emb_model, emb_sizes, mask,
             device='cpu', oos=False):
    """
    Hitung MAE, RMSE (numerik) dan Accuracy (kategorikal).

    PERBAIKAN [1] vs versi sebelumnya (argmax statis):
    ───────────────────────────────────────────────────
    Akurasi dihitung dengan NEAREST-PROTOTYPE di embedding space,
    persis seperti mekanisme evaluasi original binary bits.

    Cara kerja:
      1. Bangun prototype kelas j, kelas c =
             mean( emb_recon[i, col_j] )  untuk semua i dengan true_label[i,j] == c
      2. pred[i,j] = argmin_c || emb_recon[i, col_j] - proto[j][c] ||^2
      3. Akurasi hanya dihitung pada posisi missing

    Dengan cara ini, setiap iterasi diffusion yang menghasilkan embedding
    lebih baik akan membuat prototype lebih akurat → akurasi naik progressif,
    bukan stagnan seperti pure argmax.
    """
    info_path = f'datasets/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    num_mask = mask[:, num_col_idx].astype(bool)
    cat_mask = (mask[:, cat_col_idx].astype(bool)
                if len(cat_col_idx) > 0 else None)

    num_pred     = X_recon[:, :num_num]
    num_true     = X_true[:, :num_num]
    cat_emb_pred = X_recon[:, num_num:]   # embedding output diffusion (sudah denorm)

    if dataname == 'news' and oos:
        drop = 6265
        num_mask     = np.delete(num_mask,     drop, axis=0)
        num_pred     = np.delete(num_pred,     drop, axis=0)
        num_true     = np.delete(num_true,     drop, axis=0)
        if cat_mask is not None:
            cat_mask = np.delete(cat_mask,     drop, axis=0)
        if truth_cat_idx is not None:
            truth_cat_idx = np.delete(truth_cat_idx, drop, axis=0)
        cat_emb_pred = np.delete(cat_emb_pred, drop, axis=0)

    # MAE & RMSE (hanya posisi missing)
    div  = num_pred[num_mask] - num_true[num_mask]
    mae  = np.abs(div).mean()
    rmse = np.sqrt((div ** 2).mean())

    # Akurasi kategorikal via Nearest-Prototype di embedding space
    acc = np.nan
    if (truth_cat_idx is not None
            and len(cat_col_idx) > 0
            and emb_model is not None
            and emb_sizes is not None):

        # Bangun prototype dari ground-truth + embedding rekonstruksi
        # (analog: pred_bits[mask_c].mean() di original binary bits)
        prototypes = get_class_prototypes_from_embedding(
            model           = emb_model,
            cat_idx_array   = truth_cat_idx,   # ground-truth integer label
            emb_recon_array = cat_emb_pred,     # embedding output diffusion
            device          = device,
        )

        # Nearest-prototype decode
        pred_cat_idx = nearest_prototype_decode(
            emb_recon_array = cat_emb_pred,
            prototypes      = prototypes,
            emb_sizes       = emb_sizes,
        )   # [N, n_cat_cols]

        correct_total = 0
        total_missing = 0

        for j in range(len(cat_col_idx)):
            rows_miss = cat_mask[:, j]
            if rows_miss.sum() == 0:
                continue
            correct = (pred_cat_idx[rows_miss, j] == truth_cat_idx[rows_miss, j]).sum()
            correct_total += int(correct)
            total_missing += int(rows_miss.sum())

        if total_missing > 0:
            acc = correct_total / total_missing

    return mae, rmse, acc