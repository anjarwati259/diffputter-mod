"""
embedding.py
============

Modul ini menyediakan dua komponen utama untuk menggantikan binary encoding
pada fitur kategorik dalam pipeline imputasi berbasis diffusion:

    1. TabularEmbedding  – mengubah integer category index → embedding kontinu
    2. CategoricalDecoder – mengubah embedding kontinu → predicted label index
                            (digunakan saat evaluasi, setelah diffusion selesai)

Mengapa embedding lebih baik daripada binary encoding untuk diffusion?
----------------------------------------------------------------------
Diffusion model diasumsikan beroperasi di ruang input yang kontinu dan smooth.
Binary encoding menghasilkan vektor diskret {0,1}^b yang:
  - Tidak memiliki struktur metrik (jarak Hamming ≠ jarak semantik antar kelas)
  - Menyebabkan difusi yang tidak smooth di dimensi kategorik
  - Membuat proses denoising sulit karena transisi antar bit tidak berkorelasi
    dengan makna semantik

Embedding kontinu nn.Embedding(n_classes, emb_dim) menghasilkan vektor real
yang:
  - Bisa dilatih untuk mengkodekan similaritas semantik antar kelas
  - Menghasilkan ruang kontinu yang lebih cocok untuk forward/reverse diffusion
  - Tidak memerlukan rounding saat decoding (cukup projeksi + argmax)

Mengapa decoding dilakukan SETELAH imputasi (bukan selama diffusion)?
----------------------------------------------------------------------
Diffusion bekerja di ruang embedding kontinu. Mengkonversi ke diskret selama
proses diffusion akan:
  - Memutus aliran gradient (non-differentiable)
  - Mengubah distribusi input secara tiba-tiba (discontinuous jump)
  - Merusak properti Markov dari proses diffusion

Decoding dilakukan sekali saja di akhir, hanya untuk keperluan evaluasi.
"""

import numpy as np
import torch
import torch.nn as nn


class TabularEmbedding(nn.Module):
    """
    Mengubah fitur kategorik (integer index) menjadi embedding kontinu,
    lalu men-concat dengan fitur numerik.

    Args:
        n_categories_list (list[int]): Jumlah kategori unik per fitur kategorik.
        emb_dim (int or list[int]): Dimensi embedding. Jika int, dipakai untuk semua
                                    fitur; jika list, harus panjangnya sama dengan
                                    n_categories_list.
        num_dim (int): Dimensi fitur numerik (sebelum concat).

    Output dimension:
        num_dim + sum(emb_dims)

    Cara kerja mask saat ada missing value:
    ----------------------------------------
    Jika mask[i, j] == 1 (missing), embedding untuk fitur j pada baris i
    diganti dengan vektor nol (zero-imputation di ruang embedding).
    Ini konsisten dengan cara mask diterapkan pada fitur numerik:
        X_miss = (1 - mask) * X
    Diffusion kemudian belajar untuk merekonstruksi slot yang ter-mask.
    """

    def __init__(self, n_categories_list, emb_dim, num_dim):
        super().__init__()

        self.num_dim = num_dim
        self.n_categories_list = n_categories_list

        if isinstance(emb_dim, int):
            emb_dims = [emb_dim] * len(n_categories_list)
        else:
            assert len(emb_dim) == len(n_categories_list)
            emb_dims = list(emb_dim)

        self.emb_dims = emb_dims
        self.total_cat_dim = sum(emb_dims)
        self.out_dim = num_dim + self.total_cat_dim

        # Satu embedding layer per fitur kategorik
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cat, dim)
            for n_cat, dim in zip(n_categories_list, emb_dims)
        ])

    def forward(self, x_num, cat_idx, cat_mask=None):
        """
        Args:
            x_num    : FloatTensor (N, num_dim)  – fitur numerik (sudah dinormalisasi)
            cat_idx  : LongTensor  (N, n_cat)    – integer index per fitur kategorik
            cat_mask : FloatTensor (N, n_cat) or None
                       1.0 = missing (embedding di-zero-out), 0.0 = observed

        Returns:
            x_out    : FloatTensor (N, num_dim + total_cat_dim)
        """
        cat_embs = []
        for j, emb_layer in enumerate(self.embeddings):
            emb = emb_layer(cat_idx[:, j])          # (N, emb_dim_j)
            if cat_mask is not None:
                # Zero-out embedding pada posisi missing
                # Konsisten dengan X_miss = (1 - mask) * X pada fitur numerik
                missing = cat_mask[:, j].unsqueeze(1)    # (N, 1)
                emb = emb * (1.0 - missing)
            cat_embs.append(emb)

        cat_concat = torch.cat(cat_embs, dim=1)          # (N, total_cat_dim)
        x_out = torch.cat([x_num, cat_concat], dim=1)    # (N, out_dim)
        return x_out

    def embed_only(self, x_num, cat_idx, cat_mask=None):
        """Alias lebih ekspresif untuk forward(), digunakan di main.py."""
        return self.forward(x_num, cat_idx, cat_mask)

    def embed_cat_numpy(self, cat_idx_np, cat_mask_np=None, device='cpu'):
        """
        Convenience: menerima numpy array, mengembalikan numpy array.
        Digunakan untuk memproses batch di CPU/GPU tanpa boilerplate.

        Args:
            cat_idx_np  : np.ndarray (N, n_cat), dtype int64
            cat_mask_np : np.ndarray (N, n_cat), dtype float32 atau None
            device      : str atau torch.device

        Returns:
            cat_emb_np  : np.ndarray (N, total_cat_dim)
        """
        cat_idx_t = torch.tensor(cat_idx_np, dtype=torch.long, device=device)
        mask_t = None
        if cat_mask_np is not None:
            mask_t = torch.tensor(cat_mask_np, dtype=torch.float32, device=device)

        cat_embs = []
        with torch.no_grad():
            for j, emb_layer in enumerate(self.embeddings):
                emb = emb_layer(cat_idx_t[:, j])
                if mask_t is not None:
                    missing = mask_t[:, j].unsqueeze(1)
                    emb = emb * (1.0 - missing)
                cat_embs.append(emb)

        cat_concat = torch.cat(cat_embs, dim=1)
        return cat_concat.cpu().numpy()


class CategoricalDecoder(nn.Module):
    """
    Mendekode embedding kontinu hasil imputasi diffusion kembali ke label diskret.

    Untuk setiap fitur kategorik ke-j, ada satu linear layer:
        Linear(emb_dim_j → n_categories_j)

    Digunakan HANYA saat evaluasi, setelah diffusion selesai.

    Mengapa linear layer (bukan threshold/rounding)?
    -------------------------------------------------
    Embedding yang di-imputasi oleh diffusion tidak harus tepat sama dengan
    embedding aslinya; diffusion dapat menghasilkan vektor yang 'dekat' di
    ruang embedding. Linear layer memproyeksikan embedding tersebut ke ruang
    logit kelas, lalu argmax memilih kelas yang paling dekat.

    Ini setara dengan nearest-neighbor di embedding space, namun lebih fleksibel
    karena proyeksi linear bisa dilatih.
    """

    def __init__(self, n_categories_list, emb_dims):
        super().__init__()

        self.n_categories_list = n_categories_list
        self.emb_dims = emb_dims

        # Satu linear decoder per fitur kategorik
        self.decoders = nn.ModuleList([
            nn.Linear(emb_dim, n_cat)
            for emb_dim, n_cat in zip(emb_dims, n_categories_list)
        ])

        # Batas slice embedding untuk setiap fitur
        self.ends   = np.cumsum(emb_dims)
        self.starts = np.concatenate(([0], self.ends[:-1]))

    def forward(self, cat_emb):
        """
        Args:
            cat_emb : FloatTensor (N, total_cat_dim) – embedding slice dari X_recon

        Returns:
            list of LongTensor (N,) – predicted label index per fitur kategorik
        """
        pred_list = []
        for j, (s, e, decoder) in enumerate(zip(self.starts, self.ends, self.decoders)):
            emb_j   = cat_emb[:, s:e]         # (N, emb_dim_j)
            logits  = decoder(emb_j)           # (N, n_categories_j)
            pred_j  = torch.argmax(logits, dim=1)   # (N,)
            pred_list.append(pred_j)
        return pred_list

    def decode(self, cat_emb_np, device='cpu'):
        """
        Convenience: menerima numpy, mengembalikan list numpy array int.

        Args:
            cat_emb_np : np.ndarray (N, total_cat_dim)
            device     : str atau torch.device

        Returns:
            list of np.ndarray (N,) – predicted label index per fitur kategorik
        """
        cat_emb_t = torch.tensor(cat_emb_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_list = self.forward(cat_emb_t)
        return [p.cpu().numpy() for p in pred_list]