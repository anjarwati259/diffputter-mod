"""
mrmd_discretizer.py
===================
MRmD (Max-Relevance-Min-Divergence) Discretization
Implementasi berdasarkan paper:
  "A Max-Relevance-Min-Divergence criterion for data discretization
   with applications on naive Bayes"
  Wang et al., Pattern Recognition 149 (2024) 110236

============================================================
RINGKASAN METODE
============================================================

Masalah yang diselesaikan:
  Metode diskritisasi lama hanya memaksimalkan discriminant power
  (informasi pembeda antar kelas), sehingga data terpecah menjadi
  terlalu banyak bin kecil → generalisasi buruk.

Solusi MRmD:
  Optimasi DUA kriteria secara bersamaan (Persamaan 13 di paper):

    Ψ(Aj; C) = λ * I(Aj; C)  −  D_JS(P_t(aj) ‖ P_v(aj))

  di mana:
    • I(Aj; C)          = Mutual Information atribut-diskrit vs kelas
                          → memaksimalkan discriminant power
    • D_JS(P_t ‖ P_v)  = Jensen-Shannon Divergence distribusi
                          training vs validation
                          → memaksimalkan generalisasi
    • λ = exp(-|D*_j| / N_D)  (Persamaan 14, N_D=50)
                          → bobot adaptif: awal lebih fokus discriminant,
                            makin banyak cut point makin fokus generalisasi

Algoritma (Algorithm 1 di paper):
  Greedy top-down splitting per atribut:
  1. Mulai dari 1 bin (seluruh range)
  2. Setiap iterasi: coba semua kandidat cut point, pilih yang
     memaksimalkan Ψ
  3. Berhenti bila Ψ tidak meningkat lagi

============================================================
CARA PAKAI
============================================================

    from mrmd_discretizer import MRmDDiscretizer, MRmDNaiveBayes

    # --- Hanya diskritisasi ---
    disc = MRmDDiscretizer(val_size=0.125, N_D=50, random_state=42)
    disc.fit(X_train, y_train)
    X_train_disc = disc.transform(X_train)
    X_test_disc  = disc.transform(X_test)
    disc.summary()   # cetak cut points

    # --- Pipeline MRmD + Naive Bayes ---
    clf = MRmDNaiveBayes(random_state=42)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    # --- Kompatibel dengan scikit-learn ---
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(MRmDNaiveBayes(), X, y, cv=10)

============================================================
DEPENDENCIES
============================================================
  numpy >= 1.20
  scikit-learn >= 1.0
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


# ══════════════════════════════════════════════════════════
#  FUNGSI UTILITAS
# ══════════════════════════════════════════════════════════

def _mutual_information(a_discrete: np.ndarray, c: np.ndarray) -> float:
    """
    Hitung Mutual Information I(A; C).

    Persamaan (3) di paper:
      I(A; C) = Σ_{a,c} P(a,c) * log[ P(a,c) / (P(a) * P(c)) ]

    Parameters
    ----------
    a_discrete : np.ndarray of int
        Hasil diskritisasi satu atribut (label bin).
    c : np.ndarray
        Label kelas.

    Returns
    -------
    float : nilai MI >= 0
    """
    n = len(a_discrete)
    bins_a = np.unique(a_discrete)
    bins_c = np.unique(c)

    mi = 0.0
    for a_val in bins_a:
        mask_a = (a_discrete == a_val)
        p_a = mask_a.sum() / n
        for c_val in bins_c:
            p_ac = ((mask_a) & (c == c_val)).sum() / n
            p_c  = (c == c_val).sum() / n
            if p_ac > 0 and p_a > 0 and p_c > 0:
                mi += p_ac * np.log(p_ac / (p_a * p_c))

    return max(mi, 0.0)


def _js_divergence(p_t: np.ndarray, p_v: np.ndarray) -> float:
    """
    Hitung Jensen-Shannon Divergence D_JS(P_t ‖ P_v).

    Persamaan (4)-(6) di paper:
      P*     = ½(P_t + P_v)
      D_JS   = ½[D_KL(P_t ‖ P*) + D_KL(P_v ‖ P*)]
      D_JS ∈ [0, 1]  (bounded karena menggunakan log natural + ½)

    Parameters
    ----------
    p_t : np.ndarray of float
        Distribusi probabilitas training (harus sum=1).
    p_v : np.ndarray of float
        Distribusi probabilitas validation (harus sum=1).

    Returns
    -------
    float : JSD dalam [0, 1]
    """
    eps    = 1e-10
    p_t    = np.clip(p_t, eps, 1.0)
    p_v    = np.clip(p_v, eps, 1.0)
    p_star = 0.5 * (p_t + p_v)

    kl_t = np.sum(p_t * np.log(p_t / p_star))
    kl_v = np.sum(p_v * np.log(p_v / p_star))

    return float(np.clip(0.5 * (kl_t + kl_v), 0.0, 1.0))


def _get_distributions(a_train: np.ndarray,
                        a_val: np.ndarray):
    """
    Hitung P_t dan P_v dari label bin diskrit (training & validation).
    Semua bin yang muncul di salah satu set diikutkan.

    Returns
    -------
    p_t, p_v : np.ndarray of float, masing-masing sum=1
    """
    all_bins = np.union1d(np.unique(a_train), np.unique(a_val))
    p_t = np.array([(a_train == b).sum() for b in all_bins], dtype=float)
    p_v = np.array([(a_val   == b).sum() for b in all_bins], dtype=float)

    if p_t.sum() > 0: p_t /= p_t.sum()
    if p_v.sum() > 0: p_v /= p_v.sum()

    return p_t, p_v


def _make_bins(cut_points: np.ndarray,
               x_min: float, x_max: float) -> np.ndarray:
    """
    Bangun array edges bins lengkap: [x_min-ε, cp1, cp2, ..., x_max+ε].
    """
    lo = x_min - 1e-10
    hi = x_max + 1e-10
    if len(cut_points) == 0:
        return np.array([lo, hi])
    return np.concatenate([[lo], np.sort(cut_points), [hi]])


def _discretize(x: np.ndarray, cut_points: np.ndarray,
                x_min: float, x_max: float) -> np.ndarray:
    """
    Diskritisasi array x menggunakan cut_points.
    Mengembalikan label bin integer [0, 1, 2, ...].
    """
    if len(cut_points) == 0:
        return np.zeros(len(x), dtype=int)
    bins = _make_bins(cut_points, x_min, x_max)
    # np.digitize: nilai <= bins[1] → 0, dst.
    return (np.digitize(x, bins[1:-1])).astype(int)


# ══════════════════════════════════════════════════════════
#  KELAS UTAMA: MRmDDiscretizer
# ══════════════════════════════════════════════════════════

class MRmDDiscretizer(BaseEstimator, TransformerMixin):
    """
    MRmD (Max-Relevance-Min-Divergence) Discretizer

    Implementasi Algorithm 1 dari:
      Wang et al., Pattern Recognition 149 (2024) 110236

    Kompatibel dengan scikit-learn API (fit / transform / fit_transform).

    Parameters
    ----------
    val_size : float, default=0.125
        Proporsi data yang dijadikan validation internal.
        Paper pakai ≈ 1/9 ≈ 0.111.  Default 1/8 = 0.125.

    N_D : int, default=50
        Parameter pada fungsi λ (Persamaan 14).
        λ = exp(-|D*_j| / N_D)
        Nilai kecil → λ turun cepat → sedikit cut point.
        Nilai besar → lebih banyak cut point.

    random_state : int or None, default=None
        Seed untuk split train/val.

    verbose : bool, default=False
        Cetak progress setiap atribut.

    Attributes
    ----------
    cut_points_ : list of np.ndarray
        cut_points_[j] = cut point optimal untuk fitur ke-j.

    n_features_in_ : int
    feature_names_in_ : np.ndarray (jika input DataFrame)
    """

    def __init__(self, val_size: float = 0.125, N_D: int = 50,
                 random_state=None, verbose: bool = False):
        self.val_size     = val_size
        self.N_D          = N_D
        self.random_state = random_state
        self.verbose      = verbose

    # ── fit ────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Fit MRmD: temukan cut point optimal untuk tiap fitur.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        # Tangani pandas DataFrame
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)
            X = np.array(X, dtype=float)
        else:
            X = np.array(X, dtype=float)

        y = np.array(y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # ── Split internal: training vs validation ─────────
        rng      = np.random.RandomState(self.random_state)
        val_n    = max(1, int(n_samples * self.val_size))
        val_idx  = rng.choice(n_samples, size=val_n, replace=False)
        train_idx = np.setdiff1d(np.arange(n_samples), val_idx)

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_vl        = X[val_idx]

        if self.verbose:
            print(f"[MRmD] n_train={len(train_idx)}, n_val={len(val_idx)}, "
                  f"n_features={n_features}")

        # ── Algorithm 1: loop per atribut ──────────────────
        self.cut_points_ = []
        self.x_min_      = []   # simpan min/max dari TRAINING untuk transform
        self.x_max_      = []

        for j in range(n_features):
            x_tr_j = X_tr[:, j]
            x_vl_j = X_vl[:, j]

            # Simpan range dari X_TRAINING untuk transform nanti
            x_min = x_tr_j.min()
            x_max = x_tr_j.max()
            self.x_min_.append(x_min)
            self.x_max_.append(x_max)

            # Kandidat cut point HANYA dari X_tr (bukan X_val)
            # X_val hanya dipakai untuk menghitung JS-divergence,
            # bukan untuk menentukan posisi kandidat cut point.
            unique_tr = np.unique(x_tr_j)

            if len(unique_tr) <= 1:
                # Atribut konstan → tidak perlu diskritisasi
                self.cut_points_.append(np.array([]))
                if self.verbose:
                    print(f"  Fitur [{j}]: konstan, skip.")
                continue

            cp = self._fit_one_attribute(
                x_tr_j, x_vl_j, y_tr, unique_tr, x_min, x_max, j
            )
            self.cut_points_.append(cp)

        return self

    def _fit_one_attribute(self, x_tr, x_vl, c_tr,
                            unique_all, x_min, x_max, j_idx):
        """
        Algorithm 1 untuk satu atribut.

        Returns
        -------
        D_star_j : np.ndarray, cut point optimal (sorted)
        """
        # Baris 3-5: inisialisasi
        D_star_j = np.array([])          # D*_j ← ∅
        S_j      = unique_all.copy()     # S_j  ← U(x_j) (kandidat cut point)
        psi_max  = -np.inf               # Ψ_max ← -∞

        # Baris 6: while S_j ≠ ∅
        while len(S_j) > 0:

            best_psi = -np.inf
            best_dk  = None

            # Baris 7-11: evaluasi setiap kandidat cut point
            for dk in S_j:

                # Baris 8: D^k_j = D*_j ∪ {dk}
                D_k_j = np.append(D_star_j, dk)

                # Baris 9: diskritisasi x_j dengan D^k_j
                a_tr_disc = _discretize(x_tr, D_k_j, x_min, x_max)
                a_vl_disc = _discretize(x_vl, D_k_j, x_min, x_max)

                # Baris 10: hitung Ψ_k = λ * I(Aj;C) - D_JS(P_t ‖ P_v)
                n_cuts = len(D_star_j) + 1          # jumlah cut point baru
                lam    = np.exp(-n_cuts / self.N_D) # Persamaan (14)

                mi_val  = _mutual_information(a_tr_disc, c_tr)
                p_t, p_v = _get_distributions(a_tr_disc, a_vl_disc)
                jsd_val = _js_divergence(p_t, p_v)

                psi_k = lam * mi_val - jsd_val

                if psi_k > best_psi:
                    best_psi = psi_k
                    best_dk  = dk

            # Baris 12-15: stop jika tidak ada peningkatan
            if best_dk is None or best_psi <= psi_max:
                break

            # Baris 16-18: update state
            psi_max  = best_psi
            D_star_j = np.append(D_star_j, best_dk)
            S_j      = S_j[S_j != best_dk]

        result = np.sort(D_star_j)

        if self.verbose:
            print(f"  Fitur [{j_idx}]: {len(result)} cut points "
                  f"→ {np.round(result, 4).tolist()}")

        return result

    # ── transform ──────────────────────────────────────────

    def transform(self, X) -> np.ndarray:
        """
        Diskritisasi data X menggunakan cut point hasil fit.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X_disc : np.ndarray of int, shape (n_samples, n_features)
            Label bin mulai dari 0.
        """
        check_is_fitted(self, 'cut_points_')

        if hasattr(X, 'values'):
            X = X.values
        X = np.array(X, dtype=float)

        X_disc = np.empty(X.shape, dtype=int)

        for j, cp in enumerate(self.cut_points_):
            x_col = X[:, j]
            # BUG-FIX: pakai x_min/x_max dari TRAINING (disimpan saat fit),
            # bukan dari data yang sedang di-transform.
            # Ini memastikan bins konsisten antara train dan test.
            x_min = self.x_min_[j]
            x_max = self.x_max_[j]
            X_disc[:, j] = _discretize(x_col, cp, x_min, x_max)

        return X_disc

    # ── utilitas ───────────────────────────────────────────

    def get_n_bins(self) -> np.ndarray:
        """Jumlah bin per fitur setelah fit."""
        check_is_fitted(self, 'cut_points_')
        return np.array([len(cp) + 1 for cp in self.cut_points_])

    def summary(self):
        """Cetak tabel ringkasan cut points."""
        check_is_fitted(self, 'cut_points_')
        header = "MRmD Discretizer — Summary"
        print("=" * 60)
        print(f"{header:^60}")
        print("=" * 60)
        print(f"  {'Fitur':<20} {'# Bins':>7}   Cut Points")
        print("  " + "-" * 56)
        for j, cp in enumerate(self.cut_points_):
            if hasattr(self, 'feature_names_in_'):
                name = str(self.feature_names_in_[j])[:20]
            else:
                name = f"fitur_{j}"
            cp_str = np.round(cp, 4).tolist() if len(cp) > 0 else "[ ]"
            print(f"  {name:<20} {len(cp)+1:>7}   {cp_str}")
        print("=" * 60)
        print(f"  Total cut points: {sum(len(c) for c in self.cut_points_)}")
        print(f"  N_D={self.N_D}, val_size={self.val_size}")
        print("=" * 60)


# ══════════════════════════════════════════════════════════
#  PIPELINE: MRmD + Naive Bayes  (MRmD-NB)
# ══════════════════════════════════════════════════════════

class MRmDNaiveBayes(BaseEstimator):
    """
    Pipeline MRmD Discretizer + Categorical Naive Bayes.

    Meniru MRmD-RNB dari paper (versi lite, tanpa regularisasi penuh RNB).
    Kompatibel scikit-learn: bisa dipakai di cross_val_score, GridSearchCV, dll.

    Parameters
    ----------
    val_size     : float, default=0.125
    N_D          : int,   default=50
    random_state : int or None
    verbose      : bool,  default=False
    """

    def __init__(self, val_size=0.125, N_D=50,
                 random_state=None, verbose=False):
        self.val_size     = val_size
        self.N_D          = N_D
        self.random_state = random_state
        self.verbose      = verbose

    def fit(self, X, y):
        """
        Alur sesuai paper Section 4.1:

          X_train (input user = misal 9 folds dari 10-fold CV)
          │
          ├─ split internal oleh MRmD:
          │    ├─ X_tr  (87.5%) → MRmD criterion (I & JS-divergence)
          │    └─ X_val (12.5%) → MRmD criterion (JS-divergence) saja
          │         ↓
          │    cut_points_* ditemukan
          │
          └─ NB dilatih dari X_TRAIN PENUH (100%)
               setelah di-transform pakai cut_points_*
               (val fold hanya untuk MRmD, bukan untuk membatasi NB)

          X_test (fold ke-10) → hanya transform, tidak masuk fitting sama sekali
        """
        from sklearn.naive_bayes import CategoricalNB

        if hasattr(X, "values"):
            X = X.values
        X = np.array(X, dtype=float)
        y = np.array(y)

        # ── Step 1: Temukan cut_points_* via MRmD
        #    (X_val internal hanya untuk JS-divergence criterion)
        self.discretizer_ = MRmDDiscretizer(
            val_size=self.val_size, N_D=self.N_D,
            random_state=self.random_state, verbose=self.verbose
        )
        self.discretizer_.fit(X, y)

        # ── Step 2: Setelah cut_points_* ditemukan, latih NB dari X_TRAIN PENUH
        #    Sesuai paper: X_val internal hanya untuk MRmD criterion.
        #    NB dilatih dari semua X_train (bukan hanya 87.5%).
        X_disc_full = self.discretizer_.transform(X)
        self.classes_ = np.unique(y)
        self.clf_ = CategoricalNB()
        self.clf_.fit(X_disc_full, y)
        return self

    def predict(self, X):
        return self.clf_.predict(self.discretizer_.transform(X))

    def predict_proba(self, X):
        return self.clf_.predict_proba(self.discretizer_.transform(X))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))


# ══════════════════════════════════════════════════════════
#  DEMO (jalankan: python mrmd_discretizer.py)
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.naive_bayes import GaussianNB
    import time

    print("=" * 60)
    print("  Demo: MRmD Discretizer")
    print("  Wang et al., Pattern Recognition 149 (2024) 110236")
    print("=" * 60)

    datasets = {
        "Iris"          : load_iris(return_X_y=True),
        "Wine"          : load_wine(return_X_y=True),
        "Breast Cancer" : load_breast_cancer(return_X_y=True),
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for name, (X, y) in datasets.items():
        print(f"\n{'─'*55}")
        print(f"  {name}  ({X.shape[0]} sampel, {X.shape[1]} fitur)")
        print(f"{'─'*55}")

        # Baseline: Gaussian NB (tanpa diskritisasi)
        gnb = cross_val_score(GaussianNB(), X, y, cv=cv, scoring='accuracy')
        print(f"  GaussianNB (no disc)    : "
              f"{gnb.mean()*100:.2f}% ± {gnb.std()*100:.2f}%")

        # MRmD + Categorical NB
        t0 = time.time()
        mnb = MRmDNaiveBayes(val_size=0.125, N_D=50, random_state=42)
        sc  = cross_val_score(mnb, X, y, cv=cv, scoring='accuracy')
        print(f"  MRmD + CategoricalNB    : "
              f"{sc.mean()*100:.2f}% ± {sc.std()*100:.2f}%  "
              f"({time.time()-t0:.1f}s)")

    # ── Detail cut points untuk Iris
    print(f"\n{'─'*55}")
    print("  Detail Cut Points — Iris Dataset")
    print(f"{'─'*55}")
    X_iris, y_iris = load_iris(return_X_y=True)
    from sklearn.datasets import load_iris
    iris = load_iris()
    disc = MRmDDiscretizer(val_size=0.2, N_D=50, random_state=0, verbose=True)
    disc.fit(X_iris, y_iris)
    disc.feature_names_in_ = np.array(iris.feature_names)
    disc.summary()

    print("\n  5 sampel pertama (original vs diskrit):")
    print("  Original:")
    print(X_iris[:5])
    print("  Diskritisasi:")
    print(disc.transform(X_iris[:5]))