import os
import re
import argparse
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# -----------------------------
# Utility
# -----------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_plots_dir():
    os.makedirs("plots", exist_ok=True)


def plot_loss_curve(train_hist: List[float], val_hist: List[float], name: str):
    ensure_plots_dir()
    plt.figure()
    plt.plot(train_hist, label="train")
    plt.plot(val_hist, label="dev")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{name} loss")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join("plots", f"{name}_loss.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] Saved loss curve to {out_path}")


def plot_embeddings_pca(
    encoder: nn.Module,
    X_by_user: Dict[str, np.ndarray],
    known_users: List[str],
    unknown_users: List[str],
    device,
    is_heatmap: bool,
    name: str,
):
    ensure_plots_dir()

    from itertools import chain

    all_embeddings = []
    all_labels = []

    # Use all rows per user for visualisation
    with torch.no_grad():
        for u in chain(known_users, unknown_users):
            X = X_by_user[u]
            if X.shape[0] == 0:
                continue
            emb = compute_embeddings_encoder(encoder, X, device, is_heatmap=is_heatmap)
            all_embeddings.append(emb)
            if u in known_users:
                all_labels.extend([u] * emb.shape[0])
            else:
                all_labels.extend(["Unknown"] * emb.shape[0])

    if not all_embeddings:
        print(f"[Plot] No embeddings to plot for {name}")
        return

    E = np.vstack(all_embeddings)
    labels = np.array(all_labels, dtype=object)

    # PCA to 2D
    pca = PCA(n_components=2)
    Z = pca.fit_transform(E)

    plt.figure(figsize=(7, 6))
    unique_labels = sorted(set(labels.tolist()))
    for lab in unique_labels:
        idx = labels == lab
        plt.scatter(Z[idx, 0], Z[idx, 1], label=lab, alpha=0.7, s=20)
    plt.title(f"{name} embeddings (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join("plots", f"{name}_pca.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[Plot] Saved PCA embedding plot to {out_path}")


# -----------------------------
# Data loading & split helpers
# -----------------------------

HEATMAP_PATTERN = re.compile(r"mouse_heatmap_r(\d+)_c(\d+)")

MOUSE_STATS = [
    "mouse_avg_speed", "mouse_click_pause_mean", "mouse_click_pause_std",
    "mouse_left_hold_mean", "mouse_left_hold_std",
    "mouse_right_hold_mean", "mouse_right_hold_std",
]

KEYSTROKE_STATS = [
    "key_press_count", "key_avg_hold", "key_std_hold",
    "key_avg_dd", "key_std_dd", "key_avg_rp", "key_std_rp",
    "key_avg_rr", "key_cpm",
]

GUI_STATS = [
    "gui_focus_time", "gui_switch_count", "gui_unique_apps", "gui_window_event_count",
]


def discover_users(data_dir: str) -> List[str]:
    users = []
    for fn in os.listdir(data_dir):
        if fn.lower().endswith(".csv") and fn.lower().startswith("user"):
            users.append(os.path.splitext(fn)[0])  # "user1"
    users = sorted(users, key=lambda x: int(re.findall(r'\d+', x)[0]))
    return users


def load_user_df(data_dir: str, user_name: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{user_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    return df


def split_known_user_indices(
    n_rows: int,
    train_rows: int = 45,
    dev_within_train: int = 9,
    test_rows: int = 15,
):
    assert n_rows >= (train_rows + test_rows), f"Need at least {train_rows + test_rows} rows, found {n_rows}."
    train_end = train_rows - dev_within_train  # 36
    dev_end = train_rows                        # 45
    test_end = train_rows + test_rows          # 60

    train_idx = list(range(0, train_end))
    dev_idx   = list(range(train_end, dev_end))
    test_idx  = list(range(dev_end, test_end))
    return np.array(train_idx), np.array(dev_idx), np.array(test_idx)


def select_unknown_test_indices(n_rows: int, unknown_test_rows: int = 15) -> np.ndarray:
    if unknown_test_rows <= 0 or unknown_test_rows >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    start = n_rows - unknown_test_rows
    return np.arange(start, n_rows, dtype=np.int64)


# -----------------------------
# Feature extraction & scaling
# -----------------------------

def extract_heatmap(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    heat_cols = [c for c in df.columns if HEATMAP_PATTERN.match(c)]
    heat_cols = sorted(
        heat_cols,
        key=lambda c: (
            int(HEATMAP_PATTERN.match(c).group(1)),
            int(HEATMAP_PATTERN.match(c).group(2)),
        ),
    )
    assert len(heat_cols) == 9 * 16, f"Expected 144 heatmap columns (9x16); got {len(heat_cols)}"
    heat = df[heat_cols].values.astype(np.float32).reshape(-1, 9, 16)
    return heat, heat_cols


def extract_subset(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    exist = [c for c in cols if c in df.columns]
    if len(exist) != len(cols):
        missing = set(cols) - set(exist)
        raise ValueError(f"Missing columns: {sorted(list(missing))}")
    X = df[exist].values.astype(np.float32)
    return X, exist


def fit_partial_robust_scaler(train_arrays: List[np.ndarray]) -> Tuple[Optional[RobustScaler], Optional[np.ndarray]]:
    if len(train_arrays) == 0:
        return None, None
    A = np.concatenate(train_arrays, axis=0)
    col_min = A.min(axis=0)
    col_max = A.max(axis=0)
    eps = 1e-6
    passthrough = (col_min >= -eps) & (col_max <= 1.0 + eps)
    scale_mask = ~passthrough
    if scale_mask.sum() == 0:
        return None, scale_mask
    scaler = RobustScaler().fit(A[:, scale_mask])
    return scaler, scale_mask


def apply_partial_scaler(X: np.ndarray, scaler: Optional[RobustScaler], mask: Optional[np.ndarray]) -> np.ndarray:
    if X.shape[1] == 0:
        return X
    if scaler is None or mask is None or mask.sum() == 0:
        return X
    X_out = X.copy()
    X_out[:, mask] = scaler.transform(X_out[:, mask])
    return X_out


# -----------------------------
# Datasets per modality
# -----------------------------

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HeatmapDataset(Dataset):
    def __init__(self, H: np.ndarray, y: np.ndarray):
        self.H = torch.tensor(H, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.H.shape[0]

    def __getitem__(self, idx):
        return self.H[idx].unsqueeze(0), self.y[idx]


# -----------------------------
# Models per modality
# -----------------------------

class HeatmapEncoder(nn.Module):
    def __init__(self, emb_dim=32, p_drop=0.15):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 9 * 16, 64)
        self.out = nn.Linear(64, emb_dim)
        self.drop = nn.Dropout(p_drop)
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        z = F.relu(self.conv1(x))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = z.flatten(1)
        z = F.relu(self.fc(z))
        z = self.norm(z)
        z = self.drop(z)
        z = self.out(z)
        z = F.normalize(z, p=2, dim=1)
        return z


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim=32, hidden=128, p_drop=0.15):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim) if in_dim > 0 else None
        self.fc1 = nn.Linear(in_dim, hidden) if in_dim > 0 else None
        self.fc2 = nn.Linear(hidden, 64) if in_dim > 0 else None
        self.out = nn.Linear(64, emb_dim) if in_dim > 0 else None
        self.drop = nn.Dropout(p_drop)
        self.ln = nn.LayerNorm(64) if in_dim > 0 else None
        self.in_dim = in_dim

    def forward(self, x):
        if self.in_dim == 0:
            return F.normalize(torch.zeros((x.size(0), 32), device=x.device), p=2, dim=1)
        z = self.bn(x)
        z = F.relu(self.fc1(z))
        z = self.drop(z)
        z = F.relu(self.fc2(z))
        z = self.ln(z)
        z = self.drop(z)
        z = self.out(z)
        z = F.normalize(z, p=2, dim=1)
        return z


class ClfHead(nn.Module):
    def __init__(self, emb_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(emb_dim, n_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)


# -----------------------------
# Training utility (per modality)
# -----------------------------

def train_single_modality(
    encoder,
    head: Optional[ClfHead],
    n_classes: int,
    train_loader,
    dev_loader,
    device,
    emb_dim=32,
    max_epochs=60,
    lr=1e-3,
    weight_decay=1e-4,
    patience=8,
) -> Tuple[nn.Module, Optional[nn.Module], List[float], List[float]]:
    enc = encoder.to(device)
    h = head.to(device) if head is not None else None

    params = list(enc.parameters())
    center = None
    if n_classes >= 2 and h is not None:
        params += list(h.parameters())
    else:
        center = nn.Parameter(torch.randn(emb_dim), requires_grad=True).to(device)
        params += [center]

    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_state = None
    bad = 0

    train_hist: List[float] = []
    val_hist: List[float] = []

    def epoch_step(loader, train=True):
        if train:
            enc.train()
            if h is not None:
                h.train()
        else:
            enc.eval()
            if h is not None:
                h.eval()

        total = 0.0
        with torch.set_grad_enabled(train):
            for batch in loader:
                xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                if train:
                    opt.zero_grad()
                emb = enc(xb)
                if n_classes >= 2 and h is not None:
                    logits = h(emb)
                    loss = F.cross_entropy(logits, yb)
                else:
                    c = F.normalize(center, p=2, dim=0)
                    sim = F.cosine_similarity(emb, c.unsqueeze(0), dim=1)
                    loss = (1.0 - sim).mean()
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                total += float(loss.item()) * yb.size(0)
        return total / len(loader.dataset)

    for ep in range(1, max_epochs + 1):
        tr = epoch_step(train_loader, True)
        val = epoch_step(dev_loader, False)
        train_hist.append(tr)
        val_hist.append(val)
        print(f"[Modality Train] Epoch {ep:02d} | train_loss={tr:.4f} | val_loss={val:.4f}")
        if val + 1e-6 < best_val:
            best_val = val
            st = {"enc": enc.state_dict()}
            if h is not None:
                st["head"] = h.state_dict()
            if center is not None:
                st["center"] = center.detach().cpu().numpy()
            best_state = st
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        enc.load_state_dict(best_state["enc"])
        if h is not None and "head" in best_state:
            h.load_state_dict(best_state["head"])

    return enc, h, train_hist, val_hist


# -----------------------------
# Embedding & prototypes & thresholds
# -----------------------------

def compute_embeddings_encoder(encoder, X, device, batch_size=64, is_heatmap=False):
    if is_heatmap:
        ds = HeatmapDataset(X, np.zeros((len(X),), dtype=np.int64))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        encoder.eval()
        outs = []
        with torch.no_grad():
            for xb, _ in dl:
                emb = encoder(xb.to(device))
                outs.append(emb.cpu().numpy())
        return np.concatenate(outs, axis=0)
    else:
        ds = TabularDataset(X, np.zeros((len(X),), dtype=np.int64))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        encoder.eval()
        outs = []
        with torch.no_grad():
            for xb, _ in dl:
                emb = encoder(xb.to(device))
                outs.append(emb.cpu().numpy())
        return np.concatenate(outs, axis=0)


def class_prototypes(emb_by_user: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    protos = {}
    for u, E in emb_by_user.items():
        mu = E.mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        protos[u] = mu
    return protos


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = b / (np.linalg.norm(b) + 1e-12)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    return (a * b).sum(axis=1)


def compute_dev_thresholds(
    dev_emb_by_user: Dict[str, np.ndarray],
    protos: Dict[str, np.ndarray],
    quantile=0.95,
) -> Dict[str, float]:
    tau = {}
    for u, E in dev_emb_by_user.items():
        sims = cosine_similarity(E, protos[u])
        d = 1.0 - sims
        tau[u] = float(np.quantile(d, quantile))
    return tau


def nearest_proto_predict(
    emb: np.ndarray,
    protos: Dict[str, np.ndarray],
    tau: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    users = list(protos.keys())
    M = np.stack([protos[u] for u in users], axis=0)  # (K,D)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)

    embn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims_all = embn @ M.T  # (N,K)
    best_idx = sims_all.argmax(axis=1)
    best_sims = sims_all[np.arange(emb.shape[0]), best_idx]
    best_users = np.array([users[i] for i in best_idx], dtype=object)
    dists = 1.0 - best_sims
    pred = []
    for i in range(len(dists)):
        u = best_users[i]
        pred.append(u if dists[i] <= tau[u] else "Unknown")
    return np.array(pred, dtype=object), best_sims, best_users


# -----------------------------
# Dev weighting for fusion
# -----------------------------

def dev_macro_f1_for_modality(
    encoder,
    train_idx_by_user,
    dev_idx_by_user,
    X_by_user,
    y_map,
    device,
    is_heatmap=False,
) -> float:
    train_emb_by_user = {}
    dev_emb, dev_true = [], []
    for u in y_map.keys():
        tr_idx = train_idx_by_user[u]
        dv_idx = dev_idx_by_user[u]
        if is_heatmap:
            train_emb_by_user[u] = compute_embeddings_encoder(
                encoder, X_by_user[u][tr_idx], device, is_heatmap=True
            )
            dev_emb.append(
                compute_embeddings_encoder(
                    encoder, X_by_user[u][dv_idx], device, is_heatmap=True
                )
            )
        else:
            train_emb_by_user[u] = compute_embeddings_encoder(
                encoder, X_by_user[u][tr_idx], device, is_heatmap=False
            )
            dev_emb.append(
                compute_embeddings_encoder(
                    encoder, X_by_user[u][dv_idx], device, is_heatmap=False
                )
            )
        dev_true.extend([u] * len(dv_idx))
    protos = class_prototypes(train_emb_by_user)
    dev_emb = np.concatenate(dev_emb, axis=0)
    users = list(y_map.keys())
    M = np.stack([protos[u] for u in users], axis=0)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    devn = dev_emb / (np.linalg.norm(dev_emb, axis=1, keepdims=True) + 1e-12)
    sims_all = devn @ M.T
    best_idx = sims_all.argmax(axis=1)
    dev_pred = [users[i] for i in best_idx]
    return f1_score(dev_true, dev_pred, labels=users, average="macro")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--known_users",
        nargs="+",
        default=None,
        help="One or more users, e.g., user1 user4 (>=1, <=6 recommended)",
    )
    parser.add_argument("--train_rows", type=int, default=45)
    parser.add_argument("--dev_within_train", type=int, default=9)
    parser.add_argument("--test_rows", type=int, default=15)
    parser.add_argument("--unknown_test_rows", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau_quantile", type=float, default=0.95)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    users_all = discover_users(args.data_dir)
    if len(users_all) < 2:
        raise RuntimeError("Need at least 1 known and 1 unknown user.")

    if args.known_users is None:
        k = min(3, max(1, len(users_all) - 1))
        known_users = users_all[:k]
    else:
        known_users = list(args.known_users)
    if len(known_users) > 6:
        print("Warning: too many known users; capping to 6.")
        known_users = known_users[:6]
    unknown_users = [u for u in users_all if u not in known_users]
    if len(unknown_users) == 0:
        raise RuntimeError("No unknown users left; please ensure at least one user not in --known_users.")
    print(f"Known users ({len(known_users)}): {known_users}")
    print(f"Unknown users ({len(unknown_users)}): {unknown_users}")

    # Load dfs
    df_by_user = {u: load_user_df(args.data_dir, u) for u in users_all}

    # Extract modalities
    heat_by_user, mouse_by_user, key_by_user, gui_by_user = {}, {}, {}, {}
    for u, df in df_by_user.items():
        H, _ = extract_heatmap(df)
        M, _ = extract_subset(df, MOUSE_STATS)
        K, _ = extract_subset(df, KEYSTROKE_STATS)
        G, _ = extract_subset(df, GUI_STATS)
        heat_by_user[u] = H
        mouse_by_user[u] = M
        key_by_user[u] = K
        gui_by_user[u] = G

    # Known splits
    train_idx_by_user, dev_idx_by_user, test_idx_by_user = {}, {}, {}
    for u in known_users:
        n_rows = len(df_by_user[u])
        tr, dv, te = split_known_user_indices(
            n_rows, args.train_rows, args.dev_within_train, args.test_rows
        )
        train_idx_by_user[u] = tr
        dev_idx_by_user[u] = dv
        test_idx_by_user[u] = te

    # Build label map
    label_map = {u: i for i, u in enumerate(known_users)}

    # Fit partial scalers on known-train ONLY for tabular modalities
    def build_train_arrays(by_user_dict):
        arrs = []
        for u in known_users:
            idx = train_idx_by_user[u]
            arrs.append(by_user_dict[u][idx])
        return arrs

    mouse_scaler, mouse_mask = fit_partial_robust_scaler(build_train_arrays(mouse_by_user))
    key_scaler, key_mask = fit_partial_robust_scaler(build_train_arrays(key_by_user))
    gui_scaler, gui_mask = fit_partial_robust_scaler(build_train_arrays(gui_by_user))

    # Apply scaling globally
    for u in users_all:
        mouse_by_user[u] = apply_partial_scaler(mouse_by_user[u], mouse_scaler, mouse_mask)
        key_by_user[u] = apply_partial_scaler(key_by_user[u], key_scaler, key_mask)
        gui_by_user[u] = apply_partial_scaler(gui_by_user[u], gui_scaler, gui_mask)

    # Build per-modality datasets (known train/dev)
    def build_tab_datasets(by_user_dict):
        X_train, y_train, X_dev, y_dev = [], [], [], []
        for u in known_users:
            tr, dv = train_idx_by_user[u], dev_idx_by_user[u]
            X_train.append(by_user_dict[u][tr])
            y_train += [label_map[u]] * len(tr)
            X_dev.append(by_user_dict[u][dv])
            y_dev += [label_map[u]] * len(dv)
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.array(y_train, dtype=np.int64)
        X_dev = np.concatenate(X_dev, axis=0)
        y_dev = np.array(y_dev, dtype=np.int64)
        return X_train, y_train, X_dev, y_dev

    def build_heat_datasets():
        H_train, y_train, H_dev, y_dev = [], [], [], []
        for u in known_users:
            tr, dv = train_idx_by_user[u], dev_idx_by_user[u]
            H_train.append(heat_by_user[u][tr])
            y_train += [label_map[u]] * len(tr)
            H_dev.append(heat_by_user[u][dv])
            y_dev += [label_map[u]] * len(dv)
        return (
            np.concatenate(H_train, axis=0),
            np.array(y_train),
            np.concatenate(H_dev, axis=0),
            np.array(y_dev),
        )

    H_tr, yH_tr, H_dv, yH_dv = build_heat_datasets()
    M_tr, yM_tr, M_dv, yM_dv = build_tab_datasets(mouse_by_user)
    K_tr, yK_tr, K_dv, yK_dv = build_tab_datasets(key_by_user)
    G_tr, yG_tr, G_dv, yG_dv = build_tab_datasets(gui_by_user)

    # Create loaders
    bs = args.batch_size
    heat_train_loader = DataLoader(HeatmapDataset(H_tr, yH_tr), batch_size=bs, shuffle=True)
    heat_dev_loader = DataLoader(HeatmapDataset(H_dv, yH_dv), batch_size=bs, shuffle=False)
    mouse_train_loader = DataLoader(TabularDataset(M_tr, yM_tr), batch_size=bs, shuffle=True)
    mouse_dev_loader = DataLoader(TabularDataset(M_dv, yM_dv), batch_size=bs, shuffle=False)
    key_train_loader = DataLoader(TabularDataset(K_tr, yK_tr), batch_size=bs, shuffle=True)
    key_dev_loader = DataLoader(TabularDataset(K_dv, yK_dv), batch_size=bs, shuffle=False)
    gui_train_loader = DataLoader(TabularDataset(G_tr, yG_tr), batch_size=bs, shuffle=True)
    gui_dev_loader = DataLoader(TabularDataset(G_dv, yG_dv), batch_size=bs, shuffle=False)

    n_classes = len(known_users)

    # Build encoders + heads
    heat_encoder = HeatmapEncoder(emb_dim=32, p_drop=0.15)
    heat_head = ClfHead(32, n_classes) if n_classes >= 2 else None
    mouse_encoder = MLPEncoder(
        in_dim=mouse_by_user[known_users[0]].shape[1], emb_dim=32, hidden=128, p_drop=0.15
    )
    mouse_head = ClfHead(32, n_classes) if n_classes >= 2 else None
    key_encoder = MLPEncoder(
        in_dim=key_by_user[known_users[0]].shape[1], emb_dim=32, hidden=128, p_drop=0.15
    )
    key_head = ClfHead(32, n_classes) if n_classes >= 2 else None
    gui_encoder = MLPEncoder(
        in_dim=gui_by_user[known_users[0]].shape[1], emb_dim=32, hidden=128, p_drop=0.15
    )
    gui_head = ClfHead(32, n_classes) if n_classes >= 2 else None

    # Train each modality
    print("\n=== Train Heatmap modality ===")
    heat_encoder, heat_head, heat_tr_hist, heat_val_hist = train_single_modality(
        heat_encoder,
        heat_head,
        n_classes,
        heat_train_loader,
        heat_dev_loader,
        device,
        emb_dim=32,
        max_epochs=args.epochs,
        lr=args.lr,
        patience=8,
    )
    plot_loss_curve(heat_tr_hist, heat_val_hist, "heatmap")

    print("\n=== Train Mouse stats modality ===")
    mouse_encoder, mouse_head, mouse_tr_hist, mouse_val_hist = train_single_modality(
        mouse_encoder,
        mouse_head,
        n_classes,
        mouse_train_loader,
        mouse_dev_loader,
        device,
        emb_dim=32,
        max_epochs=args.epochs,
        lr=args.lr,
        patience=8,
    )
    plot_loss_curve(mouse_tr_hist, mouse_val_hist, "mouse")

    print("\n=== Train Keystrokes modality ===")
    key_encoder, key_head, key_tr_hist, key_val_hist = train_single_modality(
        key_encoder,
        key_head,
        n_classes,
        key_train_loader,
        key_dev_loader,
        device,
        emb_dim=32,
        max_epochs=args.epochs,
        lr=args.lr,
        patience=8,
    )
    plot_loss_curve(key_tr_hist, key_val_hist, "key")

    print("\n=== Train GUI modality ===")
    gui_encoder, gui_head, gui_tr_hist, gui_val_hist = train_single_modality(
        gui_encoder,
        gui_head,
        n_classes,
        gui_train_loader,
        gui_dev_loader,
        device,
        emb_dim=32,
        max_epochs=args.epochs,
        lr=args.lr,
        patience=8,
    )
    plot_loss_curve(gui_tr_hist, gui_val_hist, "gui")

    # Build per-modality prototypes and taus from known users (train/dev)
    def build_proto_tau(encoder, by_user_dict, is_heatmap=False):
        train_emb_by_user, dev_emb_by_user = {}, {}
        for u in known_users:
            tr = train_idx_by_user[u]
            dv = dev_idx_by_user[u]
            if is_heatmap:
                train_emb_by_user[u] = compute_embeddings_encoder(
                    encoder, by_user_dict[u][tr], device, is_heatmap=True
                )
                dev_emb_by_user[u] = compute_embeddings_encoder(
                    encoder, by_user_dict[u][dv], device, is_heatmap=True
                )
            else:
                train_emb_by_user[u] = compute_embeddings_encoder(
                    encoder, by_user_dict[u][tr], device, is_heatmap=False
                )
                dev_emb_by_user[u] = compute_embeddings_encoder(
                    encoder, by_user_dict[u][dv], device, is_heatmap=False
                )
        protos = class_prototypes(train_emb_by_user)
        tau = compute_dev_thresholds(dev_emb_by_user, protos, quantile=args.tau_quantile)
        return protos, tau

    heat_protos, heat_tau = build_proto_tau(heat_encoder, heat_by_user, is_heatmap=True)
    mouse_protos, mouse_tau = build_proto_tau(mouse_encoder, mouse_by_user, is_heatmap=False)
    key_protos, key_tau = build_proto_tau(key_encoder, key_by_user, is_heatmap=False)
    gui_protos, gui_tau = build_proto_tau(gui_encoder, gui_by_user, is_heatmap=False)

    # Compute dev macro-F1 per modality for fusion weights
    print("\n=== Compute dev macro-F1 for modality weights ===")
    mod_acc = {}
    mod_acc["heat"] = dev_macro_f1_for_modality(
        heat_encoder, train_idx_by_user, dev_idx_by_user, heat_by_user, label_map, device, is_heatmap=True
    )
    mod_acc["mouse"] = dev_macro_f1_for_modality(
        mouse_encoder, train_idx_by_user, dev_idx_by_user, mouse_by_user, label_map, device, is_heatmap=False
    )
    mod_acc["key"] = dev_macro_f1_for_modality(
        key_encoder, train_idx_by_user, dev_idx_by_user, key_by_user, label_map, device, is_heatmap=False
    )
    mod_acc["gui"] = dev_macro_f1_for_modality(
        gui_encoder, train_idx_by_user, dev_idx_by_user, gui_by_user, label_map, device, is_heatmap=False
    )
    print("Dev macro-F1 per modality:", mod_acc)
    acc_vec = np.array(
        [mod_acc["heat"], mod_acc["mouse"], mod_acc["key"], mod_acc["gui"]], dtype=np.float32
    )
    if acc_vec.sum() <= 1e-9:
        weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    else:
        weights = acc_vec / acc_vec.sum()
    print("Fusion weights (normalized):", weights.tolist())

    # Also: clustering plot (PCA) for keystroke modality
    print("\n=== PCA clustering plot for keystroke modality ===")
    plot_embeddings_pca(
        key_encoder,
        key_by_user,
        known_users,
        unknown_users,
        device,
        is_heatmap=False,
        name="keystroke",
    )

    # Fused evaluation
    def fused_predict(H, M, K, G):
        eH = compute_embeddings_encoder(heat_encoder, H, device, is_heatmap=True)
        eM = compute_embeddings_encoder(mouse_encoder, M, device, is_heatmap=False)
        eK = compute_embeddings_encoder(key_encoder, K, device, is_heatmap=False)
        eG = compute_embeddings_encoder(gui_encoder, G, device, is_heatmap=False)

        users = known_users

        def sims_to_all(emb, protos):
            Mproto = np.stack([protos[u] for u in users], axis=0)
            Mproto = Mproto / (np.linalg.norm(Mproto, axis=1, keepdims=True) + 1e-12)
            embn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
            return embn @ Mproto.T

        S_h = sims_to_all(eH, heat_protos)
        S_m = sims_to_all(eM, mouse_protos)
        S_k = sims_to_all(eK, key_protos)
        S_g = sims_to_all(eG, gui_protos)

        S_fused = (
            weights[0] * S_h
            + weights[1] * S_m
            + weights[2] * S_k
            + weights[3] * S_g
        )
        best_idx = S_fused.argmax(axis=1)
        best_sim = S_fused[np.arange(S_fused.shape[0]), best_idx]
        best_users = np.array([users[i] for i in best_idx], dtype=object)

        def get_tau_for(u):
            return (
                weights[0] * heat_tau[u]
                + weights[1] * mouse_tau[u]
                + weights[2] * key_tau[u]
                + weights[3] * gui_tau[u]
            )

        fused_tau = np.array([get_tau_for(u) for u in best_users], dtype=np.float32)

        d = 1.0 - best_sim
        pred = np.where(d <= fused_tau, best_users, "Unknown")
        return pred, best_sim, best_users

    # Build test sets
    y_true, y_pred, best_sims = [], [], []

    # Known users' held-out (next 15 rows)
    for u in known_users:
        te_idx = test_idx_by_user[u]
        pred, sims, _nearest = fused_predict(
            heat_by_user[u][te_idx],
            mouse_by_user[u][te_idx],
            key_by_user[u][te_idx],
            gui_by_user[u][te_idx],
        )
        y_true += [u] * len(te_idx)
        y_pred += pred.tolist()
        best_sims += sims.tolist()

    # Unknown: last N rows (unknown_test_rows)
    for u in unknown_users:
        idx = select_unknown_test_indices(len(df_by_user[u]), args.unknown_test_rows)
        pred, sims, _nearest = fused_predict(
            heat_by_user[u][idx],
            mouse_by_user[u][idx],
            key_by_user[u][idx],
            gui_by_user[u][idx],
        )
        y_true += ["Unknown"] * len(idx)
        y_pred += pred.tolist()
        best_sims += sims.tolist()

    # Metrics
    labels_full = known_users + ["Unknown"]
    print("\nConfusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred, labels=labels_full)
    print(pd.DataFrame(cm, index=labels_full, columns=labels_full))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred, labels=labels_full, zero_division=0, digits=3
        )
    )

    # Additional global metrics
    y_true_arr = np.array(y_true, dtype=object)
    y_pred_arr = np.array(y_pred, dtype=object)

    overall_acc = accuracy_score(y_true_arr, y_pred_arr)
    bal_acc = balanced_accuracy_score(y_true_arr, y_pred_arr)
    print(f"Overall accuracy: {overall_acc:.3f}")
    print(f"Balanced accuracy: {bal_acc:.3f}")

    # FAR / FRR for open-set authentication
    is_known_true = y_true_arr != "Unknown"
    is_known_pred = y_pred_arr != "Unknown"

    false_rejects = np.logical_and(is_known_true, ~is_known_pred).sum()
    false_accepts = np.logical_and(~is_known_true, is_known_pred).sum()

    n_known = is_known_true.sum()
    n_unknown = (~is_known_true).sum()

    frr = false_rejects / n_known if n_known > 0 else 0.0
    far = false_accepts / n_unknown if n_unknown > 0 else 0.0

    print(f"False Rejection Rate (FRR): {frr:.3f}")
    print(f"False Acceptance Rate (FAR): {far:.3f}")

    # Known-vs-Unknown ROC-AUC
    y_bin_true = np.array([0 if t == "Unknown" else 1 for t in y_true_arr], dtype=np.int32)
    try:
        auc = roc_auc_score(y_bin_true, np.array(best_sims))
        print(f"Known-vs-Unknown ROC-AUC (fused similarity): {auc:.4f}")
    except Exception as e:
        print(f"ROC-AUC error: {e}")


if __name__ == "__main__":
    main()
