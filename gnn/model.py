"""mesh_refine_gnn.py – GNN model  training loop for fast AMR
-----------------------------------------------------------------
Assumes dataset produced by make_dataset.py lives in data_raw/<split>/
Each .h5 file contains edge_src, edge_dst, x, y, plus attrs.

Dependencies
------------
    pip install torch torch_geometric h5py tqdm scikit-learn

Usage
-----
    python mesh_refine_gnn.py --data_root data_raw --hidden 128 --epochs 30

The script will
 * stream HDF5 files -> PyG Data objects
 * train a GATv2-based node-classifier that outputs P(refine)
 * print validation metrics each epoch
"""

from __future__ import annotations

import argparse, json, os, random, glob, math
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
from torch import nn, optim
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, GraphNorm
from sklearn.metrics import f1_score, precision_recall_fscore_support

# -----------------------------------------------------------------------------
# Dataset loader – streams HDF5 files on the fly (no huge RAM usage)
# -----------------------------------------------------------------------------

class AntennaGraphDataset(Dataset):
    def __init__(self, root_dir: str, split: str):
        super().__init__()
        self.files: List[str] = sorted(split)
        assert self.files, f"no .h5 files found in {root_dir}/{split}"

    def len(self):
        return len(self.files)

    def get(self, idx):
        fn = self.files[idx]
        file =  h5py.File(fn, "r")
        # node features: [xyz, matID, η, |E|, Re(E), Im(E), h, κ]
        x          = torch.from_numpy(file["x"][:]).float()               # (N × Fₙ)

        # per‐edge geometry: dx, dist, mat_i, mat_j, (…plus face‐jump if you want) 
        edge_src   = file["edge_src"][:]
        edge_dst   = file["edge_dst"][:]
        edge_attr  = torch.from_numpy(file["edge_attr"][:]).float()      # (E × Fₑ)
        edge_index = torch.from_numpy(np.array([edge_src, edge_dst])).long()

        y = torch.from_numpy(file["y"][:]).float()                        # (N)
        freq = torch.tensor([file.attrs["freq"]], dtype=torch.float32)   # (1)
        # broadcast global feature to nodes
        freq_feat = freq.repeat(x.size(0), 1)
        x = torch.cat([x, freq_feat], dim=1)

        return Data(x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                ntype=torch.tensor([0])
                )

# -----------------------------------------------------------------------------
# GNN model
# -----------------------------------------------------------------------------

from torch_geometric.nn import GINEConv

class RefineGNN(nn.Module):
    def __init__(self, in_dim, hidden=128, num_layers=4, edge_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

        # Build one MLP per GINE layer for node feature processing
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            node_mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )
            self.convs.append(GINEConv(node_mlp, edge_dim=edge_dim))
        self.norms = nn.ModuleList([GraphNorm(hidden) for _ in range(num_layers)])
        self.decoder = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_attr):
        h = self.encoder(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = torch.relu(h)
        return self.decoder(h).squeeze(-1)

# -----------------------------------------------------------------------------
# train / evaluate helpers
# -----------------------------------------------------------------------------

def bce_loss(logits, labels, pos_weight=None):
    if pos_weight is not None:
        return nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)

def evaluate(model, loader, device, threshold=0.5):
    model.eval(); ys, preds = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.edge_attr)
            prob   = torch.sigmoid(logits)
            ys.append(data.y.cpu())
            preds.append((prob > threshold).cpu())
    y_true  = torch.cat(ys).numpy()
    y_pred  = torch.cat(preds).numpy()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return dict(precision=prec, recall=rec, f1=f1)

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data_raw")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads",  type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch",  type=int, default=4, help="graphs per batch – 1 ok for big meshes")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    # random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = glob.glob("data_raw/*.h5")
    files.sort()                     # make order deterministic
    # np.random.seed(42)
    np.random.shuffle(files)         # shuffles in-place deterministically

    n = len(files)
    splits = dict(train=files[:int(.7*n)],
                val  =files[int(.7*n):int(.85*n)],
                test =files[int(.85*n):])

    train_ds = AntennaGraphDataset(args.data_root, splits["train"])
    val_ds   = AntennaGraphDataset(args.data_root, splits["val"])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)

    sample = train_ds.get(0)
    model = RefineGNN(in_dim=sample.x.size(1),
                  hidden=args.hidden,
                  num_layers=args.layers,
                  edge_dim=6).to(device)

    # pos_weight to balance positive / negative samples at node level
    pos_frac = sample.y.mean().item(); pos_weight = torch.tensor([(1-pos_frac)/pos_frac])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index, data.edge_attr)
            loss = bce_loss(logits, data.y, pos_weight=pos_weight.to(device))
            loss.backward(); optimizer.step()
            running = loss.item()
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train loss {running/len(train_loader):.4f} | "
              f"val f1 {metrics['f1']:.3f} prec {metrics['precision']:.3f} rec {metrics['recall']:.3f}")
        new_val_acc = metrics['f1']
        if epoch % 10 == 0 or new_val_acc > val_acc:
            val_acc = new_val_acc
            # save model every 10 epochs
            torch.save(model.state_dict(), f"gnn_refine_epoch{epoch:03d}.pth")

    # save model
    torch.save(model.state_dict(), "gnn_refine.pth")

if __name__ == "__main__":
    main()
