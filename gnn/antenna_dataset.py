import torch, glob, h5py, os, json, hashlib
from torch_geometric.data import InMemoryDataset, Data
from torch import Tensor
from pathlib import Path
import numpy as np

class AntennaDataset(InMemoryDataset):
    """Cache-once, load-fast dataset for AMR graphs."""

    def __init__(self, root, file_list):
        self.file_list = sorted(file_list)
        super().__init__(root, transform=None)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        # SHA1 of concatenated relative paths  → 8-char digest
        digest = hashlib.sha1(";".join(self.file_list).encode()).hexdigest()[:8]
        return [f"cached_{digest}.pt"]

    def process(self):
        data_list = []
        for fp in self.file_list:
            with h5py.File(fp, "r") as f:
                x   = torch.tensor(f["x"][:], dtype=torch.float32)
                ei  = torch.stack([torch.tensor(f["edge_src"][:]),
                                   torch.tensor(f["edge_dst"][:])]).long()
                y   = torch.tensor(f["y"][:], dtype=torch.float32)

                # concat frequency to node features (match amr_gnn.py)
                freq = torch.tensor([f.attrs["freq"]], dtype=torch.float32)
                freq_feat = freq.expand(x.size(0), 1)
                x = torch.cat([x, freq_feat], dim=1)

                g = Data(x=x, edge_index=ei, y=y)
            data_list.append(g)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    files = glob.glob("data_raw/*.h5")
    files.sort()
    np.random.seed(42)
    np.random.shuffle(files)

    n = len(files)
    splits = dict(train=files[:int(.7*n)],
                val  =files[int(.7*n):int(.85*n)],
                test =files[int(.85*n):])

    # one cache file per split
    for split, lst in splits.items():
        AntennaDataset("data_processed", lst)