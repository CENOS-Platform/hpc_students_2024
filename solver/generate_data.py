# make_dataset.py
import h5py, itertools, json, os
from tqdm import tqdm
from parametric import IfaAntennaCase, PatchAntennaCase
from utils import mesh_to_arrays
from rf_solver import SolverRF
import argparse, hashlib
import numpy as np

def compute_reference(case, freq: float, n_refine: int = 2) -> float:
    print(f"Computing reference S11 for {case.__class__.__name__} at {freq:.2e} Hz with {n_refine} refinements")
    mesh = case.generate_mesh()            # fresh coarse mesh of same geometry
    mesh.Refine()
    solver = SolverRF(mesh, case.domains, case.boundaries,
                      case.lumped_elements, {"frequency": freq}, order=1)
    solver.assemble(); solver.solve()
    solver.finalize()
    return solver.s11                      # already in dB

def param_grid():
    ifa = dict(feed_gap=[0.6e-3, 0.8e-3, 1.0e-3],
               line_width=[0.8e-3, 1.0e-3],
               substrate_eps=[2.2, 3.0])

    patch = dict(pw=[0.045, 0.049, 0.053],  # metres
                 pl=[0.038, 0.041, 0.044],
                 eps=[2.2, 4.4])

    def sweep(grid):
        keys, vals = zip(*grid.items())
        for combo in itertools.product(*vals):
            yield dict(zip(keys, combo))

    for cfg in sweep(ifa):
        yield "ifa", cfg
    for cfg in sweep(patch):
        yield "patch", cfg

def cfg_to_fname(tag: str, cfg: dict) -> str:
    kv = "_".join(f"{k}={v}" for k, v in cfg.items())
    h  = hashlib.md5(kv.encode()).hexdigest()[:4]
    return f"{tag}_{kv}_{h}.h5"

def main():
    ap = argparse.ArgumentParser(description="Generate coarse‑mesh graph dataset")
    # ap.add_argument("--split", choices=["train", "val", "test"], default="test",
    #                 help="dataset split – val/test will include reference S11")
    ap.add_argument("--out_root", default="data_raw", help="output directory root")
    ap.add_argument("--freq", type=float, default=2.45e9, help="analysis frequency [Hz]")
    ap.add_argument("--refine_ratio", type=float, default=0.2,
                    help="fraction of elements labelled as refine=1")
    ap.add_argument("--ref_levels", type=int, default=2,
                    help="number of uniform refinements for reference solve (val/test)")
    args = ap.parse_args()

    out_dir = args.out_root
    os.makedirs(out_dir, exist_ok=True)

    for tag, cfg in tqdm(list(param_grid()), ncols=88):
        fname = cfg_to_fname(tag, cfg)
        path  = os.path.join(out_dir, fname)
        if os.path.exists(path):
            continue                      # skip already generated sample

        # --- build geometry & solve on coarse mesh --------------------------
        case = IfaAntennaCase(**cfg) if tag == "ifa" else PatchAntennaCase(**cfg)
        solver = SolverRF(case.mesh, case.domains, case.boundaries,
                          case.lumped_elements, {"frequency": args.freq})
        solver.assemble(); solver.solve()

        arrays = mesh_to_arrays(case.mesh, solver, label_frac=args.refine_ratio)

        # --- write HDF5 ------------------------------------------------------
        with h5py.File(path, "w") as f:
            # coarse graph datasets
            for k, v in arrays.items():
                f.create_dataset(k, data=v, compression="gzip", compression_opts=4)

            # metadata
            f.attrs.update(tag=tag, freq=args.freq, param=json.dumps(cfg))

            # reference solve
            # ref_s11 = compute_reference(case, args.freq, n_refine=args.ref_levels)
            # g = f.create_group("ref")
            # g.create_dataset("S11", data=np.array([ref_s11], dtype=np.float32))

if __name__ == "__main__":
    main()
