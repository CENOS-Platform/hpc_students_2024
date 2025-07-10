import ngsolve, numpy as np
from itertools import combinations
from netgen.libngpy._meshing import PointId
import torch
from torch_geometric.data import Data

def _centroid(mesh, el):
    pts = []
    for nid in el.vertices:
        idx = nid.nr or (nid.nr + 1)
        pt = mesh.ngmesh.Points()[PointId(idx)].p
        pts.append(pt)
    return np.mean(pts, axis=0).astype(np.float32)

# helper for element diameter h_i
def _element_diameter(vertices):
    dmax = 0.0
    for v1, v2 in combinations(vertices, 2):
        d = np.linalg.norm(v1 - v2)
        if d > dmax:
            dmax = d
    return dmax

# helper for shape quality κ_i = (3 * inscribed_radius) / circumscribed_radius (for tetrahedra)
def _shape_quality(vertices):
    # approximate: use ratio of minimal altitude to maximal edge length
    # altitude from vertex opposite face
    # For simplicity, compute min distance from centroid to face
    cent = np.mean(vertices, axis=0)
    mins = []
    for tri in combinations(vertices, 3):
        # plane normal
        v0, v1, v2 = tri
        normal = np.cross(v1 - v0, v2 - v0)
        area = np.linalg.norm(normal) * 0.5
        if area < 1e-12:
            continue
        unit_n = normal / np.linalg.norm(normal)
        # distance of centroid to face
        d = abs(np.dot(unit_n, cent - v0))
        mins.append(d)
    if not mins:
        return 0.0
    min_alt = min(mins)
    # max edge:
    max_edge = max(np.linalg.norm(v1 - v2) for v1, v2 in combinations(vertices, 2))
    return float(min_alt / max_edge)


def mesh_to_arrays(mesh, solver, label_frac: float = 0.2):
    ncells = mesh.ne
    centres = np.zeros((ncells, 3), np.float32)
    mats = np.empty(ncells, np.int16)

    # extract field coefficients
    E_coeff = solver.electric_field
    est_err = solver.estimate_error().astype(np.float32)

    h = np.zeros(ncells, np.float32)
    kappa = np.zeros(ncells, np.float32)
    Eabs = np.zeros(ncells, np.float32)
    Ereal = np.zeros(ncells, np.float32)
    Eimag = np.zeros(ncells, np.float32)

    for el in mesh.Elements(ngsolve.VOL):
        cid = el.nr
        # centroid
        verts = [mesh.ngmesh.Points()[PointId(v.nr or (v.nr+1))].p for v in el.vertices]
        centres[cid] = np.array(verts).mean(axis=0, dtype=np.float32)
        
        # material
        mats[cid] = el.mat if isinstance(el.mat, int) else 0

        # geometry metrics
        verts_arr = np.array(verts, dtype=np.float32)
        h[cid] = _element_diameter(verts_arr)
        kappa[cid] = _shape_quality(verts_arr)

        # evaluate field at centroid
        Ec = E_coeff.vec.data[cid]
        Eabs[cid] = np.linalg.norm(Ec)
        Ereal[cid] = np.real(Ec)  # or component-wise
        Eimag[cid] = np.imag(Ec)

    # labels
    thresh = np.quantile(est_err, 1 - label_frac)
    labels = (est_err > thresh).astype(np.uint8)

    # adjacency
    src, dst = [], []
    edge_owner = {}
    for el in mesh.Elements(ngsolve.VOL):
        cid = el.nr
        verts = [v.nr - 1 for v in el.vertices]
        for v1, v2 in combinations(verts, 2):
            key = (min(v1, v2), max(v1, v2))
            other = edge_owner.setdefault(key, cid)
            if other != cid:
                src.extend([cid, other])
                dst.extend([other, cid])

    # node features: [centroid(3), mat, est_err, |E|, Re(E), Im(E), h_i, kappa]
    node_feat = np.column_stack((centres,
                                 mats.astype(np.float32)[:,None],
                                 est_err[:,None],
                                 Eabs[:,None],
                                 Ereal[:,None],
                                 Eimag[:,None],
                                 h[:,None],
                                 kappa[:,None]))
    # edge_attr as before... (optionally add face-jump: see comment)
    dx = centres[dst] - centres[src]
    dlen = np.linalg.norm(dx,axis=1,keepdims=True)
    mat_i = mats[src][:,None]
    mat_j = mats[dst][:,None]
    edge_attr = np.hstack((dx, dlen, mat_i, mat_j))

    return dict(
        edge_src=src, edge_dst=dst,
        edge_attr=edge_attr.astype(np.float32),
        x=node_feat.astype(np.float32),
        y=labels
    )


def build_graph(mesh: ngsolve.Mesh,
                solver,
                *,
                add_freq: bool = True) -> Data:
    """
    Convert a solved mesh into a PyG Data for *inference* in the AMR loop.

    Returns a Data object with fields:
      x         : (N_cells × F_node) float tensor
      edge_index: (2 × N_edges) long tensor
      edge_attr : (N_edges × F_edge) float tensor
      elem_ids  : (N_cells,) long tensor mapping graph-node i → mesh element nr
    """

    # 1) pull out arrays (no need for label_frac or y)
    arrs = mesh_to_arrays(mesh, solver, label_frac=0.0)
    # arrs now contains:
    #   'x'          → (N_cells × F_node) np.float32
    #   'edge_src'   → (N_edges,) int
    #   'edge_dst'   → (N_edges,) int
    #   'edge_attr'  → (N_edges × F_edge) np.float32
    #   'y'          → all zeros because label_frac=0

    # 2) build torch tensors
    x          = torch.from_numpy(arrs['x']).float()                      # node feats
    edge_index = torch.from_numpy(
                     np.vstack([arrs['edge_src'], arrs['edge_dst']])
                 ).long()                                                # (2,E)
    edge_attr  = torch.from_numpy(arrs['edge_attr']).float()              # (E,F_edge)

    # 3) attach frequency if you trained with it
    if add_freq:
        freq = float(solver.omega/(2*np.pi))
        freq_col = torch.full((x.size(0),1), freq)
        x = torch.cat([x, freq_col], dim=1)

    # 4) build a reverse map from graph nodes → mesh element nr
    #    mesh.Elements(ngsolve.VOL) gives you an iterator in ascending el.nr order,
    #    so we can just grab all the .nr into a 1d tensor:
    elem_ids = torch.tensor(
        [el.nr for el in mesh.Elements(ngsolve.VOL)],
        dtype=torch.long
    )

    # 5) pack into a Data and return
    return Data(x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                elem_ids=elem_ids)