"""
Graph-tabular GNN (GATv2 + tabular branch) with a sklearn-like interface, so it
plugs into the same nested scaffold-grouped CV as Extra Trees / CatBoost.

Architecture follows utils/utils_models.py::GNNTabularRegressor. What changes:
* it is fit and tuned ONLY inside the folds handed to it by nested_cv.py;
* the tabular block it receives is the fold-local preprocessed matrix
  (FoldPreprocessor), scaled with a StandardScaler fit on the training rows;
* the early-stopping validation set is carved out of the *training* rows only
  (outer test rows are never seen), grouped by scaffold when groups are given.

nested_cv.run() calls set_context(smiles, scaffolds) once; GNNRegressor.fit/predict
then look molecules up through the DataFrame index of the matrix they receive.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

ATOMS = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
HYB = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
       Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
       Chem.rdchem.HybridizationType.SP3D2]
CHI = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
       Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW]
BT = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

_CTX = {"graphs": None, "groups": None}


def _oh(v, choices):
    return [int(v == c) for c in choices] + [int(v not in choices)]


def _atom(a):
    return (_oh(a.GetSymbol(), ATOMS) + _oh(a.GetHybridization(), HYB) + _oh(a.GetChiralTag(), CHI)
            + [a.GetAtomicNum() / 100.0, a.GetTotalDegree() / 6.0, a.GetFormalCharge(),
               float(a.GetIsAromatic()), float(a.IsInRing())])


def _bond(b):
    return _oh(b.GetBondType(), BT) + [float(b.GetIsConjugated()), float(b.IsInRing()),
                                       float(b.GetStereo() != Chem.rdchem.BondStereo.STEREONONE)]


def set_context(smiles, scaffolds):
    """Build one outcome-independent graph per compound (indexed by row position)."""
    graphs = []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            raise ValueError(f"Invalid SMILES for GNN graph: {s}")
        x = torch.tensor([_atom(a) for a in m.GetAtoms()], dtype=torch.float)
        ei, ea = [], []
        for b in m.GetBonds():
            i, j, f = b.GetBeginAtomIdx(), b.GetEndAtomIdx(), _bond(b)
            ei += [[i, j], [j, i]]; ea += [f, f]
        if not ei:
            ei, ea = [[0, 0]], [[0.0] * (len(BT) + 1 + 3)]
        graphs.append((x, torch.tensor(ei).t().contiguous(), torch.tensor(ea, dtype=torch.float)))
    _CTX["graphs"], _CTX["groups"] = graphs, np.asarray(scaffolds)


def _signed_log1p(v):
    return np.sign(v) * np.log1p(np.abs(v))


def _inv(v):
    return np.sign(v) * np.expm1(np.abs(v))


class Net(nn.Module):
    def __init__(self, node_dim, edge_dim, tab_dim, hidden=128, dropout=0.15):
        super().__init__()
        self.tab_dim = tab_dim
        self.enc = nn.Linear(node_dim, hidden)
        self.c1 = GATv2Conv(hidden, hidden, edge_dim=edge_dim, add_self_loops=True)
        self.c2 = GATv2Conv(hidden, hidden, edge_dim=edge_dim, add_self_loops=True)
        self.n1, self.n2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.tab = nn.Sequential(nn.Linear(tab_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.LayerNorm(hidden // 2))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden + hidden // 2, hidden), nn.ReLU(), nn.LayerNorm(hidden),
                                  nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Linear(hidden // 2, 1))

    def forward(self, x, ei, ea, batch, tab):
        x = self.enc(x)
        x = self.drop(self.n1(F.relu(self.c1(x, ei, ea) + x)))
        x = self.drop(self.n2(F.relu(self.c2(x, ei, ea) + x)))
        g = global_mean_pool(x, batch)
        return self.head(torch.cat([g, self.tab(tab.view(-1, self.tab_dim))], dim=1))


class GNNRegressor:
    def __init__(self, hidden_dim=128, dropout=0.15, lr=1e-3, weight_decay=1e-5, batch_size=32,
                 max_epochs=60, patience=10, random_state=0):
        self.p = dict(hidden_dim=hidden_dim, dropout=dropout, lr=lr, weight_decay=weight_decay,
                      batch_size=batch_size, max_epochs=max_epochs, patience=patience)
        self.seed = random_state

    def _data(self, X, tab, y=None):
        ds = []
        for k, idx in enumerate(X.index):
            x, ei, ea = _CTX["graphs"][idx]
            d = Data(x=x, edge_index=ei, edge_attr=ea, tabular=torch.tensor(tab[k]).view(1, -1))
            if y is not None:
                d.y = torch.tensor([_signed_log1p(y[k])], dtype=torch.float)
            ds.append(d)
        return ds

    def _predict(self, loader):
        self.net.eval()
        out = []
        with torch.no_grad():
            for b in loader:
                out.append(self.net(b.x, b.edge_index, b.edge_attr, b.batch, b.tabular).view(-1).numpy())
        return _inv(np.concatenate(out))

    def fit(self, X, y):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.scaler = StandardScaler().fit(X.values)
        tab = np.clip(self.scaler.transform(X.values), -10, 10).astype(np.float32)
        y = np.asarray(y, dtype=float)
        groups = _CTX["groups"][X.index.values]
        fit_i, val_i = next(GroupShuffleSplit(1, test_size=0.15, random_state=self.seed)
                            .split(tab, groups=groups))
        p = self.p
        tr = DataLoader(self._data(X.iloc[fit_i], tab[fit_i], y[fit_i]), batch_size=p["batch_size"], shuffle=True)
        va = DataLoader(self._data(X.iloc[val_i], tab[val_i], y[val_i]), batch_size=128)
        s = tr.dataset[0]
        self.net = Net(s.x.shape[1], s.edge_attr.shape[1], tab.shape[1], p["hidden_dim"], p["dropout"])
        opt = torch.optim.AdamW(self.net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        best, best_state, stale = np.inf, None, 0
        for _ in range(p["max_epochs"]):
            self.net.train()
            for b in tr:
                opt.zero_grad()
                loss = F.mse_loss(self.net(b.x, b.edge_index, b.edge_attr, b.batch, b.tabular).view(-1), b.y.view(-1))
                loss.backward(); opt.step()
            pred = self._predict(va)
            v = float(np.sqrt(np.mean((pred - y[val_i]) ** 2)))
            if v < best:
                best, stale = v, 0
                best_state = {k: t.detach().clone() for k, t in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= p["patience"]:
                    break
        self.net.load_state_dict(best_state)
        return self

    def predict(self, X):
        tab = np.clip(self.scaler.transform(X.values), -10, 10).astype(np.float32)
        return self._predict(DataLoader(self._data(X, tab), batch_size=128))


def sample_params(rng):
    pick = lambda o: o[rng.randint(len(o))]
    return dict(hidden_dim=pick([64, 128, 192]), dropout=pick([0.1, 0.2, 0.3]),
                lr=pick([5e-4, 1e-3, 2e-3]), weight_decay=pick([1e-5, 1e-4]), batch_size=pick([32, 64]))
