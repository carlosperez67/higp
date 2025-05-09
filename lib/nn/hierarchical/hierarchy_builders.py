import torch
from torch import nn
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor

from lib.nn.hierarchical.pooling.mincut_pool import MinCutPool
import tsl

from tsl.utils import ensure_list

class MinCutHierarchyBuilder(nn.Module):
    r"""Hierarchy encoder"""

    def __init__(self,
                 n_nodes: int,
                 hidden_size: int,
                 n_clusters: float,
                 n_levels: int = 1,
                 temp_decay: float = 0.99995,
                 hard=True):
        super(MinCutHierarchyBuilder, self).__init__()

        self.n_levels = n_levels
        input_nodes = n_nodes
        pooling_layers = []
        n_clusters = ensure_list(n_clusters)
        if len(n_clusters) != n_levels - 2:
            assert len(n_clusters) == 1
            n_clusters = n_clusters * (n_levels - 2)
        for i in range(n_levels - 2):
            pooling_layers.append(MinCutPool(emb_size=hidden_size,
                                             n_nodes=input_nodes,
                                             n_clusters=n_clusters[i],
                                             hard=hard,
                                             temp_decay=temp_decay))
            input_nodes = n_clusters[i]

        self.pooling_layers = nn.ModuleList(pooling_layers)

    def forward(self, emb, edge_index, edge_weight=None):
        # emb: [nodes features]
        if isinstance(edge_index, SparseTensor):
            adj = edge_index.to_dense()
        else:
            adj = to_dense_adj(edge_index, edge_attr=edge_weight)[0].T

        # force the graph to be undirected
        adj = torch.max(adj, adj.T)

        d = torch.sum(adj, dim=-1, keepdim=True)
        d = 1 / (torch.sqrt(d) + tsl.epsilon)
        adj = d * adj * d.T

        embs = [emb]
        adjs = [adj]
        seletcs = [None]
        sizes = [emb.size(-2)]
        min_cut_loss = 0.
        reg_loss = 0.
        for i in range(self.n_levels - 2):
            # Pooling
            v, adj_, s, (mc_loss, r_loss) = \
                self.pooling_layers[i](embs[i], adjs[i])
            # Update embedding
            seletcs.append(s)
            embs.append(v)
            adjs.append(adj_)
            sizes.append(v.size(-2))
            min_cut_loss += mc_loss
            reg_loss += r_loss
        # add the last level
        embs.append(emb.mean(-2, keepdim=True))
        adjs.append(None)
        if self.n_levels > 2 and emb.dim() == 3:
            s_tot = torch.ones(emb.size(0), sizes[-1], 1, device=emb.device)
        else:
            s_tot = torch.ones(sizes[-1], 1, device=emb.device)
        seletcs.append(s_tot)
        sizes.append(1)
        return embs, adjs, seletcs, sizes, (min_cut_loss, reg_loss)


# lib/nn/hierarchical/hierarchy_builders/fixed_hierarchy_builder.py
import torch
from torch import nn
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor
import tsl


class FixedHierarchyBuilder(nn.Module):
    r"""
    Build a hierarchy from user‑defined selection matrices,
    cascadingly removing empty parent columns so that
    row(S_l) == col(S_{l-1}) after cleaning.
    """

    def __init__(self, selects):
        super().__init__()
        assert len(selects) >= 1, '`selects` cannot be empty'
        self._selects = self._clean_and_register(selects)
        self.n_levels = len(self._selects) + 2            # + level‑0 + global

    # ---------- cascade clean ----------
    def _clean_and_register(self, selects):
        clean = []
        keep_rows = slice(None)
        for l, S in enumerate(selects):
            S = S.float()[keep_rows]

            keep_cols = S.sum(0) > 0
            S = S[:, keep_cols]

            row_zero = S.sum(1) == 0
            if row_zero.any():
                S[row_zero, 0] = 1.0


            S = S / S.sum(1, keepdim=True)

            self.register_buffer(f'select_{l}', S, persistent=False)
            clean.append(S)


            keep_rows = keep_cols

        return clean

    # ---------- helper ----------
    def _iter_selects(self):
        for i in range(len(self._selects)):
            yield getattr(self, f'select_{i}')

    # ---------- forward ----------
    @torch.no_grad()
    def forward(self, emb, edge_index, edge_weight=None):
        # ---- level‑0 adjacency ----
        if isinstance(edge_index, SparseTensor):
            A = edge_index.to_dense()
        else:
            A = to_dense_adj(edge_index, edge_attr=edge_weight)[0].T
        A = torch.max(A, A.T)
        deg = A.sum(-1, keepdim=True)
        A = A * (1 / (torch.sqrt(deg) + tsl.epsilon)) * \
                (1 / (torch.sqrt(deg).T + tsl.epsilon))

        batched = emb.dim() == 3
        B = emb.size(0) if batched else None

        embs   = [emb]
        adjs   = [A]
        selects = [None]
        sizes  = [emb.size(-2)]

        # ---- iterate levels ----
        for S in self._iter_selects():
            selects.append(S.unsqueeze(0).expand(B, -1, -1) if batched else S)

            # feature pooling
            if batched:
                cl_sz  = S.sum(0).unsqueeze(0).unsqueeze(-1)
                v_next = torch.einsum('bnd,nk->bkd', embs[-1], S) / cl_sz
            else:
                cl_sz  = S.sum(0).unsqueeze(-1)
                v_next = (S.T @ embs[-1]) / cl_sz
            embs.append(v_next)
            sizes.append(v_next.size(-2))

            # adjacency pooling
            A = S.T @ A @ S
            deg = A.sum(-1, keepdim=True)
            A = A * (1 / (torch.sqrt(deg) + tsl.epsilon)) * \
                    (1 / (torch.sqrt(deg).T + tsl.epsilon))
            adjs.append(A)

        # ---- global level ----
        if batched:
            s_tot = torch.ones(B, sizes[-1], 1, device=emb.device)
            emb_global = embs[0].mean(-2, keepdim=True)
        else:
            s_tot = torch.ones(sizes[-1], 1, device=emb.device)
            emb_global = embs[0].mean(-2, keepdim=True)

        selects.append(s_tot)
        adjs.append(None)
        embs.append(emb_global)
        sizes.append(1)

        zero = torch.tensor(0., device=emb.device)
        return embs, adjs, selects, sizes, (zero, zero)