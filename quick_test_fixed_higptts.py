# filename: quick_test_fixed_higptts.py
# -----------------------------------------------------------
# 1. imports
import torch
from torch import nn
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# === Key: import the modified model ===
from lib.nn.hierarchical.models.higp_tts_model import HiGPTTSModel

# -----------------------------------------------------------
# 2. synthetic settings
BATCH      = 8      # batch size
HIST_LEN   = 12     # history window length
HORIZON    = 3      # prediction steps
N_NODES    = 20     # number of original nodes
FEAT_IN    = 4      # input feature dimension per node
FEAT_OUT   = 4      # output feature dimension (same as input)
LEVELS     = 4      # original + 2 predefined levels + global = 4
HIDDEN     = 32
EMB_SIZE   = 16

# -----------------------------------------------------------
# 3. Construct "predefined hierarchy" selects
#    Example: Level‑0  -> Level‑1: 20→6； Level‑1 -> Level‑2: 6→3
S0 = torch.zeros(N_NODES, 6)
for i in range(N_NODES):
    S0[i, i // 4] = 1          # group every 4 nodes into one cluster
S1 = torch.tensor([[1,0,0,0,0,0],
                   [1,0,0,0,0,0],
                   [0,1,0,0,0,0],
                   [0,1,0,0,0,0],
                   [0,0,1,0,0,0],
                   [0,0,1,0,0,0]]).float()   # 6→3 (example)

fixed_selects = [S0, S1]       # len=2 → LEVELS should = 2+2 = 4

# -----------------------------------------------------------
# 4. Create random data and graph
x_hist = torch.randn(200, HIST_LEN, N_NODES, FEAT_IN)   # 200 samples
y_future = torch.randn(200, HORIZON, N_NODES, FEAT_OUT)

dataset = TensorDataset(x_hist, y_future)
loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True)

# Use a fully connected graph as a simple example
edge_index = torch.combinations(torch.arange(N_NODES), r=2).T
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # bidirectional

# -----------------------------------------------------------
# 5. Instantiate model
model = HiGPTTSModel(
    input_size     = FEAT_IN,
    output_size    = FEAT_OUT,
    horizon        = HORIZON,
    n_nodes        = N_NODES,
    hidden_size    = HIDDEN,
    emb_size       = EMB_SIZE,
    levels         = LEVELS,
    n_clusters     = 4,            # only used in MinCut mode; arbitrary here
    single_sample  = False,
    fixed_selects  = fixed_selects # ⭐ key argument
).train()

optim = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# -----------------------------------------------------------
# 6. quick training loop (2 epochs)
for epoch in range(60):
    for xb, yb in loader:
        optim.zero_grad()
        y_hat, _, _, _, _ = model(
            xb, edge_index=edge_index
        )                           # forward pass
        y_hat_level0 = y_hat[:, :, :N_NODES, :]
        loss = loss_fn(y_hat_level0, yb)
        loss.backward()
        optim.step()
    print(f"Epoch {epoch+1}: loss = {loss.item():.5f}")

print("✔️  Sanity-check finished — model with FixedHierarchyBuilder works.")

model.eval()
with torch.no_grad():
    # take one batch from loader
    xb, _ = next(iter(loader))
    _, _, _, _, latents = model(xb, edge_index=edge_index)
    # latents is a list of Tensors [B, hidden_size] for each level
    np_dict = {
        f"level{i}": lat.detach().cpu().numpy()
        for i, lat in enumerate(latents)
    }
    np.savez("latents_all_levels.npz", **np_dict)
    print("Saved latents_all_levels.npz:", list(np_dict.keys()))

# 6) quick PCA + scatter of level 0
data0 = np.load("latents_all_levels.npz")["level0"]
#  e.g. shape (B, hidden_size) or (B*T, hidden_size) once flattened

# 2) Flatten into (n_samples, features)
flat = data0.reshape(-1, data0.shape[-1])

# 3) PCA → 3 components
pca = PCA(n_components=3)
z3 = pca.fit_transform(flat)  # shape (n_samples, 3)

# 4) 3D scatter
fig = plt.figure(figsize=(7,6))
ax  = fig.add_subplot(111, projection="3d")
ax.scatter(z3[:,0], z3[:,1], z3[:,2], s=15, alpha=0.7)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")
ax.set_title("Level 0 Latents (3D PCA)")
plt.tight_layout()
plt.show()