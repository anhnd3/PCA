import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

# --- PCA ---
class PCA:
    def __init__(self):
        self.mean_ = None
        self.components_ = None  # eigenvectors
        self.eigvals_ = None     # eigenvalues

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = Xc.T @ Xc / (Xc.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        self.eigvals_    = eigvals[idx]
        self.components_ = eigvecs[:, idx]

    def transform(self, X, k):
        Xc = X - self.mean_
        W  = self.components_[:, :k]
        return Xc @ W

    def explained_variance_ratio(self):
        return self.eigvals_ / self.eigvals_.sum()


# --- Load Iris dataset ---
data   = load_iris()
X, y   = data.data, data.target
labels = data.target_names

# --- Fit PCA ---
pca = PCA()
pca.fit(X)

# --- Prepare projections and variance info ---
Z2         = pca.transform(X, k=2)
Z3         = pca.transform(X, k=3)
var_ratio  = pca.explained_variance_ratio()
cum_var    = np.cumsum(var_ratio)
n_components = X.shape[1]

# --- Figure 1: 2 subplots (2D & 3D PCA) ---
fig1 = plt.figure(figsize=(12, 5))
ax1 = fig1.add_subplot(1, 2, 1)
for lbl in np.unique(y):
    ax1.scatter(Z2[y == lbl, 0], Z2[y == lbl, 1], label=labels[lbl])
ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
ax1.set_title("Iris: 2D PCA (k=2)"); ax1.legend()

ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
for lbl in np.unique(y):
    ax2.scatter(Z3[y == lbl, 0], Z3[y == lbl, 1], Z3[y == lbl, 2], label=labels[lbl])
ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.set_zlabel("PC3")
ax2.set_title("Iris: 3D PCA (k=3)")

plt.tight_layout()
plt.show()

# --- Figure 2: 3 subplots (Scree, Variance Ratio, Cumulative) ---
fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(range(1, n_components+1), pca.eigvals_)
axes[0].set_title("Scree Plot"); axes[0].set_xlabel("Component"); axes[0].set_ylabel("Eigenvalue")

axes[1].bar(range(1, n_components+1), var_ratio)
axes[1].set_title("Variance Ratio"); axes[1].set_xlabel("Component"); axes[1].set_ylabel("Explained Variance")

axes[2].plot(range(1, n_components+1), cum_var, marker='o')
axes[2].axhline(0.95, color='red', linestyle='--')
axes[2].set_title("Cumulative Variance vs k")
axes[2].set_xlabel("Number of Components"); axes[2].set_ylabel("Cumulative Variance")

plt.tight_layout()
plt.show()

# --- Benchmark: true latency & memory footprint vs k ---

def measure_latency(X, pca, k, repeats=20000):
    # Measure loop overhead
    start = time.perf_counter()
    for _ in range(repeats):
        pass
    overhead = time.perf_counter() - start
    # Measure loop + transform
    start = time.perf_counter()
    for _ in range(repeats):
        _ = pca.transform(X, k)
    total = time.perf_counter() - start
    # Net time per call (ms)
    return (total - overhead) * 1000 / repeats

ks       = list(range(1, n_components+1))
times_ms = []
mem_kb   = []

for k in ks:
    t = measure_latency(X, pca, k, repeats=20000)
    times_ms.append(t)
    W = pca.components_[:, :k]
    mem_kb.append(W.nbytes / 1024)

# --- Figure 3: latency & memory plots ---
fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))

ax3.plot(ks, times_ms, marker='o')
ax3.set_xlabel("Number of PCs (k)")
ax3.set_ylabel("Avg Transform Time (ms)")
ax3.set_title("True Projection Latency vs k")

ax4.bar(ks, mem_kb)
ax4.set_xlabel("Number of PCs (k)")
ax4.set_ylabel("Memory of W (KB)")
ax4.set_title("Memory Footprint vs k")

plt.tight_layout()
plt.show()

# --- Print numeric benchmark results ---
print("k\tLatency (ms)\tMemory (KB)")
for k, t, m in zip(ks, times_ms, mem_kb):
    print(f"{k}\t{t:.5f}\t\t{m:.2f}")

# --- 1) Reconstruction Error vs k ---
ks = list(range(1, n_components+1))
errors = []
for k in ks:
    Zk   = pca.transform(X, k)
    Xrec = Zk @ pca.components_[:, :k].T + pca.mean_
    # mean squared error per sample
    errors.append(np.mean(np.sum((X - Xrec)**2, axis=1)))

# --- 2) Biplot (scores + loadings for k=2) ---
Z2       = pca.transform(X, 2)
loadings = pca.components_[:, :2]  # shape (n_features, 2)

# --- 3) Residual Space = last 2 PCs ---
Zfull = pca.transform(X, n_components)

# --- 4) Class Separation vs k ---
def class_separation(Z, y):
    cents = [Z[y==c].mean(axis=0) for c in np.unique(y)]
    dists = [np.linalg.norm(a-b) for a, b in combinations(cents, 2)]
    return np.mean(dists)

seps = [class_separation(pca.transform(X, k), y) for k in ks]

# --- 6) Heatmap of first 4 PC loadings ---
load_heat = pca.components_[:, :4]  # (n_features, 4)

# --- Combine into one figure with 5 subplots ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
ax = axes.flatten()

# (1) Reconstruction Error
ax[0].plot(ks, errors, marker='o')
ax[0].set_title("Reconstruction Error vs k")
ax[0].set_xlabel("k"); ax[0].set_ylabel("Mean Squared Error")

# (2) Biplot
ax[1].scatter(Z2[:,0], Z2[:,1], c=y, cmap='tab10', alpha=0.6)
for i, feat in enumerate(data.feature_names):
    ax[1].arrow(0, 0, loadings[i,0]*3, loadings[i,1]*3,
                head_width=0.1, length_includes_head=True)
    ax[1].text(loadings[i,0]*3.2, loadings[i,1]*3.2, feat, color='r')
ax[1].set_title("Biplot (PC1 vs PC2)")
ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")

# (3) Residual Space (last 2 PCs)
ax[2].scatter(Zfull[:,-2], Zfull[:,-1], c=y, cmap='tab10', alpha=0.6)
ax[2].set_title(f"Residual Space (PC{n_components-1} vs PC{n_components})")
ax[2].set_xlabel(f"PC{n_components-1}"); ax[2].set_ylabel(f"PC{n_components}")

# (4) Class Separation
ax[3].plot(ks, seps, marker='o')
ax[3].set_title("Class Separation vs k")
ax[3].set_xlabel("k"); ax[3].set_ylabel("Avg. Centroid Distance")

# (6) Loadings Heatmap
im = ax[4].imshow(load_heat, aspect='auto')
ax[4].set_title("Heatmap of First 4 PC Loadings")
ax[4].set_xlabel("Principal Component")
ax[4].set_ylabel("Feature")
ax[4].set_xticks(range(4))
ax[4].set_xticklabels([f"PC{i+1}" for i in range(4)])
ax[4].set_yticks(range(len(data.feature_names)))
ax[4].set_yticklabels(data.feature_names)
fig.colorbar(im, ax=ax[4], shrink=0.6)

# Remove unused subplot
fig.delaxes(ax[5])

plt.tight_layout()
plt.show()