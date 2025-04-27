import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces, fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- PCA from scratch ---
class PCA:
    def __init__(self):
        self.mean_ = None
        self.components_ = None  # eigenvectors
        self.eigvals_ = None     # eigenvalues

    def fit(self, X):
        # Center
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # Covariance
        cov = Xc.T @ Xc / (Xc.shape[0] - 1)
        # Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        self.eigvals_    = eigvals[idx]
        self.components_ = eigvecs[:, idx]

    def transform(self, X, k):
        Xc = X - self.mean_
        return Xc @ self.components_[:, :k]

    def reconstruct(self, Z, k):
        W = self.components_[:, :k]
        return Z @ W.T + self.mean_


def run_eigenface_demo(X, y, images, h, w, title):
    # split
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)
    # fit
    pca = PCA()
    pca.fit(Xtr)

    # mean + top 9 eigenfaces
    mean_face = pca.mean_.reshape(h, w)
    eigenfaces = pca.components_.T.reshape(-1, h, w)

    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    axes[0,0].imshow(mean_face, cmap='gray')
    axes[0,0].set_title(f"{title} Mean")
    axes[0,0].axis('off')
    for i, ax in enumerate(axes.flat[1:10], start=1):
        ax.imshow(eigenfaces[i], cmap='gray')
        ax.set_title(f"PC {i}")
        ax.axis('off')
    plt.suptitle(f"{title}: Mean & Top 9 Eigenfaces")
    plt.tight_layout()
    plt.show()

    # reconstruction demo
    ks = [10, 50, 100]
    sample_idxs = np.random.choice(len(Xte), 4, replace=False)
    fig, axes = plt.subplots(4, len(ks)+1, figsize=(12, 8))
    for r, idx in enumerate(sample_idxs):
        orig = images[idx]
        axes[r,0].imshow(orig, cmap='gray')
        axes[r,0].set_title("Original")
        axes[r,0].axis('off')
        for c, k in enumerate(ks, start=1):
            Z = pca.transform(Xte[idx:idx+1], k)
            rec = pca.reconstruct(Z, k).reshape(h, w)
            axes[r,c].imshow(rec, cmap='gray')
            axes[r,c].set_title(f"k={k}")
            axes[r,c].axis('off')
    plt.suptitle(f"{title}: Reconstructions")
    plt.tight_layout()
    plt.show()

    # 1-NN accuracy vs k
    accuracies = []
    for k in ks:
        Ztr = pca.transform(Xtr, k)
        Zte = pca.transform(Xte, k)
        acc = KNeighborsClassifier(1).fit(Ztr, ytr).score(Zte, yte)
        accuracies.append(acc)
    plt.figure(figsize=(5,3))
    plt.plot(ks, accuracies, marker='o')
    plt.title(f"{title}: 1-NN Accuracy vs k")
    plt.xlabel("k"); plt.ylabel("Accuracy")
    plt.ylim(0,1); plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"{title} accuracies:", dict(zip(ks, np.round(accuracies, 3))))


if __name__ == "__main__":
    # 1) Olivetti faces
    # olivetti = fetch_olivetti_faces(shuffle=True, random_state=0)
    # run_eigenface_demo(
    #     olivetti.data,
    #     olivetti.target,
    #     olivetti.images,
    #     olivetti.images.shape[1],
    #     olivetti.images.shape[2],
    #     title="Olivetti"
    # )

    # 2) LFW faces (min 20 images/person)
    lfw = fetch_lfw_people(min_faces_per_person=100, resize=0.5, download_if_missing=True)
    run_eigenface_demo(
        lfw.data,
        lfw.target,
        lfw.images,
        lfw.images.shape[1],
        lfw.images.shape[2],
        title="LFW"
    )
