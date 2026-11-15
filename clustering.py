import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon, Circle
from preprocessing import load_and_preprocess, target_col

os.makedirs("plots", exist_ok=True)

RANDOM_SEED = 42
df, final_df = load_and_preprocess(show_eda=False)

X = final_df.drop(columns=[target_col]).values
scaler_reg = StandardScaler()
X_scaled = scaler_reg.fit_transform(X)

# پیدا کردن بهترین k بر اساس silhouette
sil_scores = {}
for kk in range(2, 9):
    km = KMeans(n_clusters=kk, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores[kk] = score

best_k = max(sil_scores, key=sil_scores.get)
print(f"Best k based on silhouette score: {best_k}, score={sil_scores[best_k]:.4f}")

# Elbow plot (inertia) برای k مختلف (1..8)
inertias = []
K_range = range(1, 9)
for kk in K_range:
    km = KMeans(n_clusters=kk, n_init=5, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
plt.figure(figsize=(6, 3))
plt.plot(list(K_range), inertias, marker='o')
plt.title("KMeans Elbow plot")
plt.xlabel("n_clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/clust_kmeans_elbow.png", dpi=300, bbox_inches="tight")
plt.show()

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# PCA 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, alpha=0.5)
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], marker='x', s=100)
plt.title("KMeans Clustering (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("plots/clust_kmeans_pca2d.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n=== Clustering (stepwise visualization) ===")
if target_col in final_df.columns:
    X_cluster = final_df.drop(columns=[target_col]).values
else:
    X_cluster = final_df.values

scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

pca_vis = PCA(n_components=2, random_state=RANDOM_SEED)
X_pca_vis = pca_vis.fit_transform(X_cluster_scaled)
explained = pca_vis.explained_variance_ratio_.sum()
print(f"PCA(2) explained variance ratio sum: {explained:.3f}")

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except Exception:
    TSNE_AVAILABLE = False

use_tsne = False
if explained < 0.30 and TSNE_AVAILABLE:
    print("PCA(2) explains little variance -> using t-SNE for visualization.")
    X_vis = TSNE(n_components=2, init='pca', random_state=RANDOM_SEED).fit_transform(X_cluster_scaled)
    use_tsne = True
else:
    X_vis = X_pca_vis

xy = X_vis
xs, ys = xy[:, 0], xy[:, 1]

# مقداردهی اولیه centroids برای نمایش مرحله‌ای (نمونهٔ تصادفی)
rng = np.random.RandomState(RANDOM_SEED)
init_idx = rng.choice(len(X_cluster_scaled), size=best_k, replace=False)
centroids_scaled = X_cluster_scaled[init_idx].astype(float).copy()
centroids_scaled += rng.normal(scale=1e-6, size=centroids_scaled.shape)
print("Initialized stepwise centroids from random sample (to ensure visible iterations).")

MAX_STEPWISE_ITERS = 12
PAUSE_BETWEEN = 1.0
TOL_STOP = 1e-3
USE_BLOCKING_SHOW = False
markers = ['o', '^', 's', 'P', 'X', 'D', 'v', '<', '>']
cmap = plt.colormaps.get_cmap("tab10")


def stepwise_kmeans_visual(Xs_scaled, centroids_scaled_init, xy_vis, k, max_iters=12, tol=1e-3, pause=1.0, block_show=False):
    cent_scaled = centroids_scaled_init.copy()
    for it in range(1, max_iters + 1):
        dists = np.linalg.norm(Xs_scaled[:, None, :] - cent_scaled[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_cent = cent_scaled.copy()
        for j in range(cent_scaled.shape[0]):
            idx = np.where(labels == j)[0]
            if idx.size > 0:
                new_cent[j] = Xs_scaled[idx].mean(axis=0)
        movement = np.linalg.norm(new_cent - cent_scaled, axis=1).max()
        cent_scaled = new_cent

        if not use_tsne:
            cent_vis = pca_vis.transform(cent_scaled)
        else:
            cent_vis = np.zeros((k, 2))
            for j in range(k):
                idx = np.where(labels == j)[0]
                if idx.size > 0:
                    cent_vis[j] = xy_vis[idx].mean(axis=0)
                else:
                    cent_vis[j] = xy_vis.mean(axis=0)

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_facecolor('#f8f8f8')
        ax.hexbin(xs, ys, gridsize=120, cmap='Greys', mincnt=1)
        try:
            ax.collections[0].set_alpha(0.12)
        except Exception:
            pass
        plotted_any = False
        for j in range(k):
            idx = np.where(labels == j)[0]
            pts_vis = xy_vis[idx]
            color = cmap(j)
            if pts_vis.shape[0] >= 3:
                try:
                    hull = ConvexHull(pts_vis)
                    hull_pts = pts_vis[hull.vertices]
                    inner = hull_pts.mean(axis=0) + (hull_pts - hull_pts.mean(axis=0)) * 1.06
                    outer = hull_pts.mean(axis=0) + (hull_pts - hull_pts.mean(axis=0)) * 1.12
                    ax.add_patch(Polygon(outer, closed=True, facecolor=color, edgecolor=None, alpha=0.10, zorder=0))
                    ax.add_patch(Polygon(inner, closed=True, facecolor=color, edgecolor=None, alpha=0.18, zorder=1))
                except Exception:
                    c = pts_vis.mean(axis=0)
                    ax.add_patch(Circle((c[0], c[1]), 0.06 * np.mean(np.ptp(xy_vis, axis=0)), facecolor=color, alpha=0.10, zorder=0))
                    ax.add_patch(Circle((c[0], c[1]), 0.03 * np.mean(np.ptp(xy_vis, axis=0)), facecolor=color, alpha=0.18, zorder=1))
            if pts_vis.shape[0] > 0:
                ax.scatter(pts_vis[:, 0], pts_vis[:, 1], s=44, marker=markers[j % len(markers)], facecolor='white', edgecolor=color, linewidth=1.1, zorder=3, label=f"cluster {j}")
                plotted_any = True

        ax.scatter(cent_vis[:, 0], cent_vis[:, 1], marker='X', s=160, c='k', zorder=4)
        ax.set_title(f"Iteration {it} — max centroid move = {movement:.6f}")
        ax.set_xlabel("vis dim 1")
        ax.set_ylabel("vis dim 2")
        if plotted_any:
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title='cluster')
        plt.tight_layout()
        plt.savefig(f"plots/clust_stepwise_iter_{it}.png", dpi=300, bbox_inches="tight")
        if block_show:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(pause)
            plt.close(fig)
        print(f"Displayed iter {it}: max_move={movement:.6f}, cluster sizes={dict(zip(*np.unique(labels, return_counts=True)))}")
        if movement < tol:
            print(f"Stopping at iter {it}: movement {movement:.6f} < tol {tol}")
            break
    return labels, cent_scaled


labels_stepwise, cent_scaled_final = stepwise_kmeans_visual(
    X_cluster_scaled,
    centroids_scaled,
    X_vis,
    k=best_k,
    max_iters=MAX_STEPWISE_ITERS,
    tol=TOL_STOP,
    pause=PAUSE_BETWEEN,
    block_show=USE_BLOCKING_SHOW
)

# اجرای KMeans واقعی برای نمودار نهایی و اضافه کردن برچسب
kmeans_final = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=RANDOM_SEED)
labels_final = kmeans_final.fit_predict(X_cluster_scaled)
print("Final clustering sizes:", dict(zip(*np.unique(labels_final, return_counts=True))))
df["Cluster"] = labels_final

# رسم نمودار نهایی
fig, ax = plt.subplots(figsize=(9, 7))
ax.set_facecolor('#f8f8f8')
ax.hexbin(xs, ys, gridsize=140, cmap='Greys', mincnt=1)
try:
    ax.collections[0].set_alpha(0.12)
except Exception:
    pass
plotted_any = False
for j in range(best_k):
    idx = np.where(labels_final == j)[0]
    pts_vis = X_vis[idx]
    color = cmap(j)
    if pts_vis.shape[0] >= 3:
        try:
            hull = ConvexHull(pts_vis)
            hull_pts = pts_vis[hull.vertices]
            outer = hull_pts.mean(axis=0) + (hull_pts - hull_pts.mean(axis=0)) * 1.12
            inner = hull_pts.mean(axis=0) + (hull_pts - hull_pts.mean(axis=0)) * 1.06
            ax.add_patch(Polygon(outer, closed=True, facecolor=color, edgecolor=None, alpha=0.10, zorder=0))
            ax.add_patch(Polygon(inner, closed=True, facecolor=color, edgecolor=None, alpha=0.22, zorder=1))
        except Exception:
            c = pts_vis.mean(axis=0)
            ax.add_patch(Circle((c[0], c[1]), 0.06 * np.mean(np.ptp(X_vis, axis=0)), facecolor=color, alpha=0.10, zorder=0))
            ax.add_patch(Circle((c[0], c[1]), 0.03 * np.mean(np.ptp(X_vis, axis=0)), facecolor=color, alpha=0.22, zorder=1))
    if pts_vis.shape[0] > 0:
        ax.scatter(pts_vis[:, 0], pts_vis[:, 1], s=44, marker=markers[j % len(markers)], facecolor='white', edgecolor=color, linewidth=1.1, zorder=3, label=f"cluster {j}")
        plotted_any = True

if not use_tsne:
    centroids_vis_final = pca_vis.transform(kmeans_final.cluster_centers_)
else:
    centroids_vis_final = np.zeros((best_k, 2))
    for j in range(best_k):
        idx = np.where(labels_final == j)[0]
        centroids_vis_final[j] = X_vis[idx].mean(axis=0) if idx.size > 0 else X_vis.mean(axis=0)

ax.scatter(centroids_vis_final[:, 0], centroids_vis_final[:, 1], marker='X', s=180, c='k', zorder=4)
if plotted_any:
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title='cluster')
ax.set_xlabel("vis dim 1")
ax.set_ylabel("vis dim 2")
ax.set_title(f"Final clustering (k={best_k})")
plt.tight_layout()
plt.savefig("plots/clust_final_convex_hull.png", dpi=300, bbox_inches="tight")
plt.show()

# اندازه هر خوشه
unique, counts = np.unique(clusters, return_counts=True)
plt.figure(figsize=(5, 3))
plt.bar(unique, counts)
plt.title("Cluster sizes (KMeans)")
plt.xlabel("Cluster label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/clust_sizes.png", dpi=300, bbox_inches="tight")
plt.show()

# میانگین ویژگی‌ها به تفکیک خوشه
cluster_summary = df.groupby("Cluster")[["Price", "Release Year", "Min Number of Players"]].mean()
print("\nCluster interpretation (mean values):")
print(cluster_summary)

cluster_summary.plot(kind="bar", figsize=(8, 4))
plt.title("Cluster profile (mean of key features)")
plt.ylabel("Mean value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/clust_feature_profiles.png", dpi=300, bbox_inches="tight")
plt.show()

# توضیح متنی هر خوشه
for cl in cluster_summary.index:
    desc = []
    for col in cluster_summary.columns:
        mean_val = cluster_summary.loc[cl, col]
        desc.append(f"{col}≈{mean_val:.1f}")
    print(f"Cluster {cl}: " + ", ".join(desc))