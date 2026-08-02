"""t-SNE of the 275-persona Big Five space -> compact JSON for the explainer.

Big Five is only 5-dimensional, so t-SNE here is a *nonlinear* re-embedding of that
5-D space (complementary to the linear PCA atlas). Colours by the 6 k-means
archetypes; also emits AA per point so the explainer can show the Assistant vs
drifted gradient. Coordinates are normalised to [0,1] for canvas plotting.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from src.bigfive import stimuli as S

ATLAS = Path("results/bigfive/llama-3.3-70b/role_profiles_atlas")
# archetype names keyed by their (unique) cluster size, from the atlas cards
NAME_BY_N = {125: "Professional helper", 75: "Competent-creative",
             36: "Otherworldly", 18: "Dark", 14: "Immature / vulnerable",
             8: "Withdrawn-ascetic"}


def main():
    prof = json.loads((ATLAS / "role_bigfive_profiles.json").read_text())["per_role"]
    names = [n for n in prof if "aa_proj" in prof[n]]
    Z = np.array([[prof[n]["z_vs_roles"][t] for t in S.TRAITS] for n in names])
    AA = np.array([prof[n]["aa_proj"] for n in names])
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)

    km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(Z)
    lab = km.labels_
    # per-cluster metadata (name via size, mean AA for the colour ramp)
    clusters = []
    for c in range(6):
        m = lab == c
        n = int(m.sum())
        clusters.append({"name": NAME_BY_N.get(n, f"cluster {c}"),
                         "n": n, "aa": round(float(AA[m].mean()), 2)})

    emb = TSNE(n_components=2, perplexity=30, learning_rate="auto",
               init="pca", random_state=0).fit_transform(Zs)
    lo, hi = emb.min(0), emb.max(0)
    norm = (emb - lo) / (hi - lo + 1e-9)

    pts = [[round(float(norm[i, 0]), 4), round(float(norm[i, 1]), 4),
            int(lab[i]), round(float(AA[i]), 2), names[i]] for i in range(len(names))]
    anchors = ["default", "assistant", "void", "eldritch", "criminal", "teenager",
               "altruist", "shaman", "ghost", "hermit"]
    out = {"clusters": clusters, "points": pts,
           "anchors": [a for a in anchors if a in names]}
    (ATLAS / "tsne.json").write_text(json.dumps(out))
    for c in sorted(range(6), key=lambda k: -clusters[k]["aa"]):
        print(f"  cluster {c} {clusters[c]['name']:22} n={clusters[c]['n']:3} meanAA={clusters[c]['aa']:+.2f}")
    print("wrote tsne.json", len(pts), "points")


if __name__ == "__main__":
    main()
