"""Track D bridge analysis: link the user map to the full role atlas.

Uses the 8+8 user map (v2) and the 275-role atlas to build a unified
user-tag -> evoked-role -> role-Big-Five chain, a tag-cluster typology, and an
explicit/implicit deep-dive.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from src.bigfive import stimuli as BF

V2 = Path("results/user_persona_mapping_v2/llama-3.3-70b")
ATLAS = Path("results/bigfive/llama-3.3-70b/role_profiles_atlas")
TAGS = ["vulnerability", "expertise", "emotional_load", "trust", "tech_literacy"]


def main():
    umap = json.loads((V2 / "persona_map.json").read_text())["personas"]
    atlas = json.loads((ATLAS / "role_bigfive_profiles.json").read_text())["per_role"]
    ids = [i for i in sorted(umap) if umap[i].get("n", 0) > 0 and umap[i].get("top_role_meancos")]
    out = {"n": len(ids)}

    # --- bridge: user's evoked-role -> that role's atlas Big Five ---
    # consistency: does the user's OWN evoked Big Five match the evoked role's atlas profile?
    consist = []
    role_bf_rows = []
    for i in ids:
        top = umap[i]["top_role_meancos"][0][0]
        if top in atlas and "z_vs_roles" in atlas[top]:
            role_z = [atlas[top]["z_vs_roles"][t] for t in BF.TRAITS]
            user_z = [umap[i]["bigfive_z"][t] for t in BF.TRAITS]
            if np.std(role_z) > 1e-9 and np.std(user_z) > 1e-9:
                consist.append(spearmanr(role_z, user_z).statistic)
            role_bf_rows.append((i, top, role_z))
    out["user_vs_evokedrole_bigfive_agreement_rho"] = round(float(np.nanmean(consist)), 3)

    # --- tag-cluster typology: (vulnerability, expertise) quadrants ---
    def quad(e):
        v = e["tags"].get("vulnerability", 5); x = e["tags"].get("expertise", 5)
        return ("vuln" if v >= 6 else "low-vuln") + " / " + ("expert" if x >= 6 else "novice")
    quads = {}
    for i in ids:
        q = quad(umap[i]); quads.setdefault(q, []).append(i)
    out["tag_clusters"] = {}
    for q, members in quads.items():
        roles = Counter(umap[m]["top_role_meancos"][0][0] for m in members)
        bf = {t: round(float(np.mean([umap[m]["bigfive_z"][t] for m in members])), 2) for t in BF.TRAITS}
        aa = round(float(np.mean([umap[m]["aa_proj"] for m in members])), 2)
        out["tag_clusters"][q] = {"n": len(members), "mean_bigfive_z": bf, "mean_aa": aa,
                                  "top_roles": roles.most_common(5)}

    # --- explicit vs implicit deep-dive ---
    ei = []
    for i in ids:
        e = umap[i].get("bigfive_by_arm", {})
        if "explicit" in e and "implicit" in e:
            ve = [e["explicit"][t] for t in BF.TRAITS]; vi = [e["implicit"][t] for t in BF.TRAITS]
            if np.std(ve) > 1e-9 and np.std(vi) > 1e-9:
                ei.append(spearmanr(ve, vi).statistic)
    out["explicit_implicit_rho"] = round(float(np.nanmean(ei)), 3)

    (V2 / "bridge_analysis.json").write_text(json.dumps(out, indent=1))
    print(f"=== user evoked-Big-Five vs evoked-role atlas-Big-Five agreement: rho={out['user_vs_evokedrole_bigfive_agreement_rho']}")
    print(f"=== explicit/implicit agreement (8+8): rho={out['explicit_implicit_rho']}")
    print("=== tag-cluster typology (vulnerability x expertise) ===")
    for q, d in out["tag_clusters"].items():
        print(f"  {q:22} n={d['n']:3} AA={d['mean_aa']:+.2f} BF={d['mean_bigfive_z']}")
        print(f"     top roles: {[r for r,_ in d['top_roles'][:4]]}")
    print("wrote bridge_analysis.json")


if __name__ == "__main__":
    main()
