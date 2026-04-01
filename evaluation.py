"""
evaluation.py — Đánh giá GraphRAG vs Traditional CF

Metrics:
  Precision@K  = |recommended ∩ relevant| / K
  Recall@K     = |recommended ∩ relevant| / |relevant|
  NDCG@K       = normalized discounted cumulative gain
  Modularity   = quality of community structure

Protocol: Leave-one-out
  Với mỗi user, ẩn review cuối cùng (test item),
  dùng phần còn lại để recommend, xem có hit không.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
from collections import defaultdict
from tqdm import tqdm
from falkordb import FalkorDB

FALKOR_HOST = os.getenv("FALKOR_HOST", "localhost")
FALKOR_PORT = int(os.getenv("FALKOR_PORT", 6379))
GRAPH_NAME  = "amazon_graph"

# ── Metric functions ──────────────────────────────────────────────────────────

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    top = recommended[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top  = recommended[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    NDCG@K — relevance = 1 nếu item trong relevant set, 0 otherwise.
    """
    dcg  = 0.0
    for i, r in enumerate(recommended[:k], 1):
        if r in relevant:
            dcg += 1.0 / math.log2(i + 1)
    # Ideal DCG
    n_hits = min(len(relevant), k)
    idcg   = sum(1.0 / math.log2(i + 1) for i in range(1, n_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    return 1.0 if any(r in relevant for r in recommended[:k]) else 0.0


# ── Leave-one-out evaluator ───────────────────────────────────────────────────

class Evaluator:
    def __init__(self, df: pd.DataFrame, partition: dict):
        """
        df        : processed reviews dataframe
        partition : user_id → community_id
        """
        self.df        = df
        self.partition = partition

    def prepare_splits(self, min_ratings: int = 5):
        """
        Leave-one-out: sort by implicit timestamp (row order),
        ẩn item cuối cùng làm test.
        """
        train_rows, test_rows = [], []
        for uid, group in self.df.groupby("user_id"):
            if len(group) < min_ratings:
                continue
            group_sorted = group.reset_index(drop=True)
            test_rows.append(group_sorted.iloc[-1])
            train_rows.append(group_sorted.iloc[:-1])

        self.train_df = pd.concat(train_rows).reset_index(drop=True)
        self.test_df  = pd.DataFrame(test_rows).reset_index(drop=True)
        print(f"[EVAL] Train: {len(self.train_df):,} | Test users: {len(self.test_df):,}")

    # ── GraphRAG recommender (community-based) ────────────────────────────────

    def graphrag_recommend(self, user_id: str, top_k: int = 10) -> list[str]:
        comm = self.partition.get(user_id)
        if comm is None:
            return []
        # Products bought in same community (excl. user's own)
        user_bought = set(
            self.train_df[self.train_df["user_id"] == user_id]["product_id"]
        )
        comm_users  = [u for u, c in self.partition.items() if c == comm and u != user_id]
        if not comm_users:
            return []
        comm_df     = self.train_df[self.train_df["user_id"].isin(comm_users)]
        prod_scores = (
            comm_df.groupby("product_id")
            .agg(score=("rating", lambda x: x.count() * x.mean()))
            ["score"]
            .sort_values(ascending=False)
        )
        recs = [p for p in prod_scores.index if p not in user_bought]
        return recs[:top_k]

    # ── SVD baseline recommender ──────────────────────────────────────────────

    def svd_recommend(self, user_id: str, top_k: int = 10,
                       _cache: dict = {}) -> list[str]:
        if "model" not in _cache:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.preprocessing import normalize
            pivot = self.train_df.pivot_table(
                index="user_id", columns="product_id",
                values="rating", fill_value=0
            )
            svd  = TruncatedSVD(n_components=min(50, pivot.shape[1]-1), random_state=42)
            U    = normalize(svd.fit_transform(pivot.values))
            pred = U @ svd.components_
            _cache["model"]    = pred
            _cache["users"]    = pivot.index.tolist()
            _cache["products"] = pivot.columns.tolist()
            _cache["pivot"]    = pivot

        if user_id not in _cache["users"]:
            return []
        idx     = _cache["users"].index(user_id)
        scores  = _cache["model"][idx].copy()
        bought  = _cache["pivot"].iloc[idx].values > 0
        scores[bought] = -999
        top_idx = scores.argsort()[::-1][:top_k]
        return [_cache["products"][i] for i in top_idx]

    # ── Run full evaluation ───────────────────────────────────────────────────

    def evaluate(self, k_values: list[int] = [5, 10, 20],
                 sample_n: int = 500) -> dict:
        self.prepare_splits()

        # Sample users để chạy nhanh
        test_users = self.test_df.sample(
            n=min(sample_n, len(self.test_df)), random_state=42
        )

        results = {
            "graphrag": defaultdict(list),
            "svd":      defaultdict(list),
        }

        for _, row in tqdm(test_users.iterrows(), total=len(test_users),
                           desc="Evaluating"):
            uid     = row["user_id"]
            test_item = row["product_id"]
            relevant  = {test_item}

            # GraphRAG
            gr_recs = self.graphrag_recommend(uid, top_k=max(k_values))
            # SVD
            sv_recs = self.svd_recommend(uid, top_k=max(k_values))

            for k in k_values:
                results["graphrag"][f"P@{k}"].append(precision_at_k(gr_recs, relevant, k))
                results["graphrag"][f"R@{k}"].append(recall_at_k(gr_recs, relevant, k))
                results["graphrag"][f"NDCG@{k}"].append(ndcg_at_k(gr_recs, relevant, k))
                results["graphrag"][f"HR@{k}"].append(hit_rate_at_k(gr_recs, relevant, k))

                results["svd"][f"P@{k}"].append(precision_at_k(sv_recs, relevant, k))
                results["svd"][f"R@{k}"].append(recall_at_k(sv_recs, relevant, k))
                results["svd"][f"NDCG@{k}"].append(ndcg_at_k(sv_recs, relevant, k))
                results["svd"][f"HR@{k}"].append(hit_rate_at_k(sv_recs, relevant, k))

        # Aggregate
        summary = {}
        for method, metrics in results.items():
            summary[method] = {
                m: round(float(np.mean(vals)), 4)
                for m, vals in metrics.items()
            }

        return summary

    # ── Print & plot ──────────────────────────────────────────────────────────

    def print_results(self, summary: dict):
        print("\n" + "=" * 65)
        print("  Evaluation Results (Leave-One-Out)")
        print("=" * 65)
        metrics = list(next(iter(summary.values())).keys())
        header  = f"{'Metric':<12}" + "".join(f"{m:<22}" for m in summary.keys())
        print(header)
        print("-" * 65)
        for m in metrics:
            row = f"{m:<12}"
            for method in summary:
                v = summary[method][m]
                row += f"{v:<22.4f}"
            print(row)
        print("=" * 65)

    def plot_results(self, summary: dict, save_path: str = None):
        methods = list(summary.keys())
        k_vals  = [5, 10, 20]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        colors = ["#534AB7", "#888780"]

        for ax, metric_prefix in zip(axes, ["P@", "NDCG@", "HR@"]):
            for method, color in zip(methods, colors):
                vals = [summary[method].get(f"{metric_prefix}{k}", 0) for k in k_vals]
                ax.plot(k_vals, vals, marker="o", label=method, color=color, linewidth=2)
            ax.set_title(f"{metric_prefix}K", fontsize=13)
            ax.set_xlabel("K")
            ax.set_xticks(k_vals)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top","right"]].set_visible(False)

        plt.suptitle("GraphRAG vs SVD Baseline", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[EVAL] Plot saved → {save_path}")
        return fig


# ── Modularity evaluation ─────────────────────────────────────────────────────

def evaluate_modularity(G, partition_louvain: dict, partition_kmeans: dict) -> dict:
    import community as cl
    return {
        "louvain_modularity": round(cl.modularity(partition_louvain, G, weight="weight"), 4),
        "kmeans_modularity":  round(cl.modularity(partition_kmeans,  G, weight="weight"), 4),
        "louvain_communities": len(set(partition_louvain.values())),
        "kmeans_communities":  len(set(partition_kmeans.values())),
    }


if __name__ == "__main__":
    from pathlib import Path
    data_path = Path(__file__).parent.parent / "data" / "reviews_processed.parquet"
    df = pd.read_parquet(data_path)

    # Load partition từ graph
    db    = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    graph = db.select_graph(GRAPH_NAME)
    res   = graph.query("MATCH (u:User) WHERE u.community_id IS NOT NULL "
                        "RETURN u.user_id, u.community_id")
    partition = {r[0]: int(r[1]) for r in res.result_set}

    evaluator = Evaluator(df, partition)
    summary   = evaluator.evaluate(k_values=[5, 10, 20], sample_n=500)
    evaluator.print_results(summary)
    evaluator.plot_results(
        summary,
        save_path=str(Path(__file__).parent.parent / "data" / "eval_results.png")
    )
