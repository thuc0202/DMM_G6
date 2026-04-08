"""
clustering.py — Weighted Louvain Community Detection trên FalkorDB graph

Điểm cải tiến so với standard Louvain:
  Edge weight = purchase_count × avg_rating  (encode cả frequency lẫn satisfaction)
  Thay vì binary edge (mua/không mua), weight phản ánh chất lượng tương tác.

Output:
  - community_id ghi ngược vào mỗi User node trong FalkorDB
  - Trả về dict: user_id → community_id
  - Evaluation: modularity score, cluster stats
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain   # python-louvain
from falkordb import FalkorDB
from collections import defaultdict, Counter

FALKOR_HOST = os.getenv("FALKOR_HOST", "localhost")
FALKOR_PORT = int(os.getenv("FALKOR_PORT", 6379))
GRAPH_NAME  = "amazon_graph"


class WeightedCommunityDetector:
    """
    Thực hiện weighted Louvain community detection.

    Cách build graph cho Louvain:
      - Nodes = Users
      - Edge (u1, u2) tồn tại nếu họ mua ≥ 1 sản phẩm chung
      - Edge weight = Σ (w_u1_p × w_u2_p) với w = rating
        (giống cosine similarity trên rating vectors, nhưng không normalize)
    """

    def __init__(self):
        self.db    = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
        self.graph = self.db.select_graph(GRAPH_NAME)
        self.partition: dict[str, int] = {}   # user_id → community_id

    # ── Step 1: Lấy user-product-rating từ graph ─────────────────────────────

    def fetch_user_ratings(self) -> pd.DataFrame:
        """Pull toàn bộ BOUGHT edges về Python."""
        res = self.graph.query("""
            MATCH (u:User)-[b:BOUGHT]->(p:Product)
            RETURN u.user_id AS user_id,
                   p.product_id AS product_id,
                   b.weight AS rating
        """)
        rows = [
            {"user_id": r[0], "product_id": r[1], "rating": float(r[2])}
            for r in res.result_set
        ]
        df = pd.DataFrame(rows)
        print(f"[CLUSTER] Fetch {len(df):,} user-product pairs")
        return df

    # ── Step 2: Build user-user similarity graph ──────────────────────────────

    def build_user_graph(self, df: pd.DataFrame) -> nx.Graph:
        """
        Build user-user graph với weighted edges.
        weight(u1,u2) = Σ_p (rating_u1_p × rating_u2_p)
        """
        # product → list of (user_id, rating)
        product_users: dict[str, list] = defaultdict(list)
        for _, row in df.iterrows():
            product_users[row["product_id"]].append(
                (row["user_id"], row["rating"])
            )

        G = nx.Graph()
        users = df["user_id"].unique().tolist()
        G.add_nodes_from(users)

        edge_weights: dict[tuple, float] = defaultdict(float)
        for pid, user_list in product_users.items():
            if len(user_list) < 2:
                continue
            for i in range(len(user_list)):
                u1, r1 = user_list[i]
                for j in range(i + 1, len(user_list)):
                    u2, r2 = user_list[j]
                    if u1 != u2:
                        key = (min(u1, u2), max(u1, u2))
                        # ★ Weighted edge: product of ratings
                        edge_weights[key] += r1 * r2

        for (u1, u2), w in edge_weights.items():
            G.add_edge(u1, u2, weight=w)

        print(f"[CLUSTER] User graph: {G.number_of_nodes():,} nodes, "
              f"{G.number_of_edges():,} edges")
        return G

    # ── Step 3: Louvain clustering ────────────────────────────────────────────

    def run_louvain(self, G: nx.Graph, resolution: float = 1.0) -> dict:
        """
        Louvain với weight='weight'.
        resolution > 1 → nhiều community nhỏ hơn
        resolution < 1 → ít community lớn hơn
        """
        partition = community_louvain.best_partition(
            G, weight="weight", resolution=resolution, random_state=42
        )
        n_communities = len(set(partition.values()))
        modularity    = community_louvain.modularity(partition, G, weight="weight")
        print(f"[CLUSTER] Louvain: {n_communities} communities, "
              f"modularity = {modularity:.4f}")
        return partition

    # ── Step 4: Ghi community_id ngược vào FalkorDB ───────────────────────────

    def write_communities_to_graph(self, partition: dict):
        batch = []
        for user_id, comm_id in partition.items():
            batch.append({"user_id": user_id, "community_id": comm_id})
            if len(batch) == 500:
                self.graph.query("""
                    UNWIND $rows AS r
                    MATCH (u:User {user_id: r.user_id})
                    SET u.community_id = r.community_id
                """, {"rows": batch})
                batch = []
        if batch:
            self.graph.query("""
                UNWIND $rows AS r
                MATCH (u:User {user_id: r.user_id})
                SET u.community_id = r.community_id
            """, {"rows": batch})
        print(f"[CLUSTER] Đã ghi community_id cho {len(partition):,} users")

    # ── Step 5: Community stats ───────────────────────────────────────────────

    def community_stats(self, partition: dict, df: pd.DataFrame) -> pd.DataFrame:
        """Tóm tắt mỗi community: size, top categories, avg rating."""
        df2 = df.copy()
        df2["community_id"] = df2["user_id"].map(partition)
        df2 = df2.dropna(subset=["community_id"])

        stats = []
        for cid, group in df2.groupby("community_id"):
            top_cats = group["product_id"].value_counts().head(3).index.tolist()
            stats.append({
                "community_id":  int(cid),
                "user_count":    group["user_id"].nunique(),
                "product_count": group["product_id"].nunique(),
                "avg_rating":    round(group["rating"].mean(), 2),
                "total_buys":    len(group),
            })
        stats_df = pd.DataFrame(stats).sort_values("user_count", ascending=False)
        return stats_df

    # ── Step 6: So sánh với K-means baseline ─────────────────────────────────

    def kmeans_baseline(self, df: pd.DataFrame, n_clusters: int = None) -> dict:
        """
        Baseline: K-means trên user feature vectors.
        Features: purchase_count, avg_rating, std_rating
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        user_features = df.groupby("user_id").agg(
            purchase_count=("product_id", "count"),
            avg_rating=("rating", "mean"),
            std_rating=("rating", "std"),
        ).fillna(0)

        if n_clusters is None:
            n_clusters = max(2, int(np.sqrt(len(user_features) / 2)))

        scaler   = StandardScaler()
        X        = scaler.fit_transform(user_features.values)
        km       = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels   = km.fit_predict(X)
        partition = dict(zip(user_features.index.tolist(), labels.tolist()))
        print(f"[CLUSTER] K-means baseline: {n_clusters} clusters")
        return partition

    def modularity_score(self, G: nx.Graph, partition: dict) -> float:
        return community_louvain.modularity(partition, G, weight="weight")

    # ── Main entrypoint ───────────────────────────────────────────────────────

    def run(self, resolution: float = 1.0) -> dict:
        print("=" * 55)
        print("  Weighted Louvain Community Detection")
        print("=" * 55)

        df = self.fetch_user_ratings()
        G  = self.build_user_graph(df)

        # Louvain (proposed)
        partition = self.run_louvain(G, resolution=resolution)
        self.partition = partition
        self.write_communities_to_graph(partition)

        stats = self.community_stats(partition, df)
        print("\n[CLUSTER] Top communities:")
        print(stats.head(10).to_string(index=False))

        # So sánh với K-means baseline
        n_communities = len(set(partition.values()))
        km_partition  = self.kmeans_baseline(df, n_clusters=n_communities)
        km_mod        = self.modularity_score(G, km_partition)
        louvain_mod   = self.modularity_score(G, partition)

        print(f"\n[EVAL] Modularity — Louvain: {louvain_mod:.4f} | "
              f"K-means: {km_mod:.4f} "
              f"(Louvain {'tốt hơn' if louvain_mod > km_mod else 'tương đương'})")

        self._stats_df = stats
        self._graph_G  = G
        return partition


def get_community_products(community_id: int, top_k: int = 10) -> list[dict]:
    """
    Lấy top products trong một community dựa trên:
      score = avg_rating × purchase_count (weighted popularity)
    Dùng trong GraphRAG retrieval.
    """
    db    = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    graph = db.select_graph(GRAPH_NAME)

    res = graph.query("""
        MATCH (u:User {community_id: $cid})-[b:BOUGHT]->(p:Product)
        WITH p,
             count(b)      AS purchase_count,
             avg(b.weight) AS avg_rating
        WHERE purchase_count >= 2
        RETURN p.product_id AS product_id,
               p.name       AS name,
               p.category   AS category,
               purchase_count,
               avg_rating,
               purchase_count * avg_rating AS score
        ORDER BY score DESC
        LIMIT $k
    """, {"cid": community_id, "k": top_k})

    return [
        {
            "product_id":     r[0],
            "name":           r[1],
            "category":       r[2],
            "purchase_count": int(r[3]),
            "avg_rating":     round(float(r[4]), 2),
            "score":          round(float(r[5]), 2),
        }
        for r in res.result_set
    ]


if __name__ == "__main__":
    detector = WeightedCommunityDetector()
    partition = detector.run()
