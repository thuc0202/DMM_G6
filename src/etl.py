"""
etl.py — ETL pipeline: Amazon Reviews CSV → FalkorDB Knowledge Graph

Graph schema:
  Nodes : User, Product, Category
  Edges : BOUGHT  (User→Product,  weight=rating)
          REVIEWED(User→Product,  weight=rating, text=review)
          BELONGS (Product→Category)
          SIMILAR (Product→Product, weight=similarity_score)
"""

import os, json, gzip, requests, math
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from falkordb import FalkorDB

# ── Config ────────────────────────────────────────────────────────────────────
FALKOR_HOST = os.getenv("FALKOR_HOST", "localhost")
FALKOR_PORT = int(os.getenv("FALKOR_PORT", 6379))
GRAPH_NAME  = "amazon_graph"
DATA_DIR    = Path(__file__).parent.parent / "data"
DATA_FILE   = DATA_DIR / "reviews_electronics.json.gz"

# Dùng Electronics 5-core subset (~50K reviews, nhỏ gọn cho demo)
DATASET_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
    "reviews_Electronics_5.json.gz"
)
MAX_REVIEWS = 30_000   # Giới hạn để demo nhanh; tăng nếu muốn đầy đủ
MIN_REVIEWS_PER_USER    = 3
MIN_REVIEWS_PER_PRODUCT = 5

# ── Helpers ───────────────────────────────────────────────────────────────────

def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    if DATA_FILE.exists():
        print(f"[ETL] Dataset đã tồn tại: {DATA_FILE}")
        return
    print(f"[ETL] Đang tải dataset từ SNAP...")
    r = requests.get(DATASET_URL, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(DATA_FILE, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Download"
    ) as bar:
        for chunk in r.iter_content(8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"[ETL] Tải xong → {DATA_FILE}")


def load_reviews(max_rows=MAX_REVIEWS) -> pd.DataFrame:
    rows = []
    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            try:
                d = json.loads(line)
                rows.append({
                    "user_id":    d.get("reviewerID", ""),
                    "product_id": d.get("asin", ""),
                    "rating":     float(d.get("overall", 3.0)),
                    "review":     d.get("reviewText", "")[:300],
                    "category":   d.get("category", ["Electronics"])[0]
                              if isinstance(d.get("category"), list)
                              else "Electronics",
                    "product_name": d.get("summary", d.get("asin", ""))[:80],
                })
            except Exception:
                continue
    df = pd.DataFrame(rows)
    # Filter sparse users / products
    uc = df["user_id"].value_counts()
    pc = df["product_id"].value_counts()
    df = df[
        df["user_id"].isin(uc[uc >= MIN_REVIEWS_PER_USER].index) &
        df["product_id"].isin(pc[pc >= MIN_REVIEWS_PER_PRODUCT].index)
    ].reset_index(drop=True)
    print(f"[ETL] {len(df):,} reviews | {df['user_id'].nunique():,} users | "
          f"{df['product_id'].nunique():,} products")
    return df


def cosine_similarity_products(df: pd.DataFrame, top_k=5) -> list[tuple]:
    """
    Tính product similarity dựa trên user-rating vectors (item-CF style).
    Trả về list (product_a, product_b, score).
    """
    pivot = df.pivot_table(
        index="user_id", columns="product_id",
        values="rating", fill_value=0
    )
    # Normalize
    norms = np.linalg.norm(pivot.values, axis=0, keepdims=True)
    norms[norms == 0] = 1
    mat = pivot.values / norms

    sim = mat.T @ mat   # product × product similarity matrix
    products = pivot.columns.tolist()
    edges = []
    for i in range(len(products)):
        row = sim[i]
        row[i] = 0  # exclude self
        top = np.argsort(row)[::-1][:top_k]
        for j in top:
            score = float(row[j])
            if score > 0.1:
                edges.append((products[i], products[j], round(score, 4)))
    return edges

# ── Graph builder ─────────────────────────────────────────────────────────────

class GraphBuilder:
    def __init__(self):
        self.db    = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
        self.graph = self.db.select_graph(GRAPH_NAME)

    def clear(self):
        try:
            self.graph.delete()
        except Exception:
            pass
        self.graph = self.db.select_graph(GRAPH_NAME)
        print("[ETL] Graph cũ đã xóa.")

    def create_indexes(self):
        for label, prop in [
            ("User",     "user_id"),
            ("Product",  "product_id"),
            ("Category", "name"),
        ]:
            try:
                self.graph.query(
                    f"CREATE INDEX FOR (n:{label}) ON (n.{prop})"
                )
            except Exception:
                pass

    def load_users(self, df: pd.DataFrame):
        users = df[["user_id"]].drop_duplicates()
        batch = []
        for _, row in users.iterrows():
            batch.append({"user_id": row["user_id"]})
            if len(batch) == 500:
                self.graph.query(
                    "UNWIND $rows AS r MERGE (:User {user_id: r.user_id})",
                    {"rows": batch}
                )
                batch = []
        if batch:
            self.graph.query(
                "UNWIND $rows AS r MERGE (:User {user_id: r.user_id})",
                {"rows": batch}
            )
        print(f"[ETL] Tạo {users.shape[0]:,} User nodes")

    def load_products(self, df: pd.DataFrame):
        products = df[["product_id","product_name","category"]].drop_duplicates("product_id")
        batch = []
        for _, row in products.iterrows():
            batch.append({
                "product_id":   row["product_id"],
                "name":         row["product_name"],
                "category":     row["category"],
            })
            if len(batch) == 500:
                self.graph.query(
                    "UNWIND $rows AS r "
                    "MERGE (n:Product {product_id: r.product_id}) "
                    "SET n.name = r.name, n.category = r.category",
                    {"rows": batch}
                )
                batch = []
        if batch:
            self.graph.query(
                "UNWIND $rows AS r "
                "MERGE (n:Product {product_id: r.product_id}) "
                "SET n.name = r.name, n.category = r.category",
                {"rows": batch}
            )
        print(f"[ETL] Tạo {products.shape[0]:,} Product nodes")

    def load_categories(self, df: pd.DataFrame):
        cats = df["category"].dropna().unique().tolist()
        for cat in cats:
            self.graph.query(
                "MERGE (:Category {name: $name})", {"name": cat}
            )
        # BELONGS_TO edges: Product → Category
        self.graph.query("""
            MATCH (p:Product), (c:Category {name: p.category})
            MERGE (p)-[:BELONGS_TO]->(c)
        """)
        print(f"[ETL] Tạo {len(cats)} Category nodes + BELONGS_TO edges")

    def load_reviews(self, df: pd.DataFrame):
        """
        Tạo BOUGHT edges với weight = rating.
        Edge weight là cốt lõi cho weighted community detection.
        """
        batch = []
        for _, row in df.iterrows():
            batch.append({
                "user_id":    row["user_id"],
                "product_id": row["product_id"],
                "weight":     float(row["rating"]),
                "review":     row["review"],
            })
            if len(batch) == 500:
                self.graph.query("""
                    UNWIND $rows AS r
                    MATCH (u:User    {user_id:    r.user_id})
                    MATCH (p:Product {product_id: r.product_id})
                    MERGE (u)-[e:BOUGHT]->(p)
                    SET e.weight = r.weight, e.review = r.review
                """, {"rows": batch})
                batch = []
        if batch:
            self.graph.query("""
                UNWIND $rows AS r
                MATCH (u:User    {user_id:    r.user_id})
                MATCH (p:Product {product_id: r.product_id})
                MERGE (u)-[e:BOUGHT]->(p)
                SET e.weight = r.weight, e.review = r.review
            """, {"rows": batch})
        print(f"[ETL] Tạo {len(df):,} BOUGHT edges")

    def load_similar(self, sim_edges: list[tuple]):
        batch = []
        for a, b, score in sim_edges:
            batch.append({"a": a, "b": b, "score": score})
            if len(batch) == 300:
                self.graph.query("""
                    UNWIND $rows AS r
                    MATCH (a:Product {product_id: r.a})
                    MATCH (b:Product {product_id: r.b})
                    MERGE (a)-[e:SIMILAR]->(b)
                    SET e.weight = r.score
                """, {"rows": batch})
                batch = []
        if batch:
            self.graph.query("""
                UNWIND $rows AS r
                MATCH (a:Product {product_id: r.a})
                MATCH (b:Product {product_id: r.b})
                MERGE (a)-[e:SIMILAR]->(b)
                SET e.weight = r.score
            """, {"rows": batch})
        print(f"[ETL] Tạo {len(sim_edges):,} SIMILAR edges")

    def add_user_stats(self, df: pd.DataFrame):
        """Thêm aggregate stats vào User nodes (dùng cho clustering features)."""
        stats = df.groupby("user_id").agg(
            purchase_count=("product_id","count"),
            avg_rating=("rating","mean"),
            std_rating=("rating","std"),
        ).fillna(0).reset_index()
        batch = []
        for _, row in stats.iterrows():
            batch.append({
                "user_id":       row["user_id"],
                "purchase_count": int(row["purchase_count"]),
                "avg_rating":     round(float(row["avg_rating"]), 2),
                "std_rating":     round(float(row["std_rating"]),  2),
            })
            if len(batch) == 500:
                self.graph.query("""
                    UNWIND $rows AS r
                    MATCH (u:User {user_id: r.user_id})
                    SET u.purchase_count = r.purchase_count,
                        u.avg_rating     = r.avg_rating,
                        u.std_rating     = r.std_rating
                """, {"rows": batch})
                batch = []
        if batch:
            self.graph.query("""
                UNWIND $rows AS r
                MATCH (u:User {user_id: r.user_id})
                SET u.purchase_count = r.purchase_count,
                    u.avg_rating     = r.avg_rating,
                    u.std_rating     = r.std_rating
            """, {"rows": batch})
        print("[ETL] User stats đã cập nhật")

    def summary(self):
        res = self.graph.query("MATCH (n) RETURN labels(n)[0] AS lbl, count(n) AS cnt")
        print("\n[ETL] === Graph Summary ===")
        for row in res.result_set:
            print(f"  {row[0]}: {row[1]:,} nodes")
        res2 = self.graph.query("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS cnt")
        for row in res2.result_set:
            print(f"  [{row[0]}]: {row[1]:,} edges")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 55)
    print("  GraphRAG ETL — Amazon Reviews → FalkorDB")
    print("=" * 55)

    download_dataset()
    df = load_reviews()

    builder = GraphBuilder()
    builder.clear()
    builder.create_indexes()

    print("\n[ETL] Đang build graph...")
    builder.load_users(df)
    builder.load_products(df)
    builder.load_categories(df)
    builder.load_reviews(df)
    builder.add_user_stats(df)

    print("\n[ETL] Tính product similarity...")
    sim_edges = cosine_similarity_products(df, top_k=5)
    builder.load_similar(sim_edges)

    builder.summary()
    print("\n[ETL] Hoàn thành! Graph sẵn sàng cho clustering.")

    # Lưu processed df để dùng lại
    df.to_parquet(DATA_DIR / "reviews_processed.parquet", index=False)
    print(f"[ETL] Data đã lưu → {DATA_DIR / 'reviews_processed.parquet'}")


if __name__ == "__main__":
    run()