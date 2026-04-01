"""
graphrag.py — GraphRAG pipeline: FalkorDB context → LLM answer

Flow:
  1. Xác định community của user (từ graph)
  2. Cypher query lấy relevant products trong community
  3. Format context string
  4. GPT-4o generate answer có grounding
"""

import os
import json
from falkordb import FalkorDB
from openai import OpenAI
from clustering import get_community_products

FALKOR_HOST = os.getenv("FALKOR_HOST", "localhost")
FALKOR_PORT = int(os.getenv("FALKOR_PORT", 6379))
GRAPH_NAME  = "amazon_graph"
OPENAI_KEY  = os.getenv("OPENAI_API_KEY", "")

SYSTEM_PROMPT = """Bạn là một recommendation assistant thông minh.
Bạn được cung cấp danh sách sản phẩm được tổng hợp từ Knowledge Graph (FalkorDB)
dựa trên cộng đồng người dùng tương tự với user đang hỏi.

Hãy:
1. Recommend các sản phẩm phù hợp nhất với câu hỏi
2. Giải thích lý do dựa trên dữ liệu (rating, purchase count)
3. Trả lời bằng tiếng Việt, thân thiện và ngắn gọn
4. Chỉ recommend sản phẩm có trong context được cung cấp

Nếu không có sản phẩm phù hợp, hãy nói thật và gợi ý cách tìm kiếm khác."""


class GraphRAGPipeline:
    def __init__(self):
        self.db     = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
        self.graph  = self.db.select_graph(GRAPH_NAME)
        self.client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

    # ── 1. Tìm community của user ─────────────────────────────────────────────

    def get_user_community(self, user_id: str) -> int | None:
        res = self.graph.query(
            "MATCH (u:User {user_id: $uid}) RETURN u.community_id AS cid",
            {"uid": user_id}
        )
        if res.result_set and res.result_set[0][0] is not None:
            return int(res.result_set[0][0])
        return None

    # ── 2. Lấy user purchase history ─────────────────────────────────────────

    def get_user_history(self, user_id: str) -> list[dict]:
        res = self.graph.query("""
            MATCH (u:User {user_id: $uid})-[b:BOUGHT]->(p:Product)
            RETURN p.product_id AS pid,
                   p.name       AS name,
                   b.weight     AS rating
            ORDER BY b.weight DESC
            LIMIT 10
        """, {"uid": user_id})
        return [
            {"product_id": r[0], "name": r[1], "rating": float(r[2])}
            for r in res.result_set
        ]

    # ── 3. Lấy candidate products từ community ────────────────────────────────

    def get_candidates(self, user_id: str, community_id: int,
                       top_k: int = 15) -> list[dict]:
        """
        Lấy products phổ biến trong community,
        loại bỏ những gì user đã mua.
        """
        # Products user đã mua
        history_res = self.graph.query("""
            MATCH (u:User {user_id: $uid})-[:BOUGHT]->(p:Product)
            RETURN p.product_id AS pid
        """, {"uid": user_id})
        bought_ids = {r[0] for r in history_res.result_set}

        candidates = get_community_products(community_id, top_k=top_k * 2)
        # Filter out already bought
        candidates = [c for c in candidates if c["product_id"] not in bought_ids]
        return candidates[:top_k]

    # ── 4. Lấy similar products (dùng SIMILAR edges) ─────────────────────────

    def get_similar_products(self, product_id: str, top_k: int = 5) -> list[dict]:
        res = self.graph.query("""
            MATCH (p:Product {product_id: $pid})-[s:SIMILAR]->(q:Product)
            RETURN q.product_id AS qid,
                   q.name       AS name,
                   q.category   AS category,
                   s.weight     AS similarity
            ORDER BY s.weight DESC
            LIMIT $k
        """, {"pid": product_id, "k": top_k})
        return [
            {"product_id": r[0], "name": r[1],
             "category": r[2], "similarity": round(float(r[3]), 3)}
            for r in res.result_set
        ]

    # ── 5. Format context cho LLM ─────────────────────────────────────────────

    def format_context(self, candidates: list[dict],
                       history: list[dict],
                       community_id: int) -> str:
        ctx = [f"Community ID: {community_id}"]
        ctx.append(f"Số sản phẩm trong context: {len(candidates)}\n")

        if history:
            ctx.append("Lịch sử mua của user (rating cao nhất):")
            for h in history[:5]:
                ctx.append(f"  - {h['name'][:60]} (rating: {h['rating']})")
            ctx.append("")

        ctx.append("Sản phẩm phổ biến trong community (sorted by score):")
        for i, c in enumerate(candidates, 1):
            ctx.append(
                f"  {i}. [{c['category']}] {c['name'][:70]}\n"
                f"     → {c['purchase_count']} lượt mua | "
                f"avg rating: {c['avg_rating']} | score: {c['score']}"
            )
        return "\n".join(ctx)

    # ── 6. Query LLM ─────────────────────────────────────────────────────────

    def ask_llm(self, question: str, context: str) -> str:
        if not self.client:
            return self._mock_answer(question, context)

        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content":
                 f"Context từ Knowledge Graph:\n{context}\n\n"
                 f"Câu hỏi: {question}"}
            ],
            max_tokens=600,
            temperature=0.3,
        )
        return resp.choices[0].message.content

    def _mock_answer(self, question: str, context: str) -> str:
        """Fallback khi không có OpenAI key — parse context và trả về top 3."""
        lines = [l for l in context.split("\n") if l.strip().startswith(("1.", "2.", "3."))]
        products = "\n".join(lines[:3]) if lines else "(Không tìm thấy sản phẩm)"
        return (
            f"[Demo mode — không có OpenAI key]\n\n"
            f"Dựa trên Knowledge Graph, top sản phẩm gợi ý:\n{products}\n\n"
            f"Câu hỏi của bạn: {question}"
        )

    # ── 7. Main recommend function ─────────────────────────────────────────────

    def recommend(self, user_id: str, question: str) -> dict:
        """
        Full GraphRAG pipeline.
        Returns: answer, context, community_id, candidates
        """
        community_id = self.get_user_community(user_id)
        history      = self.get_user_history(user_id)

        if community_id is None:
            return {
                "answer":       "User chưa có dữ liệu trong graph. Hãy chạy ETL trước.",
                "context":      "",
                "community_id": None,
                "candidates":   [],
                "history":      [],
            }

        candidates = self.get_candidates(user_id, community_id)
        context    = self.format_context(candidates, history, community_id)
        answer     = self.ask_llm(question, context)

        return {
            "answer":       answer,
            "context":      context,
            "community_id": community_id,
            "candidates":   candidates,
            "history":      history,
        }

    # ── 8. Free-form graph Q&A ────────────────────────────────────────────────

    def graph_qa(self, question: str) -> str:
        """
        Q&A không cần user_id — query graph tổng quan.
        Ví dụ: 'Sản phẩm nào được mua nhiều nhất?'
        """
        res = self.graph.query("""
            MATCH (u:User)-[b:BOUGHT]->(p:Product)
            WITH p, count(b) AS cnt, avg(b.weight) AS avg_r
            WHERE cnt >= 10
            RETURN p.name AS name, p.category AS cat,
                   cnt, round(avg_r * 100) / 100 AS avg_r
            ORDER BY cnt DESC LIMIT 10
        """)
        rows = res.result_set
        context = "Top sản phẩm được mua nhiều nhất:\n"
        for i, r in enumerate(rows, 1):
            context += f"  {i}. [{r[1]}] {r[0]} — {r[2]} lượt mua, rating: {r[3]}\n"

        return self.ask_llm(question, context)


# ── Traditional CF baseline (để so sánh) ─────────────────────────────────────

class CollaborativeFilterBaseline:
    """
    SVD-based Collaborative Filtering (baseline để compare với GraphRAG).
    """
    def __init__(self, df: "pd.DataFrame"):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        self.df = df
        self.pivot = df.pivot_table(
            index="user_id", columns="product_id",
            values="rating", fill_value=0
        )
        svd = TruncatedSVD(n_components=50, random_state=42)
        U   = svd.fit_transform(self.pivot.values)
        Vt  = svd.components_
        self.U        = normalize(U)
        self.Vt       = Vt
        self.users    = self.pivot.index.tolist()
        self.products = self.pivot.columns.tolist()
        self._pred    = self.U @ self.Vt

    def recommend(self, user_id: str, top_k: int = 10) -> list[str]:
        if user_id not in self.users:
            return []
        idx     = self.users.index(user_id)
        scores  = self._pred[idx]
        # Exclude already bought
        bought_mask = self.pivot.iloc[idx].values > 0
        scores[bought_mask] = -999
        top_idx = scores.argsort()[::-1][:top_k]
        return [self.products[i] for i in top_idx]


if __name__ == "__main__":
    import sys
    uid = sys.argv[1] if len(sys.argv) > 1 else "A1RSDE90N6RSZF"
    q   = sys.argv[2] if len(sys.argv) > 2 else "Gợi ý tai nghe tốt cho tôi"

    rag    = GraphRAGPipeline()
    result = rag.recommend(uid, q)
    print("\n=== GraphRAG Answer ===")
    print(result["answer"])
    print(f"\n[Community: {result['community_id']} | "
          f"Candidates: {len(result['candidates'])}]")
