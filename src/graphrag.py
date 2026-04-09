"""
graphrag.py — GraphRAG pipeline: FalkorDB context → Groq LLM answer
"""

import os
from falkordb import FalkorDB
from groq import Groq
from clustering import get_community_products

FALKOR_HOST = os.getenv("FALKOR_HOST", "localhost")
FALKOR_PORT = int(os.getenv("FALKOR_PORT", 6379))
GRAPH_NAME  = "amazon_graph"
GROQ_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL  = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """Ban la mot recommendation assistant thong minh.
Ban duoc cung cap danh sach san pham tu Knowledge Graph (FalkorDB)
dua tren cong dong nguoi dung tuong tu voi user dang hoi.

Hay:
1. Recommend cac san pham phu hop nhat voi cau hoi
2. Giai thich ly do dua tren du lieu (rating, purchase count)
3. Tra loi bang tieng Viet, than thien va ngan gon
4. Chi recommend san pham co trong context duoc cung cap"""


class GraphRAGPipeline:
    def __init__(self):
        self.db     = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
        self.graph  = self.db.select_graph(GRAPH_NAME)
        self.client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

    def get_user_community(self, user_id: str):
        res = self.graph.query(
            "MATCH (u:User {user_id: $uid}) RETURN u.community_id AS cid",
            {"uid": user_id}
        )
        if res.result_set and res.result_set[0][0] is not None:
            return int(res.result_set[0][0])
        return None

    def get_user_history(self, user_id: str):
        res = self.graph.query("""
            MATCH (u:User {user_id: $uid})-[b:BOUGHT]->(p:Product)
            RETURN p.product_id AS pid, p.name AS name, b.weight AS rating
            ORDER BY b.weight DESC LIMIT 10
        """, {"uid": user_id})
        return [{"product_id": r[0], "name": r[1], "rating": float(r[2])} for r in res.result_set]

    def get_candidates(self, user_id: str, community_id: int, top_k: int = 15):
        history_res = self.graph.query(
            "MATCH (u:User {user_id: $uid})-[:BOUGHT]->(p:Product) RETURN p.product_id",
            {"uid": user_id}
        )
        bought_ids  = {r[0] for r in history_res.result_set}
        candidates  = get_community_products(community_id, top_k=top_k * 2)
        return [c for c in candidates if c["product_id"] not in bought_ids][:top_k]

    def get_similar_products(self, product_id: str, top_k: int = 5):
        res = self.graph.query("""
            MATCH (p:Product {product_id: $pid})-[s:SIMILAR]->(q:Product)
            RETURN q.product_id, q.name, q.category, s.weight
            ORDER BY s.weight DESC LIMIT $k
        """, {"pid": product_id, "k": top_k})
        return [{"product_id": r[0], "name": r[1], "category": r[2], "similarity": round(float(r[3]), 3)} for r in res.result_set]

    def format_context(self, candidates, history, community_id: int) -> str:
        ctx = [f"Community ID: {community_id}", f"So san pham trong context: {len(candidates)}\n"]
        if history:
            ctx.append("Lich su mua cua user:")
            for h in history[:5]:
                ctx.append(f"  - {h['name'][:60]} (rating: {h['rating']})")
            ctx.append("")
        ctx.append("San pham pho bien trong community:")
        for i, c in enumerate(candidates, 1):
            ctx.append(f"  {i}. [{c['category']}] {c['name'][:70]}")
            ctx.append(f"     -> {c['purchase_count']} luot mua | avg rating: {c['avg_rating']} | score: {c['score']}")
        return "\n".join(ctx)

    def ask_llm(self, question: str, context: str) -> str:
        if not self.client:
            return self._mock_answer(question, context)
        resp = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Context tu Knowledge Graph:\n{context}\n\nCau hoi: {question}"}
            ],
            max_tokens=600,
            temperature=0.3,
        )
        return resp.choices[0].message.content

    def _mock_answer(self, question: str, context: str) -> str:
        lines    = [l for l in context.split("\n") if l.strip().startswith(("1.", "2.", "3."))]
        products = "\n".join(lines[:3]) if lines else "(Khong tim thay san pham)"
        return f"[Demo mode - khong co GROQ_API_KEY]\n\nTop goi y tu graph:\n{products}\n\nCau hoi: {question}"

    def recommend(self, user_id: str, question: str) -> dict:
        community_id = self.get_user_community(user_id)
        history      = self.get_user_history(user_id)
        if community_id is None:
            return {"answer": "User chua co du lieu. Chay ETL truoc.", "context": "", "community_id": None, "candidates": [], "history": []}
        candidates = self.get_candidates(user_id, community_id)
        context    = self.format_context(candidates, history, community_id)
        answer     = self.ask_llm(question, context)
        return {"answer": answer, "context": context, "community_id": community_id, "candidates": candidates, "history": history}

    def graph_qa(self, question: str) -> str:
        res = self.graph.query("""
            MATCH (u:User)-[b:BOUGHT]->(p:Product)
            WITH p, count(b) AS cnt, avg(b.weight) AS avg_r
            WHERE cnt >= 5
            RETURN p.name, p.category, cnt, round(avg_r*100)/100
            ORDER BY cnt DESC LIMIT 10
        """)
        context = "Top san pham ban chay:\n"
        for i, r in enumerate(res.result_set, 1):
            context += f"  {i}. [{r[1]}] {r[0]} - {r[2]} luot mua, rating: {r[3]}\n"
        return self.ask_llm(question, context)


class CollaborativeFilterBaseline:
    def __init__(self, df):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        self.pivot    = df.pivot_table(index="user_id", columns="product_id", values="rating", fill_value=0)
        svd           = TruncatedSVD(n_components=min(50, self.pivot.shape[1]-1), random_state=42)
        U             = normalize(svd.fit_transform(self.pivot.values))
        self.users    = self.pivot.index.tolist()
        self.products = self.pivot.columns.tolist()
        self._pred    = U @ svd.components_

    def recommend(self, user_id: str, top_k: int = 10):
        if user_id not in self.users:
            return []
        idx    = self.users.index(user_id)
        scores = self._pred[idx].copy()
        scores[self.pivot.iloc[idx].values > 0] = -999
        return [self.products[i] for i in scores.argsort()[::-1][:top_k]]


if __name__ == "__main__":
    import sys
    uid    = sys.argv[1] if len(sys.argv) > 1 else "A1RSDE90N6RSZF"
    q      = sys.argv[2] if len(sys.argv) > 2 else "Goi y tai nghe tot cho toi"
    rag    = GraphRAGPipeline()
    result = rag.recommend(uid, q)
    print("\n=== GraphRAG Answer ===")
    print(result["answer"])
    print(f"\n[Community: {result['community_id']} | Candidates: {len(result['candidates'])}]")