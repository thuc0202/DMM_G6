"""
app.py — Streamlit web UI cho GraphRAG demo

Tabs:
  1. Graph Overview  — stats + node/edge counts
  2. Recommend       — GraphRAG Q&A cho từng user
  3. Community       — visualize clusters
  4. Evaluation      — kết quả so sánh GraphRAG vs CF
  5. Cypher Explorer — chạy Cypher query tự do
"""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from falkordb import FalkorDB

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRAG — FalkorDB Demo",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Connection ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_graph():
    host = os.getenv("FALKOR_HOST", "localhost")
    port = int(os.getenv("FALKOR_PORT", 6379))
    db   = FalkorDB(host=host, port=port)
    return db.select_graph("amazon_graph")


@st.cache_data(ttl=300)
def get_graph_stats() -> dict:
    g = get_graph()
    node_res = g.query("MATCH (n) RETURN labels(n)[0] AS lbl, count(n) AS cnt")
    edge_res = g.query("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS cnt")
    nodes = {r[0]: int(r[1]) for r in node_res.result_set}
    edges = {r[0]: int(r[1]) for r in edge_res.result_set}
    return {"nodes": nodes, "edges": edges}


@st.cache_data(ttl=300)
def get_sample_users(n=200) -> list[str]:
    g   = get_graph()
    res = g.query(
        "MATCH (u:User) WHERE u.community_id IS NOT NULL "
        "RETURN u.user_id LIMIT $n", {"n": n}
    )
    return [r[0] for r in res.result_set]


@st.cache_data(ttl=300)
def get_community_overview() -> pd.DataFrame:
    g   = get_graph()
    res = g.query("""
        MATCH (u:User) WHERE u.community_id IS NOT NULL
        WITH u.community_id AS cid, count(u) AS user_count,
             avg(u.avg_rating) AS avg_rating,
             avg(u.purchase_count) AS avg_purchases
        RETURN cid, user_count, round(avg_rating*100)/100,
               round(avg_purchases*10)/10
        ORDER BY user_count DESC LIMIT 20
    """)
    return pd.DataFrame(
        res.result_set,
        columns=["Community", "Users", "Avg Rating", "Avg Purchases"]
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔗 GraphRAG Demo")
    st.markdown("**Data Mining Miniproject**")
    st.caption("FalkorDB · GraphBLAS · GPT-4o")

    st.divider()
    try:
        stats = get_graph_stats()
        st.markdown("**Graph Stats**")
        for lbl, cnt in stats["nodes"].items():
            st.metric(lbl, f"{cnt:,}")
        st.divider()
        for t, cnt in stats["edges"].items():
            st.caption(f"[{t}]: {cnt:,}")
    except Exception as e:
        st.error(f"Không kết nối được FalkorDB: {e}")

    st.divider()
    openai_key = st.text_input(
        "OpenAI API Key", type="password",
        value=os.getenv("OPENAI_API_KEY",""),
        help="Để trống = demo mode (mock answer)"
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Graph Overview",
    "💬 Recommend",
    "🏘 Communities",
    "📈 Evaluation",
    "🔍 Cypher Explorer",
])


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 1 — Graph Overview
# ════════════════════════════════════════════════════════════════════════════ #
with tab1:
    st.header("Knowledge Graph Overview")
    st.caption("Amazon Electronics Reviews → FalkorDB Heterogeneous Graph")

    try:
        stats = get_graph_stats()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Users",    f"{stats['nodes'].get('User', 0):,}")
        col2.metric("Products", f"{stats['nodes'].get('Product', 0):,}")
        col3.metric("Categories",f"{stats['nodes'].get('Category', 0):,}")
        col4.metric("BOUGHT edges",f"{stats['edges'].get('BOUGHT', 0):,}")

        st.divider()
        st.subheader("Graph Schema")
        st.code("""
(User)-[:BOUGHT   {weight: rating}      ]->(Product)
(User)-[:REVIEWED {weight: rating}      ]->(Product)
(Product)-[:BELONGS_TO                  ]->(Category)
(Product)-[:SIMILAR {weight: cos_sim}   ]->(Product)

Node properties:
  User    : user_id, purchase_count, avg_rating, community_id
  Product : product_id, name, category
  Category: name
        """, language="text")

        st.subheader("Proposed Pipeline")
        cols = st.columns(3)
        with cols[0]:
            st.info("**Step 1 — ETL**\n\nAmazon CSV → FalkorDB graph với heterogeneous nodes và weighted edges")
        with cols[1]:
            st.info("**Step 2 — Clustering ★**\n\nWeighted Louvain: edge weight = rating × purchase_count")
        with cols[2]:
            st.info("**Step 3 — GraphRAG**\n\nCypher context retrieval → GPT-4o → grounded answer")

    except Exception as e:
        st.error(str(e))


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 2 — Recommend
# ════════════════════════════════════════════════════════════════════════════ #
with tab2:
    st.header("GraphRAG Recommendation")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        sample_users = get_sample_users()
        user_id = st.selectbox("Chọn User ID", options=sample_users)

        question = st.text_area(
            "Câu hỏi",
            value="Gợi ý cho tôi tai nghe tốt trong tầm giá thấp.",
            height=100
        )
        run_btn = st.button("🔍 Recommend", type="primary", use_container_width=True)

    with col_right:
        if run_btn:
            with st.spinner("Đang query Knowledge Graph..."):
                try:
                    from graphrag import GraphRAGPipeline
                    rag    = GraphRAGPipeline()
                    result = rag.recommend(user_id, question)

                    st.success(f"Community: **{result['community_id']}**")

                    st.subheader("💬 GraphRAG Answer")
                    st.markdown(result["answer"])

                    st.divider()

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.subheader("Lịch sử mua")
                        if result["history"]:
                            hist_df = pd.DataFrame(result["history"])
                            st.dataframe(hist_df[["name","rating"]], hide_index=True)
                        else:
                            st.caption("Chưa có lịch sử.")

                    with col_b:
                        st.subheader("Top candidates từ Graph")
                        if result["candidates"]:
                            cand_df = pd.DataFrame(result["candidates"])
                            st.dataframe(
                                cand_df[["name","category","avg_rating","purchase_count","score"]],
                                hide_index=True
                            )

                    with st.expander("📋 Raw context gửi cho LLM"):
                        st.text(result["context"])

                except Exception as e:
                    st.error(f"Lỗi: {e}")
                    st.caption("Đảm bảo FalkorDB đang chạy và ETL đã được thực thi.")


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 3 — Communities
# ════════════════════════════════════════════════════════════════════════════ #
with tab3:
    st.header("Community Detection Results")
    st.caption("Weighted Louvain — edge weight = rating × purchase_count")

    try:
        comm_df = get_community_overview()
        st.dataframe(comm_df, hide_index=True, use_container_width=True)

        st.divider()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].bar(comm_df["Community"].astype(str)[:10],
                    comm_df["Users"][:10], color="#534AB7", alpha=0.85)
        axes[0].set_title("Users per community (top 10)")
        axes[0].set_xlabel("Community ID")
        axes[0].set_ylabel("User count")
        axes[0].spines[["top","right"]].set_visible(False)

        axes[1].bar(comm_df["Community"].astype(str)[:10],
                    comm_df["Avg Rating"][:10], color="#1D9E75", alpha=0.85)
        axes[1].set_title("Avg rating per community (top 10)")
        axes[1].set_xlabel("Community ID")
        axes[1].set_ylim(0, 5)
        axes[1].axhline(y=comm_df["Avg Rating"].mean(), color="#E24B4A",
                        linestyle="--", linewidth=1, label="Overall avg")
        axes[1].legend()
        axes[1].spines[["top","right"]].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Tại sao Weighted Louvain tốt hơn K-means?")
        cols = st.columns(2)
        with cols[0]:
            st.info(
                "**Louvain (proposed)**\n\n"
                "- Khai thác graph structure thực sự\n"
                "- Edge weight = rating × count (encode satisfaction)\n"
                "- Không cần chỉ định K trước\n"
                "- Modularity cao hơn → cluster chất lượng hơn"
            )
        with cols[1]:
            st.warning(
                "**K-means (baseline)**\n\n"
                "- Chỉ dùng user feature vectors\n"
                "- Bỏ qua quan hệ giữa users\n"
                "- K phải chọn thủ công\n"
                "- Không phản ánh community structure"
            )
    except Exception as e:
        st.error(str(e))


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 4 — Evaluation
# ════════════════════════════════════════════════════════════════════════════ #
with tab4:
    st.header("Evaluation: GraphRAG vs SVD Baseline")
    st.caption("Protocol: Leave-one-out | Metrics: P@K, Recall@K, NDCG@K, HR@K")

    data_path = Path(__file__).parent.parent / "data" / "reviews_processed.parquet"
    eval_img  = Path(__file__).parent.parent / "data" / "eval_results.png"

    col_run, col_info = st.columns([1, 2])
    with col_run:
        sample_n = st.slider("Số users đánh giá", 100, 1000, 300, step=100)
        run_eval = st.button("▶ Chạy Evaluation", type="primary", use_container_width=True)

    with col_info:
        st.markdown("""
        **Cách tính:**
        - Với mỗi user, ẩn sản phẩm cuối cùng làm ground-truth
        - Recommend top-K từ cả 2 methods
        - Tính Precision@K, NDCG@K, Hit Rate@K
        """)

    if run_eval:
        if not data_path.exists():
            st.error("Chưa có data. Hãy chạy `python src/etl.py` trước.")
        else:
            with st.spinner("Đang evaluate... (~1-2 phút)"):
                try:
                    from evaluation import Evaluator
                    df = pd.read_parquet(data_path)

                    g   = get_graph()
                    res = g.query(
                        "MATCH (u:User) WHERE u.community_id IS NOT NULL "
                        "RETURN u.user_id, u.community_id"
                    )
                    partition = {r[0]: int(r[1]) for r in res.result_set}

                    ev      = Evaluator(df, partition)
                    summary = ev.evaluate(k_values=[5, 10, 20], sample_n=sample_n)

                    st.subheader("Kết quả")
                    rows = []
                    for method, metrics in summary.items():
                        for m, v in metrics.items():
                            rows.append({"Method": method, "Metric": m, "Value": v})
                    res_df   = pd.DataFrame(rows)
                    pivot_df = res_df.pivot(index="Metric", columns="Method", values="Value")
                    st.dataframe(pivot_df.style.highlight_max(axis=1, color="#E1F5EE"),
                                 use_container_width=True)

                    fig = ev.plot_results(summary, save_path=str(eval_img))
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Lỗi evaluation: {e}")

    elif eval_img.exists():
        st.image(str(eval_img), caption="Kết quả evaluation gần nhất")


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 5 — Cypher Explorer
# ════════════════════════════════════════════════════════════════════════════ #
with tab5:
    st.header("Cypher Query Explorer")
    st.caption("Chạy Cypher query trực tiếp trên FalkorDB")

    examples = {
        "Top sản phẩm bán chạy": (
            "MATCH (u:User)-[b:BOUGHT]->(p:Product)\n"
            "WITH p, count(b) AS cnt, avg(b.weight) AS avg_r\n"
            "RETURN p.name, p.category, cnt, round(avg_r*100)/100 AS avg_r\n"
            "ORDER BY cnt DESC LIMIT 10"
        ),
        "User với nhiều community neighbors": (
            "MATCH (u:User)-[:BOUGHT]->(p:Product)<-[:BOUGHT]-(v:User)\n"
            "WHERE u.community_id = v.community_id AND u.user_id <> v.user_id\n"
            "RETURN u.user_id, count(DISTINCT v) AS neighbors\n"
            "ORDER BY neighbors DESC LIMIT 10"
        ),
        "Community size distribution": (
            "MATCH (u:User) WHERE u.community_id IS NOT NULL\n"
            "RETURN u.community_id AS cid, count(u) AS size\n"
            "ORDER BY size DESC LIMIT 15"
        ),
        "Similar products network": (
            "MATCH (p:Product)-[s:SIMILAR]->(q:Product)\n"
            "RETURN p.name, q.name, s.weight\n"
            "ORDER BY s.weight DESC LIMIT 10"
        ),
    }

    selected = st.selectbox("Ví dụ câu query", ["(Tự nhập)"] + list(examples.keys()))
    default_q = examples.get(selected, "MATCH (n) RETURN labels(n)[0], count(n) LIMIT 5")
    query = st.text_area("Cypher Query", value=default_q, height=130)

    if st.button("▶ Chạy", type="primary"):
        try:
            g   = get_graph()
            res = g.query(query)
            if res.result_set:
                cols_names = [h for h in res.header] if res.header else None
                df_res = pd.DataFrame(res.result_set, columns=cols_names)
                st.dataframe(df_res, use_container_width=True)
                st.caption(f"{len(df_res)} rows returned")
            else:
                st.success("Query thực thi thành công (không có kết quả trả về).")
        except Exception as e:
            st.error(f"Query error: {e}")
