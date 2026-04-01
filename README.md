# GraphRAG Knowledge Graph với FalkorDB
**Data Mining Miniproject** — Graph clustering + LLM-powered recommendation

## Cấu trúc dự án
```
graphrag_project/
├── data/                  # Dataset (Amazon Reviews subset)
├── notebooks/
│   └── demo.ipynb         # Jupyter notebook demo từng bước
├── src/
│   ├── etl.py             # Load & build graph vào FalkorDB
│   ├── clustering.py      # Weighted Louvain community detection
│   ├── graphrag.py        # GraphRAG query pipeline
│   └── evaluation.py      # Precision@K, Recall@K, NDCG
├── streamlit_app/
│   └── app.py             # Streamlit web UI
├── docker-compose.yml     # FalkorDB container
├── requirements.txt
└── README.md
```

## Cài đặt

### 1. Khởi động FalkorDB
```bash
docker-compose up -d
```

### 2. Cài dependencies
```bash
pip install -r requirements.txt
```

### 3. Cấu hình API key
```bash
export OPENAI_API_KEY="sk-..."
```

### 4. Chạy ETL (load data vào graph)
```bash
python src/etl.py
```

### 5. Chạy Streamlit app
```bash
streamlit run streamlit_app/app.py
```

### 6. Hoặc mở Jupyter Notebook
```bash
jupyter notebook notebooks/demo.ipynb
```

## Dataset
Dùng Amazon Product Reviews (Electronics subset, ~50K reviews).
Download tự động khi chạy `etl.py` lần đầu.

## Tech Stack
- **FalkorDB** — Graph database (GraphBLAS-based)
- **Python falkordb** — Python client
- **NetworkX + python-louvain** — Community detection
- **OpenAI GPT-4o** — LLM generation
- **Streamlit** — Web UI
- **pyvis** — Graph visualization
