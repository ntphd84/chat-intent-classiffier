# 🧠 Adaptive Chat Intent Classifier

This project is an end-to-end unsupervised pipeline for discovering user intent from chat data — ideal for chatbot analytics, customer support insights, or improving conversational AI models.

## 🚀 Features

- **Intent Clustering**: Automatically groups similar chat queries using sentence embeddings + clustering.
- **Embedding Selection**: Supports multiple sentence transformers.
- **Dashboard**: Visualizes clustered chats for easy exploration.

## 📂 Project Structure

```
chat-intent-classifier/
│
├── scripts/
│   ├── analyze_chats.py
│   └── adaptive_update_master_chat_class.py
│
├── dashboard/
│   └── view_dashboard.py
│
├── data/
│   └── (place your chat logs here)
│
├── output/
│   └── (clustered outputs go here)
│
├── requirements.txt
└── README.md
```

## 🔧 Requirements

- Python 3.9+
- `sentence-transformers`
- `scikit-learn`
- `hdbscan`
- `pandas`
- `streamlit`

Install them with:
```bash
pip install -r requirements.txt
```

## 📊 Run the Pipeline

1. Analyze and cluster chat data:
```bash
python scripts/analyze_chats.py
```

2. View clusters in a dashboard:
```bash
streamlit run dashboard/view_dashboard.py
```

3. Update intent mappings:
```bash
python scripts/adaptive_update_master_chat_class.py
```

## 🧩 Use Cases

- Improve chatbot FAQs
- Discover emerging customer concerns
- Speed up manual tagging
