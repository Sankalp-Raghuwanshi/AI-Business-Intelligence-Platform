# 🧠 Insight Pilot AI
### AI-Powered Business Intelligence Platform

🌐 **Live Demo:** https://insight-pilotai.streamlit.app/
📁 **Dataset:** [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 115,607 orders

---

## What is Insight Pilot AI?

Insight Pilot AI is an end-to-end business intelligence platform that lets you 
analyze 115,000+ real e-commerce orders using plain English — no SQL knowledge required.
Ask a question, get SQL, get results, get insights, get charts. All automatically.

---

## ✨ Features

### 💬 Natural Language to SQL
Type any business question → AI generates SQL → query runs → results displayed
with interactive Plotly charts and structured WHAT/WHY/ACTION business insights.

### 📊 Live KPI Dashboard
Four real-time metric cards: Total Revenue (R$19.9M), Total Orders (96,514),
Average Review Score (4.03/5), Average Delivery Days (12.0).

### 🎯 Executive Summary
One click generates a full C-suite business report — AI automatically runs 8 KPI
queries and synthesizes a structured Performance/Strengths/Risks/Recommendations report.

### 🔬 RAG Deep Analysis
Ask broad analytical questions like "Why are customers in Amazonas unhappy?" —
AI decomposes into sub-queries, retrieves data from multiple SQL executions,
and synthesizes a comprehensive cross-dataset answer grounded in real data.

### 🤖 ML Predictions (3 Models)
- **Delivery Delay Predictor** — XGBoost classifier, 64% recall on imbalanced 7.5% minority class
- **Review Score Predictor** — RandomForest multiclass, 54% recall on low-satisfaction orders  
- **Customer Segmentation** — KMeans RFM analysis, 4 behavioral clusters identified

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit |
| AI/LLM | Groq API (LLaMA 3.3 70B) |
| ML Models | XGBoost, Scikit-learn (RandomForest, KMeans) |
| Database | SQLite (exported from PostgreSQL) |
| Visualization | Plotly Express |
| Data Processing | Pandas, NumPy |
| Deployment | Streamlit Cloud |

---

## 🔑 Key Findings from the Data

- São Paulo drives **37%+ of total revenue** — highest geographic concentration
- Delivery satisfaction **collapses below 2.0** beyond 40 days
- **7.5% of orders are delayed** — systemic across all product categories
- **19 VIP customers** averaging R$26,000 per order — likely B2B segment
- Same-state delivery is the **3rd strongest delay predictor** (discovered via ML, not EDA)

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/Sankalp-Raghuwanshi/AI-Business-Intelligence-Platform.git
cd AI-Business-Intelligence-Platform

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Set up the database (requires PostgreSQL with Olist data)
python3 setup_db.py

# Train ML models
python3 train_models.py

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure
AI-Business-Intelligence-Platform/
├── app.py              # Main Streamlit app (dashboard + RAG + NL-to-SQL)
├── predict.py          # ML prediction pages (3 models)
├── train_models.py     # Model training script
├── setup_db.py         # Database export from PostgreSQL to SQLite
├── models/             # Saved trained models (.pkl files)
├── data/               # SQLite database
├── requirements.txt    # Python dependencies
├── learnings.md        # Project learnings and interview prep
└── README.md           # This file

---

## 📈 ML Model Performance

| Model | Algorithm | Key Metric |
|-------|-----------|------------|
| Delivery Delay Predictor | XGBoost | 64% recall on delayed orders |
| Review Score Predictor | RandomForest | 54% recall on low reviews |
| Customer Segmentation | KMeans (k=4) | 4 RFM behavioral clusters |

---

## 🧠 GenAI Concepts Used

- **Prompt Engineering** — Role prompting, structured output, constraint injection
- **Context Injection** — Schema and SQL results injected as facts into prompts
- **RAG over Structured Data** — SQL results as retrieval source instead of documents
- **Prompt Chaining** — 3-step chain: decompose → retrieve → synthesize
- **Grounded Generation** — AI answers from retrieved data, not training memory

---

## 🔗 Related Project

📊 [Olist EDA — SQL Exploratory Data Analysis](https://github.com/Sankalp-Raghuwanshi/olist-ecommerce-eda)
The foundational EDA that uncovered the business insights this platform is built on.

---

*Built as part of placement preparation for Data Science and Business Analytics roles.*