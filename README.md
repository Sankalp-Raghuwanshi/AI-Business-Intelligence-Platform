# 📊 AI Business Intelligence Platform

An AI-powered analytics tool that converts natural language questions 
into SQL queries, runs them on real e-commerce data, and explains 
results as actionable business insights.

## 🎯 What it does
- Type any business question in plain English
- AI generates SQL automatically using Groq LLaMA 3.3
- Results displayed as interactive Plotly charts
- AI explains findings as structured business insights (WHAT/WHY/ACTION)
- Executive Summary button generates a full C-suite business report

## 🛠️ Tech Stack
Python · Streamlit · Groq (LLaMA 3.3 70B) · SQLite · Plotly · Pandas

## 📦 Dataset
Built on the [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 115,607 orders across 9 relational tables

## 🚀 Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add Groq API key to `.env`: `GROQ_API_KEY=your_key`
4. Export data to SQLite: `python3 setup_db.py`
5. Run: `streamlit run app.py`

## 📈 Roadmap
- [x] Natural language to SQL conversion
- [x] Auto Plotly chart generation
- [x] KPI dashboard cards
- [x] Structured AI business insights
- [x] Executive summary report
- [ ] Delivery delay prediction ML model
- [ ] Review score prediction ML model
- [ ] Pre-built analysis buttons
- [ ] Streamlit Cloud deployment