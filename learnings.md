# Project Learnings — AI Business Intelligence Platform

## Day 1 — Project Setup & AI Integration (July 4, 2026)

### What I built
# Set up a Streamlit web application connected to a SQLite database
# Integrated Groq's LLaMA 3.3 model via API to convert natural language to SQL
# Built the core loop: User Question → AI generates SQL → Query runs → AI explains results

### Technical decisions and why
# *SQLite over PostgreSQL for the app* — PostgreSQL requires a running server, 
  SQLite is a single file that travels with the project. Better for portability 
  and deployment.
# **Groq over Gemini/OpenAI** — Gemini free tier hit quota limits quickly. 
  Groq offers genuinely free, fast inference with LLaMA models. 
  For a development project, this was the right tradeoff.
# **LLaMA 3.3 70B model** — The 8B model (llama3-8b-8192) was decommissioned 
  by Groq. Switched to llama-3.3-70b-versatile which is more capable anyway.
# **dotenv for API key management** — Never hardcode API keys. .env file keeps 
  secrets out of the codebase. Added .gitignore to prevent accidental commits.

### Problems I hit and how I solved them
# **Gemini quota exhausted** — Hit free tier limits during testing. 
  Switched to Groq which has more generous free limits.
# **Wrong app.py being served** — Had two app.py files (one in home folder, 
  one in project folder). Streamlit was running the wrong one. 
  Lesson: always check which file path is in the error traceback.
# **Model decommissioned error** — LLM providers deprecate models. 
  Always check the provider's deprecation docs if you get a model error.
# **SQLite export row limit** — DBeaver caps CSV exports by default. 
  Used Python + SQLAlchemy to export all 115,607 rows cleanly.

### Concepts I applied
# **Prompt Engineering** — The quality of AI output depends entirely on how 
  you phrase the prompt. Giving the AI a role ("You are a SQL expert"), 
  constraints ("use only columns in the schema"), and output format 
  ("return ONLY SQL in ```sql``` blocks") dramatically improves results.
# **Context Injection** — Instead of asking the AI to guess the database structure, I inject the schema directly into the prompt. This grounds 
  the AI in facts and reduces hallucinations.
# **Environment Variables** — API keys should never be in code. .env + python-dotenv is the standard pattern.

### What I want to improve next
# AI insight is sometimes repetitive — upgrade to Senior Business Analyst prompt
# No charts yet — add Plotly auto-charts based on result type
# No KPI summary at top — add Total Revenue, Orders, Avg Review Score cards
# Executive Summary button for full business overview
# Better error handling when AI generates invalid SQL

### Interview talking points from today
# "I chose SQLite for portability — the entire database is one file, which makes deployment and sharing trivial compared to a server-based database"
# "I hit quota limits with Gemini and switched to Groq  -this taught me to always have a fallback LLM provider and understand free tier constraints"
# "The most important lesson was prompt design — the same question to the AI gives completely different SQL quality depending on how specific the prompt is"