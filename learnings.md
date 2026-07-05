# Project Learnings — AI Business Intelligence Platform

## Overview
An AI-powered business intelligence tool that converts natural language 
questions into SQL queries, runs them on real e-commerce data, and 
explains results as business insights. Built on the Olist Brazilian 
E-Commerce Dataset (115,607 orders across 9 relational tables).

---

## Phase 1: Project Setup & Database

### What I did
- Exported master_orders table from PostgreSQL to SQLite using Python/SQLAlchemy
- SQLite chosen over PostgreSQL for portability — entire database is one file
- Set up Streamlit as the web framework
- Configured environment variables using python-dotenv for API key security

### Why SQLite over PostgreSQL for the app
PostgreSQL requires a running server process. SQLite is a single .db file 
that travels with the project — easier to deploy, share, and demo. 
Since the data doesn't change in real time, SQLite is the right tradeoff.

### Problem: DBeaver row export limit
DBeaver caps exports at 200 rows by default. Discovered this when SQLite 
only had 800 rows instead of 115,607. Fixed by exporting via Python:
pandas.read_sql() + DataFrame.to_sql() bypasses any GUI row limits.

### Concept learned: Environment Variables
Never hardcode API keys in source code. .env file + python-dotenv loads 
keys as environment variables. .gitignore prevents .env from being 
committed to GitHub. Standard industry practice.

---

## Phase 2: AI Integration (NL to SQL)

### What I built
The core loop: User types question → AI generates SQL → SQL runs on 
database → Results displayed → AI explains results as business insight.

### Why Groq over Gemini/OpenAI
- Gemini free tier hit quota within minutes of testing
- OpenAI requires credit card
- Groq offers free, fast inference with LLaMA 3.3 70B
- For development, Groq is the right choice

### Problem: Wrong model name
llama3-8b-8192 was decommissioned by Groq. Error: "model_decommissioned". 
Fix: check provider's deprecation docs, switch to llama-3.3-70b-versatile. 
Lesson: always handle model deprecation in production AI apps.

### Problem: Two app.py files
Streamlit was running /Users/sankalpsingh/app.py instead of the project 
file. Traceback showed the wrong path. Fix: delete the rogue file, always 
run streamlit from the project directory with explicit cd first.

### Concept learned: Prompt Engineering
The same question to an LLM gives completely different output quality 
depending on prompt design. Key techniques used:

1. Role assignment: "You are a SQL expert" — sets context and expertise level
2. Schema injection: providing the full database schema as context so AI 
   knows exactly what columns and tables exist
3. Output formatting: "Return ONLY SQL wrapped in ```sql``` blocks" — 
   makes parsing the response reliable
4. Constraints: "Use only columns in the schema, table name is master_orders" 
   — prevents hallucinated column names

### Concept learned: Context Injection
Instead of asking the AI to guess the database structure, inject the schema 
directly into every prompt. This grounds the AI in facts and eliminates 
hallucinations about column names or table structure. The AI reasons over 
provided context rather than training data memory.

### SQL extraction from AI response
AI doesn't always return clean SQL. Used regex to extract SQL from 
markdown code blocks:
re.search(r"```sql\n(.*?)```", text, re.DOTALL)
Fallback: find any SELECT....; pattern if code block missing.

---

## Phase 3: KPI Dashboard

### What I built
4 live metric cards at the top of the app pulling from real SQL:
- Total Revenue: R$19,929,267
- Total Orders: 96,514  
- Average Review Score: 4.03/5
- Average Delivery Days: 12.0

### Why KPIs first
KPIs give instant context. A business user landing on the dashboard 
immediately understands scale before asking any questions. This is 
standard BI dashboard design — summary at top, detail below.

### Note on order count
master_orders has 115,607 rows but only 96,514 distinct orders — because 
orders with multiple items have one row per item. COUNT(DISTINCT order_id) 
gives the correct order count.

---

## Phase 4: Structured AI Insights

### Problem with original insight prompt
The AI was just repeating the SQL result: "São Paulo generated the 
highest revenue." Descriptive but not useful for business decisions.

### Solution: Role + Structure prompting
New prompt assigns role ("Senior Business Analyst presenting to management") 
and forces exactly 3 sections:
- WHAT: What does the data show? (specific numbers)
- WHY: Why might this have happened? (business reasoning)  
- ACTION: What should management do? (concrete recommendation)

Result: AI now interprets data rather than just describing it. Same data, 
completely different business value.

### Concept learned: Structured Output Prompting
Forcing a specific output structure (numbered sections with labels) makes 
AI responses consistent and scannable. Without structure, AI varies its 
format every time making the UI unpredictable.

---

## Phase 5: Plotly Auto-Charts

### What I built
Automatic chart generation based on question type:
- Time-related keywords (month, trend, yearly) → Line chart
- Categorical + numeric columns → Horizontal bar chart
- Numeric only → Vertical bar chart
- Single row result → No chart

### Problem: SP missing from revenue chart
Showed df.head(15) before sorting — SP (highest value) was being cut off. 
Fix: always sort_values(ascending=False) before head(). 
Lesson: order of operations matters — slice after sort, never before.

### Problem: Text formatting in AI insight
Numbers running together with text (7,502,898.23Thisrepresents...). 
Caused by AI adding markdown italic formatting mid-sentence. Fix: 
explicitly instruct "Do not use any markdown formatting, plain text only."

### Concept learned: Auto-visualization logic
Chart type selection uses keyword matching on the question text. This is 
a simple but effective form of intent detection — understanding what the 
user wants to see based on what they asked, without requiring them to 
specify chart type manually. More sophisticated versions use NLP 
classification.

### Plotly Express key patterns
- px.bar() for bar charts — orientation='h' for horizontal
- px.line() for time series — markers=True shows data points
- color_continuous_scale for gradient coloring
- transparent backgrounds: plot_bgcolor="rgba(0,0,0,0)"

---

## Phase 6: Executive Summary

### What I built
A single button that:
1. Automatically runs 8 SQL queries (revenue, orders, review score, 
   delivery, top state, top category, delay%, worst state)
2. Injects all results as structured facts into an LLM prompt
3. AI generates a 4-section executive report: Performance, Strengths, 
   Risks, Recommendations

### Key architectural decision: Separate calculation from reasoning
SQL handles all number calculation. AI only does interpretation and 
recommendation. This means:
- Numbers are always accurate (SQL is deterministic)
- AI focuses on what it's good at (reasoning, not arithmetic)
- No hallucinated statistics — every number in the report came from a query

This pattern is called "grounded generation" — grounding AI output in 
retrieved facts rather than model memory.

### Result
Executive summary correctly identified:
- R$19.9M total revenue
- SP as top state (R$7.5M)
- 7.5% delayed orders as a risk
- RR state (27.9 avg delivery days) as worst performer
- Concrete recommendations tied to specific data points

### Concept learned: Prompt Chaining
The executive summary uses a chain:
Query 1 → Query 2 → ... → Query 8 → Combine results → Single LLM call
Each query feeds into the final prompt as context. More complex chains 
would feed output of one LLM call into input of the next.

---

## Technical Stack Summary
- Python 3.13
- Streamlit — web framework
- SQLite — portable local database  
- Groq API (LLaMA 3.3 70B) — LLM for SQL generation and insights
- Plotly Express — interactive charts
- Pandas — data manipulation
- python-dotenv — environment variable management
- SQLAlchemy — database connection for data export

## Roadmap (upcoming)
- [ ] Delivery delay prediction ML model (RandomForest classifier)
- [ ] Review score prediction ML model (multiclass classification)
- [ ] Pre-built analysis buttons (Revenue/Delivery/Customer analysis)
- [ ] Multi-step reasoning for "why" questions
- [ ] Streamlit Cloud deployment