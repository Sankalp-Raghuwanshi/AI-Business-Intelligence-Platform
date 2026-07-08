import streamlit as st
import sqlite3
import plotly.express as px
import pandas as pd
from groq import Groq
import re
from dotenv import load_dotenv
import os

# Page navigation
page = st.sidebar.radio("Navigation", ["📊 Dashboard", "🤖 ML Predictions"])

if page == "🤖 ML Predictions":
    exec(open("predict.py").read())
    st.stop()

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

st.set_page_config(page_title="AI Business Intelligence Platform", page_icon="📊", layout="wide")
st.title("📊 AI Business Intelligence Platform")
st.subheader("Powered by Olist E-Commerce Data")
st.markdown("---")

def get_connection():
    return sqlite3.connect("data/olist.db")

def get_schema():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(master_orders)")
    columns = cursor.fetchall()
    conn.close()
    schema = "Table: master_orders, Columns: "
    for col in columns:
        schema += col[1] + " (" + col[2] + "), "
    return schema

def get_kpis():
    conn = get_connection()
    
    total_revenue = pd.read_sql_query(
        "SELECT ROUND(SUM(payment_value), 2) as value FROM master_orders", conn
    ).iloc[0]['value']
    
    total_orders = pd.read_sql_query(
        "SELECT COUNT(DISTINCT order_id) as value FROM master_orders", conn
    ).iloc[0]['value']
    
    avg_review = pd.read_sql_query(
        "SELECT ROUND(AVG(review_score), 2) as value FROM master_orders", conn
    ).iloc[0]['value']
    
    avg_delivery = pd.read_sql_query(
        "SELECT ROUND(AVG(delivery_days), 1) as value FROM master_orders WHERE delivery_days > 0", conn
    ).iloc[0]['value']
    
    conn.close()
    return total_revenue, total_orders, avg_review, avg_delivery

def run_query(sql):
    try:
        conn = get_connection()
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)

def extract_sql(text):
    match = re.search(r"```sql\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"SELECT.*?;", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()
    return None

def generate_sql(question, schema):
    prompt = "You are a SQL expert. Database schema: " + schema + " Write a SQLite SQL query to answer: " + question + " Rules: table name is master_orders, use payment_value for revenue, use customer_state for state. Return ONLY the SQL query inside ```sql ``` blocks." + "For monthly trend questions, always group by both order_year and order_month together and order by order_year, order_month. "
    return call_llm(prompt)

def generate_insight(question, df_result):
    prompt = (
        "You are a Senior Business Analyst presenting to the management team. "
        "The user asked: " + question + " "
        "The SQL query returned this data: " + df_result.head(10).to_string() + " "
        "Provide a structured response with exactly these three parts: "
        "1. WHAT: What does this data show? (1 sentence with specific numbers) "
        "2. WHY: Why might this have happened? (1 sentence of business reasoning) "
        "3. ACTION: What should management do? (1 concrete recommendation) "
        "Keep total response under 150 words. Be specific, not generic."
        "Do not use any markdown formatting like asterisks or underscores. Plain text only. "
    )
    return call_llm(prompt)

def decompose_question(question, schema):
    prompt = (
        "You are a data analyst. A user asked this broad business question: " + question + " "
        "Break this into exactly 3 specific SQL-answerable sub-questions about the database. "
        "The database schema is: " + schema + " "
        "Return ONLY a numbered list like this format: "
        "1. [specific question] "
        "2. [specific question] "
        "3. [specific question] "
        "Each question must be answerable with a single SQL query on master_orders table. "
        "No explanations, just the 3 numbered questions."
    )
    return call_llm(prompt)

def extract_questions(text):
    lines = text.strip().split('\n')
    questions = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and '.' in line:
            question = line.split('.', 1)[1].strip()
            if question:
                questions.append(question)
    return questions[:3]

def deep_analysis(question, schema):
    results_context = ""
    sub_questions = []
    
    # Step 1: Decompose into sub-questions
    decomposed = decompose_question(question, schema)
    sub_questions = extract_questions(decomposed)
    
    if not sub_questions:
        return None, [], "Could not decompose question"
    
    # Step 2: Run SQL for each sub-question
    retrieved_data = []
    for i, sub_q in enumerate(sub_questions):
        sql_response = generate_sql(sub_q, schema)
        sql_query = extract_sql(sql_response)
        
        if sql_query:
            df, error = run_query(sql_query)
            if df is not None and len(df) > 0:
                retrieved_data.append({
                    'question': sub_q,
                    'sql': sql_query,
                    'result': df
                })
                results_context += f"\nSub-question {i+1}: {sub_q}\n"
                results_context += f"Data: {df.head(5).to_string()}\n"
    
    if not results_context:
        return None, sub_questions, "No data retrieved"
    
    # Step 3: Synthesize comprehensive answer
    synthesis_prompt = (
        "You are a Senior Business Analyst. A user asked: " + question + " "
        "You retrieved the following data from the database to answer it: "
        + results_context +
        "Write a comprehensive 4-6 sentence analysis that: "
        "1. Directly answers the user's question with specific numbers "
        "2. Explains the root causes based on the data "
        "3. Identifies the most important pattern across all retrieved data "
        "4. Gives one concrete business recommendation "
        "Use plain text only, no markdown symbols or bullet points. Be specific and data-driven."
    )
    
    synthesis = call_llm(synthesis_prompt)
    return retrieved_data, sub_questions, synthesis

def generate_executive_summary():
    conn = get_connection()
    
    # Run all KPI queries
    total_revenue = pd.read_sql_query(
        "SELECT ROUND(SUM(payment_value), 2) as value FROM master_orders", conn
    ).iloc[0]['value']
    
    total_orders = pd.read_sql_query(
        "SELECT COUNT(DISTINCT order_id) as value FROM master_orders", conn
    ).iloc[0]['value']
    
    avg_review = pd.read_sql_query(
        "SELECT ROUND(AVG(review_score), 2) as value FROM master_orders", conn
    ).iloc[0]['value']
    
    avg_delivery = pd.read_sql_query(
        "SELECT ROUND(AVG(delivery_days), 1) as value FROM master_orders WHERE delivery_days > 0", conn
    ).iloc[0]['value']
    
    top_state = pd.read_sql_query(
        "SELECT customer_state, ROUND(SUM(payment_value), 2) as revenue FROM master_orders GROUP BY customer_state ORDER BY revenue DESC LIMIT 1", conn
    ).iloc[0]
    
    top_category = pd.read_sql_query(
        "SELECT category, COUNT(*) as orders FROM master_orders WHERE category IS NOT NULL GROUP BY category ORDER BY orders DESC LIMIT 1", conn
    ).iloc[0]
    
    delayed_pct = pd.read_sql_query(
        "SELECT ROUND(100.0 * SUM(delay_flag) / COUNT(*), 1) as pct FROM master_orders", conn
    ).iloc[0]['pct']
    
    worst_state = pd.read_sql_query(
        "SELECT customer_state, ROUND(AVG(delivery_days), 1) as avg_days FROM master_orders WHERE delivery_days > 0 GROUP BY customer_state ORDER BY avg_days DESC LIMIT 1", conn
    ).iloc[0]
    
    conn.close()
    
    prompt = (
        "You are the Head of Business Intelligence presenting to the CEO. "
        "Here are the company KPIs: "
        "Total Revenue: R$" + str(total_revenue) + ", "
        "Total Orders: " + str(int(total_orders)) + ", "
        "Average Review Score: " + str(avg_review) + " out of 5, "
        "Average Delivery Time: " + str(avg_delivery) + " days, "
        "Top State by Revenue: " + str(top_state['customer_state']) + " with R$" + str(top_state['revenue']) + ", "
        "Top Category by Orders: " + str(top_category['category']) + ", "
        "Delayed Orders: " + str(delayed_pct) + "% of all orders, "
        "Worst Delivery State: " + str(worst_state['customer_state']) + " averaging " + str(worst_state['avg_days']) + " days. "
        "Write a professional executive summary with exactly these 4 sections: "
        "PERFORMANCE: Overall business performance in 2 sentences. "
        "STRENGTHS: Top 2 strengths with specific numbers. "
        "RISKS: Top 2 risks or concerns with specific numbers. "
        "RECOMMENDATIONS: 2 concrete actionable recommendations for management. "
        "Use plain text only, no markdown symbols. Maximum 250 words. Be specific and data-driven."
    )
    
    return call_llm(prompt)

def generate_chart(df, question):
    # Detect chart type based on data shape and question keywords
    question_lower = question.lower()
    
    # If only 1 row returned, no chart needed
    if len(df) <= 1:
        return None
    
    cols = df.columns.tolist()
    
    # Find numeric and text columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols:
        return None
    
    numeric_col = numeric_cols[0]
    
    # Time series — line chart
    time_keywords = ['month', 'year', 'trend', 'over time', 'monthly', 'yearly', 'daily']
    if any(word in question_lower for word in time_keywords):
        if len(cols) >= 2:
            fig = px.line(
                df,
                x=cols[0],
                y=numeric_col,
                title="📈 " + question.title(),
                markers=True,
                color_discrete_sequence=["#00C9A7"]
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white"
            )
            return fig
    
    # Categorical comparison — horizontal bar chart
    if text_cols and numeric_cols:
        fig = px.bar(
            df.sort_values(by=numeric_col, ascending=False).head(15),
            x=numeric_col,
            y=text_cols[0],
            orientation='h',
            title="📊 " + question.title(),
            color=numeric_col,
            color_continuous_scale="Teal"
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            yaxis={'categoryorder': 'total ascending'}
        )
        return fig
    
    # Pure numeric — vertical bar
    if len(numeric_cols) >= 1:
        fig = px.bar(
            df.sort_values(by=numeric_col, ascending=False).head(15),
            x=cols[0],
            y=numeric_col,
            title="📊 " + question.title(),
            color=numeric_col,
            color_continuous_scale="Teal"
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        return fig
    
    return None

schema = get_schema()

with st.sidebar:
    st.markdown("### Database Schema")
    st.code(schema)
    st.markdown("### 💡 Try asking:")
    st.markdown("- What is the revenue by state?")
    st.markdown("- What is the average review score by category?")
    st.markdown("- What is the monthly order trend in 2018?")
    st.markdown("- Which categories have the most delayed orders?")
    st.markdown("- What is the average delivery days by state?")
    st.markdown("- What is the total orders by category?")


st.markdown("### Ask a question about your data")
# Deep Analysis / RAG Section
st.markdown("### 🔬 Deep Analysis")
st.markdown("Ask broad business questions — AI retrieves relevant data automatically and synthesizes a comprehensive answer.")

deep_question = st.text_input(
    "",
    placeholder="e.g. Why are customers in Amazonas unhappy? What is driving revenue growth?",
    key="deep_q"
)

if deep_question:
    with st.spinner("🔍 Step 1: Breaking down your question..."):
        retrieved_data, sub_questions, synthesis = deep_analysis(deep_question, schema)
    
    if sub_questions:
        st.markdown("**🧩 Question decomposed into:**")
        for i, q in enumerate(sub_questions):
            st.markdown(f"{i+1}. {q}")
    
    if retrieved_data:
        with st.expander("📊 Retrieved Data (click to expand)", expanded=False):
            for item in retrieved_data:
                st.markdown(f"**{item['question']}**")
                st.dataframe(item['result'], use_container_width=True)
                st.markdown("---")
    
    if synthesis and synthesis != "No data retrieved" and synthesis != "Could not decompose question":
        st.markdown("### 🧠 Comprehensive Analysis")
        st.info(synthesis)
    else:
        st.warning("Could not retrieve enough data. Try rephrasing your question.")

st.markdown("---")
# KPI Cards
st.markdown("### 📈 Key Performance Indicators")
total_revenue, total_orders, avg_review, avg_delivery = get_kpis()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💰 Total Revenue", f"R$ {total_revenue:,.0f}")
with col2:
    st.metric("📦 Total Orders", f"{int(total_orders):,}")
with col3:
    st.metric("⭐ Avg Review Score", f"{avg_review} / 5")
with col4:
    st.metric("🚚 Avg Delivery Days", f"{avg_delivery} days")

st.markdown("---")
# Executive Summary Button
st.markdown("### 🎯 Executive Summary")
if st.button("📋 Generate Executive Summary", type="primary"):
    with st.spinner("Analyzing all KPIs and generating executive summary..."):
        summary = generate_executive_summary()
    st.markdown("#### 📊 Business Intelligence Report")
    st.info(summary)

st.markdown("---")
user_question = st.text_input("", placeholder="e.g. Which state generated the highest revenue?")

if user_question:
    st.markdown("**Your question:** " + user_question)
    with st.spinner("Generating SQL..."):
        sql_response = generate_sql(user_question, schema)
        sql_query = extract_sql(sql_response)
    if sql_query:
        with st.expander("View Generated SQL"):
            st.code(sql_query, language="sql")
        df, error = run_query(sql_query)
        if error:
            st.error("SQL Error: " + error)
        elif df is not None and len(df) > 0:
            st.markdown("### 📊 Results")

        # Auto generate chart
            chart = generate_chart(df, user_question)
            if chart:
              st.plotly_chart(chart, use_container_width=True)

        # Show data table below chart
            with st.expander("📋 View Raw Data Table"):
                st.dataframe(df, use_container_width=True)
            with st.spinner("Generating insight..."):
                insight = generate_insight(user_question, df)
            st.markdown("### AI Business Insight")
            st.success(insight)
        else:
            st.warning("No results. Try rephrasing.")
    else:
        st.error("Could not generate SQL. Try rephrasing.")

st.caption("AI-BI Platform | Olist Dataset | Streamlit + Groq")