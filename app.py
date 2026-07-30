import os

# Must be set before torch/sentence-transformers get imported anywhere down
# the chain (via agents.rag_agent). Works around a known macOS + conda issue
# where Intel MKL's OpenMP runtime (loaded by numpy/scipy) and PyTorch's own
# OpenMP runtime both try to initialize in the same process, which can
# segfault instead of raising a catchable error - especially when the model
# call happens off the main thread (as it does here, inside the
# orchestrator's thread pool).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# joblib (used by scikit-learn's predict/predict_proba) defaults to a
# multiprocessing backend (loky) that spawns worker subprocesses. On macOS,
# spawning those from inside a thread Streamlit already manages can segfault
# instead of raising a catchable error. Forcing joblib to stay
# single-process/sequential avoids that - inference on one row is fast
# enough that this costs nothing noticeable.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd
import sqlite3
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from agents.ml_agent import MLAgent
from agents.planner_agent import PlannerAgent
from agents.rag_agent import RAGAgent
from agents.report_agent import ReportAgent
from agents.sql_agent import SQLAgent
from agents.viz_agent import VizAgent
from orchestrator import BIOrchestrator

page = st.sidebar.radio("Navigation", ["\U0001F4CA Dashboard", "\U0001F916 ML Predictions"])

if page == "\U0001F916 ML Predictions":
    exec(open("predict.py").read())
    st.stop()

load_dotenv()

DB_PATH = "data/olist.db"


@st.cache_resource
def get_pipeline():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    sql_agent = SQLAgent(client, DB_PATH)
    planner = PlannerAgent(client)
    ml_agent = MLAgent()
    report_agent = ReportAgent(client)
    viz_agent = VizAgent()

    rag_agent = None
    rag_error = None
    try:
        rag_agent = RAGAgent(client)
    except ImportError as e:
        rag_error = str(e)

    orchestrator = BIOrchestrator(planner, sql_agent, ml_agent, rag_agent, viz_agent, report_agent)
    return orchestrator, sql_agent, report_agent, rag_error


orchestrator, sql_agent, report_agent, rag_error = get_pipeline()

st.set_page_config(page_title="AI Business Intelligence Platform", page_icon="\U0001F4CA", layout="wide")
st.title("\U0001F4CA AI Business Intelligence Platform")
st.subheader("Powered by Olist E-Commerce Data")
if rag_error:
    st.sidebar.warning(rag_error)
st.markdown("---")


def get_kpis():
    conn = sqlite3.connect(DB_PATH)
    total_revenue = pd.read_sql_query(
        "SELECT ROUND(SUM(payment_value), 2) as value FROM master_orders", conn
    ).iloc[0]["value"]
    total_orders = pd.read_sql_query(
        "SELECT COUNT(DISTINCT order_id) as value FROM master_orders", conn
    ).iloc[0]["value"]
    avg_review = pd.read_sql_query(
        "SELECT ROUND(AVG(review_score), 2) as value FROM master_orders", conn
    ).iloc[0]["value"]
    avg_delivery = pd.read_sql_query(
        "SELECT ROUND(AVG(delivery_days), 1) as value FROM master_orders WHERE delivery_days > 0", conn
    ).iloc[0]["value"]
    conn.close()
    return total_revenue, total_orders, avg_review, avg_delivery


schema = sql_agent.get_schema()

with st.sidebar:
    st.markdown("### Database Schema")
    st.code(schema)
    st.markdown("### \U0001F4A1 Try asking:")
    st.markdown("- What is the revenue by state?")
    st.markdown("- What is our return policy?")
    st.markdown("- Which categories have the most delayed orders?")
    st.markdown("- Will an order from SP to AM likely be delayed?")
    st.markdown("- How is the review score calculated?")

st.markdown("### Ask a question about your business")
user_question = st.text_input(
    "",
    placeholder="e.g. Which state generated the highest revenue? What's our delivery SLA?",
)

if user_question:
    with st.spinner("Planning and running agents..."):
        state = orchestrator.handle(user_question, schema)

    with st.expander("\U0001F9ED Agent plan", expanded=False):
        for task in state["plan"]:
            st.markdown(f"- **{task['agent']}**: {task['instruction']}")

    for err in state["errors"]:
        st.warning(err)

    sql_result = state["results"].get("sql")
    if sql_result and sql_result.get("success"):
        with st.expander("View generated SQL"):
            st.code(sql_result["sql"], language="sql")
        with st.expander("\U0001F4CB View raw data table"):
            st.dataframe(sql_result["dataframe"], use_container_width=True)
    elif sql_result:
        st.error("SQL agent: " + sql_result.get("error", "unknown error"))

    if state.get("chart") is not None:
        st.plotly_chart(state["chart"], use_container_width=True)

    rag_result = state["results"].get("rag")
    if rag_result and rag_result.get("success"):
        with st.expander("\U0001F4DA Retrieved sources"):
            st.markdown(", ".join(rag_result["sources"]))

    ml_result = state["results"].get("ml")
    if ml_result and not ml_result.get("success"):
        st.info("ML agent: " + ml_result.get("error", "unknown error"))

    st.markdown("### AI Business Insight")
    st.success(state["final_report"])

st.markdown("---")
st.markdown("### \U0001F4C8 Key Performance Indicators")
total_revenue, total_orders, avg_review, avg_delivery = get_kpis()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("\U0001F4B0 Total Revenue", f"R$ {total_revenue:,.0f}")
with col2:
    st.metric("\U0001F4E6 Total Orders", f"{int(total_orders):,}")
with col3:
    st.metric("\u2B50 Avg Review Score", f"{avg_review} / 5")
with col4:
    st.metric("\U0001F69A Avg Delivery Days", f"{avg_delivery} days")

st.markdown("---")
st.markdown("### \U0001F3AF Executive Summary")
if st.button("\U0001F4CB Generate Executive Summary", type="primary"):
    with st.spinner("Analyzing all KPIs and generating executive summary..."):
        kpis = {
            "Total Revenue": f"R$ {total_revenue}",
            "Total Orders": int(total_orders),
            "Average Review Score": f"{avg_review}/5",
            "Average Delivery Time": f"{avg_delivery} days",
        }
        summary = report_agent.executive_summary(kpis)
    st.markdown("#### \U0001F4CA Business Intelligence Report")
    st.info(summary)

st.caption("AI-BI Platform | Olist Dataset | Streamlit + Groq | Multi-agent architecture")
