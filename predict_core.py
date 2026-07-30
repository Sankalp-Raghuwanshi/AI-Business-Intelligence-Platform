"""
predict_core.py

Callable wrapper around the trained models used by predict.py's Streamlit
page (delay-risk classifier + review-score classifier), so the ML agent can
call the same models via one plain function instead of duplicating logic.
predict.py's tabs keep using their own form inputs exactly as before -
nothing there needs to change.

The orchestrator hands this a free-text question, not pre-filled form
fields, so this module uses the Groq LLM to extract whichever order
attributes are mentioned. Anything not mentioned falls back to a dataset
average/most-common value pulled from olist.db, so a vague question still
gets a best-effort prediction instead of failing outright.
"""

import json
import os

# Defensive: same workaround as app.py, in case this module gets imported
# or run standalone before app.py's own os.environ.setdefault calls run.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import re

# Force scikit-learn / joblib to run single-threaded. Without this, some
# models (e.g. RandomForest trained with n_jobs=-1) spawn a loky
# multiprocessing worker on .predict(), which segfaults on macOS when it
# forks a process while torch/tokenizers (loaded by the RAG agent) already
# have native threads running. Must be set before sklearn/joblib import
# anything that reads it.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import joblib
import pandas as pd
import sqlite3
from dotenv import load_dotenv
from groq import Groq
from joblib import parallel_backend

load_dotenv()

DB_PATH = "data/olist.db"
_MODEL_NAME = "llama-3.3-70b-versatile"

_client = None
_models = None
_defaults = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def _load_models() -> dict:
    global _models
    if _models is None:
        _models = {
            "delay_model": joblib.load("models/delay_model.pkl"),
            "review_model": joblib.load("models/review_model.pkl"),
            "le_customer_state": joblib.load("models/le_customer_state.pkl"),
            "le_seller_state": joblib.load("models/le_seller_state.pkl"),
            "le_category": joblib.load("models/le_category.pkl"),
        }
    return _models


def _load_defaults() -> dict:
    """Dataset averages / most-common values, used to fill in anything the
    question doesn't mention so a vague question still gets an answer."""
    global _defaults
    if _defaults is None:
        conn = sqlite3.connect(DB_PATH)
        row = pd.read_sql_query(
            """
            SELECT
                ROUND(AVG(price), 2) AS price,
                ROUND(AVG(freight_value), 2) AS freight_value,
                ROUND(AVG(payment_value), 2) AS payment_value,
                ROUND(AVG(delivery_days), 0) AS delivery_days
            FROM master_orders
            WHERE price IS NOT NULL
            """,
            conn,
        ).iloc[0]
        top_state = pd.read_sql_query(
            "SELECT customer_state FROM master_orders GROUP BY customer_state "
            "ORDER BY COUNT(*) DESC LIMIT 1",
            conn,
        ).iloc[0]["customer_state"]
        top_category = pd.read_sql_query(
            "SELECT category FROM master_orders WHERE category IS NOT NULL "
            "GROUP BY category ORDER BY COUNT(*) DESC LIMIT 1",
            conn,
        ).iloc[0]["category"]
        conn.close()
        _defaults = {
            "customer_state": top_state,
            "seller_state": top_state,
            "category": top_category,
            "price": float(row["price"]),
            "freight_value": float(row["freight_value"]),
            "payment_value": float(row["payment_value"]),
            "order_month": 6,
            "order_year": 2018,
            "delivery_days": int(row["delivery_days"]),
            "delay_flag": 0,
        }
    return _defaults


_EXTRACTION_PROMPT = """Extract order attributes mentioned in the question below.
Return ONLY valid JSON, no explanation, with this exact shape (use null for
anything not mentioned):
{{
  "target": "delay" or "review",
  "customer_state": "<2-letter state code or null>",
  "seller_state": "<2-letter state code or null>",
  "category": "<product category or null>",
  "price": <number or null>,
  "freight_value": <number or null>,
  "payment_value": <number or null>,
  "order_month": <1-12 or null>,
  "order_year": <2017 or 2018 or null>,
  "delivery_days": <number or null>,
  "delay_flag": <0 or 1 or null>
}}

"target" is "delay" if the question asks about delivery delay/lateness
risk, or "review" if it asks about predicted review score/satisfaction.
Default to "delay" if unclear.

Question: {question}
"""


def _extract_params(question: str) -> dict:
    client = _get_client()
    response = client.chat.completions.create(
        model=_MODEL_NAME,
        messages=[{"role": "user", "content": _EXTRACTION_PROMPT.format(question=question)}],
        max_tokens=300,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}


def _predict_delay(params: dict, defaults: dict, models: dict) -> dict:
    customer_state = params.get("customer_state") or defaults["customer_state"]
    seller_state = params.get("seller_state") or defaults["seller_state"]
    category = params.get("category") or defaults["category"]
    price = params.get("price") or defaults["price"]
    freight_value = params.get("freight_value") or defaults["freight_value"]
    payment_value = params.get("payment_value") or defaults["payment_value"]
    order_month = params.get("order_month") or defaults["order_month"]
    order_year = params.get("order_year") or defaults["order_year"]

    freight_ratio = freight_value / (price + 1)
    is_holiday_season = 1 if order_month in [11, 12] else 0
    seller_customer_same = 1 if customer_state == seller_state else 0

    cs_encoded = models["le_customer_state"].transform([customer_state])[0]
    ss_encoded = models["le_seller_state"].transform([seller_state])[0]
    cat_encoded = models["le_category"].transform([category])[0]

    input_data = pd.DataFrame([[
        cs_encoded, ss_encoded, cat_encoded,
        price, freight_value, payment_value,
        order_month, order_year,
        freight_ratio, is_holiday_season, seller_customer_same,
    ]], columns=[
        "customer_state", "seller_state", "category",
        "price", "freight_value", "payment_value",
        "order_month", "order_year",
        "freight_ratio", "is_holiday_season", "seller_customer_same_state",
    ])

    prediction = models["delay_model"].predict(input_data)[0]
    probability = models["delay_model"].predict_proba(input_data)[0]

    if prediction == 1:
        pct = probability[1] * 100
        explanation = (
            f"High delay risk ({pct:.1f}% probability) for a {category} order "
            f"from {seller_state} to {customer_state}."
        )
    else:
        pct = probability[0] * 100
        explanation = (
            f"Low delay risk ({pct:.1f}% probability of on-time delivery) for a "
            f"{category} order from {seller_state} to {customer_state}."
        )

    return {
        "prediction": "delayed" if prediction == 1 else "on_time",
        "probability": round(float(max(probability)), 3),
        "explanation": explanation,
        "inputs_used": {
            "customer_state": customer_state, "seller_state": seller_state,
            "category": category, "price": price, "freight_value": freight_value,
            "payment_value": payment_value, "order_month": order_month,
            "order_year": order_year,
        },
    }


def _predict_review(params: dict, defaults: dict, models: dict) -> dict:
    delivery_days = params.get("delivery_days") or defaults["delivery_days"]
    delay_flag = params.get("delay_flag")
    if delay_flag is None:
        delay_flag = defaults["delay_flag"]
    price = params.get("price") or defaults["price"]
    freight_value = params.get("freight_value") or defaults["freight_value"]
    payment_value = params.get("payment_value") or defaults["payment_value"]
    order_month = params.get("order_month") or defaults["order_month"]

    freight_ratio = freight_value / (price + 1)
    is_holiday = 1 if order_month in [11, 12] else 0

    input_data = pd.DataFrame([[
        delivery_days, delay_flag, freight_value,
        price, payment_value, order_month,
        freight_ratio, is_holiday,
    ]], columns=[
        "delivery_days", "delay_flag", "freight_value",
        "price", "payment_value", "order_month",
        "freight_ratio", "is_holiday_season",
    ])

    prediction = models["review_model"].predict(input_data)[0]
    probability = models["review_model"].predict_proba(input_data)[0]
    best_prob = max(probability)

    explanation = (
        f"Predicted review score bucket: {prediction} ({best_prob:.1%} confidence), "
        f"based on a {delivery_days}-day delivery"
        + (" that was delayed." if delay_flag else " that was on time.")
    )

    return {
        "prediction": str(prediction),
        "probability": round(float(best_prob), 3),
        "explanation": explanation,
        "inputs_used": {
            "delivery_days": delivery_days, "delay_flag": delay_flag,
            "price": price, "freight_value": freight_value,
            "payment_value": payment_value, "order_month": order_month,
        },
    }


def run_prediction(question: str) -> dict:
    models = _load_models()
    defaults = _load_defaults()
    params = _extract_params(question)

    target = params.get("target") or "delay"
    if target == "review":
        return _predict_review(params, defaults, models)
    return _predict_delay(params, defaults, models)
